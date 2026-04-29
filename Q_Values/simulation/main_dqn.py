#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真主程序
"""

import logging
import os
from datetime import timedelta
from itertools import count
from tqdm import tqdm

import pandas as pd

from core import CourierSimulator
from core import OrderGenerator
from core import RoutePlanner
from core import StateRefresher
from core import OrderDispatcherByActorCritic
from dqn import ActorCriticConfig
from utils import haversine_distance
import json

# 配置日志
logging.basicConfig(
    level=logging.CRITICAL,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class DeliverySimulation:
    def __init__(self, order_data_path, idle_timeout_minutes=30):
        # 加载数据
        self.df = pd.read_csv(order_data_path)
        self.df = self.df.dropna(subset=['order_id'])
        self.df = self.df.dropna(subset=['courier_id'])
        self.df['order_id'] = self.df['order_id'].astype(int).astype(str)
        self.df["courier_id"] = self.df["courier_id"].astype(int).astype(str)
        if not pd.api.types.is_datetime64_any_dtype(self.df['dispatch_time']):
            self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['arrive_time']):
            self.df['arrive_time'] = pd.to_datetime(self.df['arrive_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['estimate_meal_prepare_time']):
            self.df['estimate_meal_prepare_time'] = pd.to_datetime(self.df['estimate_meal_prepare_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['estimate_arrived_time']):
            self.df['estimate_arrived_time'] = pd.to_datetime(self.df['estimate_arrived_time'])
        
        # 初始化生成器
        self.is_train = True
        self.order_gen = OrderGenerator(self.df)
        self.courier_sim = CourierSimulator(self.df, idle_timeout_minutes)
        # 初始化派单器、路由规划器、状态刷新器
        self.state_refresher = StateRefresher()
        self.route_planner = RoutePlanner()
        self.dispatcher_dqn = OrderDispatcherByActorCritic(self.route_planner, ActorCriticConfig())
        
        self.model_path = None
        
        
        # 构建骑手池
        self.couriers = {}
        # 初始化骑手池
        self._init_couriers()
        # 构建订单池
        self.orders = self.order_gen.orders_by_time
        self.orders = {order['order_id']: order for orders in self.order_gen.orders_by_time.values() for order in orders}
        
        # 时间轴管理
        # 获取最早的骑手上线时间作为起始时间
        earliest_event_time = self.courier_sim.get_earliest_event_time()
        self.current_time = earliest_event_time
        
        # 待分配订单池
        self.order_pool = 0
        self.pending_orders = []
        
        # logger.info(f"仿真初始化完成，起始时间: {self.current_time}")
    
    def _init_couriers(self) -> None:
        """初始化所有骑手状态，默认全部离线"""
        all_couriers = self.courier_sim.get_all_courier_ids()
        for cid in all_couriers:
            self.couriers[cid] = {
                'courier_id': cid,
                'status': 'offline',
                'position': None,
                'order_increased': False,
                'current_orders': [],
                'route': [],
                'has_had_orders': False
            }
    
    def step(self) -> None:
        """执行一步仿真"""
        # 保存当前时间（用于返回）
        current_time = self.current_time
        # 重置当前时间步的新订单数量
        self.order_pool = 0

        for courier_id, courier in self.couriers.items():
            # 1. 刷新骑手状态(如果当前有订单)
            if courier.get('current_orders'):
                updates = self.state_refresher.refresh_courier(
                    courier, 
                    current_time,
                )
                if updates.get('courier_updates'):
                    self.couriers[courier_id].update(updates['courier_updates'])

                # 记录已完成订单
                completed_orders = []
                for order in courier['current_orders']:
                    if order['status'] == 'delivered':
                        completed_orders.append(order)
                # 如果订单有对应的骑手记录，可以在这里更新订单状态
                for order in completed_orders:
                    if order['order_id'] in self.orders:
                        self.orders[order['order_id']]['actual_pickup_time'] = pd.to_datetime(order['actual_pickup_time'])
                        self.orders[order['order_id']]['actual_delivery_time'] = pd.to_datetime(order['actual_delivery_time'])
                    # 从骑手当前订单中移除已完成的订单
                    courier['current_orders'].remove(order)
                
        # 2. 处理当前时间点的骑手上下线事件
        courier_events = self.courier_sim.get_events_at_time(self.current_time)
        for cid, etype, pos in courier_events:
            self.couriers[cid]['status'] = etype
            if etype == 'online':
                self.couriers[cid]['position'] = pos
                self.couriers[cid]['has_had_orders'] = False
        online_couriers = {cid: info for cid, info in self.couriers.items() if info['status'] == 'online'}
        # logger.info(f"当前时间点 {self.current_time} 的在线骑手数量: {len(online_couriers)}")
        
        # 3. 处理当前时间点的订单生成事件
        new_orders = self.order_gen.get_orders_at_time(self.current_time)
        if new_orders:
            self.pending_orders.extend(new_orders)
            self.order_pool = len(new_orders)
        
        # 4. 如果 pending_orders 不为空，使用派单器将订单派给最近的骑手
        if self.pending_orders:
            # 派单给ActorCritic选择的骑手
            dispatch_results = self.dispatcher_dqn.dispatch_orders(self.pending_orders, online_couriers, current_time, self.is_train)
            # logger.debug(f"当前时间点 {self.current_time} 的订单派单结果: {dispatch_results}")
            # 处理分派结果
            dispatched_orders = []
            for order in self.pending_orders:
                if order['order_id'] in dispatch_results:
                    courier_id = dispatch_results[order['order_id']]
                    # 将订单添加到骑手的当前订单列表并更新状态
                    self.couriers[courier_id]['order_increased'] = True
                    self.couriers[courier_id]['has_had_orders'] = True
                    # 更新骑手池和订单池的配对情况
                    self.couriers[courier_id]['current_orders'].append(order)
                    self.orders[order['order_id']]['courier_id'] = courier_id
                    dispatched_orders.append(order)
            # 从待分配订单池中移除已分派的订单
            for order in dispatched_orders:
                self.pending_orders.remove(order)

        # 5. 规划路径
        for courier_id, courier in self.couriers.items():
            # 检查是否有订单增加
            if courier['order_increased']:
                route = self.route_planner.plan_route(courier)
                courier['route'] = route
                courier['order_increased'] = False

        done = self.courier_sim.get_next_event_time() is None
        
        # 5. 确定下一个事件时间，推进 current_time（在返回状态后再推进）
        if not done:
            self.current_time += timedelta(seconds=30)
        
        return None, 0, done, current_time


if __name__ == '__main__':
    data_path = r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv'
    model_path = 'Q_Values\\simulation\\checkpoints\\dqn_model.pth'

    sim = DeliverySimulation(data_path)

    if os.path.exists(model_path) and sim.is_train == False:
        sim.dispatcher_dqn.ac_agent.load_model(model_path)
        if sim.dispatcher_dqn.ac_agent.is_train != sim.is_train:
            if sim.is_train:
                sim.dispatcher_dqn.ac_agent.set_train_mode()
            else:
                sim.dispatcher_dqn.ac_agent.set_eval_mode()
        logger.critical(f"已加载模型: {model_path} | 模式: {'训练' if sim.is_train else '评估'}")
    else:
        logger.critical("未找到已保存的模型，从随机初始化开始")

    start_time, end_time = sim.courier_sim.get_earliest_event_time(), sim.courier_sim.get_latest_event_time()
    total_steps = int((end_time - start_time).total_seconds() / 30) if start_time and end_time else None
    
    TIME_STEP = 30
    STEPS_PER_DAY = 86400 // TIME_STEP
    TRAIN_EVERY_N_DAYS = 1

    last_train_step = 0

    day = 1
    day_labels = []
    daily_metrics = {
        'delivery_duration': [],
        'overtime_rate': [],
        'avg_overtime': [],
        'avg_revenue': []
    }

    i = 1
    for _ in tqdm(count() if not total_steps else range(total_steps), desc="仿真进度"):
        state, reward, done, current_time = sim.step()
        
        if i - last_train_step >= STEPS_PER_DAY * TRAIN_EVERY_N_DAYS and sim.is_train:
            orders_df = pd.DataFrame([o for o in sim.orders.values() if pd.notna(o.get('actual_delivery_time')) and o.get('actual_delivery_time') <= current_time])
            if len(orders_df) > 0:
                orders_df['dispatch_time'] = pd.to_datetime(orders_df['dispatch_time'])
                orders_df['estimate_arrived_time'] = pd.to_datetime(orders_df['estimate_arrived_time'])
                orders_df['actual_delivery_time'] = pd.to_datetime(orders_df['actual_delivery_time'])

                orders_df['revenue'] = 3 + orders_df.apply(
                    lambda row: haversine_distance(
                        (row['sender_lng'], row['sender_lat']),
                        (row['recipient_lng'], row['recipient_lat'])
                    ) / 1000, axis=1
                )

                delivered_orders = orders_df[orders_df['actual_delivery_time'].notna()].copy()
                delivered_orders['overtime_minutes'] = (delivered_orders['actual_delivery_time'] - delivered_orders['estimate_arrived_time']).dt.total_seconds() / 60
                delivered_orders['overtime_minutes'] = delivered_orders['overtime_minutes'].apply(lambda x: max(0, x))
                delivered_orders['delivery_duration'] = (delivered_orders['actual_delivery_time'] - delivered_orders['dispatch_time']).dt.total_seconds() / 60

                timeout_mask = delivered_orders['actual_delivery_time'] > delivered_orders['estimate_arrived_time']
                timeout_orders = delivered_orders[timeout_mask]

                overtime_rate = len(timeout_orders) / len(delivered_orders)
                avg_overtime = timeout_orders['overtime_minutes'].mean() if len(timeout_orders) > 0 else 0.0

                def apply_penalty(row):
                    overtime = row['overtime_minutes']
                    if overtime <= 10:
                        return row['revenue'] * 0.9
                    elif overtime <= 20:
                        return row['revenue'] * 0.6
                    else:
                        return row['revenue'] * 0.5

                delivered_orders['actual_revenue'] = delivered_orders.apply(apply_penalty, axis=1)
                avg_revenue = delivered_orders['actual_revenue'].mean()
                avg_delivery_duration = delivered_orders['delivery_duration'].mean()

                day_labels.append(day)
                daily_metrics['delivery_duration'].append(avg_delivery_duration)
                daily_metrics['overtime_rate'].append(overtime_rate)
                daily_metrics['avg_overtime'].append(avg_overtime)
                daily_metrics['avg_revenue'].append(avg_revenue)

                logger.critical(
                    f"===== 第 {day} 天结束，训练前指标 =====\n"
                    f"已送达订单数: {len(delivered_orders)}\n"
                    f"平均配送时长: {avg_delivery_duration:.2f}分钟\n"
                    f"超时率: {overtime_rate:.2%}\n"
                    f"平均超时时长: {avg_overtime:.2f}分钟\n"
                    f"骑手平均订单收益: {avg_revenue:.2f}")

            if len(sim.dispatcher_dqn.ac_agent.replay_buffer) >= sim.dispatcher_dqn.ac_agent.config.batch_size:
                logger.info(f"===== 第 {i // STEPS_PER_DAY} 天结束，训练网络 =====")
                sim.dispatcher_dqn.ac_agent.update_critic_offline()
                sim.dispatcher_dqn.ac_agent.update_actor_offline()
                last_train_step = i
            day += 1
        
        logger.info(
            f"Step {i}: time={current_time}\n"
            f"----------------------------------\n"
            f"order_pool={sim.order_pool}\n" 
            f"pending_orders={len(sim.pending_orders)}\n" 
            f"online_couriers={sum(1 for c in sim.couriers.values() if c['status']=='online')}\n"
            f"delivering_orders={sum(len(c['current_orders']) for c in sim.couriers.values())}\n"
            f"completed_orders={sum(1 for o in sim.orders.values() if o.get('actual_delivery_time') <= current_time)}\n"
            f"----------------------------------"
            )
        # if i == 20:
        if done:
            if len(sim.dispatcher_dqn.ac_agent.replay_buffer) >= sim.dispatcher_dqn.ac_agent.config.batch_size and sim.is_train:
                logger.info(f"===== 仿真结束，最终训练网络 =====")
                sim.dispatcher_dqn.ac_agent.update_critic_offline()
                sim.dispatcher_dqn.ac_agent.update_actor_offline()
            logger.info(f"仿真结束，总耗时: {current_time - sim.df['dispatch_time'].min()}")
            break
        i += 1

    if day_labels and sim.is_train:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))

            axes[0, 0].plot(day_labels, daily_metrics['delivery_duration'], marker='o', linestyle='-', color='tab:blue')
            axes[0, 0].set_xlabel('天数')
            axes[0, 0].set_ylabel('平均配送时长 (分钟)')
            axes[0, 0].set_title('每日平均配送时长')
            axes[0, 0].grid(True, alpha=0.3)

            axes[0, 1].plot(day_labels, daily_metrics['overtime_rate'], marker='s', linestyle='-', color='tab:red')
            axes[0, 1].set_xlabel('天数')
            axes[0, 1].set_ylabel('超时率')
            axes[0, 1].set_title('每日超时率')
            axes[0, 1].grid(True, alpha=0.3)

            axes[1, 0].plot(day_labels, daily_metrics['avg_overtime'], marker='^', linestyle='-', color='tab:orange')
            axes[1, 0].set_xlabel('天数')
            axes[1, 0].set_ylabel('平均超时时长 (分钟)')
            axes[1, 0].set_title('每日平均超时时长')
            axes[1, 0].grid(True, alpha=0.3)

            axes[1, 1].plot(day_labels, daily_metrics['avg_revenue'], marker='d', linestyle='-', color='tab:green')
            axes[1, 1].set_xlabel('天数')
            axes[1, 1].set_ylabel('骑手平均订单收益')
            axes[1, 1].set_title('每日骑手平均订单收益')
            axes[1, 1].grid(True, alpha=0.3)

            plt.tight_layout()
            metrics_plot_path = model_path.replace('.pth', '_daily_metrics.png')
            plt.savefig(metrics_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            logger.critical(f"每日指标曲线已保存到 {metrics_plot_path}")
        except ImportError:
            logger.warning("matplotlib 未安装，无法绘制每日指标曲线")
    elif not sim.is_train:
        logger.critical("评估模式：不记录每日指标")

    # 保存指标到checkpoints目录
    checkpoint_dir = os.path.dirname(model_path)
    metrics_file = os.path.join(checkpoint_dir, 'daily_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'day_labels': day_labels,
            'daily_metrics': daily_metrics
        }, f, ensure_ascii=False, indent=2)

    orders_df = pd.DataFrame([o for o in sim.orders.values() if pd.notna(o.get('order_id'))])
    total_orders = len(orders_df)
    if total_orders > 0:
        orders_df['revenue'] = 3 + orders_df.apply(
            lambda row: haversine_distance(
                (row['sender_lng'], row['sender_lat']),
                (row['recipient_lng'], row['recipient_lat'])
            ) / 1000, axis=1
        )

        delivered_orders = orders_df[orders_df['actual_delivery_time'].notna()].copy()
        delivered_orders['overtime_minutes'] = (delivered_orders['actual_delivery_time'] - delivered_orders['estimate_arrived_time']).dt.total_seconds() / 60
        delivered_orders['overtime_minutes'] = delivered_orders['overtime_minutes'].apply(lambda x: max(0, x))

        timeout_orders = delivered_orders[delivered_orders['actual_delivery_time'] > delivered_orders['estimate_arrived_time']]
        timeout_count = len(timeout_orders)
        timeout_rate = timeout_count / total_orders

        def apply_penalty(row):
            overtime = row['overtime_minutes']
            if overtime <= 10:
                return row['revenue'] * 0.9
            elif overtime <= 20:
                return row['revenue'] * 0.6
            else:
                return row['revenue'] * 0.5

        delivered_orders['actual_revenue'] = delivered_orders.apply(apply_penalty, axis=1)
        avg_revenue = delivered_orders['actual_revenue'].mean()

        delivered_orders['delivery_duration'] = (delivered_orders['actual_delivery_time'] - delivered_orders['dispatch_time']).dt.total_seconds() / 60
        avg_delivery_duration = delivered_orders['delivery_duration'].mean()

        timeout_orders = delivered_orders[delivered_orders['actual_delivery_time'] > delivered_orders['estimate_arrived_time']]
        timeout_count = len(timeout_orders)
        timeout_rate = timeout_count / total_orders
        avg_overtime = timeout_orders['overtime_minutes'].mean() if timeout_count > 0 else 0

        logger.critical(f"总订单数: {total_orders}\n"
                        f"已送达订单数: {len(delivered_orders)}\n"
                        f"平均配送时长: {avg_delivery_duration:.2f}分钟\n"
                        f"超时订单数: {timeout_count}\n"
                        f"超时率: {timeout_rate:.2%}\n"
                        f"平均超时时长: {avg_overtime:.2f}分钟\n"
                        f"骑手平均订单收益: {avg_revenue:.2f}")
    else:
        logger.critical("没有订单数据，无法计算超时率")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    sim.dispatcher_dqn.save_model(model_path)
    logger.critical(f"DQN模型已保存到 {model_path}")

    if sim.is_train and len(sim.dispatcher_dqn.ac_agent.critic_losses) > 0:
        loss_plot_path = model_path.replace('.pth', '_losses.png')
        sim.dispatcher_dqn.ac_agent.plot_losses(loss_plot_path)
        logger.critical(f"损失曲线已保存到 {loss_plot_path}")
    elif not sim.is_train:
        logger.critical("评估模式：不保存损失曲线（无训练数据）")
