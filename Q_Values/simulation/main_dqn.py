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
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """自定义 logging handler，通过 tqdm.write() 输出日志，
    使进度条始终保持在终端最底部。"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class DeliverySimulation:
    def __init__(self, order_data_path, idle_timeout_minutes=30, date_filter=None, is_train=True):
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

        # 筛选特定日期（在时间列转换之后，在生成器初始化之前）
        if date_filter is not None:
            self.df = self.df[self.df['dt'].isin(date_filter)]
            logger.info(f"按日期筛选: {date_filter}，剩余 {len(self.df)} 条订单")
        
        # 初始化生成器
        self.is_train = is_train
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
                    self.couriers[courier_id]['has_had_orders'] = True
                    self.orders[order['order_id']]['courier_id'] = courier_id
                    dispatched_orders.append(order)
            # 从待分配订单池中移除已分派的订单
            for order in dispatched_orders:
                self.pending_orders.remove(order)

        done = self.courier_sim.get_next_event_time() is None
        
        # 5. 确定下一个事件时间，推进 current_time（在返回状态后再推进）
        if not done:
            self.current_time += timedelta(seconds=30)
        
        return done, current_time


# ===================== 辅助函数 =====================


def _compute_revenue(row):
    """计算订单基础收入（起步价3元 + 里程费）"""
    distance_km = haversine_distance(
        (row['sender_lng'], row['sender_lat']),
        (row['recipient_lng'], row['recipient_lat'])
    ) / 1000
    return 3 + distance_km


def _apply_overtime_penalty(row):
    """根据超时时长对订单收入进行惩罚"""
    overtime = row['overtime_minutes']
    if overtime <= 10:
        return row['revenue'] * 0.9
    elif overtime <= 20:
        return row['revenue'] * 0.6
    else:
        return row['revenue'] * 0.5


def _orders_to_dataframe(orders_dict, delivered_only=False, current_time=None):
    """将订单字典转为DataFrame并计算基础收入"""
    records = [
        o for o in orders_dict.values()
        if pd.notna(o.get('order_id'))
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if delivered_only and 'actual_delivery_time' in df.columns:
        df = df[df['actual_delivery_time'].notna()]
        if current_time is not None:
            df = df[df['actual_delivery_time'] <= current_time]

    if 'revenue' not in df.columns:
        df['revenue'] = df.apply(_compute_revenue, axis=1)

    return df


def _compute_delivery_metrics(orders_df):
    """从订单DataFrame计算完整的配送指标"""
    if orders_df.empty:
        return None

    delivered = orders_df[orders_df['actual_delivery_time'].notna()].copy()
    if delivered.empty:
        return None

    delivered['overtime_minutes'] = (
        delivered['actual_delivery_time'] - delivered['estimate_arrived_time']
    ).dt.total_seconds() / 60
    delivered['overtime_minutes'] = delivered['overtime_minutes'].clip(lower=0)

    delivered['delivery_duration'] = (
        delivered['actual_delivery_time'] - delivered['dispatch_time']
    ).dt.total_seconds() / 60

    delivered['actual_revenue'] = delivered.apply(_apply_overtime_penalty, axis=1)

    timeout_mask = delivered['actual_delivery_time'] > delivered['estimate_arrived_time']
    timeout_orders = delivered[timeout_mask]

    return {
        'total': len(orders_df),
        'delivered_count': len(delivered),
        'timeout_count': len(timeout_orders),
        'avg_delivery_duration': delivered['delivery_duration'].mean(),
        'overtime_rate': len(timeout_orders) / len(delivered),
        'avg_overtime': timeout_orders['overtime_minutes'].mean() if len(timeout_orders) > 0 else 0.0,
        'avg_revenue': delivered['actual_revenue'].mean(),
    }


def _load_or_init_model(sim, model_path):
    """加载已有模型或从随机初始化开始"""
    if os.path.exists(model_path) and not sim.is_train:
        sim.dispatcher_dqn.ac_agent.load_model(model_path)
        if sim.dispatcher_dqn.ac_agent.is_train != sim.is_train:
            if sim.is_train:
                sim.dispatcher_dqn.ac_agent.set_train_mode()
            else:
                sim.dispatcher_dqn.ac_agent.set_eval_mode()
        logger.info(f"已加载模型: {model_path} | 模式: {'训练' if sim.is_train else '评估'}")
    else:
        logger.info("未找到已保存的模型，从随机初始化开始")


def _train_ac_agent(sim, steps_per_day):
    """训练AC网络（如果replay buffer足够）"""
    buffer_size = len(sim.dispatcher_dqn.ac_agent.replay_buffer)
    batch_size = sim.dispatcher_dqn.ac_agent.config.batch_size
    if buffer_size >= batch_size:
        logger.info(f"===== 训练网络 (buffer: {buffer_size}/{sim.dispatcher_dqn.ac_agent.config.buffer_capacity}) =====")
        sim.dispatcher_dqn.ac_agent.update_critic_offline()
        sim.dispatcher_dqn.ac_agent.update_actor_offline()
        return True
    logger.info(f"ReplayBuffer 数据不足: {buffer_size} < {batch_size}，跳过训练")
    return False


def _log_step_info(sim, step, current_time):
    """记录每个step的仿真状态"""
    online_count = sum(1 for c in sim.couriers.values() if c['status'] == 'online')
    delivering_count = sum(len(c['current_orders']) for c in sim.couriers.values())
    completed_count = sum(
        1 for o in sim.orders.values()
        if o.get('actual_delivery_time') is not None and o['actual_delivery_time'] <= current_time
    )
    buffer_size = len(sim.dispatcher_dqn.ac_agent.replay_buffer)

    logger.info(
        f"Step {step}: time={current_time}\n"
        f"----------------------------------\n"
        f"order_pool={sim.order_pool}\n"
        f"pending_orders={len(sim.pending_orders)}\n"
        f"online_couriers={online_count}\n"
        f"delivering_orders={delivering_count}\n"
        f"completed_orders={completed_count}\n"
        f"replay_buffer={buffer_size}\n"
        f"----------------------------------"
    )


def _log_period_summary(period, metrics):
    """记录每3小时周期结束时的指标汇总"""
    if metrics is None:
        return
    logger.info(
        f"===== 第 {period} 个周期(3h)结束，训练前指标 =====\n"
        f"已送达订单数: {metrics['delivered_count']}\n"
        f"平均配送时长: {metrics['avg_delivery_duration']:.2f}分钟\n"
        f"超时率: {metrics['overtime_rate']:.2%}\n"
        f"平均超时时长: {metrics['avg_overtime']:.2f}分钟\n"
        f"骑手平均订单收益: {metrics['avg_revenue']:.2f}"
    )


def _log_final_summary(metrics):
    """记录仿真结束时的最终指标汇总"""
    if metrics is None:
        logger.info("没有订单数据，无法计算超时率")
        return
    logger.info(
        f"总订单数: {metrics['total']}\n"
        f"已送达订单数: {metrics['delivered_count']}\n"
        f"平均配送时长: {metrics['avg_delivery_duration']:.2f}分钟\n"
        f"超时订单数: {metrics['timeout_count']}\n"
        f"超时率: {metrics['overtime_rate']:.2%}\n"
        f"平均超时时长: {metrics['avg_overtime']:.2f}分钟\n"
        f"骑手平均订单收益: {metrics['avg_revenue']:.2f}"
    )


def _plot_period_metrics(period_labels, period_metrics, save_path):
    """绘制每3小时周期指标变化曲线"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        plot_configs = [
            (0, 0, period_metrics['delivery_duration'], 'Avg Delivery Duration (min)', '3-Hour Avg Delivery Duration', 'tab:blue', 'o'),
            (0, 1, period_metrics['overtime_rate'], 'Overtime Rate', '3-Hour Overtime Rate', 'tab:red', 's'),
            (1, 0, period_metrics['avg_overtime'], 'Avg Overtime (min)', '3-Hour Avg Overtime', 'tab:orange', '^'),
            (1, 1, period_metrics['avg_revenue'], 'Avg Revenue per Order', '3-Hour Avg Revenue', 'tab:green', 'd'),
        ]

        for row, col, data, ylabel, title, color, marker in plot_configs:
            ax = axes[row, col]
            ax.plot(period_labels, data, marker=marker, linestyle='-', color=color)
            ax.set_xlabel('3-Hour Period')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"周期指标曲线已保存到 {save_path}")
    except ImportError:
        logger.warning("matplotlib 未安装，无法绘制周期指标曲线")


def _save_period_metrics_json(period_labels, period_metrics, save_dir):
    """保存每3小时周期指标到JSON文件"""
    metrics_file = os.path.join(save_dir, 'period_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'period_labels': period_labels,
            'period_metrics': period_metrics
        }, f, ensure_ascii=False, indent=2)


def _save_loss_curve(sim, model_path):
    """保存训练损失曲线"""
    if sim.is_train and len(sim.dispatcher_dqn.ac_agent.critic_losses) > 0:
        loss_plot_path = model_path.replace('.pth', '_losses.png')
        sim.dispatcher_dqn.ac_agent.plot_losses(loss_plot_path)
        logger.info(f"损失曲线已保存到 {loss_plot_path}")
    elif not sim.is_train:
        logger.info("评估模式：不保存损失曲线（无训练数据）")


# ===================== 主流程 =====================


def main():
    data_path = r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv'
    model_path = r'Q_Values\simulation\dqn\checkpoints\dqn_model.pth'

    TIME_STEP = 30
    STEPS_PER_DAY = 86400 // TIME_STEP
    TRAIN_EVERY_N_DAYS = 0.125
    TRAIN_INTERVAL = int(STEPS_PER_DAY * TRAIN_EVERY_N_DAYS)
    IS_TRAIN = True

    # 按日期筛选：设为 None 使用全部数据，或传入整数列表
    DATE_FILTER = None

    sim = DeliverySimulation(data_path, date_filter=DATE_FILTER, is_train=IS_TRAIN)
    _load_or_init_model(sim, model_path)

    start_time, end_time = sim.courier_sim.get_earliest_event_time(), sim.courier_sim.get_latest_event_time()
    total_steps = int((end_time - start_time).total_seconds() / TIME_STEP) if start_time and end_time else None

    last_train_step = 0
    period = 1
    period_labels = []
    period_metrics = {
        'delivery_duration': [],
        'overtime_rate': [],
        'avg_overtime': [],
        'avg_revenue': [],
    }

    step_iter = range(1, total_steps + 1) if total_steps else count(1)

    # 替换日志处理器，使日志通过 tqdm.write() 输出，进度条始终在最底部
    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s'))
    logger.root.handlers = [tqdm_handler]

    for step_num in tqdm(step_iter, desc="仿真进度"):
        done, current_time = sim.step()

        # -- 每3小时记录指标并训练 --
        if step_num - last_train_step >= TRAIN_INTERVAL and sim.is_train:
            orders_df = _orders_to_dataframe(sim.orders, delivered_only=True, current_time=current_time)
            metrics = _compute_delivery_metrics(orders_df)
            if metrics is not None:
                period_labels.append(period)
                period_metrics['delivery_duration'].append(metrics['avg_delivery_duration'])
                period_metrics['overtime_rate'].append(metrics['overtime_rate'])
                period_metrics['avg_overtime'].append(metrics['avg_overtime'])
                period_metrics['avg_revenue'].append(metrics['avg_revenue'])
                _log_period_summary(period, metrics)

            if _train_ac_agent(sim, STEPS_PER_DAY):
                last_train_step = step_num
            period += 1

        _log_step_info(sim, step_num, current_time)

        if done:
            if sim.is_train:
                _train_ac_agent(sim, STEPS_PER_DAY)
            logger.info(f"仿真结束，总耗时: {current_time - sim.df['dispatch_time'].min()}")
            break

    # ==================== 后处理 ====================
    checkpoint_dir = os.path.dirname(model_path)
    os.makedirs(checkpoint_dir, exist_ok=True)

    if period_labels and sim.is_train:
        metrics_plot_path = model_path.replace('.pth', '_period_metrics.png')
        _plot_period_metrics(period_labels, period_metrics, metrics_plot_path)
        _save_period_metrics_json(period_labels, period_metrics, checkpoint_dir)
    elif not sim.is_train:
        logger.info("评估模式：不记录周期指标")

    all_orders_df = _orders_to_dataframe(sim.orders)
    final_metrics = _compute_delivery_metrics(all_orders_df)
    _log_final_summary(final_metrics)

    sim.dispatcher_dqn.save_model(model_path)
    logger.info(f"DQN模型已保存到 {model_path}")

    _save_loss_curve(sim, model_path)


if __name__ == '__main__':
    main()
