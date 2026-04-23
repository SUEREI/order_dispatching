#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真主程序
"""

import logging
from datetime import timedelta
from itertools import count
from tqdm import tqdm

import pandas as pd

from core import CourierSimulator
from core import OrderGenerator
from core import OrderDispatcherByDistance, OrderDispatcherByRealMatch
from core import RoutePlanner
from core import StateRefresher
from utils import haversine_distance

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
        self.order_gen = OrderGenerator(self.df)
        self.courier_sim = CourierSimulator(self.df, idle_timeout_minutes)
        # 初始化派单器、路由规划器、状态刷新器
        self.state_refresher = StateRefresher()
        self.dispatcher_distance = OrderDispatcherByDistance()
        self.dispatcher_real = OrderDispatcherByRealMatch(self.df)
        self.route_planner = RoutePlanner()
        
        
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
                'position': None,  # 上线时会设置初始位置
                'order_increased': False,
                'current_orders': [],
                'route': []
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
        online_couriers = {cid: info for cid, info in self.couriers.items() if info['status'] == 'online'}
        # logger.info(f"当前时间点 {self.current_time} 的在线骑手数量: {len(online_couriers)}")
        
        # 3. 处理当前时间点的订单生成事件
        new_orders = self.order_gen.get_orders_at_time(self.current_time)
        if new_orders:
            self.pending_orders.extend(new_orders)
            self.order_pool = len(new_orders)
        
        # 4. 如果 pending_orders 不为空，使用派单器将订单派给最近的骑手
        if self.pending_orders:
            # 派单给最近的骑手
            dispatch_results = self.dispatcher_distance.dispatch_orders_by_distance(self.pending_orders, online_couriers)
            # logger.debug(f"当前时间点 {self.current_time} 的订单派单结果: {dispatch_results}")
            # 处理分派结果
            dispatched_orders = []
            for order in self.pending_orders:
                if order['order_id'] in dispatch_results:
                    courier_id = dispatch_results[order['order_id']]
                    # 将订单添加到骑手的当前订单列表并更新状态
                    self.couriers[courier_id]['order_increased'] = True
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
    sim = DeliverySimulation(r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv')

    start_time, end_time = sim.courier_sim.get_earliest_event_time(), sim.courier_sim.get_latest_event_time()
    total_steps = int((end_time - start_time).total_seconds() / 30) if start_time and end_time else None

    i = 1
    for _ in tqdm(count() if not total_steps else range(total_steps), desc="仿真进度"):
        state, reward, done, current_time = sim.step()
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
            logger.info(f"仿真结束，总耗时: {current_time - sim.df['dispatch_time'].min()}")
            break
        i += 1

    # 保存订单结果为 CSV
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
    # orders_df.to_csv('Q_Values\simulation\data_processed\simulated_orders.csv', index=False)
    # logger.info(f"订单结果已保存到 Q_Values/simulation/data_processed/simulated_orders.csv")
