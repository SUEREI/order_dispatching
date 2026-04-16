#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真主程序
"""

import logging
from datetime import timedelta

import pandas as pd
import numpy as np

from core import CourierSimulator
from core import OrderGenerator
from core import OrderDispatcher
from core import RoutePlanner
from core import StateRefresher

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class DeliverySimulation:
    def __init__(self, order_data_path, idle_timeout_minutes=30):
        # 加载数据
        self.df = pd.read_csv(order_data_path)
        self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        self.df['arrive_time'] = pd.to_datetime(self.df['arrive_time'])
        
        # 初始化生成器
        self.order_gen = OrderGenerator(self.df)
        self.courier_sim = CourierSimulator(self.df, idle_timeout_minutes)
        # 初始化派单器、路由规划器、状态刷新器
        self.dispatcher = OrderDispatcher()
        self.route_planner = RoutePlanner()
        self.state_refresher = StateRefresher()
        
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

        # self.order_timestamps = self.order_gen.get_all_timestamps()
        # self.order_index = 0  # 指向下一个要处理的订单时间
        
        # 待分配订单池
        self.order_pool = 0
        self.pending_orders = []
        
        logger.info(f"仿真初始化完成，起始时间: {self.current_time}")
    
    def _init_couriers(self) -> None:
        """初始化所有骑手状态，默认全部离线"""
        all_couriers = self.courier_sim.get_all_courier_ids()
        for cid in all_couriers:
            self.couriers[cid] = {
                'courier_id': cid,
                'status': 'offline',
                'position': None,  # 上线时会设置初始位置
                'order_increased': False,
                'current_orders': []
            }
    
    def step(self, action=None) -> None:
        """执行一步仿真"""
        # 保存当前时间（用于返回）
        current_time = self.current_time
        # 重置当前时间步的新订单数量
        self.order_pool = 0
        
        # 1. 处理当前时间点的骑手上下线事件
        courier_events = self.courier_sim.get_events_at_time(self.current_time)
        for cid, etype, pos, orders in courier_events:
            self.couriers[cid]['status'] = etype
            if etype == 'online':
                self.couriers[cid]['position'] = pos
                self.couriers[cid]['current_orders'] = orders
        online_couriers = {cid: info for cid, info in self.couriers.items() if info['status'] == 'online'}
        # logger.info(f"当前时间点 {self.current_time} 的在线骑手数量: {len(online_couriers)}")
        
        # 2. 处理当前时间点的订单生成事件
        new_orders = self.order_gen.get_orders_at_time(self.current_time)
        if new_orders:
            self.pending_orders.extend(new_orders)
            self.order_pool = len(new_orders)
        
        # 3. 如果 pending_orders 不为空，使用派单器将订单派给最近的骑手
        if self.pending_orders:
            # 派单给最近的骑手
            dispatch_results = self.dispatcher.dispatch_orders(self.pending_orders, online_couriers)
            # logger.debug(f"当前时间点 {self.current_time} 的订单派单结果: {dispatch_results}")
            # 处理分派结果
            dispatched_orders = []
            for order in self.pending_orders:
                if order['order_id'] in dispatch_results:
                    courier_id = dispatch_results[order['order_id']]
                    # 将订单添加到骑手的当前订单列表并更新状态
                    self.couriers[courier_id]['order_increased'] = True
                    self.couriers[courier_id]['current_orders'].append(order)
                    self.orders[order['order_id']]['courier_id'] = courier_id
                    dispatched_orders.append(order)
            # 从待分配订单池中移除已分派的订单
            for order in dispatched_orders:
                self.pending_orders.remove(order)

        
        for courier_id, courier in self.couriers.items():
            if courier['status'] == 'online' and courier.get('current_orders'):
                # 4. 规划路径
                # 检查是否有订单增加
                if courier['order_increased']:
                    route = self.route_planner.plan_route(courier)
                    courier['route'] = route
                    courier['order_increased'] = False

                # 5. 刷新骑手状态
                # 检查是否有订单已到达deliver_time，如果是则移除
                completed_orders = []
                for order in courier['current_orders']:
                    if current_time >= order['actual_delivery_time']:
                        completed_orders.append(order)
                # 从骑手当前订单中移除已完成的订单
                for order in completed_orders:
                    courier['current_orders'].remove(order)
                    # 如果订单有对应的骑手记录，可以在这里更新订单状态
                    if order['order_id'] in self.orders:
                        self.orders[order['order_id']]['actual_pickup_time'] = order['actual_pickup_time']
                        self.orders[order['order_id']]['actual_delivery_time'] = order['actual_delivery_time']
                
                # 如果所有订单都已完成，清除路径规划
                if not courier['current_orders']:
                    courier['route'] = []
                orders_dict = {order['order_id']: order for order in courier['current_orders']}
                updates = self.state_refresher.refresh_courier(
                    courier, 
                    orders_dict,
                    current_time,
                    30,
                    courier.get('route', [])
                )
                if updates.get('courier_updates'):
                    self.couriers[courier_id].update(updates['courier_updates'])
                
        done = self.courier_sim.get_next_event_time() is None
        
        # 5. 确定下一个事件时间，推进 current_time（在返回状态后再推进）
        if not done:
            self.current_time += timedelta(seconds=30)
        
        return None, 0, done, current_time


if __name__ == '__main__':
    sim = DeliverySimulation(r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv')
    
    # # 运行几步观察事件
    # for i in range(1, 21):
    #     state, reward, done, current_time = sim.step()
    #     # logger.info(f"Step {i}: time={current_time}, order_pool={sim.order_pool}, pending_orders={len(sim.pending_orders)}, online_couriers={sum(1 for c in sim.couriers.values() if c['status']=='online')}")    
    #     if done:
    #         break



    # 运行直到结束
    i = 1
    while True:
        state, reward, done, current_time = sim.step()
        print(f"Step {i}: time={current_time}, order_pool={sim.order_pool}, pending_orders={len(sim.pending_orders)}, online_couriers={sum(1 for c in sim.couriers.values() if c['status']=='online')}")    
        if done:
            break
        i += 1

    # 保存订单结果为 CSV
    orders_df = pd.DataFrame(list(sim.orders.values()))
    orders_df.to_csv('../Data/simulated_orders.csv', index=False)
    logger.info(f"订单结果已保存到 ../Data/simulated_orders.csv")