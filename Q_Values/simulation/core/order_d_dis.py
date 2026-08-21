#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单分派器 - 基于距离贪心策略

将每个待分配订单派发给距离该订单取餐点最近的在线骑手，
作为仿真系统的基线派单策略，与真实匹配派单（order_d_real）、
DQN/Actor-Critic 派单（order_d_dqn）进行对比。
"""

import logging
from typing import Dict, List

from utils import haversine_distance

logger = logging.getLogger(__name__)


class OrderDispatcherByDistance:
    """基于距离贪心策略的订单分派器

    对每个待分配订单，从在线且未达订单上限的骑手中，
    选择距离订单取餐点（sender_lng, sender_lat）最近的骑手进行派单。
    """

    def __init__(self, max_orders_per_courier: int = 4):
        """
        Args:
            max_orders_per_courier: 单个骑手同时承接的最大订单数（容量上限）
        """
        self.max_orders_per_courier = max_orders_per_courier
        logger.info(
            f"距离贪心订单分派器初始化完成，骑手最大订单数: {max_orders_per_courier}"
        )

    def dispatch_orders_by_distance(self, orders: List[Dict], couriers: Dict[str, Dict]) -> Dict[str, str]:
        """按距离贪心策略批量派单

        Args:
            orders: 待分配订单列表，每个订单需包含
                    order_id / sender_lng / sender_lat
            couriers: 在线骑手字典 {courier_id: courier_dict}

        Returns:
            dispatch_results: {order_id: courier_id} 派单结果映射，
                              无可用骑手的订单不会出现在结果中
        """
        dispatch_results = {}
        # 临时订单计数：同一时间步内多个订单依次派单时，
        # 实时更新骑手已分配订单数，保证不超出容量上限
        temp_order_counts = {}

        for order in orders:
            order_id = str(order['order_id'])
            order_pos = (order['sender_lng'], order['sender_lat'])

            # 1. 筛选可用骑手：在线、有位置、未达订单上限
            available = []
            for cid, courier in couriers.items():
                if courier.get('status') != 'online':
                    continue
                if courier.get('position') is None:
                    continue
                current_count = len(courier.get('current_orders', [])) + temp_order_counts.get(cid, 0)
                if current_count >= self.max_orders_per_courier:
                    continue
                available.append((cid, courier))

            if not available:
                logger.debug(f"订单 {order_id} 无可用骑手，暂不分配")
                continue

            # 2. 选择距离取餐点最近的骑手
            best_cid = None
            best_dist = float('inf')
            for cid, courier in available:
                dist = haversine_distance(courier['position'], order_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_cid = cid

            # 3. 记录派单结果并更新临时容量
            dispatch_results[order_id] = best_cid
            temp_order_counts[best_cid] = temp_order_counts.get(best_cid, 0) + 1
            logger.debug(f"订单 {order_id} 派给骑手 {best_cid}，距离 {best_dist:.1f} 米")

        return dispatch_results
