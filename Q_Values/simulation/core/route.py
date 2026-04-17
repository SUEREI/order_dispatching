#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骑手路径规划器模块

该模块实现骑手的路径规划功能，包括：
- 根据订单状态判断可达点
- 使用贪婪算法规划路径
- 确保满足先取后送约束
"""

import logging
from typing import Dict, List, Tuple, Any

from utils.geo_utils import haversine_distance

# 配置日志
logger = logging.getLogger(__name__)


class RoutePlanner:
    """
    骑手路径规划器
    该类负责根据骑手当前订单状态，规划骑手的配送路径。
    使用贪婪算法，优先选择距离最近的可达点，同时满足先取后送约束。
    """
    def __init__(self):
        """初始化路径规划器"""
        logger.info("路径规划器初始化完成")

    def get_accessible_points(self, orders: List[Dict[str, Any]], courier_id: str) -> List[Dict[str, Any]]:
        """
        获取骑手需要经过的可达点列表
        根据订单状态判断哪些点需要经过：
        - 订单状态为'pending'（已分配未取餐）：取餐点可达，送餐点不可达
        - 订单状态为'picked_up'（已取餐未送餐）：送餐点可达，取餐点不可达
        - 订单状态为'delivered'（已送达）：两点均不可达
        """
        accessible_points = []

        for order in orders:
            order_id = order['order_id']
            status = order.get('status', 'pending')
            if status == 'pending':
                # 订单未取餐，取餐点可达
                pickup_point = {
                    'point_type': 'pickup',
                    'order_id': order_id,
                    'position': (order['sender_lng'], order['sender_lat'])
                }
                accessible_points.append(pickup_point)
            elif status == 'picked_up':
                # 订单已取餐，送餐点可达
                delivery_point = {
                    'point_type': 'delivery',
                    'order_id': order_id,
                    'position': (order['recipient_lng'], order['recipient_lat'])
                }
                accessible_points.append(delivery_point)

        logger.debug(f"{courier_id} 可达点: {len(accessible_points)} 个")

        return accessible_points

    def select_nearest_point(
        self,
        current_pos: Tuple[float, float],
        accessible_points: List[Dict[str, Any]],
        courier_id: str
    ) -> Dict[str, Any]:
        """从可达点中选择距离当前位置最近的点"""
        if not accessible_points:
            return None
        # 计算到每个可达点的距离
        min_distance = float('inf')
        nearest_point = None

        for point in accessible_points:
            distance = haversine_distance(current_pos, point['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        logger.debug(f"{courier_id} 选择最近点: {nearest_point['order_id']}-{nearest_point['point_type']}, 距离: {min_distance:.2f}KM")
        
        return nearest_point

    def plan_route(self, courier: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        为骑手规划配送路径
        使用贪婪算法规划路径：
        1. 获取可达点列表
        2. 选择距离当前位置最近的可达点
        3. 如果是取餐点，将对应订单的送餐点添加到可达点
        4. 重复步骤2-3直到所有点都访问完毕
        """
        # 获取骑手当前位置和订单列表
        current_pos = courier['position']
        orders = courier.get('current_orders', [])

        # 初始化路径和可达点
        route = []
        sequence = 1
        # 获取初始可达点
        accessible_points = self.get_accessible_points(orders, courier['courier_id'])
        # 构建订单字典，用于快速查找
        orders_dict = {order['order_id']: order for order in orders}

        # 贪婪算法规划路径
        while accessible_points:
            # 选择最近的可达点
            nearest_point = self.select_nearest_point(current_pos, accessible_points, courier['courier_id'])
            # 添加到路径
            route_point = {
                'sequence': sequence,
                'point_type': nearest_point['point_type'],
                'order_id': nearest_point['order_id'],
                'position': nearest_point['position']
            }
            route.append(route_point)
            # 更新当前位置和序号
            current_pos = nearest_point['position']
            sequence += 1

            # 如果是取餐点，将对应订单的送餐点添加到可达点
            if nearest_point['point_type'] == 'pickup':
                order_id = nearest_point['order_id']
                order = orders_dict[order_id]
                delivery_point = {
                    'point_type': 'delivery',
                    'order_id': order_id,
                    'position': (order['recipient_lng'], order['recipient_lat'])
                }
                accessible_points.append(delivery_point)

            # 从可达点移除已访问的点
            accessible_points = [
                p for p in accessible_points
                if not (p['order_id'] == nearest_point['order_id'] and
                        p['point_type'] == nearest_point['point_type'])
            ]

            logger.debug(f"{courier['courier_id']} 路径规划完成，共 {len(route)} 个点")

        return route

    def validate_route(self, route: List[Dict[str, Any]]) -> bool:
        """
        验证路径是否满足先取后送约束
        对于每个订单，取餐点的访问顺序必须小于送餐点的访问顺序
        """
        # 记录每个订单取餐点的访问顺序
        pickup_sequences = {}
        delivery_sequences = {}

        for point in route:
            order_id = point['order_id']
            sequence = point['sequence']

            if point['point_type'] == 'pickup':
                pickup_sequences[order_id] = sequence
            else:
                delivery_sequences[order_id] = sequence

        # 检查每个订单是否满足先取后送
        for order_id in pickup_sequences:
            if order_id in delivery_sequences:
                if pickup_sequences[order_id] >= delivery_sequences[order_id]:
                    logger.warning(f"订单 {order_id} 不满足先取后送约束")
                    return False

        logger.debug("路径验证通过，满足先取后送约束")
        return True

