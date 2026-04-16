#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单分派器
"""
import logging
from typing import Tuple, List, Dict
import numpy as np
from utils import haversine_distance

# 配置日志
logger = logging.getLogger(__name__)


class OrderDispatcher:
    """订单分派器"""
    def __init__(self):
        """初始化分派器"""  
        pass
    
    def dispatch_to_nearest_courier(self, order, available_couriers) -> Tuple[str, float]:
        """将订单分派给最近的可用骑手"""
        if not available_couriers:
            return None
        
        # 获取订单起点位置
        order_pos = (order['sender_lng'], order['sender_lat'])
        # 计算每个骑手到订单起点的距离
        courier_distances = []
        for courier_id, courier in available_couriers.items():
            if courier['position'] is not None:
                distance = haversine_distance(order_pos, courier['position'])
                courier_distances.append((courier['courier_id'], distance))
        if not courier_distances:
            return None
        # 找出距离最近的骑手
        nearest_courier = min(courier_distances, key=lambda x: x[1])

        return nearest_courier
    
    def dispatch_orders(self, orders, couriers) -> Dict[str, str]:
        """批量处理订单分派"""
        dispatch_results = {}
        for order in orders:
            result = self.dispatch_to_nearest_courier(order, couriers)
            if result:
                courier_id, distance = result
                dispatch_results[order['order_id']] = courier_id
                logger.debug(f"订单 {order['order_id']} 分派给骑手 {courier_id}，距离: {distance:.4f}")
            else:
                logger.debug(f"订单 {order['order_id']} 无法分派，无可用骑手")
        
        return dispatch_results
