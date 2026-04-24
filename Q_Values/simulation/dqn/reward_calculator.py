"""
奖励计算器

计算边际收益作为即时奖励
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime


class RewardCalculator:
    """
    奖励计算器
    
    计算边际收益作为即时奖励：
    - 单个订单基础收益 = 3 + 配送距离(KM)
    - 超时惩罚：超时10分钟内收益打8折；超时10-20分钟打5折；超时20分钟以上收益为0
    - 边际奖励：r = R_new - R_old
    """
    
    def __init__(self, base_reward: float = 3.0):
        """
        初始化奖励计算器
        
        Args:
            base_reward: 基础收益，默认为3.0
        """
        self.base_reward = base_reward
    
    def calculate_order_reward(
        self,
        delivery_distance: float,
        overtime_minutes: float = 0.0
    ) -> float:
        """
        计算单个订单的收益（考虑超时惩罚）
        
        Args:
            delivery_distance: 配送距离（KM）
            overtime_minutes: 超时时间（分钟）
        
        Returns:
            float: 订单收益
        """
        # 基础收益
        base = self.base_reward + delivery_distance
        
        # 超时惩罚
        if overtime_minutes <= 10:
            # 超时10分钟内：收益打8折
            return base * 0.8
        elif overtime_minutes <= 20:
            # 超时10-20分钟：收益打5折
            return base * 0.5
        else:
            # 超时20分钟以上：收益为0
            return 0.0
    
    def calculate_total_reward(
        self,
        orders: List[Dict[str, Any]],
        current_time: datetime
    ) -> float:
        """
        计算所有订单的总收益
        
        Args:
            orders: 订单列表
            current_time: 当前时间
        
        Returns:
            float: 总收益
        """
        total_reward = 0.0
        
        for order in orders:
            # 只计算已送达的订单
            if order.get('status') != 'delivered':
                continue
            
            # 计算配送距离
            sender_lng = order.get('sender_lng', 0)
            sender_lat = order.get('sender_lat', 0)
            recipient_lng = order.get('recipient_lng', 0)
            recipient_lat = order.get('recipient_lat', 0)
            
            delivery_distance = self._haversine_distance(
                (sender_lng, sender_lat),
                (recipient_lng, recipient_lat)
            )
            
            # 计算超时时间
            estimate_arrived_time = order.get('estimate_arrived_time')
            actual_delivery_time = order.get('actual_delivery_time')
            
            overtime_minutes = 0.0
            if estimate_arrived_time and actual_delivery_time:
                if actual_delivery_time > estimate_arrived_time:
                    overtime_minutes = (actual_delivery_time - estimate_arrived_time).total_seconds() / 60.0
            
            # 计算订单收益
            order_reward = self.calculate_order_reward(delivery_distance, overtime_minutes)
            total_reward += order_reward
        
        return total_reward
    
    def calculate_marginal_reward(
        self,
        orders_before: List[Dict[str, Any]],
        orders_after: List[Dict[str, Any]],
        current_time: datetime
    ) -> float:
        """
        计算边际奖励
        
        Args:
            orders_before: 接单前的订单列表
            orders_after: 接单后的订单列表
            current_time: 当前时间
        
        Returns:
            float: 边际奖励 r = R_new - R_old
        """
        R_old = self.calculate_total_reward(orders_before, current_time)
        R_new = self.calculate_total_reward(orders_after, current_time)
        
        return R_new - R_old
    
    def _haversine_distance(
        self,
        coord1: Tuple[float, float],
        coord2: Tuple[float, float]
    ) -> float:
        """
        计算两点之间的球面距离（KM）
        
        Args:
            coord1: (lng, lat)
            coord2: (lng, lat)
        
        Returns:
            float: 距离（KM）
        """
        from math import radians, sin, cos, sqrt, atan2
        
        lng1, lat1 = coord1
        lng2, lat2 = coord2
        
        # 将经纬度转换为弧度
        lat1, lng1, lat2, lng2 = map(radians, [lat1, lng1, lat2, lng2])
        
        # Haversine公式
        dlat = lat2 - lat1
        dlng = lng2 - lng1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # 地球半径（KM）
        R = 6371.0
        
        return R * c
