"""
动作向量构建器

将订单信息转换为5维动作向量，并管理候选动作集
"""

import numpy as np
from typing import Dict, List, Any
from datetime import datetime


class ActionBuilder:
    """
    动作向量构建器
    
    将订单信息转换为符合规范的5维动作向量：
    - 维度 0-1：新订单取餐点经度、纬度
    - 维度 2-3：新订单送餐点经度、纬度
    - 维度 4：该订单的可用配送时长（归一化）
    """
    
    def __init__(
        self,
        lng_min: float = 174.38,
        lng_max: float = 175.00,
        lat_min: float = 45.74,
        lat_max: float = 46.08,
        max_delivery_time: float = 3600.0  # 最大配送时长（秒），默认1小时
    ):
        """
        初始化动作构建器
        
        Args:
            lng_min, lng_max: 经度归一化范围
            lat_min, lat_max: 纬度归一化范围
            max_delivery_time: 最大配送时长（秒），用于归一化
        """
        self.lng_min = lng_min
        self.lng_max = lng_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.max_delivery_time = max_delivery_time
    
    def normalize_lng(self, lng: float) -> float:
        """归一化经度"""
        return (lng - self.lng_min) / (self.lng_max - self.lng_min)
    
    def normalize_lat(self, lat: float) -> float:
        """归一化纬度"""
        return (lat - self.lat_min) / (self.lat_max - self.lat_min)
    
    def normalize_delivery_time(self, delivery_seconds: float) -> float:
        """归一化配送时长到0~1"""
        normalized = delivery_seconds / self.max_delivery_time
        return min(1.0, normalized)
    
    def build_action(
        self,
        order: Dict[str, Any],
        current_time: datetime
    ) -> np.ndarray:
        """
        构建5维动作向量
        
        Args:
            order: 订单信息字典
            current_time: 当前时间
        
        Returns:
            np.ndarray: 5维动作向量
        """
        action = np.zeros(5, dtype=np.float32)
        
        # 维度 0-1：新订单取餐点经度、纬度
        action[0] = self.normalize_lng(order.get('sender_lng', 0))
        action[1] = self.normalize_lat(order.get('sender_lat', 0))
        
        # 维度 2-3：新订单送餐点经度、纬度
        action[2] = self.normalize_lng(order.get('recipient_lng', 0))
        action[3] = self.normalize_lat(order.get('recipient_lat', 0))
        
        # 维度 4：该订单的可用配送时长（归一化）
        estimate_arrived_time = order.get('estimate_arrived_time')
        if estimate_arrived_time:
            delivery_time = (estimate_arrived_time - current_time).total_seconds()
            action[4] = self.normalize_delivery_time(delivery_time)
        
        return action
    
    def build_candidate_actions(
        self,
        candidate_orders: List[Dict[str, Any]],
        current_time: datetime
    ) -> np.ndarray:
        """
        构建候选动作集
        
        Args:
            candidate_orders: 候选订单列表
            current_time: 当前时间
        
        Returns:
            np.ndarray: 候选动作集，shape为(N, 5)，N为候选订单数量
        """
        if not candidate_orders:
            return np.zeros((0, 5), dtype=np.float32)
        
        actions = []
        for order in candidate_orders:
            action = self.build_action(order, current_time)
            actions.append(action)
        
        return np.array(actions, dtype=np.float32)
    
    def build_reject_action(self) -> np.ndarray:
        """
        构建拒单动作（全0向量）
        
        Returns:
            np.ndarray: 5维全0向量
        """
        return np.zeros(5, dtype=np.float32)
