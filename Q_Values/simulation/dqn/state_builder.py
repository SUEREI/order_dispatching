"""
状态向量构建器

将骑手和订单信息转换为25维状态向量
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from datetime import datetime


class StateBuilder:
    """
    状态向量构建器
    
    将骑手和订单信息转换为符合规范的25维状态向量：
    - 维度 0-1：骑手当前经度、纬度
    - 维度 2：当前时间步的时间戳（归一化）
    - 维度 3-18：路径点1~8的经纬度（共16维）
    - 维度 19-26：路径点1~8对应的"距离超时的剩余时间"（共8维）
    """
    
    STATE_DIM = 27
    
    def __init__(
        self,
        lng_min: float = 174.38,
        lng_max: float = 175.00,
        lat_min: float = 45.74,
        lat_max: float = 46.08,
        time_start: datetime = None,
        time_end: datetime = None,
        max_time_remaining: float = 3600.0  # 最大剩余时间（秒），默认1小时
    ):
        """
        初始化状态构建器
        
        Args:
            lng_min, lng_max: 经度归一化范围
            lat_min, lat_max: 纬度归一化范围
            time_start, time_end: 时间归一化范围
            max_time_remaining: 最大剩余时间（秒），用于归一化
        """
        self.lng_min = lng_min
        self.lng_max = lng_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.time_start = time_start
        self.time_end = time_end
        self.max_time_remaining = max_time_remaining
    
    def normalize_lng(self, lng: float) -> float:
        """归一化经度"""
        return (lng - self.lng_min) / (self.lng_max - self.lng_min)
    
    def normalize_lat(self, lat: float) -> float:
        """归一化纬度"""
        return (lat - self.lat_min) / (self.lat_max - self.lat_min)
    
    def normalize_time(self, time: datetime) -> float:
        """归一化时间到0~1"""
        if self.time_start is None or self.time_end is None:
            return 0.5
        total_seconds = (self.time_end - self.time_start).total_seconds()
        current_seconds = (time - self.time_start).total_seconds()
        return current_seconds / total_seconds
    
    def normalize_remaining_time(self, remaining_seconds: float) -> float:
        """归一化剩余时间到0~1，负数设为0"""
        normalized = remaining_seconds / self.max_time_remaining
        return max(0.0, min(1.0, normalized))
    
    def build_state(
        self,
        courier: Dict[str, Any],
        route: List[Dict[str, Any]],
        current_time: datetime,
        orders_dict: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """
        构建27维状态向量
        
        Args:
            courier: 骑手信息字典
            route: 路径规划结果（最多8个点）
            current_time: 当前时间
            orders_dict: 订单字典（用于获取订单时间信息）
        
        Returns:
            np.ndarray: 27维状态向量
        """
        state = np.zeros(27, dtype=np.float32)
        
        # 维度 0-1：骑手当前经度、纬度
        courier_pos = courier.get('position', (0, 0))
        state[0] = self.normalize_lng(courier_pos[0])
        state[1] = self.normalize_lat(courier_pos[1])
        
        # 维度 2：当前时间步的时间戳（归一化）
        state[2] = self.normalize_time(current_time)
        
        # 维度 3-18：路径点1~8的经纬度（共16维）
        # 维度 19-26：路径点1~8对应的"距离超时的剩余时间"（共8维）
        max_route_points = 8
        for i, route_point in enumerate(route[:max_route_points]):
            if i >= max_route_points:
                break
            
            pos = route_point.get('position', (0, 0))
            order_id = route_point.get('order_id')
            point_type = route_point.get('point_type')
            
            # 经纬度
            state[3 + i*2] = self.normalize_lng(pos[0])
            state[4 + i*2] = self.normalize_lat(pos[1])
            
            # 剩余时间
            if order_id and order_id in orders_dict:
                order = orders_dict[order_id]
                remaining_time = self._calculate_remaining_time(
                    order, point_type, current_time
                )
                state[19 + i] = self.normalize_remaining_time(remaining_time)
        
        return state
    
    def _calculate_remaining_time(
        self,
        order: Dict[str, Any],
        point_type: str,
        current_time: datetime
    ) -> float:
        """
        计算距离超时的剩余时间（秒）
        
        Args:
            order: 订单信息
            point_type: 'pickup' 或 'delivery'
            current_time: 当前时间
        
        Returns:
            float: 剩余时间（秒），负数表示已超时
        """
        if point_type == 'pickup':
            # 取餐点：剩余时间 = 预计备餐时间 - 当前时间
            estimate_prepare_time = order.get('estimate_meal_prepare_time')
            if estimate_prepare_time:
                remaining = (estimate_prepare_time - current_time).total_seconds()
                return remaining
        else:
            # 送餐点：剩余时间 = 预计送达时间 - 当前时间
            estimate_arrived_time = order.get('estimate_arrived_time')
            if estimate_arrived_time:
                remaining = (estimate_arrived_time - current_time).total_seconds()
                return remaining
        
        return 0.0
