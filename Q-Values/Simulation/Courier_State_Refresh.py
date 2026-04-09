import pandas as pd
import logging
from Route_Planer import RoutePlanner

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CourierStateRefresh:
    """骑手状态刷新器"""
    
    def __init__(self, speed=30):
        """
        初始化骑手状态刷新器
        
        Args:
            speed: 骑手骑行速度（单位：KM/h）
        """
        self.speed = speed
        self.route_planner = RoutePlanner()
    
    def refresh_courier_state(self, courier, current_time, courier_id=None):
        """
        刷新骑手状态
        
        Args:
            courier: 骑手信息字典
            current_time: 当前时间
            courier_id: 骑手ID
            
        Returns:
            dict: 更新后的骑手状态
        """
        # 检查骑手是否有当前订单
        if not courier['current_orders']:
            return courier
        
        # 检查是否已有路径规划
        if 'current_route' not in courier or not courier['current_route']:
            # 规划新路径
            start_pos = courier['position']
            courier['current_route'] = self.route_planner.plan_route(start_pos, courier['current_orders'])
            courier['route_index'] = 0
            courier['last_update_time'] = current_time
            if courier_id:
                logger.debug(f"为骑手 {courier_id} 规划新路径: {courier['current_route']}")
        
        # 计算自上次更新以来的时间差（分钟）
        time_diff = (current_time - courier['last_update_time']).total_seconds() / 60
        
        # 处理路径
        route = courier['current_route']
        route_index = courier['route_index']
        
        while route_index < len(route) and time_diff > 0:
            current_pos = courier['position']
            next_point = route[route_index]
            next_pos = (next_point[0], next_point[1])
            point_type = next_point[2]
            order_id = next_point[3]
            
            # 计算到下一个点的距离
            distance = self.route_planner.calculate_distance(current_pos, next_pos)
            # 计算所需时间
            time_needed = (distance / 1000) / self.speed * 60  # 转换为分钟
            
            if time_needed <= time_diff:
                # 到达下一个点
                courier['position'] = next_pos
                courier['last_update_time'] = current_time - pd.Timedelta(minutes=time_diff - time_needed)
                time_diff -= time_needed
                route_index += 1
                
                # 处理取餐或送餐事件
                if point_type == 'pickup':
                    # 检查餐是否备好
                    order = next(o for o in courier['current_orders'] if o['order_id'] == order_id)
                    prepare_time = self._get_food_prepare_time(order, current_time)
                    if prepare_time > 0:
                        # 等待备餐
                        courier['last_update_time'] += pd.Timedelta(minutes=prepare_time)
                        time_diff -= prepare_time
                        if courier_id:
                            logger.debug(f"骑手 {courier_id} 在取餐点等待 {prepare_time} 分钟")
                    if courier_id:
                        logger.debug(f"骑手 {courier_id} 取餐: 订单 {order_id}")
                elif point_type == 'dropoff':
                    # 送达订单
                    if courier_id:
                        logger.debug(f"骑手 {courier_id} 送达: 订单 {order_id}")
            else:
                # 部分移动
                ratio = time_diff / time_needed
                new_lng = current_pos[0] + (next_pos[0] - current_pos[0]) * ratio
                new_lat = current_pos[1] + (next_pos[1] - current_pos[1]) * ratio
                courier['position'] = (new_lng, new_lat)
                courier['last_update_time'] = current_time
                time_diff = 0
        
        # 更新路径索引
        courier['route_index'] = route_index
        
        # 检查是否完成所有订单
        if route_index >= len(route):
            # 清空当前订单和路径
            courier['current_orders'] = []
            courier['current_route'] = []
            courier['route_index'] = 0
            if courier_id:
                logger.debug(f"骑手 {courier_id} 完成所有订单")
        
        return courier
    
    def _get_food_prepare_time(self, order, current_time):
        """
        获取餐品准备时间
        
        Args:
            order: 订单信息
            current_time: 当前时间
            
        Returns:
            float: 准备时间（分钟）
        """
        # 这里可以根据实际情况实现餐品准备时间的计算
        # 简单起见，这里返回一个随机值（0-5分钟）
        import random
        return random.uniform(0, 5)
    
    def refresh_all_couriers(self, couriers, current_time):
        """
        刷新所有骑手状态
        
        Args:
            couriers: 骑手状态字典
            current_time: 当前时间
            
        Returns:
            dict: 更新后的骑手状态字典
        """
        for courier_id, courier in couriers.items():
            if courier['status'] == 'online':
                couriers[courier_id] = self.refresh_courier_state(courier, current_time, courier_id)
        
        return couriers
