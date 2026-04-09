import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RoutePlanner:
    """路径规划器"""
    
    def __init__(self):
        """初始化路径规划器"""
        pass
    
    def calculate_distance(self, pos1, pos2):
        """
        计算两个位置之间的欧几里得距离
        
        Args:
            pos1: (lng, lat) 第一个位置
            pos2: (lng, lat) 第二个位置
            
        Returns:
            float: 距离
        """
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def plan_route(self, start_pos, orders):
        """
        为骑手规划路径，采用贪婪算法，满足先取餐后送餐的前提
        
        Args:
            start_pos: (lng, lat) 骑手起始位置
            orders: 订单列表，每个订单包含 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat' 字段
            
        Returns:
            list: 路径点列表，格式为 [(lng, lat, type), ...]，其中 type 为 'pickup' 或 'dropoff'
        """
        if not orders:
            return []
        
        route = []
        current_pos = start_pos
        remaining_orders = orders.copy()
        picked_up_orders = set()  # 已取餐的订单ID集合
        
        # 构建所有可能的路径点
        all_points = []
        for order in orders:
            pickup_pos = (order['sender_lng'], order['sender_lat'])
            dropoff_pos = (order['recipient_lng'], order['recipient_lat'])
            all_points.append({
                'order_id': order['order_id'],
                'pos': pickup_pos,
                'type': 'pickup',
                'order': order
            })
            all_points.append({
                'order_id': order['order_id'],
                'pos': dropoff_pos,
                'type': 'dropoff',
                'order': order
            })
        
        # 贪婪算法规划路径
        while all_points:
            # 计算每个点到当前位置的距离
            point_distances = []
            for point in all_points:
                # 如果是送餐点，检查是否已经取餐
                if point['type'] == 'dropoff' and point['order_id'] not in picked_up_orders:
                    continue  # 跳过未取餐的送餐点
                distance = self.calculate_distance(current_pos, point['pos'])
                point_distances.append((point, distance))
            
            if not point_distances:
                break  # 没有可访问的点
            
            # 找到最近的点
            nearest_point, _ = min(point_distances, key=lambda x: x[1])
            
            # 添加到路径
            route.append((nearest_point['pos'][0], nearest_point['pos'][1], nearest_point['type'], nearest_point['order_id']))
            current_pos = nearest_point['pos']
            
            # 更新已取餐订单集合
            if nearest_point['type'] == 'pickup':
                picked_up_orders.add(nearest_point['order_id'])
            
            # 从所有点中移除该点
            all_points.remove(nearest_point)
        
        logger.debug(f"规划路径: {route}")
        return route
    
    def calculate_route_time(self, route, speed=30):
        """
        计算路径所需时间（单位：分钟）
        
        Args:
            route: 路径点列表
            speed: 骑行速度（单位：KM/h）
            
        Returns:
            float: 所需时间（分钟）
        """
        if not route:
            return 0
        
        total_distance = 0
        for i in range(1, len(route)):
            pos1 = (route[i-1][0], route[i-1][1])
            pos2 = (route[i][0], route[i][1])
            total_distance += self.calculate_distance(pos1, pos2)
        
        # 假设坐标单位为米，转换为公里
        total_distance_km = total_distance / 1000
        time_hours = total_distance_km / speed
        time_minutes = time_hours * 60
        
        return time_minutes
