import numpy as np
import logging

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderDispatcher:
    """订单派单器"""
    
    def __init__(self):
        """初始化派单器"""
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
    
    def dispatch_to_nearest_courier(self, order, available_couriers):
        """
        将订单派给最近的可用骑手
        
        Args:
            order: 订单对象，包含 'sender_lng' 和 'sender_lat' 字段
            available_couriers: 可用骑手列表，每个骑手包含 'courier_id' 和 'position' 字段
            
        Returns:
            tuple: (courier_id, distance) 如果有可用骑手
            None: 如果没有可用骑手
        """
        if not available_couriers:
            return None
        
        # 获取订单起点位置
        order_pos = (order['sender_lng'], order['sender_lat'])
        
        # 计算每个骑手到订单起点的距离
        courier_distances = []
        for courier in available_couriers:
            if courier['position'] is not None:
                distance = self.calculate_distance(order_pos, courier['position'])
                courier_distances.append((courier['courier_id'], distance))
        
        if not courier_distances:
            return None
        
        # 找出距离最近的骑手
        nearest_courier = min(courier_distances, key=lambda x: x[1])
        return nearest_courier
    
    def dispatch_orders(self, orders, couriers):
        """
        批量处理订单派单
        
        Args:
            orders: 待派订单列表
            couriers: 骑手状态字典，格式为 {courier_id: {'status': 'online'/'offline', 'position': (lng, lat), ...}}
            
        Returns:
            dict: 派单结果，格式为 {order_id: courier_id}
        """
        # 过滤出在线且有位置信息的骑手
        available_couriers = []
        for courier_id, status in couriers.items():
            if status['status'] == 'online' and status['position'] is not None:
                available_couriers.append({
                    'courier_id': courier_id,
                    'position': status['position']
                })
        
        dispatch_results = {}
        
        for order in orders:
            result = self.dispatch_to_nearest_courier(order, available_couriers)
            if result:
                courier_id, distance = result
                dispatch_results[order['order_id']] = courier_id
                # logger.debug(f"订单 {order['order_id']} 派给骑手 {courier_id}，距离: {distance:.4f}")
            else:
                logger.debug(f"订单 {order['order_id']} 无法派单，无可用骑手")
        
        return dispatch_results
