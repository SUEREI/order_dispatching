import pandas as pd
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderStateRefresh:
    """订单状态刷新器"""
    
    def __init__(self):
        """初始化订单状态刷新器"""
        pass
    
    def refresh_order_state(self, order, event_type, current_time):
        """
        刷新订单状态
        
        Args:
            order: 订单信息字典
            event_type: 事件类型，'pickup' 或 'dropoff'
            current_time: 当前时间
            
        Returns:
            dict: 更新后的订单状态
        """
        if event_type == 'pickup':
            # 记录取餐时间
            if 'pickup_time' not in order:
                order['pickup_time'] = current_time
                order['status'] = 'picked_up'
                logger.debug(f"订单 {order['order_id']} 于 {current_time} 被取餐")
        elif event_type == 'dropoff':
            # 记录送达时间
            if 'delivery_time' not in order:
                order['delivery_time'] = current_time
                order['status'] = 'delivered'
                # 计算配送时间
                if 'pickup_time' in order:
                    delivery_duration = (current_time - order['pickup_time']).total_seconds() / 60
                    order['delivery_duration'] = delivery_duration
                    logger.debug(f"订单 {order['order_id']} 于 {current_time} 送达，配送时间: {delivery_duration:.2f} 分钟")
                else:
                    logger.debug(f"订单 {order['order_id']} 于 {current_time} 送达")
        
        return order
    
    def refresh_orders_from_couriers(self, couriers, current_time):
        """
        从骑手状态刷新订单状态
        
        Args:
            couriers: 骑手状态字典
            current_time: 当前时间
            
        Returns:
            dict: 所有订单的状态
        """
        orders = {}
        
        for courier_id, courier in couriers.items():
            if courier['status'] == 'online' and 'current_route' in courier and courier['current_route']:
                route = courier['current_route']
                route_index = courier['route_index']
                
                # 处理已完成的路径点
                for i in range(route_index):
                    point = route[i]
                    order_id = point[3]
                    event_type = point[2]
                    
                    # 找到对应的订单
                    for order in courier['current_orders']:
                        if order['order_id'] == order_id:
                            order = self.refresh_order_state(order, event_type, current_time)
                            orders[order_id] = order
                            break
        
        return orders
    
    def get_order_status_summary(self, orders):
        """
        获取订单状态摘要
        
        Args:
            orders: 订单字典
            
        Returns:
            dict: 订单状态统计
        """
        status_counts = {
            'pending': 0,
            'picked_up': 0,
            'delivered': 0
        }
        
        for order in orders.values():
            status = order.get('status', 'pending')
            if status in status_counts:
                status_counts[status] += 1
        
        return status_counts
