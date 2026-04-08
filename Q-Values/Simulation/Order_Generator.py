import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderGenerator:
    """订单生成器"""
    
    def __init__(self, data_path):
        """
        初始化订单生成器
        
        Args:
            data_path: 清洗后的数据文件路径
        """
        logger.info(f"正在加载订单数据: {data_path}")
        self.df = pd.read_csv(data_path)
        self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        self.orders_by_time = self._group_orders_by_time()
        self.current_time = None
        logger.info(f"订单数据加载完成，共 {len(self.df)} 条订单")
    
    def _group_orders_by_time(self):
        """
        按时间分组订单
        
        Returns:
            dict: 时间到订单列表的映射
        """
        orders_by_time = {}
        for _, row in self.df.iterrows():
            time_key = row['dispatch_time']
            if time_key not in orders_by_time:
                orders_by_time[time_key] = []
            orders_by_time[time_key].append({
                'order_id': row['order_id'],
                'sender_lng': row['sender_lng'],
                'sender_lat': row['sender_lat'],
                'recipient_lng': row['recipient_lng'],
                'recipient_lat': row['recipient_lat'],
                'dispatch_time': row['dispatch_time'],
                'estimate_meal_prepare_time': row['estimate_meal_prepare_time'],
                'estimate_arrived_time': row['estimate_arrived_time']
            })
        return orders_by_time
    
    def get_orders_at_time(self, current_time):
        """
        获取指定时间的订单
        
        Args:
            current_time: 当前时间
            
        Returns:
            list: 订单列表
        """
        self.current_time = current_time
        return self.orders_by_time.get(current_time, [])
    
    def get_all_timestamps(self):
        """
        获取所有时间戳
        
        Returns:
            list: 时间戳列表
        """
        return sorted(self.orders_by_time.keys())
    
    def get_time_range(self):
        """
        获取时间范围
        
        Returns:
            tuple: (开始时间, 结束时间)
        """
        timestamps = self.get_all_timestamps()
        if timestamps:
            return timestamps[0], timestamps[-1]
        return None, None


if __name__ == '__main__':
    # 测试订单生成器
    generator = OrderGenerator('Q-values/Data/Data_Processed/cleaned_waybill_info.csv')
    start_time, end_time = generator.get_time_range()
    logger.info(f"时间范围: {start_time} 到 {end_time}")
    
    # 测试获取指定时间的订单
    test_time = datetime(2022, 10, 17, 10, 1, 0)
    orders = generator.get_orders_at_time(test_time)
    logger.info(f"{test_time} 时刻的订单数量: {len(orders)}")
    if orders:
        logger.info(f"订单: {orders}")
