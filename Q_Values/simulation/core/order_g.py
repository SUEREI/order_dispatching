#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单生成器
"""
import logging
from typing import List, Tuple, Dict

import pandas as pd
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)


class OrderGenerator:
    """订单生成器"""
    def __init__(self, order_df):
        """初始化订单生成器"""
        logger.info(f"正在加载订单数据")
        self.df = order_df
        self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        self.orders_by_time = self._group_orders_by_time()
        self.current_time = None
        logger.info(f"订单数据加载完成，共 {len(self.df)} 条订单")
    
    def _group_orders_by_time(self) -> Dict[datetime, List[Dict]]:
        """按时间分组订单"""
        orders_by_time = {}
        for _, row in self.df.iterrows():
            time_key = row['dispatch_time']
            if time_key not in orders_by_time:
                orders_by_time[time_key] = []
            orders_by_time[time_key].append({
                'order_id': row['order_id'],
                'sender_lng': row['sender_lng'] / 10**6,
                'sender_lat': row['sender_lat'] / 10**6,
                'recipient_lng': row['recipient_lng'] / 10**6,
                'recipient_lat': row['recipient_lat'] / 10**6,
                'dispatch_time': pd.to_datetime(row['dispatch_time']),
                'estimate_meal_prepare_time': pd.to_datetime(row['estimate_meal_prepare_time']),
                'estimate_arrived_time': pd.to_datetime(row['estimate_arrived_time']),
                'status': 'pending',
                'courier_id': None,
                'actual_pickup_time': pd.to_datetime('2099-12-31 23:59:59'),
                'actual_delivery_time': pd.to_datetime('2099-12-31 23:59:59')
            })

        return orders_by_time
        
    def get_orders_at_time(self, current_time: datetime) -> List[Dict]:
        """获取指定时间的订单"""
        self.current_time = current_time

        return self.orders_by_time.get(current_time, [])
    
    def get_all_timestamps(self) -> List[datetime]:
        """获取所有时间戳"""

        return sorted(self.orders_by_time.keys())
    
    def get_time_range(self) -> Tuple[datetime, datetime]:
        """获取时间范围"""
        timestamps = self.get_all_timestamps()
        if timestamps:
            return timestamps[0], timestamps[-1]

        return None, None
