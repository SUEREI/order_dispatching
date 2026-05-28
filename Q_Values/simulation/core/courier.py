#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
骑手模拟器
根据历史订单数据模拟骑手的上下线事件
"""

from typing import List, Tuple
import logging

import pandas as pd
import numpy as np
from datetime import timedelta, datetime

logger = logging.getLogger(__name__)


class CourierSimulator:
    """骑手模拟器"""
    def __init__(self, order_df, idle_timeout_minutes=30, buffer_time=5):
        """初始化骑手模拟器"""
        self.df = order_df
        self.idle_timeout = timedelta(minutes=idle_timeout_minutes)
        self.buffer_time = timedelta(minutes=buffer_time)
        self.events = []
        
        self._build_events()
        self.event_index = 0  # 用于逐步获取事件
        
        logger.info(f"骑手模拟器初始化完成，共生成 {len(self.events)} 个上下线事件")
        logger.info(f"涉及骑手数: {self.df['courier_id'].nunique()}")

    def _build_events(self) -> None:
        """按骑手分组，扫描订单序列，生成上线/下线事件"""
        # 确保时间列是datetime
        self.df = self.df.sort_values(['courier_id', 'dispatch_time'])
        
        events = []
        for courier_id, group in self.df.groupby('courier_id'):
            # 获取订单时间序列
            orders = group[['dispatch_time', 'arrive_time', 'grab_lng', 'grab_lat', 'recipient_lng', 'recipient_lat', 'courier_id_daily_avg_rejected_rate']].values
            first_dispatch = pd.Timestamp(orders[0][0])
            last_arrive = pd.Timestamp(orders[0][1])
            grab_lng = orders[0][2]
            grab_lat = orders[0][3]
            recipient_lng = orders[0][4]
            recipient_lat = orders[0][5]
            courier_id_daily_avg_rejected_rate = orders[0][6]
            
            # 骑手第一次上线：假设在第一单派单时间之前 idle_timeout 时刻上线（避免刚上线立刻接单的突兀）
            # 也可直接用 first_dispatch，但略微前置更平滑
            online_time = first_dispatch - self.buffer_time
            events.append((online_time, courier_id, 'online', (grab_lng, grab_lat), courier_id_daily_avg_rejected_rate))
            
            # 扫描后续订单，判断是否有空闲间隔超过阈值
            for i in range(1, len(orders)):
                current_dispatch = pd.Timestamp(orders[i][0])
                # 计算当前订单前几行数据中的最晚时间节点
                prev_arrives = [pd.Timestamp(orders[j][1]) for j in range(i)]
                last_arrive = max(prev_arrives)
                max_arrive_idx = prev_arrives.index(last_arrive)  # 找到最晚时间对应的索引
                
                gap = current_dispatch - last_arrive
                grab_lng = orders[i][2]
                grab_lat = orders[i][3]
                recipient_lng = orders[max_arrive_idx][4]  # 使用最晚到达订单的recipient坐标
                recipient_lat = orders[max_arrive_idx][5]
                
                if gap > self.idle_timeout:
                    # 骑手在上一单完成后 idle_timeout 时刻下线
                    offline_time = last_arrive
                    events.append((offline_time, courier_id, 'offline', (recipient_lng, recipient_lat), courier_id_daily_avg_rejected_rate))

                    # 然后在下一单开始前 idle_timeout 时刻重新上线
                    online_time = current_dispatch
                    events.append((online_time, courier_id, 'online', (grab_lng, grab_lat), courier_id_daily_avg_rejected_rate))
            
            # 最后一单后的下线事件
            final_offline = max(pd.Timestamp(orders[-1][1]), last_arrive)
            events.append((final_offline, courier_id, 'offline', (recipient_lng, recipient_lat), courier_id_daily_avg_rejected_rate))
        
        # 全局按时间排序
        events.sort(key=lambda x: x[0])
        self.events = events
        # 显示第一个骑手的events作为示例
        if events:
            first_courier_id = events[0][1]
            courier_events = [e for e in events if e[1] == first_courier_id]
            logger.debug(f"骑手 {first_courier_id} 的事件:")
            for event in courier_events:
                logger.debug(f"  时间: {event[0]}, 类型: {event[2]}, 位置: {event[3]}, 拒绝率: {event[4]}")
        
    def get_events_at_time(self, current_time: datetime) -> List[Tuple[str, str, Tuple[str, str], float]]:
        """增量获取某时间点的骑手上下线情况"""
        triggered = []
        while self.event_index < len(self.events):
            t, courier_id, etype, pos, rejected_rate = self.events[self.event_index]
            pos = (pos[0] / 10**6, pos[1] / 10**6)
            if t == current_time:
                # 记录该骑手的当前状态
                triggered.append((courier_id, etype, pos, rejected_rate))
                self.event_index += 1
            else:
                break

        return triggered

    def get_next_event_time(self) -> datetime:
        """返回下一个事件发生的时间，若没有则返回 None"""
        if self.event_index < len(self.events):
            return self.events[self.event_index][0].to_pydatetime()
            
        return None

    def get_all_courier_ids(self):
        """返回所有出现过的骑手ID"""

        return list(set([ev[1] for ev in self.events]))
    
    def get_earliest_event_time(self) -> datetime:
        """返回最早的事件时间"""
        if self.events:
            return min(event[0] for event in self.events)

        return None
    
    def get_latest_event_time(self) -> datetime:
        """返回最晚的事件时间"""
        if self.events:
            return max(event[0] for event in self.events)

        return None
