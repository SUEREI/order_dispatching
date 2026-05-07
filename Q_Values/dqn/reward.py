#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励计算模块

每个决策步（30s）内可能发生多件事，累加它们的奖励：
- 订单完成奖励: 准时 +1, 超时 -1 或 -0.1 * 超时分钟
- 骑手行驶成本: -0.1 * 行驶公里数
- 取餐完成: +0.1
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RewardCalculator:
    """
    奖励计算器

    对比骑手在 30s 前后的状态差异，统计该步内发生的事件并计算奖励。
    """

    def __init__(self, overtime_penalty_mode: str = "linear", linear_penalty_coef: float = 0.1):
        """
        Args:
            overtime_penalty_mode: "binary" (超时 -1) 或 "linear" (-coef * 超时分钟)
            linear_penalty_coef: 线性惩罚系数，默认 0.1
        """
        self.overtime_penalty_mode = overtime_penalty_mode
        self.linear_penalty_coef = linear_penalty_coef

    def compute_reward(
        self,
        courier_before: Dict[str, Any],
        courier_after: Dict[str, Any],
        distance_traveled_km: float,
        current_time: datetime,
        orders_lookup: Dict[str, Dict[str, Any]],
    ) -> float:
        """
        计算骑手在 30s 步内的总奖励

        Args:
            courier_before: 该步开始前的骑手状态快照
            courier_after:  该步结束后的骑手状态
            distance_traveled_km: 该步内骑手行驶距离（公里）
            current_time: 当前仿真时间（该步结束时间）
            orders_lookup: 全局订单查找表 {order_id: order}

        Returns:
            总奖励值
        """
        reward = 0.0

        orders_before = {o["order_id"]: o for o in courier_before.get("current_orders", [])}
        orders_after = {o["order_id"]: o for o in courier_after.get("current_orders", [])}

        completed_order_ids = set(orders_before.keys()) - set(orders_after.keys())
        for oid in completed_order_ids:
            order = orders_lookup.get(oid)
            if order is None:
                continue
            estimate_arrived = order.get("estimate_arrived_time")
            actual_delivery = order.get("actual_delivery_time")
            if estimate_arrived is not None and actual_delivery is not None:
                overtime_minutes = max(0, (actual_delivery - estimate_arrived).total_seconds() / 60.0)
                if overtime_minutes <= 0:
                    reward += 1.0
                else:
                    if self.overtime_penalty_mode == "binary":
                        reward += -1.0
                    else:
                        reward += max(-1.0, -self.linear_penalty_coef * overtime_minutes)
            else:
                reward += 1.0

        picked_up_ids = set()
        for oid, o_after in orders_after.items():
            o_before = orders_before.get(oid)
            if o_before is None:
                continue
            if o_before.get("status") == "pending" and o_after.get("status") == "picked_up":
                picked_up_ids.add(oid)

        for oid in picked_up_ids:
            reward += 0.1

        reward += -0.1 * distance_traveled_km

        return reward

    def compute_reward_from_events(
        self,
        completed_order_infos: List[Dict[str, Any]],
        picked_up_order_ids: List[str],
        distance_traveled_km: float,
    ) -> float:
        """
        通过事件列表直接计算奖励

        Args:
            completed_order_infos: 已完成订单信息列表，每项含
                {"order_id": str, "on_time": bool, "overtime_minutes": float}
            picked_up_order_ids: 已取餐订单 ID 列表
            distance_traveled_km: 行驶距离（公里）

        Returns:
            总奖励值
        """
        reward = 0.0

        for info in completed_order_infos:
            if info.get("on_time", True):
                reward += 1.0
            else:
                overtime_minutes = info.get("overtime_minutes", 0)
                if self.overtime_penalty_mode == "binary":
                    reward += -1.0
                else:
                    reward += max(-1.0, -self.linear_penalty_coef * overtime_minutes)

        reward += 0.1 * len(picked_up_order_ids)
        reward += -0.1 * distance_traveled_km

        return reward
