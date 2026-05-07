#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 DQN (Actor-Critic) 的订单派单器模块

功能:
- 构建 25 维骑手状态向量
- 构建 5 维订单动作向量
- 合法性过滤 (容量检查)
- 基于 Q 值的贪心派单

状态向量 (固定 27 维):
  维度 0-1:   骑手当前经度、纬度
  维度 2:     当前时间步的时间戳 (归一化 0~1)
  维度 3-18:  路径点 1~8 的经纬度 (共 16 维), 不足 8 个点用 0 填充
  维度 19-26: 路径点对应的 "距离超时的剩余时间" (8 个值, 对应前 8 个路径点)
              取餐点 = 0, 送餐点 = (预计送达时间 - 当前时间) / max_timeout

动作向量 (固定 5 维):
  维度 0-1: 新订单取餐点经度、纬度
  维度 2-3: 新订单送餐点经度、纬度
  维度 4:   该订单的可用配送时长 (归一化)
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OrderDispatcherByActorCritic:
    """
    DQN 订单派单器

    使用 Actor-Critic 网络的 Q(s, a) 值进行订单分配决策。
    支持训练模式 (softmax 采样) 与应用模式 (argmax 贪心)。
    """

    def __init__(
        self,
        agent: ActorCriticAgent,
        config: ActorCriticConfig = None,
    ):
        self.agent = agent
        self.config = config or agent.config

    def build_state_vector(
        self,
        courier: Dict[str, Any],
        current_time: datetime,
        orders_lookup: Dict[str, Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        构建 27 维骑手状态向量

        Args:
            courier: 骑手字典, 含 position, route, current_orders 等
            current_time: 当前仿真时间
            orders_lookup: order_id -> order 的全局订单查找表

        Returns:
            (27,) float32 状态向量
        """
        state_dim = self.config.STATE_DIM
        state = np.zeros(state_dim, dtype=np.float32)

        pos = courier.get("position")
        if pos is not None:
            state[0] = float(pos[0])
            state[1] = float(pos[1])

        total_seconds_in_day = 24.0 * 3600.0
        time_fraction = (
            current_time.hour * 3600.0
            + current_time.minute * 60.0
            + current_time.second
        ) / total_seconds_in_day
        state[2] = np.clip(time_fraction, 0.0, 1.0)

        route = courier.get("route", [])
        if not route:
            route = []

        max_points = 8
        for i, route_point in enumerate(route[:max_points]):
            rp_pos = route_point.get("position")
            if rp_pos is not None:
                state[3 + i * 2] = float(rp_pos[0])
                state[4 + i * 2] = float(rp_pos[1])

        max_time_dim = state_dim - 19
        if max_time_dim > 0 and orders_lookup is not None:
            for i, route_point in enumerate(route[:max_time_dim]):
                point_type = route_point.get("point_type", "")
                if point_type == "pickup":
                    state[19 + i] = 0.0
                elif point_type == "delivery":
                    order_id = route_point.get("order_id")
                    if order_id and order_id in orders_lookup:
                        order = orders_lookup[order_id]
                        estimate_arrived = order.get("estimate_arrived_time")
                        if estimate_arrived is not None:
                            remaining_seconds = (
                                estimate_arrived - current_time
                            ).total_seconds()
                            normalized = remaining_seconds / self.config.MAX_TIME_TO_TIMEOUT
                            state[19 + i] = np.clip(normalized, 0.0, 1.0)

        return state

    def build_action_vector(self, order: Dict[str, Any]) -> np.ndarray:
        """
        构建 5 维动作向量

        Args:
            order: 订单字典, 含 sender_lng, sender_lat, recipient_lng, recipient_lat,
                   dispatch_time, estimate_arrived_time

        Returns:
            (5,) float32 动作向量
        """
        action = np.zeros(self.config.ACTION_DIM, dtype=np.float32)
        action[0] = float(order["sender_lng"])
        action[1] = float(order["sender_lat"])
        action[2] = float(order["recipient_lng"])
        action[3] = float(order["recipient_lat"])

        dispatch_time = order.get("dispatch_time")
        estimate_arrived = order.get("estimate_arrived_time")
        if dispatch_time is not None and estimate_arrived is not None:
            available_seconds = (estimate_arrived - dispatch_time).total_seconds()
            action[4] = np.clip(
                available_seconds / self.config.MAX_TIME_TO_TIMEOUT, 0.0, 1.0
            )

        return action

    def get_valid_couriers(
        self,
        order: Dict[str, Any],
        online_couriers: Dict[str, Dict[str, Any]],
    ) -> List[str]:
        """
        过滤出可以接该订单的在线骑手

        合法性检查: 骑手当前持单量 < MAX_ORDERS (最大持单量 4)

        Args:
            order: 候选订单
            online_couriers: {courier_id: courier_dict}

        Returns:
            合法骑手 ID 列表
        """
        valid_courier_ids = []
        for cid, courier in online_couriers.items():
            current_order_count = len(courier.get("current_orders", []))
            if current_order_count < self.config.MAX_ORDERS:
                valid_courier_ids.append(cid)
        return valid_courier_ids

    def dispatch_orders_by_dqn(
        self,
        pending_orders: List[Dict[str, Any]],
        couriers: Dict[str, Dict[str, Any]],
        current_time: datetime,
    ) -> Dict[str, str]:
        """
        基于 DQN 的 Q 值贪心派单 (订单选骑手)

        算法:
        1. 过滤出在线骑手
        2. 对每个待分配订单:
           a. 构建所有合法骑手的状态向量 (batch)
           b. 使用 Agent 的 compute_q_values_for_riders 批量计算 Q(s_i, a)
           c. 收集所有 (courier_id, order, Q) 三元组
        3. 按 Q 值降序排列
        4. 贪心分配: 每个骑手最多接 MAX_ORDERS 单, 每单只分配给一个骑手

        Args:
            pending_orders: 待分配订单列表
            couriers: 所有骑手字典 {courier_id: courier}
            current_time: 当前仿真时间

        Returns:
            {order_id: courier_id} 派单映射
        """
        online_couriers = {
            cid: c
            for cid, c in couriers.items()
            if c.get("status") == "online" and c.get("position") is not None
        }

        if not online_couriers or not pending_orders:
            return {}

        order_lookup = {o["order_id"]: o for o in pending_orders}

        candidates: List[Tuple[float, str, str]] = []

        for order in pending_orders:
            action_vec = self.build_action_vector(order)
            valid_cids = self.get_valid_couriers(order, online_couriers)

            if not valid_cids:
                continue

            rider_states = []
            for cid in valid_cids:
                courier = online_couriers[cid]
                state_vec = self.build_state_vector(courier, current_time, order_lookup)
                rider_states.append(state_vec)

            rider_states_batch = np.array(rider_states, dtype=np.float32)
            q_vals = self.agent.compute_q_values_for_riders(rider_states_batch, action_vec)
            for i, cid in enumerate(valid_cids):
                candidates.append((float(q_vals[i]), cid, order["order_id"]))

        candidates.sort(key=lambda x: x[0], reverse=True)

        assigned_orders: Dict[str, str] = {}
        courier_order_counts: Dict[str, int] = {
            cid: len(couriers.get(cid, {}).get("current_orders", []))
            for cid in online_couriers
        }

        for q_value, cid, order_id in candidates:
            if order_id in assigned_orders:
                continue
            if courier_order_counts.get(cid, 0) >= self.config.MAX_ORDERS:
                continue

            assigned_orders[order_id] = cid
            courier_order_counts[cid] = courier_order_counts.get(cid, 0) + 1

        return assigned_orders
