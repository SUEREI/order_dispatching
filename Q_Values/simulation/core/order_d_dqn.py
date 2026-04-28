#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Actor-Critic 的订单分派器模块

该模块实现基于 Actor-Critic 强化学习算法的订单分派器，包括：
- 状态向量构建（25维：骑手位置+时间+路径点+超时剩余时间）
- 动作向量构建（5维：新订单取餐点+送餐点+可用配送时长）
- 候选骑手筛选（距离排序Top-K）
- Q值计算与动作选择（训练模式：Softmax采样，评估模式：Argmax）
- 边际收益奖励计算
- 经验回放与离线交替训练
"""

import copy
import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from dqn.dqn_network import ActorCriticAgent, ActorCriticConfig
from core.route import RoutePlanner
from utils import haversine_distance

logger = logging.getLogger(__name__)


class OrderDispatcherByActorCritic:
    """基于 Actor-Critic 的订单分派器"""

    STATE_DIM = 25
    ACTION_DIM = 5

    MAX_ROUTE_POINTS = 8
    ROUTE_POINT_DIM = 16

    IDX_COURIER_LNG = 0
    IDX_COURIER_LAT = 1
    IDX_CURRENT_TIME = 2
    IDX_ROUTE_START = 3
    IDX_ROUTE_END = 18
    IDX_TIMEOUT_START = 19
    IDX_TIMEOUT_END = 25

    COURIER_SPEED = 30.0

    def __init__(self, route_planner: RoutePlanner, config: ActorCriticConfig = None):
        """初始化 Actor-Critic 分派器"""
        self.route_planner = route_planner
        self.config = config or ActorCriticConfig()
        self.ac_agent = ActorCriticAgent(self.STATE_DIM, self.ACTION_DIM, self.config)
        self.temp_order_counts: Dict[str, int] = {}

        logger.info("Actor-Critic 订单分派器初始化完成")

    def dispatch_orders(
        self,
        orders: List[Dict],
        couriers: Dict[str, Dict],
        current_time: datetime,
        is_training: bool = True
    ) -> Dict[str, str]:
        """批量派单

        Args:
            orders: 待分配订单列表
            couriers: 在线骑手字典 {courier_id: courier_dict}
            current_time: 当前仿真时间
            is_training: 是否为训练模式

        Returns:
            派单结果映射 {order_id: courier_id}
        """
        if is_training:
            return self._dispatch_orders_train(orders, couriers, current_time)
        else:
            return self._dispatch_orders_eval(orders, couriers, current_time)

    def _dispatch_orders_train(
        self,
        orders: List[Dict],
        couriers: Dict[str, Dict],
        current_time: datetime
    ) -> Dict[str, str]:
        """训练模式批量派单"""
        dispatch_results = {}
        self.temp_order_counts = {cid: 0 for cid in couriers}

        for order in orders:
            candidates = self._select_candidates(order, couriers)
            if not candidates:
                logger.debug(f"订单 {order['order_id']} 无可用候选骑手，跳过")
                continue

            candidate_ids = [cid for cid, _ in candidates]
            candidate_couriers = [c for _, c in candidates]

            try:
                states = np.array([
                    self._build_state(c, current_time)
                    for c in candidate_couriers
                ], dtype=np.float32)

                candidate_actions = np.array([
                    self._build_action(order, current_time)
                    for _ in candidate_couriers
                ], dtype=np.float32)

                selected_action, log_prob = self.ac_agent.select_action(
                    states[0],
                    candidate_actions
                )

                chosen_index = 0
                for i, aid in enumerate(candidate_ids):
                    if np.allclose(selected_action, candidate_actions[i]):
                        chosen_index = i
                        break

                chosen_id = candidate_ids[chosen_index]
                chosen_courier = candidate_couriers[chosen_index]

                reward = self._calculate_reward(chosen_courier, order, current_time)
                next_state = self._build_next_state(chosen_courier, order, current_time)

                current_state = states[chosen_index]
                self.ac_agent.replay_buffer.push(current_state, selected_action, reward, next_state)

                dispatch_results[order['order_id']] = chosen_id
                self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1

                logger.debug(
                    f"订单 {order['order_id']} 分派给骑手 {chosen_id}，"
                    f"奖励: {reward:.4f}，log_prob: {log_prob:.4f}"
                )

            except Exception as e:
                logger.warning(f"Actor-Critic 推理失败({e})，退化为距离贪心策略")
                chosen_id = candidate_ids[0]
                dispatch_results[order['order_id']] = chosen_id
                self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1
                continue

        return dispatch_results

    def _dispatch_orders_eval(
        self,
        orders: List[Dict],
        couriers: Dict[str, Dict],
        current_time: datetime
    ) -> Dict[str, str]:
        """评估模式批量派单（仅推理，不训练）"""
        dispatch_results = {}
        self.temp_order_counts = {cid: 0 for cid in couriers}

        for order in orders:
            candidates = self._select_candidates(order, couriers)
            if not candidates:
                logger.debug(f"订单 {order['order_id']} 无可用候选骑手，跳过")
                continue

            candidate_ids = [cid for cid, _ in candidates]
            candidate_couriers = [c for _, c in candidates]

            try:
                states = np.array([
                    self._build_state(c, current_time)
                    for c in candidate_couriers
                ], dtype=np.float32)

                candidate_actions = np.array([
                    self._build_action(order, current_time)
                    for _ in candidate_couriers
                ], dtype=np.float32)

                selected_action, _ = self.ac_agent.select_action(
                    states[0],
                    candidate_actions
                )

                chosen_index = 0
                for i, aid in enumerate(candidate_ids):
                    if np.allclose(selected_action, candidate_actions[i]):
                        chosen_index = i
                        break

                chosen_id = candidate_ids[chosen_index]

                dispatch_results[order['order_id']] = chosen_id
                self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1

                logger.debug(
                    f"订单 {order['order_id']} 分派给骑手 {chosen_id}"
                )

            except Exception as e:
                logger.warning(f"Actor-Critic 推理失败({e})，退化为距离贪心策略")
                chosen_id = candidate_ids[0]
                dispatch_results[order['order_id']] = chosen_id
                self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1
                continue

        return dispatch_results

    def _select_candidates(
        self,
        order: Dict,
        couriers: Dict[str, Dict]
    ) -> List[Tuple[str, Dict]]:
        """筛选候选骑手"""
        order_pos = (order['sender_lng'], order['sender_lat'])
        max_orders = self.config.max_orders_per_courier
        candidate_count = self.config.candidate_count

        courier_distances = []
        for cid, courier in couriers.items():
            if courier.get('status') != 'online':
                continue
            if courier.get('position') is None:
                continue

            current_count = len(courier.get('current_orders', []))
            temp_count = self.temp_order_counts.get(cid, 0)
            if current_count + temp_count >= max_orders:
                continue

            distance = haversine_distance(order_pos, courier['position'])
            courier_distances.append((cid, courier, distance))

        courier_distances.sort(key=lambda x: x[2])
        candidates = [(cid, c) for cid, c, _ in courier_distances[:candidate_count]]

        return candidates

    def _build_state(self, courier: Dict, current_time: datetime) -> np.ndarray:
        """构建25维状态向量

        维度分配：
        - 0-1: 骑手当前经度、纬度
        - 2: 当前时间步的时间戳（归一化，如一天中的比例 0~1）
        - 3-18: 路径点1~8的经纬度（共16维）
        - 19-24: 路径点1~8对应的距离超时的剩余时间（共6维）

        Args:
            courier: 骑手字典
            current_time: 当前仿真时间

        Returns:
            25维状态向量
        """
        state = np.zeros(self.STATE_DIM, dtype=np.float32)

        if courier.get('position'):
            state[self.IDX_COURIER_LNG] = self._normalize_lng(courier['position'][0])
            state[self.IDX_COURIER_LAT] = self._normalize_lat(courier['position'][1])

        time_ratio = (current_time.hour * 3600 + current_time.minute * 60 + current_time.second) / 86400.0
        state[self.IDX_CURRENT_TIME] = time_ratio

        route = courier.get('route', [])
        for i, route_point in enumerate(route[:self.MAX_ROUTE_POINTS]):
            lng, lat = route_point['position']
            state[self.IDX_ROUTE_START + i * 2] = self._normalize_lng(lng)
            state[self.IDX_ROUTE_START + i * 2 + 1] = self._normalize_lat(lat)

            timeout_remaining = self._calculate_timeout_remaining(route_point, current_time)
            state[self.IDX_TIMEOUT_START + i] = timeout_remaining

        return state

    def _build_action(self, order: Dict, current_time: datetime) -> np.ndarray:
        """构建5维动作向量

        维度分配：
        - 0-1: 新订单取餐点经度、纬度
        - 2-3: 新订单送餐点经度、纬度
        - 4: 该订单的可用配送时长（归一化）

        Args:
            order: 订单字典
            current_time: 仿真当前时间

        Returns:
            5维动作向量
        """
        action = np.zeros(self.ACTION_DIM, dtype=np.float32)

        action[0] = self._normalize_lng(order['sender_lng'])
        action[1] = self._normalize_lat(order['sender_lat'])
        action[2] = self._normalize_lng(order['recipient_lng'])
        action[3] = self._normalize_lat(order['recipient_lat'])

        available_time = self._calculate_available_time(order, current_time)
        action[4] = available_time / self.config.max_delivery_time

        return action

    def _build_next_state(
        self,
        courier: Dict,
        order: Dict,
        current_time: datetime
    ) -> np.ndarray:
        """构建骑手接收新订单后的下一状态向量"""
        temp_orders = copy.deepcopy(courier.get('current_orders', []))
        new_order = copy.deepcopy(order)
        new_order['status'] = 'pending'
        temp_orders.append(new_order)

        temp_courier = {
            'courier_id': courier['courier_id'],
            'position': courier['position'],
            'current_orders': temp_orders
        }

        new_route = self.route_planner.plan_route(temp_courier)
        temp_courier['route'] = new_route
        return self._build_state(temp_courier, current_time)

    def _calculate_available_time(self, order: Dict, current_time: datetime) -> float:
        """计算新订单的可用配送时长"""
        estimate_arrived = order.get('estimate_arrived_time')
        if estimate_arrived is None or (isinstance(estimate_arrived, pd.Timestamp) and pd.isna(estimate_arrived)):
            return self.config.max_delivery_time
        try:
            delta = (estimate_arrived - current_time).total_seconds()
            return max(0.0, delta)
        except Exception:
            return self.config.max_delivery_time

    def _calculate_timeout_remaining(
        self,
        route_point: Dict,
        current_time: datetime
    ) -> float:
        """计算路径点的超时剩余时间（归一化到 0~1）

        Args:
            route_point: 路径点字典
            current_time: 当前仿真时间

        Returns:
            归一化的超时剩余时间（0~1），负数（已超时）设为0
        """
        if route_point.get('point_type') == 'pickup':
            return 0
        else:
            estimate_arrived = route_point.get('estimate_arrived_time')
            if estimate_arrived:
                remaining = (estimate_arrived - current_time).total_seconds()
                return max(0.0, min(1.0, remaining / self.config.max_delivery_time))

        return 0.0

    def _normalize_lng(self, lng: float) -> float:
        """归一化经度"""
        lng_range = self.config.lng_range
        if lng_range == 0:
            logger.debug("经度范围为0，无法归一化")
            return 0.0
        return (lng - self.config.lng_center) / lng_range

    def _normalize_lat(self, lat: float) -> float:
        """归一化纬度"""
        lat_range = self.config.lat_range
        if lat_range == 0:
            logger.debug("纬度范围为0，无法归一化")
            return 0.0
        return (lat - self.config.lat_center) / lat_range

    def _calculate_reward(
        self,
        courier: Dict,
        order: Dict,
        current_time: datetime
    ) -> float:
        """计算边际收益奖励

        奖励 = 骑手接收新订单后的预期总收益 - 接收前的预期总收益

        Args:
            courier: 骑手字典
            order: 新订单字典
            current_time: 当前仿真时间

        Returns:
            奖励值
        """
        current_revenue = self._calculate_courier_revenue(courier, current_time)

        temp_orders = copy.deepcopy(courier.get('current_orders', []))
        new_order = copy.deepcopy(order)
        new_order['status'] = 'pending'
        temp_orders.append(new_order)

        temp_courier = {
            'courier_id': courier['courier_id'],
            'position': courier['position'],
            'current_orders': temp_orders
        }

        new_route = self.route_planner.plan_route(temp_courier)
        if not new_route:
            logger.debug(f"骑手 {courier['courier_id']} 加入订单 {order['order_id']} 后路径规划失败")
            return -1.0

        temp_courier['route'] = new_route
        adjusted_revenue = self._calculate_courier_revenue(temp_courier, current_time)

        reward = adjusted_revenue - current_revenue
        return reward

    def _calculate_courier_revenue(
        self,
        courier: Dict,
        current_time: datetime
    ) -> float:
        """计算骑手在当前路径规划下的预期总收益

        Args:
            courier: 骑手字典（含current_orders和route）
            current_time: 当前仿真时间

        Returns:
            预期总收益
        """
        total_revenue = 0.0
        orders = courier.get('current_orders', [])
        route = courier.get('route', [])

        speed = self.COURIER_SPEED
        current_pos = courier.get('position')
        if current_pos is None:
            return 0.0

        delivery_times = {}
        elapsed_time = 0.0

        for route_point in route:
            target_pos = route_point['position']
            distance = haversine_distance(current_pos, target_pos)
            travel_time = distance / (speed * 1000 / 3600)
            elapsed_time += travel_time
            current_pos = target_pos

            order_id = route_point['order_id']
            if route_point['point_type'] == 'delivery':
                delivery_times[order_id] = current_time + __import__('datetime').timedelta(
                    seconds=elapsed_time + 180
                )

        for order in orders:
            order_id = order['order_id']
            base = self.config.base_revenue
            pickup_pos = (order['sender_lng'], order['sender_lat'])
            delivery_pos = (order['recipient_lng'], order['recipient_lat'])
            dist_km = haversine_distance(pickup_pos, delivery_pos) / 1000
            order_revenue = base + dist_km * self.config.distance_revenue_coeff

            if order_id in delivery_times:
                estimate_arrived = order.get('estimate_arrived_time')
                if estimate_arrived is not None and not pd.isna(estimate_arrived):
                    overtime_minutes = (delivery_times[order_id] - estimate_arrived).total_seconds() / 60
                    if overtime_minutes > 0:
                        if overtime_minutes <= 10:
                            order_revenue *= 0.9
                        elif overtime_minutes <= 20:
                            order_revenue *= 0.6
                        else:
                            order_revenue *= 0.5

            total_revenue += order_revenue

        return total_revenue

    def set_train_mode(self) -> None:
        """设置为训练模式"""
        self.ac_agent.set_train_mode()

    def set_eval_mode(self) -> None:
        """设置为评估/应用模式"""
        self.ac_agent.set_eval_mode()

    def update_critic_offline(self) -> float:
        """离线训练 Critic（Step 2）"""
        return self.ac_agent.update_critic_offline()

    def update_actor_offline(self) -> float:
        """离线训练 Actor（Step 3）"""
        return self.ac_agent.update_actor_offline()

    def save_model(self, path: str) -> None:
        """保存 Actor-Critic 模型参数到文件"""
        self.ac_agent.save_model(path)

    def load_model(self, path: str) -> None:
        """从文件加载 Actor-Critic 模型参数"""
        self.ac_agent.load_model(path)
