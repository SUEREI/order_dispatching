#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于DQN的订单分派器模块

该模块实现基于Deep Q-Network强化学习算法的订单分派器，包括：
- 状态向量构建（骑手路径+新订单特征）
- 候选骑手筛选（距离排序Top-K）
- Q值计算与ε-贪婪动作选择
- 边际收益奖励计算
- 经验回放与在线训练
"""

import copy
import logging
import random
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from dqn.dqn_network import DQNAgent, DQNConfig
from core.route import RoutePlanner
from utils import haversine_distance

logger = logging.getLogger(__name__)


class OrderDispatcherByDQN:
    """基于DQN的订单分派器"""

    # 状态向量维度常量
    MAX_ROUTE_POINTS = 8       # 最多8个路径点（4订单×2点）
    ROUTE_POINT_DIM = 16       # 8点×2坐标 = 16维
    NEW_ORDER_COORD_DIM = 4    # 新订单4个坐标 = 4维
    AVAILABLE_TIME_DIM = 1     # 可用配送时长 = 1维
    STATE_DIM = 21             # 总维度

    def __init__(self, route_planner: RoutePlanner, config: DQNConfig = None):
        """初始化DQN分派器"""
        self.route_planner = route_planner
        self.config = config or DQNConfig()
        self.dqn_agent = DQNAgent(self.STATE_DIM, self.config)
        # 临时骑手订单计数，用于批量派单时实时更新容量
        self.temp_order_counts: Dict[str, int] = {}

        logger.info("DQN订单分派器初始化完成")

    def dispatch_orders_by_dqn(
        self,
        orders: List[Dict],
        couriers: Dict[str, Dict],
        current_time: datetime
    ) -> Dict[str, str]:
        """批量DQN派单"""
        dispatch_results = {}
        # 重置临时订单计数
        self.temp_order_counts = {cid: 0 for cid in couriers}

        for order in orders:
            # 1. 筛选候选骑手
            candidates = self._select_candidates(order, couriers)
            if not candidates:
                logger.debug(f"订单 {order['order_id']} 无可用候选骑手，跳过")
                continue

            candidate_ids = [cid for cid, _ in candidates]
            candidate_couriers = [c for _, c in candidates]

            # 2. 构建状态向量并批量计算Q值
            try:
                states = np.array([
                    self._build_state(c, order, current_time)
                    for c in candidate_couriers
                ], dtype=np.float32)
                q_values = self.dqn_agent.batch_predict(states)
            except Exception as e:
                # 退化策略：选择距离最近的骑手（候选集已按距离排序）
                logger.warning(f"DQN推理失败({e})，退化为距离贪心策略")
                chosen_id = candidate_ids[0]
                dispatch_results[order['order_id']] = chosen_id
                self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1
                continue

            # 3. ε-贪婪选择骑手
            chosen_id = self._choose_action(q_values, candidate_ids)

            # 4. 计算奖励
            chosen_courier = candidate_couriers[candidate_ids.index(chosen_id)]
            reward = self._calculate_reward(chosen_courier, order, current_time)

            # 5. 构建next_state（骑手加入新订单后的状态）
            next_state = self._build_next_state(chosen_courier, order, current_time)

            # 6. 计算done：骑手当前订单为空且曾经有过订单时为True（刚上线时不为done）
            current_order_count = len(chosen_courier.get('current_orders', []))
            has_had_orders = chosen_courier.get('has_had_orders', False)
            done = (current_order_count == 0 and has_had_orders)

            # 7. 存储经验
            chosen_index = candidate_ids.index(chosen_id)
            current_state = states[chosen_index]
            self.dqn_agent.replay_buffer.push(current_state, chosen_index, reward, next_state, done)

            # 7. 尝试训练
            if len(self.dqn_agent.replay_buffer) >= self.config.batch_size:
                batch = self.dqn_agent.replay_buffer.sample(self.config.batch_size)
                loss = self.dqn_agent.update(batch)
                logger.debug(f"DQN训练损失: {loss:.6f}")

            # 8. 更新派单结果和临时订单计数
            dispatch_results[order['order_id']] = chosen_id
            self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1

            logger.debug(
                f"订单 {order['order_id']} 分派给骑手 {chosen_id}，"
                f"Q值: {q_values[chosen_index]:.4f}，奖励: {reward:.4f}，ε: {self.dqn_agent.epsilon:.4f}"
            )

        return dispatch_results

    def dispatch_orders_by_dqn_inference(
        self,
        orders: List[Dict],
        couriers: Dict[str, Dict],
        current_time: datetime
    ) -> Dict[str, str]:
        """批量DQN派单（仅推理，不训练）

        用于已训练模型的测试或实际使用，直接利用Q值选择最优动作。

        Args:
            orders: 待分配订单列表
            couriers: 在线骑手字典 {courier_id: courier_dict}
            current_time: 当前仿真时间

        Returns:
            派单结果映射 {order_id: courier_id}
        """
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
                    self._build_state(c, order, current_time)
                    for c in candidate_couriers
                ], dtype=np.float32)
                q_values = self.dqn_agent.batch_predict(states)
            except Exception as e:
                logger.warning(f"DQN推理失败({e})，退化为距离贪心策略")
                chosen_id = candidate_ids[0]
                dispatch_results[order['order_id']] = chosen_id
                self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1
                continue

            chosen_index = int(np.argmax(q_values))
            chosen_id = candidate_ids[chosen_index]

            dispatch_results[order['order_id']] = chosen_id
            self.temp_order_counts[chosen_id] = self.temp_order_counts.get(chosen_id, 0) + 1

            logger.debug(
                f"订单 {order['order_id']} 分派给骑手 {chosen_id}，"
                f"Q值: {q_values[chosen_index]:.4f}"
            )

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
            # 过滤：必须online
            if courier.get('status') != 'online':
                continue
            # 过滤：position不为None
            if courier.get('position') is None:
                continue
            # 过滤：当前订单数+临时计数 < 上限
            current_count = len(courier.get('current_orders', []))
            temp_count = self.temp_order_counts.get(cid, 0)
            if current_count + temp_count >= max_orders:
                continue

            distance = haversine_distance(order_pos, courier['position'])
            courier_distances.append((cid, courier, distance))

        # 按距离升序排序，取前K名
        courier_distances.sort(key=lambda x: x[2])
        candidates = [(cid, c) for cid, c, _ in courier_distances[:candidate_count]]

        return candidates

    def _build_state(
        self,
        courier: Dict,
        order: Dict,
        current_time: datetime
    ) -> np.ndarray:
        """构建状态向量"""
        state = np.zeros(self.STATE_DIM, dtype=np.float32)

        # 1. 编码路径点坐标（16维）
        route = courier.get('route', [])
        for i, route_point in enumerate(route[:self.MAX_ROUTE_POINTS]):
            lng, lat = route_point['position']
            state[i * 2] = self._normalize_lng(lng)
            state[i * 2 + 1] = self._normalize_lat(lat)

        # 2. 编码新订单坐标（4维）
        state[self.ROUTE_POINT_DIM] = self._normalize_lng(order['sender_lng'])
        state[self.ROUTE_POINT_DIM + 1] = self._normalize_lat(order['sender_lat'])
        state[self.ROUTE_POINT_DIM + 2] = self._normalize_lng(order['recipient_lng'])
        state[self.ROUTE_POINT_DIM + 3] = self._normalize_lat(order['recipient_lat'])

        # 3. 编码可用配送时长（1维）
        available_time = self._calculate_available_time(order, current_time)
        state[self.ROUTE_POINT_DIM + self.NEW_ORDER_COORD_DIM] = available_time / self.config.max_delivery_time

        return state

    def _build_next_state(
        self,
        courier: Dict,
        order: Dict,
        current_time: datetime
    ) -> np.ndarray:
        """构建骑手接收新订单后的下一状态向量"""
        # 深拷贝骑手订单列表
        temp_orders = copy.deepcopy(courier.get('current_orders', []))
        # 加入新订单（status='pending'）
        new_order = copy.deepcopy(order)
        new_order['status'] = 'pending'
        temp_orders.append(new_order)

        # 构造临时骑手字典
        temp_courier = {
            'courier_id': courier['courier_id'],
            'position': courier['position'],
            'current_orders': temp_orders
        }

        # 重新规划路径
        new_route = self.route_planner.plan_route(temp_courier)

        # 用新路径构建状态
        temp_courier_with_route = {**temp_courier, 'route': new_route}
        return self._build_state(temp_courier_with_route, order, current_time)

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

    def _choose_action(self, q_values: List[float], candidate_ids: List[str]) -> str:
        """ε-贪婪动作选择

        Args:
            q_values: 各候选骑手的Q值列表
            candidate_ids: 候选骑手ID列表

        Returns:
            选择的骑手ID
        """
        if random.random() < self.dqn_agent.epsilon:
            # 探索：随机选择
            chosen_index = random.randint(0, len(candidate_ids) - 1)
        else:
            # 利用：选择Q值最大的
            chosen_index = int(np.argmax(q_values))

        # 衰减epsilon
        self.dqn_agent.epsilon = max(
            self.config.epsilon_min,
            self.dqn_agent.epsilon * self.config.epsilon_decay
        )

        return candidate_ids[chosen_index]

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
        # 1. 计算当前收益
        current_revenue = self._calculate_courier_revenue(courier, current_time)

        # 2. 临时加入新订单，计算调整后收益
        temp_orders = copy.deepcopy(courier.get('current_orders', []))
        new_order = copy.deepcopy(order)
        new_order['status'] = 'pending'
        temp_orders.append(new_order)

        temp_courier = {
            'courier_id': courier['courier_id'],
            'position': courier['position'],
            'current_orders': temp_orders
        }

        # 重新规划路径
        new_route = self.route_planner.plan_route(temp_courier)
        if not new_route:
            # 路径规划失败，给予负奖励
            logger.debug(f"骑手 {courier['courier_id']} 加入订单 {order['order_id']} 后路径规划失败")
            return -1.0

        temp_courier['route'] = new_route
        adjusted_revenue = self._calculate_courier_revenue(temp_courier, current_time)

        # 3. 边际奖励
        reward = adjusted_revenue - current_revenue
        return reward

    def _calculate_courier_revenue(
        self,
        courier: Dict,
        current_time: datetime
    ) -> float:
        """计算骑手在当前路径规划下的预期总收益

        对每个订单：
        - 基础收益 = base_revenue (3元)
        - 配送距离 = haversine_distance(取餐点, 送餐点) / 1000 (km)
        - 订单收益 = 基础收益 + 配送距离 × distance_revenue_coeff
        - 若超时则施加惩罚

        Args:
            courier: 骑手字典（含current_orders和route）
            current_time: 当前仿真时间

        Returns:
            预期总收益
        """
        total_revenue = 0.0
        orders = courier.get('current_orders', [])
        route = courier.get('route', [])

        # 构建订单字典
        orders_dict = {o['order_id']: o for o in orders}

        # 模拟沿路径行进，估算各订单送达时间
        COURIER_SPEED = 30.0  # km/h，与StateRefresher一致
        current_pos = courier.get('position')
        if current_pos is None:
            return 0.0

        # 估算各订单的送达时间
        delivery_times = {}
        elapsed_time = 0.0  # 从current_time开始累计的秒数

        for route_point in route:
            target_pos = route_point['position']
            distance = haversine_distance(current_pos, target_pos)  # 米
            travel_time = distance / (COURIER_SPEED * 1000 / 3600)  # 秒
            elapsed_time += travel_time
            current_pos = target_pos

            order_id = route_point['order_id']
            if route_point['point_type'] == 'delivery':
                # 送达时间 = 当前时间 + 累计时间 + 3分钟等待
                delivery_times[order_id] = current_time + __import__('datetime').timedelta(
                    seconds=elapsed_time + 180
                )

        # 计算各订单收益
        for order in orders:
            order_id = order['order_id']
            # 基础收益
            base = self.config.base_revenue
            # 配送距离
            pickup_pos = (order['sender_lng'], order['sender_lat'])
            delivery_pos = (order['recipient_lng'], order['recipient_lat'])
            dist_km = haversine_distance(pickup_pos, delivery_pos) / 1000
            order_revenue = base + dist_km * self.config.distance_revenue_coeff

            # 超时惩罚
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

    def save_model(self, path: str):
        """保存DQN模型参数到文件"""
        self.dqn_agent.save(path)

    def load_model(self, path: str):
        """从文件加载DQN模型参数"""
        self.dqn_agent.load(path)
