#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单分派器 - 基于边际收益的启发式派单

派单流程：
1. 筛选出订单取餐点附近最近的 N 名在线骑手（默认 300，由 top_k_with_dist 配置）
2. 对每个候选骑手，模拟临时加入该订单并重新规划路径，
   按照与 DQN/Actor-Critic 完全相同的收益计算方式（_calculate_courier_value，
   支持收益型/时间型，见 reward_mode）计算边际收益
3. 选择边际收益最大的骑手进行派单

该派单器继承 OrderDispatcherByActorCritic，复用其收益计算、候选筛选、
拒单预测等全部方法，保证与神经网络派单的收益口径一致，仅派单决策
逻辑不同（贪心选择边际收益最大，而非神经网络输出）。
"""

import logging
import random
from datetime import datetime
from typing import Dict, List, Optional

from core.route import RoutePlanner
from core.order_d_dqn import OrderDispatcherByActorCritic
from dqn.dqn_network import ActorCriticConfig

logger = logging.getLogger(__name__)


class OrderDispatcherByMarginalRevenue(OrderDispatcherByActorCritic):
    """基于边际收益的启发式订单分派器

    对每个待分配订单：
    1. 用 _select_candidates 筛选最近的前 N 名骑手（与 DQN 相同的候选筛选）
    2. 对每个候选骑手模拟加单，计算边际收益（与 DQN 相同的奖励计算方式）
    3. 选择边际收益最大的骑手派单；开启拒单仿真时按预测接单概率拒单
    """

    def __init__(
        self,
        route_planner: RoutePlanner,
        config: ActorCriticConfig = None,
        grab_prediction: Optional[bool] = None,
        candidate_selection: Optional[str] = 'distance',
        dt_prob_threshold: float = 0.5,
    ):
        """初始化基于边际收益的启发式派单器

        Args:
            route_planner: 路径规划器
            config: 超参数配置
            grab_prediction: 是否开启拒单仿真；为 None 时从 config.yaml 读取
            candidate_selection: 候选骑手筛选方式，默认 'distance'（纯距离筛选）
            dt_prob_threshold: decision_tree_threshold 模式阈值（保留参数）
        """
        super().__init__(
            route_planner,
            config=config,
            grab_prediction=grab_prediction,
            candidate_selection=candidate_selection,
            dt_prob_threshold=dt_prob_threshold,
        )
        logger.info("基于边际收益的启发式派单器初始化完成")

    def _calc_marginal_reward(
        self,
        courier: Dict,
        order: Dict,
        current_time: datetime
    ) -> float:
        """计算给骑手临时加入该订单后的边际收益（不修改骑手实际状态）

        边际收益 = (加单后总价值 - 加单前总价值) - 距离惩罚系数 × 新增距离(km)

        与 DQN 派单中 reward 的计算方式完全一致。

        Args:
            courier: 骑手字典（含current_orders和route）
            order: 待分配订单
            current_time: 当前仿真时间

        Returns:
            边际收益
        """
        value_before = self._calculate_courier_value(courier, current_time)
        distance_before = self._calculate_route_distance(courier)

        # 保存原状态，模拟加单
        old_orders = list(courier.get('current_orders', []))
        old_route = list(courier.get('route', []))

        courier['current_orders'].append(order)
        courier['route'] = self.route_planner.plan_route(courier)

        value_after = self._calculate_courier_value(courier, current_time)
        distance_after = self._calculate_route_distance(courier)

        # 恢复骑手原状态
        courier['current_orders'] = old_orders
        courier['route'] = old_route

        marginal_value = value_after - value_before
        extra_distance_km = (distance_after - distance_before) / 1000.0
        reward = marginal_value - self.config.distance_penalty_coeff * extra_distance_km

        return reward

    def dispatch_orders(
        self,
        orders: List[Dict],
        couriers: Dict[str, Dict],
        current_time: datetime,
        is_training: bool = True
    ) -> Dict[str, str]:
        """按边际收益最大进行批量派单

        Args:
            orders: 待分配订单列表
            couriers: 在线骑手字典 {courier_id: courier_dict}
            current_time: 当前仿真时间
            is_training: 保留参数（启发式方法无需训练）

        Returns:
            派单结果映射 {order_id: courier_id}，拒单订单为 courier_id_reject
        """
        dispatch_results = {}
        # 临时订单计数：同一时间步内多个订单依次派单时，实时更新容量
        temp_order_counts = {}

        for order in orders:
            order_id = str(order['order_id'])

            # 1. 筛选候选骑手（距离最近的 N 名，与 DQN 相同）
            candidates = self._select_candidates(order, couriers)
            if not candidates:
                logger.debug(f"订单 {order_id} 无可用候选骑手，跳过")
                continue

            # 过滤临时计数已满的骑手（同一步内先派单的订单占用了容量）
            feasible = []
            for cid, courier in candidates:
                current_count = len(courier.get('current_orders', [])) + temp_order_counts.get(cid, 0)
                if current_count < self.config.max_orders_per_courier:
                    feasible.append((cid, courier))

            if not feasible:
                logger.debug(f"订单 {order_id} 候选骑手均已满载，跳过")
                continue

            # 2. 对每个候选骑手计算边际收益，选择最大的
            best_cid = None
            best_reward = float('-inf')
            for cid, courier in feasible:
                try:
                    marginal = self._calc_marginal_reward(courier, order, current_time)
                except Exception as e:
                    logger.warning(f"订单 {order_id} 计算骑手 {cid} 边际收益失败({e})，跳过该骑手")
                    continue
                if marginal > best_reward:
                    best_reward = marginal
                    best_cid = cid

            if best_cid is None:
                logger.debug(f"订单 {order_id} 所有候选骑手边际收益计算失败，跳过")
                continue

            # 3. 拒单仿真（与 DQN 派单一致）
            if self.grab_prediction:
                try:
                    grab_prob = self.predict_grab_probability_dt(order, couriers[best_cid])
                    grab_prob = grab_prob * self.grab_cof
                    if grab_prob > 1:
                        grab_prob = 1
                    elif grab_prob < 0:
                        grab_prob = 0
                except Exception as e:
                    logger.warning(f"订单 {order_id} 拒单概率预测失败({e})，按不拒单处理")
                    grab_prob = 1.0

                if random.random() > grab_prob:
                    logger.debug(f"订单 {order_id} 骑手 {best_cid} 拒单，概率: {grab_prob:.4f}")
                    dispatch_results[order_id] = best_cid + '_reject'
                    continue

            # 4. 派单并更新临时容量
            dispatch_results[order_id] = best_cid
            temp_order_counts[best_cid] = temp_order_counts.get(best_cid, 0) + 1
            logger.debug(f"订单 {order_id} 派给骑手 {best_cid}，边际收益: {best_reward:.4f}")

        return dispatch_results
