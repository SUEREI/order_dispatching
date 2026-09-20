#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单分派器 - 基于距离贪心策略

将每个待分配订单派发给距离该订单取餐点最近的在线骑手，
作为仿真系统的基线派单策略，与真实匹配派单（order_d_real）、
DQN/Actor-Critic 派单（order_d_dqn）进行对比。

继承 OrderDispatcherByActorCritic，复用其拒单仿真（接单概率预测）：
距离贪心选中最近骑手后，按决策树预测的接单概率判定是否拒单，
保证与 DQN/Actor-Critic 派单的拒单口径一致。
"""

import logging
import random
from typing import Dict, List, Optional

from core.order_d_dqn import OrderDispatcherByActorCritic
from core.route import RoutePlanner
from dqn.dqn_network import ActorCriticConfig
from utils import haversine_distance

logger = logging.getLogger(__name__)


class OrderDispatcherByDistance(OrderDispatcherByActorCritic):
    """基于距离贪心策略的订单分派器

    对每个待分配订单，从在线且未达订单上限的骑手中，
    选择距离订单取餐点（sender_lng, sender_lat）最近的骑手进行派单；
    开启拒单仿真时，按预测接单概率判定拒单。
    """

    def __init__(
        self,
        route_planner: RoutePlanner,
        config: ActorCriticConfig = None,
        grab_prediction: Optional[bool] = None,
        max_orders_per_courier: int = 4,
    ):
        """初始化距离贪心分派器

        Args:
            route_planner: 路径规划器（用于继承 Actor-Critic 派单器的拒单预测能力）
            config: 超参数配置
            grab_prediction: 是否开启拒单仿真；为 None 时从 config.yaml 读取
            max_orders_per_courier: 单个骑手同时承接的最大订单数（容量上限）
        """
        super().__init__(
            route_planner,
            config=config,
            grab_prediction=grab_prediction,
            candidate_selection='distance',
        )
        self.max_orders_per_courier = max_orders_per_courier
        logger.info(
            f"距离贪心订单分派器初始化完成，骑手最大订单数: {max_orders_per_courier}"
        )

    def dispatch_orders_by_distance(self, orders: List[Dict], couriers: Dict[str, Dict]) -> Dict[str, str]:
        """按距离贪心策略批量派单

        Args:
            orders: 待分配订单列表，每个订单需包含
                    order_id / sender_lng / sender_lat
            couriers: 在线骑手字典 {courier_id: courier_dict}

        Returns:
            dispatch_results: {order_id: courier_id} 派单结果映射，
                              拒单订单为 courier_id_reject，
                              无可用骑手的订单不会出现在结果中
        """
        dispatch_results = {}
        # 临时订单计数：同一时间步内多个订单依次派单时，
        # 实时更新骑手已分配订单数，保证不超出容量上限
        temp_order_counts = {}

        for order in orders:
            order_id = str(order['order_id'])
            order_pos = (order['sender_lng'], order['sender_lat'])

            # 1. 筛选可用骑手：在线、有位置、未达订单上限
            available = []
            for cid, courier in couriers.items():
                if courier.get('status') != 'online':
                    continue
                if courier.get('position') is None:
                    continue
                current_count = len(courier.get('current_orders', [])) + temp_order_counts.get(cid, 0)
                if current_count >= self.max_orders_per_courier:
                    continue
                available.append((cid, courier))

            if not available:
                logger.debug(f"订单 {order_id} 无可用骑手，暂不分配")
                continue

            # 2. 选择距离取餐点最近的骑手
            best_cid = None
            best_dist = float('inf')
            for cid, courier in available:
                dist = haversine_distance(courier['position'], order_pos)
                if dist < best_dist:
                    best_dist = dist
                    best_cid = cid

            # 3. 拒单仿真（与 DQN/Actor-Critic 派单一致）
            if self.grab_prediction:
                try:
                    grab_prob = self.predict_grab_probability_dt(order, couriers[best_cid], best_dist)
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

            # 4. 记录派单结果并更新临时容量
            dispatch_results[order_id] = best_cid
            temp_order_counts[best_cid] = temp_order_counts.get(best_cid, 0) + 1
            logger.debug(f"订单 {order_id} 派给骑手 {best_cid}，距离 {best_dist:.1f} 米")

        return dispatch_results
