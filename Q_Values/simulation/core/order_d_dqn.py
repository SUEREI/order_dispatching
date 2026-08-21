#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于 Actor-Critic 的订单分派器模块

该模块实现基于 Actor-Critic 强化学习算法的订单分派器，包括：
- 状态向量构建（27维：骑手位置+时间+路径点+超时剩余时间）
- 动作向量构建（5维：新订单取餐点+送餐点+可用配送时长）
- 候选骑手筛选（距离排序Top-K）
- Q值计算与动作选择（训练模式：Softmax采样，评估模式：Argmax）
- 边际收益奖励计算
- 经验回放与离线交替训练
"""

import copy
import joblib
import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from dqn.dqn_network import ActorCriticAgent, ActorCriticConfig
from core.route import RoutePlanner
from utils import haversine_distance, haversine_distance_vectorized

# 确保项目根目录在 sys.path 中，用于加载 SPOT 模型
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from SPOT.core.smart_decision_tree import SimilaritySetDecisionTree

# pickle 文件中类引用记录为 __main__.SimilaritySetDecisionTree，
# 需要注册到 __main__ 模块才能正常反序列化
sys.modules['__main__'].SimilaritySetDecisionTree = SimilaritySetDecisionTree

logger = logging.getLogger(__name__)


class OrderDispatcherByActorCritic:
    """基于 Actor-Critic 的订单分派器"""

    STATE_DIM = 27
    ACTION_DIM = 5
    EXTRA_DIM = 5

    MAX_ROUTE_POINTS = 8
    ROUTE_POINT_DIM = 16

    IDX_COURIER_LNG = 0
    IDX_COURIER_LAT = 1
    IDX_CURRENT_TIME = 2
    IDX_ROUTE_START = 3
    IDX_ROUTE_END = 18
    IDX_TIMEOUT_START = 19
    IDX_TIMEOUT_END = 26

    COURIER_SPEED = 30.0

    DT_FEATURE_NAMES = [
        'fatigue',
        'dispatch_to_sender_delivery_distance',
        'sender_to_recipient_delivery_distance',
        'ongoing_order_count',
        'courier_prefer_reject',
        'poi_prefer_reject',
        # 'max_ongoing_sender_to_recipient_distance',
        # 'courier_id_daily_avg_rejected_rate',
        # 'poi_id_daily_avg_rejected_rate',
    ]

    @staticmethod
    def _get_dt_model_path() -> Path:
        """从 config.yaml 读取模型文件名，返回模型完整路径"""
        try:
            import yaml
            config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            model_name = cfg.get('candidate_model', 'decision_tree_model.pkl')
        except Exception:
            model_name = 'decision_tree_model.pkl'
        return (
            Path(__file__).resolve().parent.parent.parent.parent
            / 'SPOT' / 'core' / 'pkl_s' / model_name
        )

    def __init__(
        self,
        route_planner: RoutePlanner,
        config: ActorCriticConfig = None,
        grab_prediction: bool = True,
        candidate_selection: Optional[str] = None,
        dt_prob_threshold: float = 0.5,
    ):
        """初始化 Actor-Critic 分派器

        Args:
            route_planner: 路径规划器
            config: 超参数配置
            candidate_selection: 候选骑手筛选方式
                - 'distance': 按距离排序
                - 'decision_tree': 先按距离筛选，再按决策树概率排序取 top-K
                - 'decision_tree_threshold': 先按距离筛选，再按决策树概率 > 阈值筛选
                为 None 时从 config/config.yaml 读取。
            dt_prob_threshold: decision_tree_threshold 模式的决策树概率阈值
        """
        self.route_planner = route_planner
        self.config = config or ActorCriticConfig()
        self.ac_agent = ActorCriticAgent(self.STATE_DIM, self.ACTION_DIM, self.config)

        if candidate_selection is None:
            config_path = Path(__file__).resolve().parent.parent / 'config' / 'config.yaml'
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
            candidate_selection = cfg.get('candidate_method', 'distance')
            self.grab_cof = cfg.get('grab_cof', 1)
        else:
            self.grab_cof = 1

        logger.info(f"接单概率系数: {self.grab_cof}")

        self.candidate_selection = candidate_selection
        self.grab_prediction = grab_prediction
        self.dt_prob_threshold = dt_prob_threshold

        if candidate_selection in ('decision_tree', 'decision_tree_threshold'):
            model_path = self._get_dt_model_path()
            logger.info(f"尝试加载决策树模型: {model_path}")
            self.dt_model = joblib.load(model_path)
            logger.info(f"决策树模型已加载: {model_path}")

        logger.info(f"Actor-Critic 订单分派器初始化完成，候选筛选方式: {candidate_selection}")

    def _predict_dt_score(self, features_df: pd.DataFrame) -> np.ndarray:
        """兼容决策树分类器（predict_proba）和回归器（predict）的评分方法

        Args:
            features_df: 特征数据框

        Returns:
            评分数组，值越大表示越优
        """
        if hasattr(self.dt_model, 'predict_proba'):
            # DecisionTreeClassifier: 取正类概率
            return self.dt_model.predict_proba(features_df)[:, 1]
        else:
            # SimilaritySetDecisionTree (回归器): 直接用预测值
            return self.dt_model.predict(features_df)

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
            is_training: 是否为训练模式（影响是否将经验存入回放缓冲区）

        Returns:
            派单结果映射 {order_id: courier_id}
        """
        dispatch_results = {}

        for order in orders:
            candidates = self._select_candidates(order, couriers)
            if not candidates:
                logger.debug(f"订单 {order['order_id']} 无可用候选骑手，跳过")
                continue

            candidate_ids = [cid for cid, _ in candidates]
            candidate_couriers = [c for _, c in candidates]

            try:
                courier_states = np.array([
                    self._build_state(c, current_time)
                    for c in candidate_couriers
                ], dtype=np.float32)

                order_action = self._build_action(order, current_time)

                selected_action, log_prob, chosen_index = self.ac_agent.select_courier(
                    courier_states,
                    order_action
                )

                chosen_id = candidate_ids[chosen_index]
                chosen_courier = candidate_couriers[chosen_index]

                state_ = courier_states[chosen_index]
                extra_feat = self._build_extra(order, chosen_courier)
                behavior_vector = np.concatenate([state_, order_action, extra_feat])
                # grab_prob = self.predict_grab_probability(behavior_vector)
                grab_prob = self.predict_grab_probability_dt(order, chosen_courier)
                grab_prob = grab_prob * self.grab_cof
                # 通过控制接单概率系数，调整整体接单概率分布
                if grab_prob > 1:
                    grab_prob = 1
                elif grab_prob < 0:
                    grab_prob = 0
                logger.debug(f"订单 {order['order_id']} 预测接单概率: {grab_prob:.4f}")
                if self.grab_prediction and random.random() > grab_prob:
                    logger.debug(f"订单 {order['order_id']} 骑手 {chosen_id} 接单，概率: {grab_prob:.4f}")
                    dispatch_results[order['order_id']] = chosen_id+'_reject'
                    continue

                dispatch_results[order['order_id']] = chosen_id

                if is_training:
                    revenue_before = self._calculate_courier_revenue(chosen_courier, current_time)
                    distance_before = self._calculate_route_distance(chosen_courier)

                chosen_courier['current_orders'].append(order)
                chosen_courier['route'] = self.route_planner.plan_route(chosen_courier)

                if is_training:
                    revenue_after = self._calculate_courier_revenue(chosen_courier, current_time)
                    distance_after = self._calculate_route_distance(chosen_courier)
                    marginal_revenue = revenue_after - revenue_before
                    extra_distance_km = (distance_after - distance_before) / 1000.0
                    reward = marginal_revenue - self.config.distance_penalty_coeff * extra_distance_km
                    next_state = self._build_state(chosen_courier, current_time)
                    current_state = courier_states[chosen_index]
                    self.ac_agent.replay_buffer.push(current_state, selected_action, reward, next_state)
                    logger.debug(
                        f"订单 {order['order_id']} 分派给骑手 {chosen_id}，"
                        f"奖励: {reward:.4f}，log_prob: {log_prob:.4f}"
                    )
                else:
                    logger.debug(f"订单 {order['order_id']} 分派给骑手 {chosen_id}")

            except Exception as e:
                logger.warning(f"Actor-Critic 推理失败({e})，退化为距离贪心策略")
                chosen_id = candidate_ids[0]
                dispatch_results[order['order_id']] = chosen_id
                continue

        return dispatch_results

    def _select_candidates(
        self,
        order: Dict,
        couriers: Dict[str, Dict]
    ) -> List[Tuple[str, Dict]]:
        """筛选候选骑手"""
        max_orders = self.config.max_orders_per_courier
        candidate_count = self.config.candidate_count
        dt_candidate_count = self.config.dt_candidate_count

        courier_candidates = []
        for cid, courier in couriers.items():
            if courier.get('status') != 'online':
                continue
            if courier.get('position') is None:
                continue
            current_count = len(courier.get('current_orders', []))
            if current_count >= max_orders:
                continue
            courier_candidates.append((cid, courier))

        if not courier_candidates:
            return []

        if self.candidate_selection == 'decision_tree':
            return self._select_candidates_by_dt(order, courier_candidates, dt_candidate_count)
        elif self.candidate_selection == 'decision_tree_threshold':
            return self._select_candidates_by_dt_threshold(order, courier_candidates, candidate_count)
        result = self._select_candidates_by_distance(order, courier_candidates, candidate_count)
        return [(cid, c) for cid, c, _ in result]

    def _select_candidates_by_distance(
        self,
        order: Dict,
        courier_candidates: List[Tuple[str, Dict]],
        candidate_count: int,
    ) -> List[Tuple[str, Dict, float]]:
        """按距离排序筛选候选骑手

        Returns:
            [(cid, courier, dispatch_distance)]
        """
        order_pos = (order['sender_lng'], order['sender_lat'])
        courier_positions = [c['position'] for _, c in courier_candidates]

        distances = haversine_distance_vectorized(order_pos, np.array(courier_positions))
        courier_distances = [
            (cid, courier, dist)
            for (cid, courier), dist in zip(courier_candidates, distances)
        ]
        courier_distances.sort(key=lambda x: x[2])
        candidates = courier_distances[:candidate_count]

        return candidates

    def _select_candidates_by_dt(
        self,
        order: Dict,
        courier_candidates: List[Tuple[str, Dict]],
        dt_candidate_count: int,
    ) -> List[Tuple[str, Dict]]:
        """先按距离取候选，再按决策树概率排序取前 dt_candidate_count 个"""
        top_k_with_dist = self._select_candidates_by_distance(order, courier_candidates, dt_candidate_count)

        if not top_k_with_dist:
            return []

        features_list = np.array([
            self._build_dt_features(order, courier, dispatch_distance=dist)
            for cid, courier, dist in top_k_with_dist
        ])
        features_df = pd.DataFrame(features_list, columns=self.DT_FEATURE_NAMES)
        scores = self._predict_dt_score(features_df)

        score_ranked = sorted(
            zip(top_k_with_dist, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        candidates = [
            (cid, courier)
            for (cid, courier, _), _ in score_ranked[:dt_candidate_count]
        ]

        return candidates

    def _select_candidates_by_dt_threshold(
        self,
        order: Dict,
        courier_candidates: List[Tuple[str, Dict]],
        candidate_count: int,
    ) -> List[Tuple[str, Dict]]:
        """先按距离取候选，再用决策树预测，只保留概率/得分 > 阈值的骑手"""
        top_k_with_dist = self._select_candidates_by_distance(order, courier_candidates, candidate_count)

        if not top_k_with_dist:
            return []

        features_list = np.array([
            self._build_dt_features(order, courier, dispatch_distance=dist)
            for cid, courier, dist in top_k_with_dist
        ])
        features_df = pd.DataFrame(features_list, columns=self.DT_FEATURE_NAMES)
        scores = self._predict_dt_score(features_df)

        candidates = [
            (cid, courier)
            for (cid, courier, _), score in zip(top_k_with_dist, scores)
            if score > self.dt_prob_threshold
        ]

        # logger.info(
        #     features_df
        # )

        return candidates

    def _build_state(self, courier: Dict, current_time: datetime) -> np.ndarray:
        """构建27维状态向量

        维度分配：
        - 0-1: 骑手当前经度、纬度
        - 2: 当前时间步的时间戳（归一化，如一天中的比例 0~1）
        - 3-18: 路径点1~8的经纬度（共16维）
        - 19-26: 路径点1~8对应的距离超时的剩余时间（共6维）

        Args:
            courier: 骑手字典
            current_time: 当前仿真时间

        Returns:
            27维状态向量
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


    def _build_extra(self, order: Dict, courier: Dict) -> np.ndarray:
        """构建额外特征向量"""
        extra = np.zeros(self.EXTRA_DIM, dtype=np.float32)

        extra[0] = float(courier.get('fatigue', 0.0)) / 120.0
        extra[1] = float(order.get('is_weekend', 0))
        extra[2] = float(order.get('is_prebook', 0))
        extra[3] = float(courier.get('rejected_rate', 0))
        extra[4] = float(order.get('poi_id_daily_avg_rejected_rate', 0))

        return extra

    def _build_dt_features(self, order: Dict, courier: Dict, dispatch_distance: float = None) -> np.ndarray:
        features = np.zeros(6, dtype=np.float32)

        features[0] = float(courier.get('fatigue', 0.0))

        if dispatch_distance is not None:
            features[1] = dispatch_distance / 1000.0  # 转换为公里
        else:
            courier_pos = courier.get('position')
            if courier_pos is not None:
                order_sender = (order['sender_lng'], order['sender_lat'])
                features[1] = haversine_distance(courier_pos, order_sender) / 1000.0  # 转换为公里

        order_recipient = (order['recipient_lng'], order['recipient_lat'])
        order_sender = (order['sender_lng'], order['sender_lat'])
        features[2] = haversine_distance(order_sender, order_recipient) / 1000.0  # 转换为公里

        features[3] = float(len(courier.get('current_orders', [])))

        features[4] = float(courier.get('courier_prefer_reject', 0))
        features[5] = float(order.get('poi_prefer_reject', 0))

        # max_dist = 0.0
        # for o in courier.get('current_orders', []):
        #     o_sender = (o['sender_lng'], o['sender_lat'])
        #     o_recipient = (o['recipient_lng'], o['recipient_lat'])
        #     d = haversine_distance(o_sender, o_recipient)
        #     if d > max_dist:
        #         max_dist = d
        # features[4] = max_dist
        # features[5] = float(courier.get('rejected_rate', 0))
        # features[6] = float(order.get('poi_id_daily_avg_rejected_rate', 0))

        return features

    def predict_grab_probability(self, behavior_vector: np.ndarray) -> float:
        """基于 GrabPredictionMLP 预测骑手接单概率

        加载预训练的 GrabPredictionMLP 模型（训练于 behavior.py），
        对传入的 37 维行为向量 [state_27 | action_5 | extra_5] 进行推理。

        Args:
            behavior_vector: 37 维行为向量

        Returns:
            grab_prob: 骑手接单概率 (0~1)
        """
        import torch
        from pathlib import Path
        from core.behavior import GrabPredictionMLP

        # 自动判断是否有 GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not hasattr(self, '_grab_model'):
            model_path = (
                Path(__file__).resolve().parent.parent
                / 'behavior' / 'grab_prediction_mlp_0_3.pth'
            )
            self._grab_model = GrabPredictionMLP(feature_dim=len(behavior_vector))
            
            # 加载模型 + 移到 GPU
            self._grab_model.load_state_dict(
                torch.load(model_path, weights_only=True)
            )
            self._grab_model.to(self.device)  # 模型放 GPU
            self._grab_model.eval()

        # 数据也放 GPU
        x = torch.FloatTensor(behavior_vector).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            grab_prob = self._grab_model.predict_proba(x).item()

        return grab_prob

    def predict_grab_probability_dt(self, order: Dict, courier: Dict, dispatch_distance: float = None) -> float:
        """基于决策树模型预测骑手接单概率

        Args:
            order: 订单字典
            courier: 骑手字典
            dispatch_distance: 派单距离（米），若为 None 则自动计算

        Returns:
            grab_prob: 骑手接单概率 (0~1)
        """
        if not hasattr(self, 'gb_model'):
            model_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / 'SPOT' / 'core' / 'pkl_s' / 'decision_tree_model_grab.pkl'
            )
            self.gb_model = joblib.load(model_path)
            logger.info(f"决策树模型已加载: {model_path}")

        features = self._build_dt_features(order, courier, dispatch_distance)
        features_df = pd.DataFrame([features], columns=self.DT_FEATURE_NAMES)

        prob = self.gb_model.predict_proba(features_df)[0, 1]

        return float(prob)

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

    def _calculate_route_distance(
        self,
        courier: Dict
    ) -> float:
        """计算骑手当前路径的总距离（米）

        Args:
            courier: 骑手字典（含route）

        Returns:
            路径总距离（米）
        """
        route = courier.get('route', [])
        current_pos = courier.get('position')
        if current_pos is None or not route:
            return 0.0

        total_distance = 0.0
        for route_point in route:
            target_pos = route_point['position']
            total_distance += haversine_distance(current_pos, target_pos)
            current_pos = target_pos

        return total_distance

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
                        if overtime_minutes <= 5:
                            order_revenue *= 0.5
                        elif overtime_minutes <= 10:
                            order_revenue *= 0.2
                        else:
                            order_revenue = -5

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
