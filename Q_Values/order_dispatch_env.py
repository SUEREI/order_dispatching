#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
订单分派强化学习环境 - Gymnasium 标准接口

将现有的订单分派仿真环境封装为 Gymnasium 兼容的环境，
以便使用 Stable Baselines3 等第三方库进行训练。

环境设计:
- Observation: 骑手状态向量 (STATE_DIM 维)
- Action: 离散动作空间，表示选择哪个骑手分配订单
- Reward: 基于配送效率、准时率等指标计算
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from simulation.core import (
    CourierSimulator,
    OrderGenerator,
    OrderDispatcherByDistance,
    RoutePlanner,
    StateRefresher
)
from dqn import RewardCalculator
from dqn import DoneMarker
from simulation.utils import haversine_distance

logger = logging.getLogger(__name__)


class OrderDispatchEnv(gym.Env):
    """
    订单分派强化学习环境
    
    符合 Gymnasium 标准接口，可用于 Stable Baselines3 训练。
    
    状态空间 (Observation):
        - 使用 Dict 空间，包含:
          - "courier_states": (N, STATE_DIM) N个在线骑手的状态
          - "order_features": (M, ACTION_DIM) M个待分配订单的特征
          - "mask": (N,) 骑手有效性掩码
        
    动作空间 (Action):
        - Discrete(N * M): 选择 (骑手i, 订单j) 的组合
        - 或使用 MultiDiscrete: [骑手索引, 订单索引]
    
    奖励 (Reward):
        - 正奖励: 准时送达、配送效率高
        - 负奖励: 超时、配送距离长
    """
    
    metadata = {"render_modes": ["human", "console"]}
    
    def __init__(
        self,
        order_data_path: str,
        state_dim: int = 27,
        action_dim: int = 5,
        max_couriers: int = 50,
        max_orders: int = 20,
        max_orders_per_courier: int = 4,
        idle_timeout_minutes: int = 30,
        max_time_to_timeout: float = 3600.0,
        render_mode: Optional[str] = None,
    ):
        """
        初始化环境
        
        Args:
            order_data_path: 订单数据文件路径
            state_dim: 状态向量维度
            action_dim: 动作向量维度
            max_couriers: 最大骑手数量（用于空间定义）
            max_orders: 最大待分配订单数量
            max_orders_per_courier: 每个骑手最大持单量
            idle_timeout_minutes: 骑手空闲超时时间（分钟）
            max_time_to_timeout: 最大超时时间（秒）
            render_mode: 渲染模式
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_couriers = max_couriers
        self.max_orders = max_orders
        self.max_orders_per_courier = max_orders_per_courier
        self.max_time_to_timeout = max_time_to_timeout
        
        # 定义动作空间: (骑手索引, 订单索引)
        self.action_space = spaces.MultiDiscrete([max_couriers, max_orders])
        
        # 定义观测空间
        self.observation_space = spaces.Dict({
            "courier_states": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_couriers, state_dim),
                dtype=np.float32
            ),
            "order_features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(max_orders, action_dim),
                dtype=np.float32
            ),
            "courier_mask": spaces.Box(
                low=0, high=1,
                shape=(max_couriers,),
                dtype=np.float32
            ),
            "order_mask": spaces.Box(
                low=0, high=1,
                shape=(max_orders,),
                dtype=np.float32
            ),
        })
        
        self.render_mode = render_mode
        
        # 加载数据
        self._load_data(order_data_path)
        
        # 初始化仿真组件
        self.order_gen = OrderGenerator(self.df)
        self.courier_sim = CourierSimulator(self.df, idle_timeout_minutes)
        self.state_refresher = StateRefresher()
        self.route_planner = RoutePlanner()
        self.reward_calc = RewardCalculator()
        self.done_marker = DoneMarker(idle_timeout_minutes)
        
        # 距离派单器（用于 warm-up 或 fallback）
        self.distance_dispatcher = OrderDispatcherByDistance()
        
        # 初始化内部状态
        self._init_internal_state()
        
    def _load_data(self, path: str):
        """加载订单数据"""
        import pandas as pd
        df = pd.read_csv(path)
        df = df.dropna(subset=["order_id", "courier_id"])
        df["order_id"] = df["order_id"].astype(int).astype(str)
        df["courier_id"] = df["courier_id"].astype(int).astype(str)
        for col in ["dispatch_time", "arrive_time", "estimate_meal_prepare_time", "estimate_arrived_time"]:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
        self.df = df
        
    def _init_internal_state(self):
        """初始化内部仿真状态"""
        self.couriers: Dict[str, Dict[str, Any]] = {}
        all_couriers = self.courier_sim.get_all_courier_ids()
        for cid in all_couriers:
            self.couriers[cid] = {
                "courier_id": cid,
                "status": "offline",
                "position": None,
                "order_increased": False,
                "current_orders": [],
                "route": [],
                "special_status": "none",
                "special_status_end_time": None,
            }
        
        self.orders: Dict[str, Dict[str, Any]] = {}
        for order_list in self.order_gen.orders_by_time.values():
            for order in order_list:
                oid = order["order_id"]
                self.orders[oid] = dict(order)
        
        self.pending_orders: List[Dict[str, Any]] = []
        
        earliest = self.courier_sim.get_earliest_event_time()
        self.current_time: datetime = earliest if earliest is not None else datetime.now()
        
        # 统计信息
        self.total_dispatch = 0
        self.total_reward = 0.0
        self.completed_orders = 0
        self.on_time_orders = 0
        self.overdue_orders = 0
        self.step_count = 0
        
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        重置环境
        
        Returns:
            observation: 初始观测
            info: 额外信息
        """
        super().reset(seed=seed)
        
        # 重置仿真组件
        self.order_gen = OrderGenerator(self.df)
        self.courier_sim = CourierSimulator(self.df, 30)
        self.done_marker.reset()
        
        # 重置内部状态
        self._init_internal_state()
        
        # 获取初始观测
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "console":
            print(f"[Reset] 时间: {self.current_time}, 在线骑手: {len(self._get_online_couriers())}")
        
        return observation, info
    
    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        执行一步动作
        
        Args:
            action: [骑手索引, 订单索引]
        
        Returns:
            observation: 新观测
            reward: 奖励值
            terminated: 是否终止（任务完成）
            truncated: 是否截断（超时等）
            info: 额外信息
        """
        self.step_count += 1
        
        # 解析动作
        courier_idx = int(action[0])
        order_idx = int(action[1])
        
        # 获取在线骑手和待分配订单
        online_couriers = self._get_online_couriers()
        courier_ids = list(online_couriers.keys())
        pending_order_ids = [o["order_id"] for o in self.pending_orders]
        
        # 执行派单
        reward = 0.0
        if courier_idx < len(courier_ids) and order_idx < len(self.pending_orders):
            courier_id = courier_ids[courier_idx]
            order = self.pending_orders[order_idx]
            order_id = order["order_id"]
            
            # 检查骑手是否可以接单
            courier = self.couriers[courier_id]
            if len(courier.get("current_orders", [])) < self.max_orders_per_courier:
                # 执行派单
                self.couriers[courier_id]["current_orders"].append(dict(self.orders[order_id]))
                self.couriers[courier_id]["order_increased"] = True
                self.orders[order_id]["courier_id"] = courier_id
                self.orders[order_id]["status"] = "assigned"
                
                # 路径规划
                route = self.route_planner.plan_route(self.couriers[courier_id])
                self.couriers[courier_id]["route"] = route
                self.couriers[courier_id]["order_increased"] = False
                
                # 从待分配列表移除
                self.pending_orders = [o for o in self.pending_orders if o["order_id"] != order_id]
                
                self.total_dispatch += 1
                self.done_marker.update_activity(courier_id, self.current_time)
        
        # 仿真步进
        step_reward = self._simulation_step()
        reward += step_reward
        self.total_reward += reward
        
        # 检查终止条件
        terminated = self._check_terminated()
        truncated = False
        
        # 获取新观测
        observation = self._get_observation()
        info = self._get_info()
        
        if self.render_mode == "console":
            print(f"[Step {self.step_count}] 派单: {courier_idx}->{order_idx}, "
                  f"奖励: {reward:.4f}, 完成: {self.completed_orders}")
        
        return observation, reward, terminated, truncated, info
    
    def _get_online_couriers(self) -> Dict[str, Dict[str, Any]]:
        """获取在线且可用的骑手"""
        return {
            cid: c for cid, c in self.couriers.items()
            if c["status"] == "online" 
            and c["position"] is not None
            and not self.done_marker.is_done(cid)
        }
    
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """构建观测"""
        online_couriers = self._get_online_couriers()
        courier_ids = list(online_couriers.keys())
        
        # 骑手状态
        courier_states = np.zeros((self.max_couriers, self.state_dim), dtype=np.float32)
        courier_mask = np.zeros(self.max_couriers, dtype=np.float32)
        
        for i, cid in enumerate(courier_ids[:self.max_couriers]):
            courier = self.couriers[cid]
            courier_states[i] = self._build_courier_state(courier)
            courier_mask[i] = 1.0
        
        # 订单特征
        order_features = np.zeros((self.max_orders, self.action_dim), dtype=np.float32)
        order_mask = np.zeros(self.max_orders, dtype=np.float32)
        
        for i, order in enumerate(self.pending_orders[:self.max_orders]):
            order_features[i] = self._build_order_feature(order)
            order_mask[i] = 1.0
        
        return {
            "courier_states": courier_states,
            "order_features": order_features,
            "courier_mask": courier_mask,
            "order_mask": order_mask,
        }
    
    def _build_courier_state(self, courier: Dict[str, Any]) -> np.ndarray:
        """构建骑手状态向量"""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # 位置
        pos = courier.get("position")
        if pos is not None:
            state[0] = float(pos[0])
            state[1] = float(pos[1])
        
        # 时间
        total_seconds_in_day = 24.0 * 3600.0
        time_fraction = (
            self.current_time.hour * 3600.0
            + self.current_time.minute * 60.0
            + self.current_time.second
        ) / total_seconds_in_day
        state[2] = np.clip(time_fraction, 0.0, 1.0)
        
        # 路径点
        route = courier.get("route", [])
        max_points = 8
        for i, route_point in enumerate(route[:max_points]):
            rp_pos = route_point.get("position")
            if rp_pos is not None:
                state[3 + i * 2] = float(rp_pos[0])
                state[4 + i * 2] = float(rp_pos[1])
        
        # 路径点剩余时间
        max_time_dim = self.state_dim - 19
        if max_time_dim > 0:
            for i, route_point in enumerate(route[:max_time_dim]):
                point_type = route_point.get("point_type", "")
                if point_type == "pickup":
                    state[19 + i] = 0.0
                elif point_type == "delivery":
                    order_id = route_point.get("order_id")
                    if order_id and order_id in self.orders:
                        order = self.orders[order_id]
                        estimate_arrived = order.get("estimate_arrived_time")
                        if estimate_arrived is not None:
                            remaining_seconds = (
                                estimate_arrived - self.current_time
                            ).total_seconds()
                            normalized = remaining_seconds / self.max_time_to_timeout
                            state[19 + i] = np.clip(normalized, 0.0, 1.0)
        
        return state
    
    def _build_order_feature(self, order: Dict[str, Any]) -> np.ndarray:
        """构建订单特征向量"""
        feature = np.zeros(self.action_dim, dtype=np.float32)
        
        feature[0] = float(order.get("sender_lng", 0))
        feature[1] = float(order.get("sender_lat", 0))
        feature[2] = float(order.get("recipient_lng", 0))
        feature[3] = float(order.get("recipient_lat", 0))
        
        dispatch_time = order.get("dispatch_time")
        estimate_arrived = order.get("estimate_arrived_time")
        if dispatch_time is not None and estimate_arrived is not None:
            available_seconds = (estimate_arrived - dispatch_time).total_seconds()
            feature[4] = np.clip(
                available_seconds / self.max_time_to_timeout, 0.0, 1.0
            )
        
        return feature
    
    def _simulation_step(self) -> float:
        """执行仿真步进（30秒）"""
        current_time = self.current_time
        
        # 处理骑手上下线
        courier_events = self.courier_sim.get_events_at_time(current_time)
        for cid, etype, pos in courier_events:
            self.couriers[cid]["status"] = etype
            if etype == "online":
                self.couriers[cid]["position"] = pos
                self.done_marker.reset_courier(cid)
            elif etype == "offline":
                self.couriers[cid]["position"] = None
        
        # 生成新订单
        new_orders = self.order_gen.get_orders_at_time(current_time)
        if new_orders:
            self.pending_orders.extend(new_orders)
            for o in new_orders:
                oid = o["order_id"]
                self.orders[oid] = dict(o)
        
        # 骑手移动
        reward = 0.0
        courier_completed: Dict[str, List[str]] = {}
        courier_distance: Dict[str, float] = {}
        
        for cid, courier in self.couriers.items():
            if courier.get("current_orders") and courier["status"] == "online":
                pos_before = courier.get("position")
                
                updates = self.state_refresher.refresh_courier(courier, current_time)
                if updates.get("courier_updates"):
                    self.couriers[cid].update(updates["courier_updates"])
                
                pos_after = self.couriers[cid].get("position")
                dist_km = 0.0
                if pos_before and pos_after:
                    dist_m = haversine_distance(pos_before, pos_after)
                    dist_km = dist_m / 1000.0
                courier_distance[cid] = dist_km
                
                # 检查完成的订单
                completed_ids = []
                for o in courier["current_orders"][:]:
                    if o.get("status") == "delivered":
                        completed_ids.append(o["order_id"])
                        if o["order_id"] in self.orders:
                            self.orders[o["order_id"]].update({
                                "actual_delivery_time": o.get("actual_delivery_time"),
                                "actual_pickup_time": o.get("actual_pickup_time"),
                            })
                        courier["current_orders"].remove(o)
                        self.done_marker.update_activity(cid, current_time + timedelta(seconds=30))
                        self.completed_orders += 1
                        
                        # 计算奖励
                        estimate_arrived = self.orders[o["order_id"]].get("estimate_arrived_time")
                        actual_delivery = o.get("actual_delivery_time")
                        if estimate_arrived is not None and actual_delivery is not None:
                            if actual_delivery <= estimate_arrived:
                                self.on_time_orders += 1
                            else:
                                self.overdue_orders += 1
                        
                        # 计算该订单的奖励
                        order_reward = self.reward_calc.compute_reward_from_events(
                            completed_order_infos=[{
                                "order_id": o["order_id"],
                                "on_time": actual_delivery <= estimate_arrived if actual_delivery and estimate_arrived else True,
                                "overtime_minutes": max(0, (actual_delivery - estimate_arrived).total_seconds() / 60.0) if actual_delivery and estimate_arrived else 0,
                            }],
                            picked_up_order_ids=[],
                            distance_traveled_km=dist_km,
                        )
                        reward += order_reward
                
                courier_completed[cid] = completed_ids
        
        # 推进时间
        self.current_time = current_time + timedelta(seconds=30)
        
        return reward
    
    def _check_terminated(self) -> bool:
        """检查是否终止"""
        if self.courier_sim.get_next_event_time() is None:
            all_done = True
            for cid, c in self.couriers.items():
                if c["status"] == "online" and not self.done_marker.is_done(cid):
                    all_done = False
                    break
            if all_done:
                return True
        return False
    
    def _get_info(self) -> Dict[str, Any]:
        """获取额外信息"""
        return {
            "current_time": self.current_time,
            "total_dispatch": self.total_dispatch,
            "total_reward": self.total_reward,
            "completed_orders": self.completed_orders,
            "on_time_orders": self.on_time_orders,
            "overdue_orders": self.overdue_orders,
            "pending_orders": len(self.pending_orders),
            "online_couriers": len(self._get_online_couriers()),
            "step_count": self.step_count,
        }
    
    def render(self):
        """渲染环境"""
        if self.render_mode == "console":
            info = self._get_info()
            print(f"\n{'='*60}")
            print(f"时间: {info['current_time']}")
            print(f"在线骑手: {info['online_couriers']}, 待分配订单: {info['pending_orders']}")
            print(f"总派单: {info['total_dispatch']}, 完成: {info['completed_orders']}")
            print(f"准时: {info['on_time_orders']}, 超时: {info['overdue_orders']}")
            print(f"累计奖励: {info['total_reward']:.4f}")
            print(f"{'='*60}\n")
    
    def close(self):
        """关闭环境"""
        pass
    
    def get_valid_actions(self) -> np.ndarray:
        """
        获取当前状态下的合法动作掩码
        
        Returns:
            (max_couriers, max_orders) 布尔掩码
        """
        online_couriers = self._get_online_couriers()
        courier_ids = list(online_couriers.keys())
        
        mask = np.zeros((self.max_couriers, self.max_orders), dtype=bool)
        
        for i, cid in enumerate(courier_ids[:self.max_couriers]):
            courier = self.couriers[cid]
            # 检查骑手是否可以接单
            if len(courier.get("current_orders", [])) < self.max_orders_per_courier:
                for j in range(min(len(self.pending_orders), self.max_orders)):
                    mask[i, j] = True
        
        return mask
