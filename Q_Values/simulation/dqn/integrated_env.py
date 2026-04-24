"""
集成仿真环境

将Actor-Critic网络与仿真系统完整集成
"""

import numpy as np
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timedelta
from copy import deepcopy

from agent import Agent
from state_builder import StateBuilder
from action_builder import ActionBuilder
from reward_calculator import RewardCalculator

# 导入仿真系统核心模块
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.courier import CourierSimulator
from core.order_g import OrderGenerator
from core.route import RoutePlanner
from core.state import StateRefresher

logger = logging.getLogger(__name__)


class IntegratedSimulationEnv:
    """
    集成仿真环境
    
    将Actor-Critic强化学习网络与即时配送仿真系统完整集成，
    实现基于强化学习的订单派发决策。
    """
    
    def __init__(
        self,
        order_df,
        agent: Agent,
        state_builder: StateBuilder,
        action_builder: ActionBuilder,
        reward_calculator: RewardCalculator,
        max_orders_per_courier: int = 4,
        idle_timeout_minutes: int = 30
    ):
        """
        初始化集成仿真环境
        
        Args:
            order_df: 订单数据DataFrame
            agent: Actor-Critic Agent
            state_builder: 状态向量构建器
            action_builder: 动作向量构建器
            reward_calculator: 奖励计算器
            max_orders_per_courier: 骑手最大持单量
            idle_timeout_minutes: 骑手空闲超时时间（分钟）
        """
        # 初始化仿真系统核心模块
        self.courier_simulator = CourierSimulator(
            order_df, 
            idle_timeout_minutes=idle_timeout_minutes
        )
        self.order_generator = OrderGenerator(order_df)
        self.route_planner = RoutePlanner()
        self.state_refresher = StateRefresher()
        
        # 初始化强化学习模块
        self.agent = agent
        self.state_builder = state_builder
        self.action_builder = action_builder
        self.reward_calculator = reward_calculator
        
        # 配置参数
        self.max_orders_per_courier = max_orders_per_courier
        
        # 仿真状态
        self.couriers = {}  # 骑手状态字典
        self.orders = {}    # 订单状态字典
        self.current_time = None
        
        # 统计信息
        self.total_orders = 0
        self.accepted_orders = 0
        self.rejected_orders = 0
        
        logger.info("集成仿真环境初始化完成")
    
    def reset(self) -> None:
        """
        重置仿真环境
        
        清空所有骑手和订单状态，准备开始新的仿真。
        """
        self.couriers = {}
        self.orders = {}
        self.current_time = None
        self.total_orders = 0
        self.accepted_orders = 0
        self.rejected_orders = 0
        
        # 重置仿真系统
        self.courier_simulator.event_index = 0
        
        logger.info("仿真环境已重置")
    
    def step(self, current_time: datetime) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        执行一个时间步的仿真
        
        Args:
            current_time: 当前时间
        
        Returns:
            List[Tuple]: 经验列表，每个元素为(state, action, reward, next_state)
        """
        self.current_time = current_time
        experiences = []
        
        # 1. 处理骑手上下线事件
        self._process_courier_events(current_time)
        
        # 2. 获取新订单
        new_orders = self.order_generator.get_orders_at_time(current_time)
        self.total_orders += len(new_orders)
        
        # 3. 为每个在线骑手处理订单派发
        for courier_id, courier in self.couriers.items():
            if courier.get('status') != 'online':
                continue
            
            # 获取候选订单（过滤掉容量已满的骑手）
            candidate_orders = self._get_candidate_orders(courier, new_orders)
            
            if not candidate_orders:
                continue
            
            # 构建状态向量
            route = self.route_planner.plan_route(courier)
            state = self.state_builder.build_state(
                courier, route, current_time, self.orders
            )
            
            # 构建候选动作集
            candidate_actions = self.action_builder.build_candidate_actions(
                candidate_orders, current_time
            )
            
            # 添加拒单动作
            reject_action = self.action_builder.build_reject_action()
            candidate_actions = np.vstack([candidate_actions, reject_action])
            
            # 使用Agent选择动作
            selected_action = self.agent.select_action(
                state, candidate_actions, mode='train' if self.agent.is_train else 'eval'
            )
            
            # 判断是否拒单
            if np.allclose(selected_action, reject_action):
                # 拒单
                self.rejected_orders += 1
                reward = 0.0
                next_state = state  # 状态不变
            else:
                # 接单：找到对应的订单
                selected_order_idx = self._find_selected_order(
                    selected_action, candidate_orders, current_time
                )
                
                if selected_order_idx >= 0:
                    selected_order = candidate_orders[selected_order_idx]
                    
                    # 计算边际奖励
                    orders_before = deepcopy(list(courier.get('current_orders', [])))
                    
                    # 模拟接单
                    courier['current_orders'].append(selected_order)
                    self.orders[selected_order['order_id']] = selected_order
                    
                    orders_after = courier['current_orders']
                    
                    # 计算奖励
                    reward = self.reward_calculator.calculate_marginal_reward(
                        orders_before, orders_after, current_time
                    )
                    
                    self.accepted_orders += 1
                    
                    # 构建下一状态
                    route_next = self.route_planner.plan_route(courier)
                    next_state = self.state_builder.build_state(
                        courier, route_next, current_time, self.orders
                    )
                else:
                    # 未找到对应订单，视为拒单
                    self.rejected_orders += 1
                    reward = 0.0
                    next_state = state
            
            # 存储经验
            experiences.append((state, selected_action, reward, next_state))
        
        # 4. 更新骑手和订单状态
        self._update_states(current_time)
        
        return experiences
    
    def _process_courier_events(self, current_time: datetime) -> None:
        """处理骑手上下线事件"""
        events = self.courier_simulator.get_events_at_time(current_time)
        
        for courier_id, event_type, position in events:
            if event_type == 'online':
                # 骑手上线
                self.couriers[courier_id] = {
                    'courier_id': courier_id,
                    'status': 'online',
                    'position': position,
                    'current_orders': [],
                    'special_status': 'none',
                    'special_status_end_time': None
                }
                logger.debug(f"骑手 {courier_id} 上线，位置: {position}")
            
            elif event_type == 'offline':
                # 骑手下线
                if courier_id in self.couriers:
                    self.couriers[courier_id]['status'] = 'offline'
                    logger.debug(f"骑手 {courier_id} 下线")
    
    def _get_candidate_orders(
        self,
        courier: Dict[str, Any],
        new_orders: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        获取骑手的候选订单
        
        过滤掉容量已满的骑手无法接单的订单。
        """
        current_order_count = len(courier.get('current_orders', []))
        
        if current_order_count >= self.max_orders_per_courier:
            return []
        
        return new_orders
    
    def _find_selected_order(
        self,
        selected_action: np.ndarray,
        candidate_orders: List[Dict[str, Any]],
        current_time: datetime
    ) -> int:
        """
        找到选中的订单索引
        
        Args:
            selected_action: 选中的动作向量
            candidate_orders: 候选订单列表
            current_time: 当前时间
        
        Returns:
            int: 订单索引，未找到返回-1
        """
        # 构建所有候选订单的动作向量
        for i, order in enumerate(candidate_orders):
            action = self.action_builder.build_action(order, current_time)
            if np.allclose(selected_action, action, atol=1e-3):
                return i
        
        return -1
    
    def _update_states(self, current_time: datetime) -> None:
        """更新骑手和订单状态"""
        # 这里需要调用state.py中的状态更新逻辑
        # 由于state.py的实现较为复杂，这里简化处理
        # 实际应用中需要完整实现状态更新逻辑
        pass
    
    def run_simulation(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        time_step: int = 30
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray]]:
        """
        运行完整仿真
        
        Args:
            start_time: 开始时间，默认为最早事件时间
            end_time: 结束时间，默认为最晚事件时间
            time_step: 时间步长（秒）
        
        Returns:
            List[Tuple]: 所有经验数据
        """
        if start_time is None:
            start_time = self.courier_simulator.get_earliest_event_time()
        if end_time is None:
            end_time = self.courier_simulator.get_latest_event_time()
        
        self.reset()
        all_experiences = []
        
        current_time = start_time
        while current_time <= end_time:
            experiences = self.step(current_time)
            all_experiences.extend(experiences)
            
            current_time += timedelta(seconds=time_step)
        
        logger.info(f"仿真完成，总订单: {self.total_orders}, "
                   f"接单: {self.accepted_orders}, 拒单: {self.rejected_orders}")
        
        return all_experiences
