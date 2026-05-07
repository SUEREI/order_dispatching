"""
训练主程序
结合仿真环境和DQN智能体，训练网络
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# 导入DQN模块
from dqn.agent import DQNAgent

# 导入仿真模块
from simulation.core.courier import CourierSimulator
from simulation.core.order_g import OrderGenerator
from simulation.core.route import RoutePlanner
from simulation.core.state import StateManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TrainingConfig:
    """训练配置"""
    def __init__(self):
        # 网络参数
        self.state_dim = 27
        self.action_dim = 5
        
        # 学习率
        self.lr_actor = 1e-4
        self.lr_critic = 1e-3
        
        # 折扣因子
        self.gamma = 0.99
        
        # 经验池容量
        self.buffer_capacity = 100000
        
        # Softmax温度参数
        self.temperature = 1.0
        
        # 训练参数
        self.batch_size = 64
        self.max_epochs = 100
        self.convergence_threshold = 10
        
        # 仿真参数
        self.time_step = 30  # 秒
        self.max_days = 10  # 最大训练天数
        
        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


class OrderDispatchSimulator:
    """订单派单仿真环境"""
    
    def __init__(self, order_df, config):
        """
        初始化仿真环境
        
        Args:
            order_df: 订单数据DataFrame
            config: 训练配置
        """
        self.config = config
        
        # 初始化仿真模块
        self.courier_sim = CourierSimulator(order_df)
        self.order_gen = OrderGenerator(order_df)
        self.route_planner = RoutePlanner()
        self.state_manager = StateManager()
        
        # 时间管理
        self.timestamps = self.order_gen.get_all_timestamps()
        if not self.timestamps:
            raise ValueError("订单数据为空")
        
        self.start_time = self.timestamps[0]
        self.end_time = self.timestamps[-1]
        self.current_time = self.start_time
        
        # 骑手状态：{courier_id: {'status': 'online'/'offline', 'position': (lng, lat), 'orders': []}}
        self.couriers = {}
        
        # 待派单订单池
        self.pending_orders = []
        
        # 已完成订单
        self.completed_orders = []
        
        # 统计信息
        self.total_orders = 0
        self.on_time_orders = 0
        self.late_orders = 0
        
        logger.info(f"仿真环境初始化完成: {self.start_time} -> {self.end_time}")
    
    def reset(self):
        """重置仿真环境"""
        self.current_time = self.start_time
        self.couriers = {}
        self.pending_orders = []
        self.completed_orders = []
        self.total_orders = 0
        self.on_time_orders = 0
        self.late_orders = 0
        
        # 重置仿真模块
        self.courier_sim.event_index = 0
        
        return self._get_state()
    
    def step(self, action):
        """
        执行一步仿真
        
        Args:
            action: 派单决策，格式为(order, courier_id)或None
        
        Returns:
            next_state: 下一状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 处理骑手上下线事件
        courier_events = self.courier_sim.get_events_at_time(self.current_time)
        for courier_id, event_type, position in courier_events:
            if event_type == 'online':
                self.couriers[courier_id] = {
                    'status': 'online',
                    'position': position,
                    'orders': []
                }
            elif event_type == 'offline':
                if courier_id in self.couriers:
                    self.couriers[courier_id]['status'] = 'offline'
        
        # 获取新订单
        new_orders = self.order_gen.get_orders_at_time(self.current_time)
        self.pending_orders.extend(new_orders)
        self.total_orders += len(new_orders)
        
        # 执行派单决策
        reward = 0.0
        if action is not None:
            order, courier_id = action
            if courier_id in self.couriers and self.couriers[courier_id]['status'] == 'online':
                # 分配订单给骑手
                self.couriers[courier_id]['orders'].append(order)
                order['courier_id'] = courier_id
                self.pending_orders.remove(order)
                
                # 计算即时奖励（简化版：给一个小的派单奖励）
                reward += 0.1
        
        # 推进时间（处理取餐、送达事件）
        self._advance_time()
        
        # 计算奖励（超时惩罚等）
        reward += self._calculate_rewards()
        
        # 推进到下一时间步
        next_time_idx = self.timestamps.index(self.current_time) + 1 if self.current_time in self.timestamps else -1
        if next_time_idx < len(self.timestamps):
            self.current_time = self.timestamps[next_time_idx]
        else:
            self.current_time += timedelta(seconds=self.config.time_step)
        
        # 检查是否结束
        done = self.current_time >= self.end_time
        
        # 获取下一状态
        next_state = self._get_state()
        
        # 额外信息
        info = {
            'pending_orders': len(self.pending_orders),
            'online_couriers': sum(1 for c in self.couriers.values() if c['status'] == 'online'),
            'on_time_rate': self.on_time_orders / max(self.total_orders, 1)
        }
        
        return next_state, reward, done, info
    
    def _advance_time(self):
        """推进时间，处理取餐和送达事件"""
        for courier_id, courier_data in self.couriers.items():
            if courier_data['status'] != 'online':
                continue
            
            # 处理订单送达（简化处理）
            orders_to_remove = []
            for order in courier_data['orders']:
                # 检查是否送达
                if order.get('actual_delivery_time') is None:
                    # 简化：假设每个时间步有概率送达
                    if np.random.random() < 0.1:  # 10%概率送达
                        order['actual_delivery_time'] = self.current_time
                        orders_to_remove.append(order)
                        self.completed_orders.append(order)
                        
                        # 检查是否准时
                        if order['actual_delivery_time'] <= order['estimate_arrived_time']:
                            self.on_time_orders += 1
                        else:
                            self.late_orders += 1
            
            for order in orders_to_remove:
                courier_data['orders'].remove(order)
    
    def _calculate_rewards(self):
        """计算奖励"""
        reward = 0.0
        
        # 超时惩罚
        for order in self.pending_orders:
            if self.current_time > order['estimate_arrived_time']:
                # 超时惩罚：-0.1 * 超时分钟数
                overtime_minutes = (self.current_time - order['estimate_arrived_time']).total_seconds() / 60
                reward -= 0.01 * min(overtime_minutes, 10)  # 限制最大惩罚
        
        return reward
    
    def _get_state(self):
        """获取当前状态"""
        # 简化状态：返回当前时间、待派单数量、在线骑手数量等
        state_info = {
            'current_time': self.current_time,
            'pending_orders': self.pending_orders,
            'couriers': self.couriers,
            'timestamp_seconds': (self.current_time - self.start_time).total_seconds()
        }
        return state_info
    
    def get_candidate_orders(self):
        """获取候选订单列表"""
        # 动作空间剪枝：过滤掉容量已满的骑手
        candidates = []
        for order in self.pending_orders:
            # 检查是否有可用骑手
            for courier_id, courier_data in self.couriers.items():
                if courier_data['status'] == 'online' and len(courier_data['orders']) < 8:
                    candidates.append((order, courier_id))
                    break  # 每个订单只考虑一个骑手（简化）
        
        return candidates


def train(config, checkpoint_dir='Q_Values/dqn/checkpoints'):
    """
    训练主流程
    
    Args:
        config: 训练配置
        checkpoint_dir: 模型保存目录
    """
    logger.info("=" * 50)
    logger.info("开始训练强化学习订单派单Agent")
    logger.info("=" * 50)
    
    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 加载订单数据（这里需要根据实际情况修改）
    # 示例：从CSV加载数据
    try:
        order_df = pd.read_csv('data/orders.csv')
        order_df['dispatch_time'] = pd.to_datetime(order_df['dispatch_time'])
        order_df['arrive_time'] = pd.to_datetime(order_df['arrive_time'])
        logger.info(f"加载订单数据: {len(order_df)} 条")
    except Exception as e:
        logger.warning(f"无法加载订单数据: {e}")
        logger.info("使用模拟数据进行训练")
        # 创建模拟数据
        order_df = create_mock_orders()
    
    # 初始化仿真环境
    env = OrderDispatchSimulator(order_df, config)
    
    # 初始化智能体
    agent = DQNAgent(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        lr_actor=config.lr_actor,
        lr_critic=config.lr_critic,
        gamma=config.gamma,
        buffer_capacity=config.buffer_capacity,
        temperature=config.temperature,
        max_epochs=config.max_epochs,
        convergence_threshold=config.convergence_threshold,
        device=config.device
    )
    
    logger.info(f"智能体初始化完成，设备: {config.device}")
    
    # 训练循环
    for day in range(config.max_days):
        logger.info(f"\n{'='*50}")
        logger.info(f"训练第 {day+1}/{config.max_days} 天")
        logger.info(f"{'='*50}")
        
        # 重置环境
        state_info = env.reset()
        
        # 仿真循环
        episode_reward = 0
        episode_steps = 0
        
        while True:
            # 构建状态向量（简化版）
            # 实际应用中需要根据骑手和订单信息构建完整的27维状态向量
            state = np.zeros(config.state_dim, dtype=np.float32)
            state[0] = 116.4  # 骑手经度（示例）
            state[1] = 39.9   # 骑手纬度（示例）
            state[2] = state_info['timestamp_seconds'] / (24 * 3600)  # 时间戳归一化
            
            # 获取候选订单
            candidates = env.get_candidate_orders()
            
            # 选择动作
            if len(candidates) > 0:
                # 将候选订单转换为智能体需要的格式
                candidate_orders = []
                for order, courier_id in candidates:
                    candidate_orders.append({
                        'order': order,
                        'courier_id': courier_id,
                        'sender_lng': order['sender_lng'],
                        'sender_lat': order['sender_lat'],
                        'receiver_lng': order['recipient_lng'],
                        'receiver_lat': order['recipient_lat'],
                        'available_time': 30  # 假设可用时长30分钟
                    })
                
                selected_order, q_value = agent.select_action(state, candidate_orders, mode='train')
                
                if selected_order:
                    action = (selected_order['order'], selected_order['courier_id'])
                else:
                    action = None
            else:
                action = None
            
            # 执行动作
            next_state_info, reward, done, info = env.step(action)
            
            # 构建下一状态向量
            next_state = np.zeros(config.state_dim, dtype=np.float32)
            next_state[0] = 116.4
            next_state[1] = 39.9
            next_state[2] = next_state_info['timestamp_seconds'] / (24 * 3600)
            
            # 构建动作向量
            if action:
                order, courier_id = action
                action_vec = agent.build_action_vector(
                    sender_lng=order['sender_lng'],
                    sender_lat=order['sender_lat'],
                    receiver_lng=order['recipient_lng'],
                    receiver_lat=order['recipient_lat'],
                    available_time=30
                )
            else:
                action_vec = np.zeros(config.action_dim, dtype=np.float32)
            
            # 存储经验
            agent.store_transition(state, action_vec, reward, next_state, done)
            
            episode_reward += reward
            episode_steps += 1
            
            # 更新状态
            state_info = next_state_info
            
            # 训练网络（每隔一定步数）
            if len(agent.buffer) >= config.batch_size and episode_steps % 100 == 0:
                loss_actor, loss_critic = agent.train(batch_size=config.batch_size)
                if loss_actor and loss_critic:
                    logger.debug(f"Step {episode_steps}: Actor Loss={loss_actor:.4f}, Critic Loss={loss_critic:.4f}")
            
            # 检查是否结束
            if done:
                break
        
        # 日志记录
        logger.info(f"Day {day+1} 完成:")
        logger.info(f"  总步数: {episode_steps}")
        logger.info(f"  累计奖励: {episode_reward:.2f}")
        logger.info(f"  准时率: {info['on_time_rate']:.2%}")
        logger.info(f"  经验池大小: {len(agent.buffer)}")
        
        # 训练网络（每轮结束后）
        if len(agent.buffer) >= config.batch_size:
            logger.info("开始训练网络...")
            for _ in range(10):  # 训练10轮
                loss_actor, loss_critic = agent.train(batch_size=config.batch_size)
                if loss_actor and loss_critic:
                    logger.debug(f"  Actor Loss={loss_actor:.4f}, Critic Loss={loss_critic:.4f}")
        
        # 保存模型
        agent.save_model(checkpoint_dir, day + 1)
        logger.info(f"模型已保存至: {checkpoint_dir}")
    
    logger.info("\n" + "=" * 50)
    logger.info("训练完成!")
    logger.info("=" * 50)


def create_mock_orders():
    """创建模拟订单数据"""
    logger.info("创建模拟订单数据...")
    
    orders = []
    start_time = datetime(2024, 1, 1, 8, 0, 0)
    
    for i in range(1000):
        dispatch_time = start_time + timedelta(minutes=i*2)
        arrive_time = dispatch_time + timedelta(minutes=30)
        
        orders.append({
            'order_id': f'order_{i}',
            'courier_id': f'courier_{i % 20}',  # 20个骑手
            'dispatch_time': dispatch_time,
            'arrive_time': arrive_time,
            'sender_lng': 116400000 + np.random.randint(-10000, 10000),
            'sender_lat': 39900000 + np.random.randint(-10000, 10000),
            'recipient_lng': 116400000 + np.random.randint(-10000, 10000),
            'recipient_lat': 39900000 + np.random.randint(-10000, 10000),
            'grab_lng': 116400000,
            'grab_lat': 39900000,
            'estimate_meal_prepare_time': 15,
            'estimate_arrived_time': arrive_time
        })
    
    df = pd.DataFrame(orders)
    logger.info(f"创建模拟订单: {len(df)} 条")
    return df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练强化学习订单派单Agent')
    parser.add_argument('--lr_actor', type=float, default=1e-4, help='Actor学习率')
    parser.add_argument('--lr_critic', type=float, default=1e-3, help='Critic学习率')
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小')
    parser.add_argument('--max_days', type=int, default=10, help='最大训练天数')
    parser.add_argument('--temperature', type=float, default=1.0, help='Softmax温度参数')
    parser.add_argument('--checkpoint_dir', type=str, default='Q_Values/dqn/checkpoints', help='模型保存目录')
    
    args = parser.parse_args()
    
    # 创建配置
    config = TrainingConfig()
    config.lr_actor = args.lr_actor
    config.lr_critic = args.lr_critic
    config.gamma = args.gamma
    config.batch_size = args.batch_size
    config.max_days = args.max_days
    config.temperature = args.temperature
    
    # 开始训练
    train(config, checkpoint_dir=args.checkpoint_dir)


if __name__ == "__main__":
    main()
