"""
DQN智能体实现
包含经验池ReplayBuffer和智能体DQNAgent
"""

import random
import logging
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .dqn_network import ActorNetwork, CriticNetwork


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    经验池：存储和采样状态转移数据
    """
    
    def __init__(self, capacity=100000):
        """
        初始化经验池
        
        Args:
            capacity: 最大容量，默认10万条
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """
        存储经验元组
        
        Args:
            state: 当前状态
            action: 执行动作
            reward: 获得奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """
        随机采样batch_size条经验
        
        Args:
            batch_size: 采样数量
        
        Returns:
            states, actions, rewards, next_states, dones: 批量数据
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"经验池容量不足: 请求{batch_size}，当前{len(self.buffer)}")
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        """返回当前经验池大小"""
        return len(self.buffer)
    
    def clear(self):
        """清空经验池"""
        self.buffer.clear()


class DQNAgent:
    """
    DQN智能体：实现动作选择、训练和模型管理
    """
    
    def __init__(self, state_dim=27, action_dim=5, 
                 lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, buffer_capacity=100000,
                 temperature=1.0, max_epochs=100,
                 convergence_threshold=10,
                 device='cpu'):
        """
        初始化DQN智能体
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr_actor: Actor学习率
            lr_critic: Critic学习率
            gamma: 折扣因子
            buffer_capacity: 经验池容量
            temperature: Softmax温度参数
            max_epochs: 每轮最大训练epoch
            convergence_threshold: 收敛判定阈值
            device: 计算设备
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.temperature = temperature
        self.max_epochs = max_epochs
        self.convergence_threshold = convergence_threshold
        self.device = torch.device(device)
        
        # 初始化网络
        self.actor = ActorNetwork(state_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(state_dim).to(self.device)
        self.critic_target = CriticNetwork(state_dim).to(self.device)
        
        # 同步目标网络
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 初始化优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 初始化经验池
        self.buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # 训练记录
        self.loss_history = {'actor': [], 'critic': []}
        self.reward_history = []
    
    def build_state_vector(self, courier_lng, courier_lat, timestamp, 
                          path_points, max_time=24*3600):
        """
        构建27维状态向量
        
        Args:
            courier_lng: 骑手经度
            courier_lat: 骑手纬度
            timestamp: 当前时间戳（秒）
            path_points: 路径点列表，每个元素为{lng, lat, distance_to_timeout}
            max_time: 最大时间（秒），默认24小时
        
        Returns:
            state: 27维状态向量
        """
        # 维度0-1：骑手位置
        state = [courier_lng, courier_lat]
        
        # 维度2：时间戳（归一化到[0,1]）
        normalized_time = timestamp / max_time
        normalized_time = np.clip(normalized_time, 0, 1)
        state.append(normalized_time)
        
        # 维度3-18：路径点经纬度（最多8个点）
        # 维度19-26：距离超时剩余时间（归一化）
        path_lng_lat = []
        path_distance = []
        
        if path_points and len(path_points) > 0:
            for point in path_points[:8]:  # 最多取8个点
                path_lng_lat.append(point.get('lng', 0))
                path_lng_lat.append(point.get('lat', 0))
                # 归一化距离超时时间
                distance = point.get('distance_to_timeout', 0)
                normalized_distance = distance / 60.0  # 假设最大60分钟
                normalized_distance = np.clip(normalized_distance, 0, 1)
                path_distance.append(normalized_distance)
        
        # 填充不足8个的点
        while len(path_lng_lat) < 16:
            path_lng_lat.append(0)
        while len(path_distance) < 8:
            path_distance.append(0)
        
        state.extend(path_lng_lat)
        state.extend(path_distance)
        
        return np.array(state, dtype=np.float32)
    
    def build_action_vector(self, sender_lng, sender_lat, 
                           receiver_lng, receiver_lat, available_time):
        """
        构建5维动作向量
        
        Args:
            sender_lng: 取餐点经度
            sender_lat: 取餐点纬度
            receiver_lng: 送餐点经度
            receiver_lat: 送餐点纬度
            available_time: 可用时长（分钟）
        
        Returns:
            action: 5维动作向量
        """
        # 归一化可用时长（假设最大120分钟）
        normalized_time = available_time / 120.0
        normalized_time = np.clip(normalized_time, 0, 1)
        
        action = [
            sender_lng,
            sender_lat,
            receiver_lng,
            receiver_lat,
            normalized_time
        ]
        
        return np.array(action, dtype=np.float32)
    
    def select_action(self, state, candidate_orders, mode='train'):
        """
        选择动作
        
        Args:
            state: 27维状态向量
            candidate_orders: 候选订单列表
            mode: 'train'训练模式 or 'application'应用模式
        
        Returns:
            selected_order: 选择的订单
            q_value: 对应的Q值
        """
        if not candidate_orders or len(candidate_orders) == 0:
            return None, None
        
        # 转换状态为tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # (1, 27)
        
        # 计算所有候选订单的Q值
        q_values = []
        for order in candidate_orders:
            action = self.build_action_vector(
                sender_lng=order.get('sender_lng', 0),
                sender_lat=order.get('sender_lat', 0),
                receiver_lng=order.get('receiver_lng', 0),
                receiver_lat=order.get('receiver_lat', 0),
                available_time=order.get('available_time', 30)
            )
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)  # (1, 5)
            
            with torch.no_grad():
                q_value = self.actor(state_tensor, action_tensor)
            
            # 检查NaN/Inf
            if torch.isnan(q_value) or torch.isinf(q_value):
                logger.error(f"网络输出NaN/Inf: Q={q_value.item()}")
                q_value = torch.tensor([[0.0]], device=self.device)
            
            q_values.append(q_value.item())
        
        q_values_tensor = torch.FloatTensor(q_values).to(self.device)
        
        # 选择动作
        if mode == 'train':
            # Softmax概率采样
            if self.temperature > 0:
                probs = F.softmax(q_values_tensor / self.temperature, dim=0)
            else:
                # 所有Q值相同，均匀采样
                probs = torch.ones_like(q_values_tensor) / len(q_values)
            
            # 检查所有Q值是否相同
            if torch.all(q_values_tensor == q_values_tensor[0]):
                logger.info("所有Q值相同，使用均匀概率采样")
                probs = torch.ones_like(q_values_tensor) / len(q_values)
            
            action_idx = torch.multinomial(probs, 1).item()
        else:
            # Argmax选择
            action_idx = torch.argmax(q_values_tensor).item()
        
        return candidate_orders[action_idx], q_values[action_idx]
    
    def store_transition(self, state, action, reward, next_state, done):
        """
        存储经验转移
        
        Args:
            state: 当前状态
            action: 执行动作
            reward: 获得奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.push(state, action, reward, next_state, done)
        self.reward_history.append(reward)
    
    def train(self, batch_size=64):
        """
        交替训练Actor和Critic网络
        
        Args:
            batch_size: 批量大小
        
        Returns:
            loss_actor: Actor损失
            loss_critic: Critic损失
        """
        if len(self.buffer) < batch_size:
            logger.warning(f"经验池数据不足，跳过训练: {len(self.buffer)} < {batch_size}")
            return None, None
        
        # 采样训练数据
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Phase 1: 训练Critic（冻结Actor）
        loss_critic = self._train_critic_phase(states, actions, rewards, next_states, dones)
        
        # Phase 2: 训练Actor（冻结Critic）
        loss_actor = self._train_actor_phase(states, actions, rewards, next_states, dones)
        
        # 更新Critic目标网络
        self._update_target_network()
        
        # 记录损失
        self.loss_history['actor'].append(loss_actor)
        self.loss_history['critic'].append(loss_critic)
        
        return loss_actor, loss_critic
    
    def _train_critic_phase(self, states, actions, rewards, next_states, dones):
        """
        训练Critic网络
        
        Args:
            states, actions, rewards, next_states, dones: 批量数据
        
        Returns:
            loss: Critic损失
        """
        # 冻结Actor
        for param in self.actor.parameters():
            param.requires_grad = False
        
        convergence_count = 0
        prev_loss = float('inf')
        final_loss = 0
        
        for epoch in range(self.max_epochs):
            # 计算V(s)
            v_values = self.critic(states)
            
            # 计算TD目标
            with torch.no_grad():
                v_target_values = self.critic_target(next_states)
                td_targets = rewards + self.gamma * v_target_values * (1 - dones)
            
            # 计算Critic损失
            loss_v = F.mse_loss(v_values, td_targets)
            
            # 反向传播
            self.critic_optimizer.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
            self.critic_optimizer.step()
            
            current_loss = loss_v.item()
            final_loss = current_loss
            
            # 收敛检查
            if current_loss >= prev_loss:
                convergence_count += 1
            else:
                convergence_count = 0
            prev_loss = current_loss
            
            if convergence_count >= self.convergence_threshold:
                logger.info(f"Critic收敛于epoch {epoch}")
                break
        
        # 解冻Actor
        for param in self.actor.parameters():
            param.requires_grad = True
        
        return final_loss
    
    def _train_actor_phase(self, states, actions, rewards, next_states, dones):
        """
        训练Actor网络
        
        Args:
            states, actions, rewards, next_states, dones: 批量数据
        
        Returns:
            loss: Actor损失
        """
        # 冻结Critic
        for param in self.critic.parameters():
            param.requires_grad = False
        
        convergence_count = 0
        prev_loss = float('inf')
        final_loss = 0
        
        for epoch in range(self.max_epochs):
            # 计算Q(s, a)
            q_values = self.actor(states, actions)
            
            # 使用当前Critic计算TD目标
            with torch.no_grad():
                v_values = self.critic(next_states)
                td_targets = rewards + self.gamma * v_values * (1 - dones)
            
            # 计算Actor损失
            loss_q = F.mse_loss(q_values, td_targets)
            
            # 反向传播
            self.actor_optimizer.zero_grad()
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optimizer.step()
            
            current_loss = loss_q.item()
            final_loss = current_loss
            
            # 收敛检查
            if current_loss >= prev_loss:
                convergence_count += 1
            else:
                convergence_count = 0
            prev_loss = current_loss
            
            if convergence_count >= self.convergence_threshold:
                logger.info(f"Actor收敛于epoch {epoch}")
                break
        
        # 解冻Critic
        for param in self.critic.parameters():
            param.requires_grad = True
        
        return final_loss
    
    def _update_target_network(self):
        """更新Critic目标网络参数"""
        self.critic_target.load_state_dict(self.critic.state_dict())
    
    def save_model(self, path, day_count):
        """
        保存模型
        
        Args:
            path: 保存路径
            day_count: 训练天数
        """
        timestamp = datetime.now().strftime("%Y%m%d")
        
        # 保存Actor
        torch.save({
            'state_dict': self.actor.state_dict(),
            'optimizer': self.actor_optimizer.state_dict(),
            'day_count': day_count
        }, f"{path}/actor_{timestamp}_{day_count}.pth")
        
        # 保存Critic
        torch.save({
            'state_dict': self.critic.state_dict(),
            'optimizer': self.critic_optimizer.state_dict(),
            'day_count': day_count
        }, f"{path}/critic_{timestamp}_{day_count}.pth")
        
        # 同时保存最优模型
        torch.save({
            'state_dict': self.actor.state_dict(),
            'optimizer': self.actor_optimizer.state_dict(),
            'day_count': day_count
        }, f"{path}/actor_best.pth")
        
        torch.save({
            'state_dict': self.critic.state_dict(),
            'optimizer': self.critic_optimizer.state_dict(),
            'day_count': day_count
        }, f"{path}/critic_best.pth")
        
        # 保存训练曲线
        self._save_training_curves(path)
        
        logger.info(f"模型保存成功: {path}/actor_{timestamp}_{day_count}.pth")
    
    def load_model(self, path):
        """
        加载模型
        
        Args:
            path: 模型路径
        
        Returns:
            success: 是否成功
        """
        try:
            # 加载Actor
            actor_checkpoint = torch.load(f"{path}/actor_best.pth", map_location=self.device)
            self.actor.load_state_dict(actor_checkpoint['state_dict'])
            self.actor_optimizer.load_state_dict(actor_checkpoint['optimizer'])
            
            # 加载Critic
            critic_checkpoint = torch.load(f"{path}/critic_best.pth", map_location=self.device)
            self.critic.load_state_dict(critic_checkpoint['state_dict'])
            self.critic_optimizer.load_state_dict(critic_checkpoint['optimizer'])
            
            # 同步目标网络
            self.critic_target.load_state_dict(self.critic.state_dict())
            
            logger.info("模型加载成功")
            return True
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            return False
    
    def _save_training_curves(self, path):
        """
        保存训练曲线图
        
        Args:
            path: 保存路径
        """
        # 损失曲线
        if len(self.loss_history['actor']) > 0:
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(self.loss_history['actor'], label='Actor Loss')
            plt.plot(self.loss_history['critic'], label='Critic Loss')
            plt.xlabel('Training Steps')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.grid(True)
            
            # 奖励曲线
            plt.subplot(1, 2, 2)
            if len(self.reward_history) > 0:
                # 使用滑动平均平滑奖励曲线
                window_size = min(100, len(self.reward_history))
                if window_size > 1:
                    rewards_smooth = np.convolve(
                        self.reward_history, 
                        np.ones(window_size)/window_size, 
                        mode='valid'
                    )
                    plt.plot(rewards_smooth, label='Smoothed Reward', alpha=0.8)
                plt.plot(self.reward_history, label='Reward', alpha=0.3)
                plt.xlabel('Time Steps')
                plt.ylabel('Reward')
                plt.title('Reward Curve')
                plt.legend()
                plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{path}/loss_curve.png", dpi=300)
            plt.close()
        
        # 单独保存奖励曲线
        if len(self.reward_history) > 0:
            plt.figure(figsize=(10, 6))
            window_size = min(100, len(self.reward_history))
            if window_size > 1:
                rewards_smooth = np.convolve(
                    self.reward_history, 
                    np.ones(window_size)/window_size, 
                    mode='valid'
                )
                plt.plot(rewards_smooth, label='Smoothed Reward', linewidth=2)
            plt.plot(self.reward_history, label='Reward', alpha=0.3)
            plt.xlabel('Time Steps')
            plt.ylabel('Reward')
            plt.title('Reward Curve')
            plt.legend()
            plt.grid(True)
            plt.savefig(f"{path}/reward_curve.png", dpi=300)
            plt.close()


if __name__ == "__main__":
    # 测试智能体
    print("测试DQN智能体...")
    
    agent = DQNAgent(device='cpu')
    
    # 测试状态向量构建
    state = agent.build_state_vector(
        courier_lng=116.4,
        courier_lat=39.9,
        timestamp=12*3600,  # 12:00
        path_points=[
            {'lng': 116.5, 'lat': 40.0, 'distance_to_timeout': 30},
            {'lng': 116.6, 'lat': 40.1, 'distance_to_timeout': 20}
        ]
    )
    print(f"状态向量shape: {state.shape}, 前3维: {state[:3]}")
    
    # 测试动作向量构建
    action = agent.build_action_vector(
        sender_lng=116.5,
        sender_lat=40.0,
        receiver_lng=116.6,
        receiver_lat=40.1,
        available_time=30
    )
    print(f"动作向量shape: {action.shape}")
    
    # 测试动作选择
    candidate_orders = [
        {'sender_lng': 116.5, 'sender_lat': 40.0, 'receiver_lng': 116.6, 'receiver_lat': 40.1, 'available_time': 30},
        {'sender_lng': 116.7, 'sender_lat': 40.2, 'receiver_lng': 116.8, 'receiver_lat': 40.3, 'available_time': 40}
    ]
    selected_order, q_value = agent.select_action(state, candidate_orders, mode='train')
    print(f"选择订单: {selected_order}, Q值: {q_value}")
    
    print("智能体测试完成!")
