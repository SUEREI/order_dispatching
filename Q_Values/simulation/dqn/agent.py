"""
Agent集成模块

该模块实现了Agent类，封装Actor-Critic网络和ReplayBuffer，提供统一的训练和推理接口，
实现离线交替训练机制。
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import os

from actor_network import ActorNetwork
from critic_network import CriticNetwork
from replay_buffer import ReplayBuffer


class Agent:
    """
    Agent集成类
    
    封装Actor-Critic网络和ReplayBuffer，提供统一的训练和推理接口。
    支持训练模式（Softmax采样）和应用模式（Argmax），实现离线交替训练机制。
    
    训练流程：
    1. 数据收集：完成一整天仿真后，将所有步骤的转移元组存入ReplayBuffer
    2. 冻结Actor，全量训练Critic：使用TD误差更新Critic网络
    3. 冻结Critic，全量训练Actor：以最大化Q值为目标更新Actor网络
    4. 循环：进入下一天的仿真，重复步骤1-3
    
    Attributes:
        actor (ActorNetwork): Actor网络
        critic (CriticNetwork): Critic网络
        replay_buffer (ReplayBuffer): 经验回放缓冲区
        gamma (float): 折扣因子
        device (torch.device): 计算设备
        is_train (bool): 是否为训练模式
    
    Example:
        >>> agent = Agent(state_dim=25, action_dim=5, gamma=0.99)
        >>> # 训练模式
        >>> action = agent.select_action(state, candidate_actions, mode='train')
        >>> # 应用模式
        >>> action = agent.select_action(state, candidate_actions, mode='eval')
        >>> # 离线训练
        >>> critic_loss, actor_loss = agent.train_step()
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 5,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 10000,
        device: str = 'auto',
        is_train: bool = True
    ) -> None:
        """
        初始化Agent
        
        Args:
            state_dim (int): 状态向量维度，默认为25
            action_dim (int): 动作向量维度，默认为5
            hidden_dims (List[int]): 隐藏层维度列表，默认为[128, 64]
            activation (str): 激活函数类型，默认为'relu'
            learning_rate (float): 学习率，默认为1e-3
            gamma (float): 折扣因子，默认为0.99
            buffer_capacity (int): 经验回放缓冲区容量，默认为10000
            device (str): 计算设备，'auto'为自动选择，默认为'auto'
            is_train (bool): 是否为训练模式，默认为True
        """
        # 设备选择
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 初始化网络
        self.actor = ActorNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate
        ).to(self.device)
        
        self.critic = CriticNetwork(
            state_dim=state_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            learning_rate=learning_rate
        ).to(self.device)
        
        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(
            capacity=buffer_capacity,
            state_dim=state_dim,
            action_dim=action_dim
        )
        
        # 保存配置参数
        self.gamma = gamma
        self.is_train = is_train
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def select_action(
        self,
        state: np.ndarray,
        candidate_actions: np.ndarray,
        mode: Optional[str] = None,
        temperature: float = 1.0
    ) -> np.ndarray:
        """
        根据当前状态和候选动作选择动作
        
        训练模式使用Softmax概率采样，应用模式使用Argmax选择最优动作。
        
        Args:
            state (np.ndarray): 当前状态，shape为(state_dim,)
            candidate_actions (np.ndarray): 候选动作集合，shape为(N, action_dim)，N为候选动作数量
            mode (Optional[str]): 模式选择，'train'为训练模式，'eval'为应用模式。
                                  如果为None，则使用self.is_train判断，默认为None
            temperature (float): Softmax温度参数，控制探索程度，值越大探索性越强，默认为1.0
        
        Returns:
            np.ndarray: 选中的动作向量，shape为(action_dim,)
        
        Note:
            - 训练模式：使用Softmax概率分布采样，增加探索性
            - 应用模式：使用Argmax选择最优动作，确定性决策
            - 温度参数：值越大，探索性越强；值越小，越倾向于选择最优动作
        """
        # 确定模式
        if mode is None:
            mode = 'train' if self.is_train else 'eval'
        
        # 转换为torch.Tensor并移至device
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # shape: (1, state_dim)
        candidate_actions_tensor = torch.FloatTensor(candidate_actions).to(self.device)  # shape: (N, action_dim)
        
        # 扩展state以匹配候选动作数量
        batch_size = candidate_actions_tensor.shape[0]
        states_expanded = state_tensor.expand(batch_size, -1)  # shape: (N, state_dim)
        
        # 计算所有候选动作的Q值
        with torch.no_grad():
            q_values = self.actor(states_expanded, candidate_actions_tensor)  # shape: (N, 1)
            q_values = q_values.squeeze(1)  # shape: (N,)
        
        if mode == 'train':
            # 训练模式：Softmax概率采样
            probs = F.softmax(q_values / temperature, dim=0)
            action_idx = torch.multinomial(probs, num_samples=1).item()
        else:
            # 应用模式：Argmax选择最优动作
            action_idx = torch.argmax(q_values).item()
        
        # 返回选中的动作
        selected_action = candidate_actions[action_idx]
        return selected_action
    
    def update_critic_offline(self, batch_size: Optional[int] = None) -> float:
        """
        离线训练Critic网络
        
        冻结Actor网络，使用TD误差训练Critic网络。
        
        Args:
            batch_size (Optional[int]): 采样批次大小。如果为None，则使用全部数据，默认为None
        
        Returns:
            float: Critic网络的损失值
        
        Note:
            训练流程：
            1. 冻结Actor网络
            2. 从ReplayBuffer采样数据（全量或指定批次）
            3. 计算TD目标：y = r + γ·V(s')·(1-done)
            4. 计算MSE损失：loss = mean((V(s) - y)²)
            5. 反向传播更新Critic参数
            6. 解冻Actor网络
        """
        # 冻结Actor网络
        self.actor.freeze()
        
        # 获取数据
        if batch_size is None:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_all()
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 移至device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算当前状态价值
        v_s = self.critic(states)
        
        # 计算下一状态价值
        with torch.no_grad():
            v_s_next = self.critic(next_states)
            # 计算TD目标：y = r + γ·V(s')·(1-done)
            td_target = rewards + self.gamma * v_s_next * (1 - dones)
        
        # 计算MSE损失
        loss = F.mse_loss(v_s, td_target)
        
        # 反向传播
        self.critic.optimizer.zero_grad()
        loss.backward()
        self.critic.optimizer.step()
        
        # 解冻Actor网络
        self.actor.unfreeze()
        
        return loss.item()
    
    def update_actor_offline(self, batch_size: Optional[int] = None) -> float:
        """
        离线训练Actor网络
        
        冻结Critic网络，以最大化Q值为目标训练Actor网络。
        
        Args:
            batch_size (Optional[int]): 采样批次大小。如果为None，则使用全部数据，默认为None
        
        Returns:
            float: Actor网络的损失值
        
        Note:
            训练流程：
            1. 冻结Critic网络
            2. 从ReplayBuffer采样数据
            3. 计算Q值：Q(s, a)
            4. 计算损失：loss = -mean(Q(s, a))（最大化Q值）
            5. 反向传播更新Actor参数
            6. 解冻Critic网络
        """
        # 冻结Critic网络
        self.critic.freeze()
        
        # 获取数据
        if batch_size is None:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample_all()
        else:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 移至device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # 计算Q值
        q_values = self.actor(states, actions)
        
        # 计算损失：最大化Q值，即最小化-Q值
        loss = -torch.mean(q_values)
        
        # 反向传播
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        
        # 解冻Critic网络
        self.critic.unfreeze()
        
        return loss.item()
    
    def train_step(self, batch_size: Optional[int] = None) -> Tuple[float, float]:
        """
        执行一轮完整的交替训练
        
        先训练Critic网络，再训练Actor网络。
        
        Args:
            batch_size (Optional[int]): 采样批次大小。如果为None，则使用全部数据，默认为None
        
        Returns:
            Tuple[float, float]: 返回(Critic损失值, Actor损失值)
        
        Note:
            交替训练流程：
            1. 冻结Actor，训练Critic
            2. 冻结Critic，训练Actor
        """
        critic_loss = self.update_critic_offline(batch_size)
        actor_loss = self.update_actor_offline(batch_size)
        return (critic_loss, actor_loss)
    
    def save_model(self, filepath: str) -> None:
        """
        保存模型参数和优化器状态
        
        Args:
            filepath (str): 模型保存路径
        
        Raises:
            OSError: 如果无法创建目录或保存文件
        
        Note:
            保存内容包括：
            - Actor网络参数
            - Critic网络参数
            - Actor优化器状态
            - Critic优化器状态
        """
        # 确保目录存在
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # 构建checkpoint
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic_optimizer': self.critic.optimizer.state_dict(),
            'gamma': self.gamma,
            'is_train': self.is_train
        }
        
        # 保存模型
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str) -> None:
        """
        加载模型参数和优化器状态
        
        Args:
            filepath (str): 模型文件路径
        
        Raises:
            FileNotFoundError: 如果模型文件不存在
            RuntimeError: 如果模型文件格式错误
        
        Note:
            加载内容包括：
            - Actor网络参数
            - Critic网络参数
            - Actor优化器状态
            - Critic优化器状态
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            # 加载网络参数
            self.actor.load_state_dict(checkpoint['actor_state_dict'])
            self.critic.load_state_dict(checkpoint['critic_state_dict'])
            
            # 加载优化器状态
            self.actor.optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic.optimizer.load_state_dict(checkpoint['critic_optimizer'])
            
            # 加载配置参数
            if 'gamma' in checkpoint:
                self.gamma = checkpoint['gamma']
            if 'is_train' in checkpoint:
                self.is_train = checkpoint['is_train']
                
        except RuntimeError as e:
            raise RuntimeError(f"模型文件格式错误: {e}")
    
    def set_train_mode(self, is_train: bool) -> None:
        """
        设置训练模式
        
        Args:
            is_train (bool): True为训练模式，False为应用模式
        """
        self.is_train = is_train
        if is_train:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()
