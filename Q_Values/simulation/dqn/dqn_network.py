#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DQN神经网络模块

该模块实现DQN强化学习所需的核心组件，包括：
- DQNConfig: 超参数配置类
- ReplayBuffer: 经验回放缓冲区
- DQNNetwork: Q值神经网络
- DQNAgent: DQN智能体（含训练逻辑）
"""

import logging
import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    """DQN超参数配置"""
    state_dim: int = 21
    hidden_dim: int = 256
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_capacity: int = 10000
    batch_size: int = 128
    tau: float = 0.01
    epsilon_start: float = 0.5
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    max_orders_per_courier: int = 4
    candidate_count: int = 20
    base_revenue: float = 3.0
    distance_revenue_coeff: float = 1.0
    max_delivery_time: float = 3600.0
    lng_center: float = 0.0
    lng_range: float = 1.0
    lat_center: float = 0.0
    lat_range: float = 1.0
    reward_scale: float = 0.2


class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool = False):
        """存储一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一批经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Q值神经网络"""

    def __init__(self, state_dim: int, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent:
    """DQN智能体"""

    def __init__(self, state_dim: int, config: DQNConfig = None):
        self.config = config or DQNConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_step = 0
        self.total_loss = 0.0

        self.policy_net = DQNNetwork(state_dim, self.config.hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, self.config.hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.lr)
        self.replay_buffer = ReplayBuffer(self.config.buffer_capacity)
        self.epsilon = self.config.epsilon_start

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        logger.critical(f"DQN训练设备: {self.device} | {device_name}")
        logger.critical(f"DQN网络结构: state_dim={state_dim}, hidden_dim={self.config.hidden_dim}")
        logger.critical(f"经验回放缓冲区容量: {self.config.buffer_capacity}, 批次大小: {self.config.batch_size}")

    def batch_predict(self, states: np.ndarray) -> List[float]:
        """批量Q值预测

        Args:
            states: 形状(N, state_dim)的状态向量数组

        Returns:
            N个Q值列表
        """
        self.policy_net.eval()
        with torch.no_grad():
            tensor_states = torch.FloatTensor(states).to(self.device)
            q_values = self.policy_net(tensor_states).squeeze(-1)
        self.policy_net.train()
        return q_values.cpu().numpy().tolist()

    def update(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> float:
        """网络参数更新

        Args:
            batch: (states, actions, rewards, next_states, dones)

        Returns:
            损失值
        """
        self.training_step += 1
        states, actions, rewards, next_states, dones = batch

        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device) * self.config.reward_scale
        next_states_t = torch.FloatTensor(next_states).to(self.device)
        dones_t = torch.FloatTensor(dones).to(self.device)

        # 当前Q值: Q(s, a)
        current_q = self.policy_net(states_t).squeeze(-1)

        # 目标Q值: r + γ * Q_target(s', argmax_a Q(s', a))
        with torch.no_grad():
            next_q = self.target_net(next_states_t).squeeze(-1)
        target_q = rewards_t + (1 - dones_t) * self.config.gamma * next_q

        # MSE损失
        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 软更新目标网络
        self.soft_update_target()

        self.total_loss += loss.item()
        if self.training_step % 100 == 0:
            avg_loss = self.total_loss / 100
            self.total_loss = 0.0
            logger.critical(f"DQN训练进度: Step {self.training_step} | 平均损失: {avg_loss:.6f} | ε: {self.epsilon:.4f}")

        return loss.item()

    def soft_update_target(self):
        """软更新目标网络参数: θ_target = τ*θ_policy + (1-τ)*θ_target"""
        tau = self.config.tau
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """保存网络参数到文件"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon': self.epsilon
        }, path)
        logger.info(f"DQN模型已保存到 {path}")

    def load(self, path: str):
        """从文件加载网络参数"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint.get('epsilon', self.config.epsilon_start)
        logger.info(f"DQN模型已从 {path} 加载")
