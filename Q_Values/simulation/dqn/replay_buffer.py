"""
经验回放缓冲区模块

该模块实现了经验回放缓冲区（Replay Buffer），用于存储仿真过程中产生的经验数据，
支持随机采样和FIFO（先进先出）策略。
"""

import torch
import numpy as np
from collections import deque
from typing import Tuple, Optional
import random


class ReplayBuffer:
    """
    经验回放缓冲区类
    
    使用deque实现FIFO策略的经验回放缓冲区，存储状态、动作、奖励、下一状态和终止标志。
    当缓冲区满时，自动覆盖最早的数据。
    
    Attributes:
        buffer (deque): 经验数据存储队列
        capacity (int): 缓冲区容量上限
        state_dim (int): 状态向量维度
        action_dim (int): 动作向量维度
    
    Example:
        >>> buffer = ReplayBuffer(capacity=10000, state_dim=25, action_dim=5)
        >>> buffer.push(state, action, reward, next_state, done)
        >>> states, actions, rewards, next_states, dones = buffer.sample(32)
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        state_dim: int = 25,
        action_dim: int = 5
    ) -> None:
        """
        初始化经验回放缓冲区
        
        Args:
            capacity (int): 缓冲区容量上限，默认为10000
            state_dim (int): 状态向量维度，默认为25
            action_dim (int): 动作向量维度，默认为5
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """
        添加单条经验数据到缓冲区
        
        Args:
            state (np.ndarray): 当前状态，shape为(state_dim,)
            action (np.ndarray): 执行的动作，shape为(action_dim,)
            reward (float): 获得的即时奖励
            next_state (np.ndarray): 下一状态，shape为(state_dim,)
            done (bool): 是否为终止状态
        
        Note:
            数据以元组形式存储：(state, action, reward, next_state, done)
            当缓冲区满时，自动移除最早的数据（FIFO策略）
        """
        # 确保数据为numpy数组格式
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state, dtype=np.float32)
        
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def push_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray
    ) -> None:
        """
        批量添加经验数据到缓冲区
        
        Args:
            states (np.ndarray): 状态数组，shape为(batch_size, state_dim)
            actions (np.ndarray): 动作数组，shape为(batch_size, action_dim)
            rewards (np.ndarray): 奖励数组，shape为(batch_size,)
            next_states (np.ndarray): 下一状态数组，shape为(batch_size, state_dim)
            dones (np.ndarray): 终止标志数组，shape为(batch_size,)
        """
        batch_size = len(states)
        for i in range(batch_size):
            self.push(states[i], actions[i], rewards[i], next_states[i], dones[i])
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从缓冲区随机采样一批经验数据
        
        Args:
            batch_size (int): 采样批次大小
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                返回包含五个张量的元组：
                - states: 状态张量，shape为(batch_size, state_dim)
                - actions: 动作张量，shape为(batch_size, action_dim)
                - rewards: 奖励张量，shape为(batch_size, 1)
                - next_states: 下一状态张量，shape为(batch_size, state_dim)
                - dones: 终止标志张量，shape为(batch_size, 1)
        
        Raises:
            ValueError: 如果缓冲区中的数据少于batch_size
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"缓冲区数据不足：当前有{len(self.buffer)}条数据，但请求采样{batch_size}条")
        
        # 随机采样
        batch = random.sample(list(self.buffer), batch_size)
        
        # 解包数据
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为torch.Tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)  # shape: (batch_size, 1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1)  # shape: (batch_size, 1)
        
        return states, actions, rewards, next_states, dones
    
    def sample_all(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        获取缓冲区中的全部经验数据
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                返回包含五个张量的元组：
                - states: 状态张量，shape为(buffer_size, state_dim)
                - actions: 动作张量，shape为(buffer_size, action_dim)
                - rewards: 奖励张量，shape为(buffer_size, 1)
                - next_states: 下一状态张量，shape为(buffer_size, state_dim)
                - dones: 终止标志张量，shape为(buffer_size, 1)
        
        Raises:
            ValueError: 如果缓冲区为空
        """
        if len(self.buffer) == 0:
            raise ValueError("缓冲区为空，无法获取数据")
        
        # 获取全部数据
        batch = list(self.buffer)
        
        # 解包数据
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # 转换为torch.Tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones, dtype=np.float32)).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """
        获取缓冲区当前存储的经验数量
        
        Returns:
            int: 缓冲区中的经验数量
        """
        return len(self.buffer)
    
    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空
        
        Returns:
            bool: 如果缓冲区为空返回True，否则返回False
        """
        return len(self.buffer) == 0
    
    def clear(self) -> None:
        """
        清空缓冲区中的所有数据
        """
        self.buffer.clear()
