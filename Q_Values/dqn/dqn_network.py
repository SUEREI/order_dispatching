"""
DQN网络定义
包含Actor网络（计算Q值）和Critic网络（计算V值）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """
    Actor网络：计算Q(s, a)值
    输入：状态向量(27维) + 动作向量(5维) → 拼接为32维
    输出：Q值标量
    """
    
    def __init__(self, state_dim=27, action_dim=5):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态向量维度，默认27维
            action_dim: 动作向量维度，默认5维
        """
        super(ActorNetwork, self).__init__()
        
        input_dim = state_dim + action_dim  # 32维
        
        # 网络结构：Dense(256)→BN→ReLU→Dense(256)→BN→ReLU→Dense(128)→ReLU→Dense(1)
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重
        使用Xavier均匀初始化权重，常数初始化偏置为0
        """
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state, action):
        """
        前向传播：计算Q(s, a)
        
        Args:
            state: 状态向量，shape=(batch_size, 27)
            action: 动作向量，shape=(batch_size, 5)
        
        Returns:
            Q值，shape=(batch_size, 1)
        """
        # 拼接状态和动作
        x = torch.cat([state, action], dim=-1)  # (batch_size, 32)
        
        # 前向传播
        x = F.relu(self.bn1(self.fc1(x)))  # (batch_size, 256)
        x = F.relu(self.bn2(self.fc2(x)))  # (batch_size, 256)
        x = F.relu(self.fc3(x))            # (batch_size, 128)
        q_value = self.fc4(x)              # (batch_size, 1)
        
        return q_value


class CriticNetwork(nn.Module):
    """
    Critic网络：计算V(s)值
    输入：状态向量(27维)
    输出：V值标量
    """
    
    def __init__(self, state_dim=27):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态向量维度，默认27维
        """
        super(CriticNetwork, self).__init__()
        
        # 网络结构：Dense(256)→BN→ReLU→Dense(256)→BN→ReLU→Dense(128)→ReLU→Dense(1)
        self.fc1 = nn.Linear(state_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重
        使用Xavier均匀初始化权重，常数初始化偏置为0
        """
        for layer in [self.fc1, self.fc2, self.fc3, self.fc4]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, state):
        """
        前向传播：计算V(s)
        
        Args:
            state: 状态向量，shape=(batch_size, 27)
        
        Returns:
            V值，shape=(batch_size, 1)
        """
        # 前向传播
        x = F.relu(self.bn1(self.fc1(state)))  # (batch_size, 256)
        x = F.relu(self.bn2(self.fc2(x)))      # (batch_size, 256)
        x = F.relu(self.fc3(x))                # (batch_size, 128)
        v_value = self.fc4(x)                  # (batch_size, 1)
        
        return v_value


if __name__ == "__main__":
    # 测试网络结构
    batch_size = 32
    state_dim = 27
    action_dim = 5
    
    # 测试Actor网络
    actor = ActorNetwork(state_dim, action_dim)
    state = torch.randn(batch_size, state_dim)
    action = torch.randn(batch_size, action_dim)
    q_value = actor(state, action)
    print(f"Actor网络测试: 输入shape=({batch_size}, {state_dim + action_dim}), 输出shape={q_value.shape}")
    
    # 测试Critic网络
    critic = CriticNetwork(state_dim)
    v_value = critic(state)
    print(f"Critic网络测试: 输入shape=({batch_size}, {state_dim}), 输出shape={v_value.shape}")
    
    # 统计参数量
    actor_params = sum(p.numel() for p in actor.parameters())
    critic_params = sum(p.numel() for p in critic.parameters())
    print(f"Actor参数量: {actor_params:,}")
    print(f"Critic参数量: {critic_params:,}")
