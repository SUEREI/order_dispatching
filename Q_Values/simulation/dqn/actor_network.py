"""
Actor网络模块

该模块实现了Actor网络（策略网络/DMN），用于根据状态-动作对评估Q值。
Actor网络接收状态和动作的拼接作为输入，输出该状态-动作对的Q值。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class ActorNetwork(nn.Module):
    """
    Actor网络类（策略网络/DMN）
    
    该网络用于评估状态-动作对的Q值Q(s,a)。输入为状态和动作的拼接，
    输出为Q值标量。网络结构为多层全连接网络（MLP）。
    
    Attributes:
        state_dim (int): 状态向量维度
        action_dim (int): 动作向量维度
        hidden_dims (List[int]): 隐藏层维度列表
        activation (str): 激活函数类型
        layers (nn.ModuleList): 网络层列表
        activation_fn: 激活函数
        optimizer: 优化器
    
    Example:
        >>> actor = ActorNetwork(state_dim=25, action_dim=5, hidden_dims=[128, 64])
        >>> state = torch.randn(32, 25)  # batch_size=32
        >>> action = torch.randn(32, 5)
        >>> q_values = actor(state, action)  # shape: (32, 1)
    """
    
    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 5,
        hidden_dims: List[int] = [128, 64],
        activation: str = 'relu',
        learning_rate: float = 1e-3
    ) -> None:
        """
        初始化Actor网络
        
        Args:
            state_dim (int): 状态向量维度，默认为25
            action_dim (int): 动作向量维度，默认为5
            hidden_dims (List[int]): 隐藏层维度列表，默认为[128, 64]
            activation (str): 激活函数类型，支持'relu'、'leaky_relu'、'tanh'，默认为'relu'
            learning_rate (float): 学习率，默认为1e-3
        
        Raises:
            ValueError: 如果激活函数类型不支持
        """
        super(ActorNetwork, self).__init__()
        
        # 保存配置参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        
        # 构建网络层
        input_dim = state_dim + action_dim
        layer_dims = [input_dim] + hidden_dims + [1]
        
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
        
        # 选择激活函数
        if activation == 'relu':
            self.activation_fn = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation_fn = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation_fn = nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数类型: {activation}，请选择 'relu', 'leaky_relu' 或 'tanh'")
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算状态-动作对的Q值
        
        Args:
            state (torch.Tensor): 状态张量，shape为(batch_size, state_dim)
            action (torch.Tensor): 动作张量，shape为(batch_size, action_dim)
        
        Returns:
            torch.Tensor: Q值张量，shape为(batch_size, 1)
        
        Raises:
            RuntimeError: 如果输入维度不匹配
        """
        # 维度检查
        if state.dim() != 2 or action.dim() != 2:
            raise RuntimeError(f"输入维度错误：state应为2维，action应为2维，但得到state.dim()={state.dim()}, action.dim()={action.dim()}")
        
        if state.shape[0] != action.shape[0]:
            raise RuntimeError(f"批次大小不匹配：state的批次大小为{state.shape[0]}，action为{action.shape[0]}")
        
        # 拼接状态和动作
        x = torch.cat([state, action], dim=1)
        
        # 通过隐藏层和激活函数
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation_fn(x)
        
        # 输出层（不接激活函数）
        x = self.layers[-1](x)
        
        return x
    
    def freeze(self) -> None:
        """
        冻结网络参数
        
        将所有参数的requires_grad设为False，用于交替训练时冻结Actor网络。
        在训练Critic时，需要冻结Actor以防止其参数被更新。
        """
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """
        解冻网络参数
        
        将所有参数的requires_grad设为True，用于恢复Actor网络的训练。
        在训练Critic完成后，需要解冻Actor以进行后续训练。
        """
        for param in self.parameters():
            param.requires_grad = True
