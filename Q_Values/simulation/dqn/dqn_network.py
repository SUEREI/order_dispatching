#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actor-Critic 神经网络模块

该模块实现 Actor-Critic 强化学习所需的核心组件，包括：
- ActorConfig: 超参数配置类
- ReplayBuffer: 经验回放缓冲区（存储动作向量）
- ActorNetwork: 策略网络/动作网络，输入 (state + action)，输出 Q(s,a)
- CriticNetwork: 价值网络，输入 state，输出 V(s)
- ActorCriticAgent: Actor-Critic 智能体（含离线交替训练逻辑）
"""

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


@dataclass
class ActorCriticConfig:
    """Actor-Critic 超参数配置"""
    state_dim: int = 25
    action_dim: int = 5
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    lr_actor: float = 1e-3
    lr_critic: float = 1e-4
    gamma: float = 0.98
    buffer_capacity: int = 100000
    batch_size: int = 128
    critic_epochs: int = 100
    actor_epochs: int = 100
    batches_per_epoch: int = 50
    target_update_rate: float = 0.005
    grad_clip_norm: float = 1.0
    max_orders_per_courier: int = 4
    candidate_count: int = 20
    base_revenue: float = 3.0
    distance_revenue_coeff: float = 1.5
    max_delivery_time: float = 1800.0
    lng_center: float = 174.387
    lng_max: float = 174.995
    lng_range: float = lng_max - lng_center
    lat_center: float = 45.728
    lat_max: float = 46.073
    lat_range: float = lat_max - lat_center
    reward_scale: float = 1.0

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [128, 64, 32]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [128, 64, 32]


class ReplayBuffer:
    """经验回放缓冲区

    用于存储离线训练数据。
    存储格式: (state, action_vector, reward, next_state)
    注意: 不存储 done 标志，因为奖励是即时边际结算。
    """

    def __init__(self, capacity: int):
        """初始化经验回放缓冲区

        Args:
            capacity: 缓冲区最大容量
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray
    ) -> None:
        """存储一条经验

        Args:
            state: 状态向量 (state_dim,)
            action: 动作向量 (action_dim,)，具体5维动作而非动作索引
            reward: 即时奖励
            next_state: 下一状态向量 (state_dim,)
        """
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """随机采样一批经验

        Args:
            batch_size: 批次大小

        Returns:
            (states, actions, rewards, next_states) 元组
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32)
        )

    def __len__(self) -> int:
        """返回缓冲区当前长度"""
        return len(self.buffer)


class ActorNetwork(nn.Module):
    """Actor 网络（策略网络/DMN）

    输入: 状态向量与动作向量的拼接 (state_dim + action_dim)
    输出: 每个动作的 Q 值 Q(s, a)

    结构: MLP (多层全连接层) + ReLU 激活函数
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        """初始化 Actor 网络

        Args:
            state_dim: 状态向量维度
            action_dim: 动作向量维度
            hidden_dims: 隐藏层维度列表，默认为 [128, 64, 32]
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播计算 Q(s, a)

        Args:
            state: 状态张量 (Batch, state_dim) 或 (state_dim,)
            action: 动作张量 (Batch, action_dim) 或 (action_dim,)

        Returns:
            Q 值张量 (Batch, 1) 或 (1,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        x = torch.cat([state, action], dim=-1)
        q_value = self.net(x)

        if squeeze_output:
            q_value = q_value.squeeze(0)

        return q_value


class CriticNetwork(nn.Module):
    """Critic 网络（价值网络）

    输入: 状态向量
    输出: 状态价值 V(s)

    结构: MLP (多层全连接层) + ReLU 激活函数
    """

    def __init__(self, state_dim: int, hidden_dims: List[int] = None):
        """初始化 Critic 网络

        Args:
            state_dim: 状态向量维度
            hidden_dims: 隐藏层维度列表，默认为 [128, 64]
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播计算 V(s)

        Args:
            state: 状态张量 (Batch, state_dim) 或 (state_dim,)

        Returns:
            状态价值张量 (Batch, 1) 或 (1,)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        value = self.net(state)

        if squeeze_output:
            value = value.squeeze(0)

        return value


class ActorCriticAgent:
    """Actor-Critic 智能体

    支持离线交替训练模式：
    - Step 2: 冻结 Actor，全量训练 Critic
    - Step 3: 冻结 Critic，全量训练 Actor

    支持训练模式（Softmax 采样）和应用模式（Argmax）切换。
    """

    def __init__(
        self,
        state_dim: int = 25,
        action_dim: int = 5,
        config: ActorCriticConfig = None
    ):
        """初始化 Actor-Critic 智能体

        Args:
            state_dim: 状态向量维度
            action_dim: 动作向量维度
            config: 超参数配置
        """
        self.config = config or ActorCriticConfig()
        self.state_dim = self.config.state_dim
        self.action_dim = self.config.action_dim
        self.gamma = self.config.gamma
        self.is_train = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.actor = ActorNetwork(
            self.state_dim,
            self.action_dim,
            self.config.actor_hidden_dims
        ).to(self.device)

        self.critic = CriticNetwork(
            self.state_dim,
            self.config.critic_hidden_dims
        ).to(self.device)

        self.target_critic = CriticNetwork(
            self.state_dim,
            self.config.critic_hidden_dims
        ).to(self.device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config.lr_critic)

        self.replay_buffer = ReplayBuffer(self.config.buffer_capacity)

        self.training_step = 0
        self.critic_loss_sum = 0.0
        self.actor_loss_sum = 0.0
        
        self.critic_losses = []
        self.actor_losses = []

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        logger.critical(f"Actor-Critic 训练设备: {self.device} | {device_name}")
        logger.critical(f"Actor 网络结构: state_dim={self.state_dim}, action_dim={self.action_dim}, "
                        f"hidden_dims={self.config.actor_hidden_dims}")
        logger.critical(f"Critic 网络结构: state_dim={self.state_dim}, "
                        f"hidden_dims={self.config.critic_hidden_dims}")
        logger.critical(f"经验回放缓冲区容量: {self.config.buffer_capacity}, 批次大小: {self.config.batch_size}")

    def set_train_mode(self) -> None:
        """设置为训练模式"""
        self.is_train = True
        self.actor.train()
        self.critic.train()

    def set_eval_mode(self) -> None:
        """设置为评估/应用模式"""
        self.is_train = False
        self.actor.eval()
        self.critic.eval()

    def select_action(
        self,
        state: np.ndarray,
        candidate_actions: np.ndarray
    ) -> Tuple[np.ndarray, Optional[float], int]:
        """选择动作

        训练模式: 使用 Softmax 概率采样
        应用模式: 直接输出 Q 值最高的动作（Argmax），不计算 log_prob

        Args:
            state: 状态数组 (state_dim,) 或 (N, state_dim)
            candidate_actions: 候选动作数组 (N, action_dim)，N个5维动作向量

        Returns:
            (selected_action, log_prob, chosen_index):
            - 选中的动作向量 (action_dim,)
            - 训练模式下返回 log_prob，评估模式下返回 None
            - 选中的候选索引
        """
        if state.ndim == 1:
            state = state[np.newaxis, :]

        state_tensor = torch.FloatTensor(state).to(self.device)
        actions_tensor = torch.FloatTensor(candidate_actions).to(self.device)

        with torch.no_grad():
            q_values = self.actor(state_tensor, actions_tensor).squeeze(-1)

        if self.is_train:
            probs = F.softmax(q_values, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            chosen_index = action_idx.item()
            selected_action = candidate_actions[chosen_index]
            return selected_action, log_prob.item(), chosen_index
        else:
            chosen_index = torch.argmax(q_values).item()
            selected_action = candidate_actions[chosen_index]
            return selected_action, None, chosen_index

    def batch_predict(
        self,
        states: np.ndarray,
        candidate_actions: np.ndarray
    ) -> np.ndarray:
        """批量预测 Q 值

        用于批量计算多个状态-动作对的 Q 值。

        Args:
            states: 状态数组 (Batch, state_dim)
            candidate_actions: 候选动作数组 (Batch, N, action_dim) 或 (N, action_dim)

        Returns:
            Q 值数组 (Batch,) 或 (N,)
        """
        self.actor.eval()
        with torch.no_grad():
            if candidate_actions.ndim == 2:
                states_tensor = torch.FloatTensor(states).to(self.device)
                actions_tensor = torch.FloatTensor(candidate_actions).to(self.device)

                if states_tensor.size(0) == 1 and actions_tensor.size(0) > 1:
                    states_tensor = states_tensor.expand(actions_tensor.size(0), -1)

                q_values = self.actor(states_tensor, actions_tensor).squeeze(-1)
            else:
                batch_size, num_candidates, action_dim = candidate_actions.shape
                states_tensor = torch.FloatTensor(states).to(self.device)
                actions_tensor = torch.FloatTensor(candidate_actions).to(self.device)

                q_values_list = []
                for i in range(batch_size):
                    q = self.actor(states_tensor[i], actions_tensor[i]).squeeze(-1)
                    q_values_list.append(q)

                q_values = torch.stack(q_values_list)

        self.actor.train()
        return q_values.cpu().numpy()

    def update_critic_offline(self) -> float:
        """离线训练 Critic（Step 2）

        冻结 Actor 网络参数，全量训练 Critic。
        使用 Target Critic 网络计算 TD Target，提高训练稳定性。
        TD Target: y = r_t + γ * V_target(s_{t+1})
        Critic Loss: Loss_critic = MSE(V_critic(s_t), y)

        Returns:
            平均 Critic 损失
        """
        if len(self.replay_buffer) < self.config.batch_size:
            logger.warning("ReplayBuffer 数据不足，跳过 Critic 训练")
            return 0.0

        self.actor.eval()
        self.critic.train()
        self.target_critic.eval()

        total_loss = 0.0
        num_batches = 0

        batches_per_epoch = self.config.batches_per_epoch
        for epoch in range(self.config.critic_epochs):
            epoch_loss = 0.0
            for _ in range(batches_per_epoch):
                batch = self.replay_buffer.sample(self.config.batch_size)
                states, actions, rewards, next_states = batch

                states_t = torch.FloatTensor(states).to(self.device)
                rewards_t = torch.FloatTensor(rewards).to(self.device) * self.config.reward_scale
                next_states_t = torch.FloatTensor(next_states).to(self.device)

                with torch.no_grad():
                    next_values = self.target_critic(next_states_t).squeeze(-1)

                td_targets = rewards_t + self.gamma * next_values
                td_targets = td_targets.detach()

                current_values = self.critic(states_t).squeeze(-1)

                loss = F.mse_loss(current_values, td_targets)

                self.critic_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.grad_clip_norm)
                self.critic_optimizer.step()

                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                self.training_step += 1
                self.critic_losses.append(loss.item())

            avg_epoch_loss = epoch_loss / max(batches_per_epoch, 1)
            logger.critical(f"Critic 训练 | Epoch {epoch+1}/{self.config.critic_epochs} | "
                        f"Avg Loss: {avg_epoch_loss:.6f}")

            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.config.target_update_rate * param.data +
                                            (1 - self.config.target_update_rate) * target_param.data)

        avg_loss = total_loss / max(num_batches, 1)

        return avg_loss

    def update_actor_offline(self) -> float:
        """离线训练 Actor（Step 3）

        冻结 Critic 网络参数，全量训练 Actor。
        将 Actor (Q网络) 的输出拟合到 Critic 提供的 TD Target：
        Loss_actor = MSE(Q(s_t, a_t), r_t + γ * V_critic(s_{t+1}))

        Returns:
            平均 Actor 损失
        """
        if len(self.replay_buffer) < self.config.batch_size:
            logger.warning("ReplayBuffer 数据不足，跳过 Actor 训练")
            return 0.0

        self.actor.train()
        self.critic.eval()

        total_loss = 0.0
        num_batches = 0

        batches_per_epoch = self.config.batches_per_epoch
        for epoch in range(self.config.actor_epochs):
            epoch_loss = 0.0
            for _ in range(batches_per_epoch):
                batch = self.replay_buffer.sample(self.config.batch_size)
                states, actions, rewards, next_states = batch

                states_t = torch.FloatTensor(states).to(self.device)
                actions_t = torch.FloatTensor(actions).to(self.device)
                rewards_t = torch.FloatTensor(rewards).to(self.device) * self.config.reward_scale
                next_states_t = torch.FloatTensor(next_states).to(self.device)

                with torch.no_grad():
                    next_values = self.critic(next_states_t).squeeze(-1)
                    td_targets = rewards_t + self.gamma * next_values

                q_values = self.actor(states_t, actions_t).squeeze(-1)

                loss = F.mse_loss(q_values, td_targets)

                self.actor_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.grad_clip_norm)
                self.actor_optimizer.step()

                total_loss += loss.item()
                epoch_loss += loss.item()
                num_batches += 1
                self.training_step += 1

                self.actor_losses.append(loss.item())

            avg_epoch_loss = epoch_loss / max(batches_per_epoch, 1)
            logger.critical(f"Actor 训练 | Epoch {epoch+1}/{self.config.actor_epochs} | "
                        f"Avg Loss: {avg_epoch_loss:.6f}")
        
        avg_loss = total_loss / max(num_batches, 1)

        return avg_loss

    def save_model(self, path: str, save_actor: bool = True, save_critic: bool = True) -> None:
        """保存模型权重到文件

        Args:
            path: 保存路径
            save_actor: 是否保存 Actor 网络
            save_critic: 是否保存 Critic 网络
        """
        save_dict = {}
        if save_actor:
            save_dict['actor'] = self.actor.state_dict()
        if save_critic:
            save_dict['critic'] = self.critic.state_dict()
            save_dict['target_critic'] = self.target_critic.state_dict()
        save_dict['is_train'] = self.is_train
        save_dict['critic_losses'] = self.critic_losses
        save_dict['actor_losses'] = self.actor_losses
        buffer_data = []
        for state, action, reward, next_state in self.replay_buffer.buffer:
            buffer_data.append((state.tolist(), action.tolist(), reward, next_state.tolist()))
        save_dict['replay_buffer'] = buffer_data

        torch.save(save_dict, path)
        logger.info(f"Actor-Critic 模型已保存到 {path} | "
                    f"actor={save_actor}, critic={save_critic}, "
                    f"critic_losses={len(self.critic_losses)}, actor_losses={len(self.actor_losses)}, "
                    f"replay_buffer={len(self.replay_buffer)}")

    def plot_losses(self, save_path: str = None) -> None:
        """绘制训练损失曲线

        Args:
            save_path: 保存路径，如果为 None 则显示图表
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            axes[0].plot(self.critic_losses, alpha=0.6, label='Critic Loss')
            if len(self.critic_losses) > 10:
                smoothed = []
                window = 10
                for i in range(len(self.critic_losses)):
                    start = max(0, i - window + 1)
                    smoothed.append(sum(self.critic_losses[start:i+1]) / (i - start + 1))
                axes[0].plot(smoothed, label='Critic Loss (Smoothed)', linewidth=2)
            axes[0].set_xlabel('Batch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Critic Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            axes[1].plot(self.actor_losses, alpha=0.6, label='Actor Loss', color='orange')
            if len(self.actor_losses) > 10:
                smoothed = []
                window = 10
                for i in range(len(self.actor_losses)):
                    start = max(0, i - window + 1)
                    smoothed.append(sum(self.actor_losses[start:i+1]) / (i - start + 1))
                axes[1].plot(smoothed, label='Actor Loss (Smoothed)', linewidth=2, color='red')
            axes[1].set_xlabel('Batch')
            axes[1].set_ylabel('Loss')
            axes[1].set_title('Actor Loss')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                logger.info(f"损失曲线已保存到 {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib 未安装，无法绘制损失曲线")

    def load_model(self, path: str, load_actor: bool = True, load_critic: bool = True) -> None:
        """从文件加载模型权重

        Args:
            path: 加载路径
            load_actor: 是否加载 Actor 网络（应用模式下通常只需加载 Actor）
            load_critic: 是否加载 Critic 网络
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)

        if load_actor and 'actor' in checkpoint:
            self.actor.load_state_dict(checkpoint['actor'])
            logger.info(f"Actor 网络权重已从 {path} 加载")
        if load_critic and 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
            logger.info(f"Critic 网络权重已从 {path} 加载")
        if load_critic and 'target_critic' in checkpoint:
            self.target_critic.load_state_dict(checkpoint['target_critic'])
            logger.info(f"Target Critic 网络权重已从 {path} 加载")

        if 'is_train' in checkpoint:
            self.is_train = checkpoint['is_train']
            if self.is_train:
                self.set_train_mode()
            else:
                self.set_eval_mode()

        if 'critic_losses' in checkpoint:
            self.critic_losses = list(checkpoint['critic_losses'])
        else:
            self.critic_losses = []
        if 'actor_losses' in checkpoint:
            self.actor_losses = list(checkpoint['actor_losses'])
        else:
            self.actor_losses = []

        if 'replay_buffer' in checkpoint:
            buffer_data = checkpoint['replay_buffer']
            self.replay_buffer.buffer.clear()
            for state_list, action_list, reward, next_state_list in buffer_data:
                self.replay_buffer.buffer.append((
                    np.array(state_list, dtype=np.float32),
                    np.array(action_list, dtype=np.float32),
                    reward,
                    np.array(next_state_list, dtype=np.float32)
                ))
            logger.info(f"恢复经验回放缓冲区: {len(self.replay_buffer)} 条经验")
        else:
            logger.info(f"未找到已保存的经验回放缓冲区，从空开始")


