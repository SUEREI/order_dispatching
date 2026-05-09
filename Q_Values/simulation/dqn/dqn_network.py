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
    state_dim: int = 27
    action_dim: int = 5
    actor_hidden_dims: List[int] = None
    critic_hidden_dims: List[int] = None
    lr_actor: float = 1e-3
    lr_critic: float = 5e-4
    gamma: float = 0.99
    buffer_capacity: int = 70000
    batch_size: int = 128
    critic_epochs: int = 200
    actor_epochs: int = 100
    batches_per_epoch: int = buffer_capacity // batch_size
    target_update_rate: float = 0.01
    grad_clip_norm: float = 1.0
    early_stop_patience: int = 3
    early_stop_threshold: float = 1e-3
    max_orders_per_courier: int = 4
    candidate_count: int = 300
    base_revenue: float = 3.0
    distance_revenue_coeff: float = 1.5
    max_delivery_time: float = 3600.0
    lng_center: float = 174.387
    lng_max: float = 174.995
    lng_range: float = lng_max - lng_center
    lat_center: float = 45.728
    lat_max: float = 46.073
    lat_range: float = lat_max - lat_center
    reward_scale: float = 1.0
    distance_penalty_coeff: float = 0.3

    def __post_init__(self):
        if self.actor_hidden_dims is None:
            self.actor_hidden_dims = [256, 128, 64]
        if self.critic_hidden_dims is None:
            self.critic_hidden_dims = [256, 128, 64]


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
        state_dim: int = 27,
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
        self.epoch_critic_losses = []
        self.epoch_actor_losses = []

        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
        logger.info(f"Actor-Critic 训练设备: {self.device} | {device_name}")
        logger.info(f"Actor 网络结构: state_dim={self.state_dim}, action_dim={self.action_dim}, "
                        f"hidden_dims={self.config.actor_hidden_dims}")
        logger.info(f"Critic 网络结构: state_dim={self.state_dim}, "
                        f"hidden_dims={self.config.critic_hidden_dims}")
        logger.info(f"经验回放缓冲区容量: {self.config.buffer_capacity}, 批次大小: {self.config.batch_size}")

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

    def select_courier(
        self,
        courier_states: np.ndarray,
        order_action: np.ndarray
    ) -> Tuple[np.ndarray, Optional[float], int]:
        """选择骑手

        一个订单动作（order_action）对应多个骑手状态（courier_states），
        为订单选择最合适的骑手。

        训练模式: 使用 Softmax 概率采样
        应用模式: 直接输出 Q 值最高的骑手（Argmax），不计算 log_prob

        Args:
            courier_states: 骑手状态数组 (N, state_dim)，N个骑手的状态向量
            order_action: 订单动作向量 (action_dim,)，5维动作向量（仅一份）

        Returns:
            (selected_action, log_prob, chosen_index):
            - 选中的骑手对应的动作向量 (action_dim,)
            - 训练模式下返回 log_prob，评估模式下返回 None
            - 选中的骑手索引
        """
        if courier_states.ndim == 1:
            courier_states = courier_states[np.newaxis, :]

        num_couriers = courier_states.shape[0]

        state_tensor = torch.FloatTensor(courier_states).to(self.device)
        action_tensor = torch.FloatTensor(order_action).to(self.device)
        actions_tensor = action_tensor.unsqueeze(0).expand(num_couriers, -1)

        with torch.no_grad():
            q_values = self.actor(state_tensor, actions_tensor).squeeze(-1)

        if self.is_train:
            probs = F.softmax(q_values, dim=0)
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx)
            chosen_index = action_idx.item()
            selected_action = order_action
            return selected_action, log_prob.item(), chosen_index
        else:
            chosen_index = torch.argmax(q_values).item()
            selected_action = order_action
            return selected_action, None, chosen_index

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
        prev_epoch_loss = None
        patience_counter = 0

        batches_per_epoch = self.config.batches_per_epoch
        for epoch in range(self.config.critic_epochs):
            epoch_loss = 0.0
            epoch_var_residual = 0.0
            epoch_var_td = 0.0
            num_var_batches = 0
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

                with torch.no_grad():
                    residuals = current_values - td_targets
                    epoch_var_residual += torch.var(residuals).item()
                    epoch_var_td += torch.var(td_targets).item()
                    num_var_batches += 1

            avg_epoch_loss = epoch_loss / max(batches_per_epoch, 1)
            self.epoch_critic_losses.append(avg_epoch_loss)
            avg_var_residual = epoch_var_residual / max(num_var_batches, 1)
            avg_var_td = epoch_var_td / max(num_var_batches, 1)
            explained_var = 1.0 - avg_var_residual / max(avg_var_td, 1e-8)
            logger.info(f"Critic 训练 | Epoch {epoch+1}/{self.config.critic_epochs} | "
                        f"Avg Loss: {avg_epoch_loss:.6f} | "
                        f"V(s) Var: {avg_var_td:.4f} | "
                        f"Explained Var: {explained_var:.4f}")

            with torch.no_grad():
                for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
                    target_param.data.copy_(self.config.target_update_rate * param.data +
                                            (1 - self.config.target_update_rate) * target_param.data)

            if prev_epoch_loss is not None:
                relative_change = abs(avg_epoch_loss - prev_epoch_loss) / max(abs(prev_epoch_loss), 1e-8)
                if relative_change < self.config.early_stop_threshold:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        logger.info(f"Critic 训练 | 损失波动小于阈值 ({self.config.early_stop_threshold})，"
                                    f"连续 {self.config.early_stop_patience} 个 epoch，提前停止")
                        break
                else:
                    patience_counter = 0
            prev_epoch_loss = avg_epoch_loss

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
        prev_epoch_loss = None
        patience_counter = 0

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
                    next_values = self.target_critic(next_states_t).squeeze(-1)
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
            self.epoch_actor_losses.append(avg_epoch_loss)
            logger.info(f"Actor 训练 | Epoch {epoch+1}/{self.config.actor_epochs} | "
                        f"Avg Loss: {avg_epoch_loss:.6f}")

            if prev_epoch_loss is not None:
                relative_change = abs(avg_epoch_loss - prev_epoch_loss) / max(abs(prev_epoch_loss), 1e-8)
                if relative_change < self.config.early_stop_threshold:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stop_patience:
                        logger.info(f"Actor 训练 | 损失波动小于阈值 ({self.config.early_stop_threshold})，"
                                    f"连续 {self.config.early_stop_patience} 个 epoch，提前停止")
                        break
                else:
                    patience_counter = 0
            prev_epoch_loss = avg_epoch_loss
        
        avg_loss = total_loss / max(num_batches, 1)

        return avg_loss

    def save_model(self, path: str, save_actor: bool = True, save_critic: bool = True) -> None:
        """仅保存网络参数到文件"""
        save_dict = {}
        if save_actor:
            save_dict['actor'] = self.actor.state_dict()
        if save_critic:
            save_dict['critic'] = self.critic.state_dict()
            save_dict['target_critic'] = self.target_critic.state_dict()

        torch.save(save_dict, path)
        logger.info(f"网络参数已保存到 {path} | actor={save_actor}, critic={save_critic}")

    def plot_losses(self, save_path: str = None) -> None:
        """绘制训练损失曲线（以 epoch 为单位）

        Args:
            save_path: 保存路径，如果为 None 则显示图表
        """
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

            axes[0].plot(self.epoch_critic_losses, alpha=0.6, label='Critic Loss')
            if len(self.epoch_critic_losses) > 5:
                smoothed = []
                window = 5
                for i in range(len(self.epoch_critic_losses)):
                    start = max(0, i - window + 1)
                    smoothed.append(sum(self.epoch_critic_losses[start:i+1]) / (i - start + 1))
                axes[0].plot(smoothed, label='Critic Loss (Smoothed)', linewidth=2)
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title('Critic Loss')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(self.epoch_actor_losses, alpha=0.6, label='Actor Loss', color='orange')
            if len(self.epoch_actor_losses) > 5:
                smoothed = []
                window = 5
                for i in range(len(self.epoch_actor_losses)):
                    start = max(0, i - window + 1)
                    smoothed.append(sum(self.epoch_actor_losses[start:i+1]) / (i - start + 1))
                axes[1].plot(smoothed, label='Actor Loss (Smoothed)', linewidth=2, color='red')
            axes[1].set_xlabel('Epoch')
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
        """仅从文件加载网络参数"""
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


