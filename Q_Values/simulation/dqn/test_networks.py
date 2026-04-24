"""
测试脚本：验证Actor-Critic网络和Agent的正确性
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import torch
import numpy as np
from Q_Values.simulation.dqn.actor_network import ActorNetwork
from Q_Values.simulation.dqn.critic_network import CriticNetwork
from Q_Values.simulation.dqn.replay_buffer import ReplayBuffer
from Q_Values.simulation.dqn.agent import Agent


def test_actor_network():
    """测试ActorNetwork"""
    print("=" * 50)
    print("测试 ActorNetwork")
    print("=" * 50)
    
    # 创建网络
    actor = ActorNetwork(state_dim=25, action_dim=5, hidden_dims=[128, 64])
    print(f"[OK] ActorNetwork创建成功")
    print(f"  - 输入维度: {actor.state_dim + actor.action_dim}")
    print(f"  - 隐藏层: {actor.hidden_dims}")
    print(f"  - 激活函数: {actor.activation}")
    
    # 测试前向传播
    batch_size = 32
    state = torch.randn(batch_size, 25)
    action = torch.randn(batch_size, 5)
    q_values = actor(state, action)
    print(f"[OK] 前向传播测试通过")
    print(f"  - 输入: state({state.shape}), action({action.shape})")
    print(f"  - 输出: q_values({q_values.shape})")
    
    # 测试冻结/解冻
    actor.freeze()
    frozen_params = all(not p.requires_grad for p in actor.parameters())
    print(f"[OK] 冻结测试: {'通过' if frozen_params else '失败'}")
    
    actor.unfreeze()
    unfrozen_params = all(p.requires_grad for p in actor.parameters())
    print(f"[OK] 解冻测试: {'通过' if unfrozen_params else '失败'}")
    print()


def test_critic_network():
    """测试CriticNetwork"""
    print("=" * 50)
    print("测试 CriticNetwork")
    print("=" * 50)
    
    # 创建网络
    critic = CriticNetwork(state_dim=25, hidden_dims=[128, 64])
    print(f"[OK] CriticNetwork创建成功")
    print(f"  - 输入维度: {critic.state_dim}")
    print(f"  - 隐藏层: {critic.hidden_dims}")
    print(f"  - 激活函数: {critic.activation}")
    
    # 测试前向传播
    batch_size = 32
    state = torch.randn(batch_size, 25)
    value = critic(state)
    print(f"[OK] 前向传播测试通过")
    print(f"  - 输入: state({state.shape})")
    print(f"  - 输出: value({value.shape})")
    
    # 测试冻结/解冻
    critic.freeze()
    frozen_params = all(not p.requires_grad for p in critic.parameters())
    print(f"[OK] 冻结测试: {'通过' if frozen_params else '失败'}")
    
    critic.unfreeze()
    unfrozen_params = all(p.requires_grad for p in critic.parameters())
    print(f"[OK] 解冻测试: {'通过' if unfrozen_params else '失败'}")
    print()


def test_replay_buffer():
    """测试ReplayBuffer"""
    print("=" * 50)
    print("测试 ReplayBuffer")
    print("=" * 50)
    
    # 创建缓冲区
    buffer = ReplayBuffer(capacity=1000, state_dim=25, action_dim=5)
    print(f"[OK] ReplayBuffer创建成功")
    print(f"  - 容量: {buffer.capacity}")
    print(f"  - 状态维度: {buffer.state_dim}")
    print(f"  - 动作维度: {buffer.action_dim}")
    
    # 测试单条数据添加
    state = np.random.randn(25).astype(np.float32)
    action = np.random.randn(5).astype(np.float32)
    reward = 1.0
    next_state = np.random.randn(25).astype(np.float32)
    done = False
    
    buffer.push(state, action, reward, next_state, done)
    print(f"[OK] 单条数据添加测试通过")
    print(f"  - 缓冲区大小: {len(buffer)}")
    
    # 测试批量数据添加
    batch_size = 100
    states = np.random.randn(batch_size, 25).astype(np.float32)
    actions = np.random.randn(batch_size, 5).astype(np.float32)
    rewards = np.random.randn(batch_size).astype(np.float32)
    next_states = np.random.randn(batch_size, 25).astype(np.float32)
    dones = np.random.randint(0, 2, batch_size).astype(np.float32)
    
    buffer.push_batch(states, actions, rewards, next_states, dones)
    print(f"[OK] 批量数据添加测试通过")
    print(f"  - 缓冲区大小: {len(buffer)}")
    
    # 测试随机采样
    sampled_data = buffer.sample(32)
    states_s, actions_s, rewards_s, next_states_s, dones_s = sampled_data
    print(f"[OK] 随机采样测试通过")
    print(f"  - states: {states_s.shape}")
    print(f"  - actions: {actions_s.shape}")
    print(f"  - rewards: {rewards_s.shape}")
    print(f"  - next_states: {next_states_s.shape}")
    print(f"  - dones: {dones_s.shape}")
    
    # 测试获取全量数据
    all_data = buffer.sample_all()
    states_a, actions_a, rewards_a, next_states_a, dones_a = all_data
    print(f"[OK] 获取全量数据测试通过")
    print(f"  - 数据量: {states_a.shape[0]}")
    print()


def test_agent():
    """测试Agent"""
    print("=" * 50)
    print("测试 Agent")
    print("=" * 50)
    
    # 创建Agent
    agent = Agent(
        state_dim=25,
        action_dim=5,
        hidden_dims=[128, 64],
        gamma=0.99,
        buffer_capacity=1000,
        device='cpu',
        is_train=True
    )
    print(f"[OK] Agent创建成功")
    print(f"  - 设备: {agent.device}")
    print(f"  - 折扣因子: {agent.gamma}")
    print(f"  - 训练模式: {agent.is_train}")
    
    # 测试动作选择（训练模式）
    state = np.random.randn(25).astype(np.float32)
    candidate_actions = np.random.randn(10, 5).astype(np.float32)  # 10个候选动作
    
    action_train = agent.select_action(state, candidate_actions, mode='train')
    print(f"[OK] 训练模式动作选择测试通过")
    print(f"  - 状态维度: {state.shape}")
    print(f"  - 候选动作数量: {candidate_actions.shape[0]}")
    print(f"  - 选中动作维度: {action_train.shape}")
    
    # 测试动作选择（应用模式）
    action_eval = agent.select_action(state, candidate_actions, mode='eval')
    print(f"[OK] 应用模式动作选择测试通过")
    print(f"  - 选中动作维度: {action_eval.shape}")
    
    # 添加一些经验数据到缓冲区
    for _ in range(100):
        s = np.random.randn(25).astype(np.float32)
        a = np.random.randn(5).astype(np.float32)
        r = np.random.randn()
        s_next = np.random.randn(25).astype(np.float32)
        d = np.random.randint(0, 2)
        agent.replay_buffer.push(s, a, r, s_next, d)
    
    print(f"[OK] 经验数据添加完成")
    print(f"  - 缓冲区大小: {len(agent.replay_buffer)}")
    
    # 测试离线训练
    critic_loss, actor_loss = agent.train_step(batch_size=32)
    print(f"[OK] 离线训练测试通过")
    print(f"  - Critic损失: {critic_loss:.6f}")
    print(f"  - Actor损失: {actor_loss:.6f}")
    
    # 测试模型保存和加载
    model_path = "Q_Values/simulation/dqn/checkpoints/test_model.pt"
    agent.save_model(model_path)
    print(f"[OK] 模型保存测试通过")
    print(f"  - 保存路径: {model_path}")
    
    # 创建新Agent并加载模型
    agent_new = Agent(state_dim=25, action_dim=5, device='cpu')
    agent_new.load_model(model_path)
    print(f"[OK] 模型加载测试通过")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("开始测试 Actor-Critic 强化学习系统")
    print("=" * 50 + "\n")
    
    try:
        test_actor_network()
        test_critic_network()
        test_replay_buffer()
        test_agent()
        
        print("=" * 50)
        print("[OK] 所有测试通过！")
        print("=" * 50)
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"[ERROR] 测试失败: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
