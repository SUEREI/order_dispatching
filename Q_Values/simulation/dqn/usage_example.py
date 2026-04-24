"""
Actor-Critic强化学习系统使用示例

本示例展示如何使用Actor-Critic网络进行即时配送订单派发的强化学习训练。
"""

import numpy as np
from Q_Values.simulation.dqn.agent import Agent


def example_training_workflow():
    """
    完整的训练工作流示例
    
    展示如何：
    1. 初始化Agent
    2. 进行仿真数据收集
    3. 执行离线交替训练
    4. 保存和加载模型
    """
    print("=" * 60)
    print("Actor-Critic强化学习训练示例")
    print("=" * 60)
    
    # 1. 初始化Agent
    print("\n[步骤1] 初始化Agent")
    agent = Agent(
        state_dim=25,           # 状态向量维度
        action_dim=5,           # 动作向量维度
        hidden_dims=[128, 64],  # 隐藏层维度
        activation='relu',      # 激活函数
        learning_rate=1e-3,     # 学习率
        gamma=0.99,             # 折扣因子
        buffer_capacity=10000,  # 经验回放缓冲区容量
        device='auto',          # 自动选择设备（GPU优先）
        is_train=True           # 训练模式
    )
    print(f"Agent初始化完成，设备: {agent.device}")
    
    # 2. 仿真数据收集（模拟一天的仿真）
    print("\n[步骤2] 仿真数据收集")
    num_steps = 1000  # 一天的仿真步数
    
    for step in range(num_steps):
        # 模拟当前状态（25维）
        state = np.random.randn(25).astype(np.float32)
        
        # 模拟候选动作集（假设有5个候选订单）
        num_candidates = 5
        candidate_actions = np.random.randn(num_candidates, 5).astype(np.float32)
        
        # 使用训练模式选择动作（Softmax采样）
        selected_action = agent.select_action(
            state=state,
            candidate_actions=candidate_actions,
            mode='train',
            temperature=1.0
        )
        
        # 模拟环境反馈
        next_state = np.random.randn(25).astype(np.float32)
        reward = np.random.randn()  # 实际应用中由环境计算边际收益
        done = False  # 实际应用中根据是否到达终止状态设置
        
        # 存储经验到ReplayBuffer
        agent.replay_buffer.push(state, selected_action, reward, next_state, done)
    
    print(f"数据收集完成，缓冲区大小: {len(agent.replay_buffer)}")
    
    # 3. 离线交替训练
    print("\n[步骤3] 离线交替训练")
    num_epochs = 10  # 训练轮数
    
    for epoch in range(num_epochs):
        # 执行一轮交替训练（先Critic后Actor）
        critic_loss, actor_loss = agent.train_step(batch_size=128)
        
        if (epoch + 1) % 2 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}: "
                  f"Critic Loss = {critic_loss:.6f}, "
                  f"Actor Loss = {actor_loss:.6f}")
    
    # 4. 保存模型
    print("\n[步骤4] 保存模型")
    model_path = "Q_Values/simulation/dqn/checkpoints/trained_model.pt"
    agent.save_model(model_path)
    print(f"模型已保存至: {model_path}")
    
    # 5. 加载模型并切换到应用模式
    print("\n[步骤5] 加载模型并切换到应用模式")
    agent_eval = Agent(
        state_dim=25,
        action_dim=5,
        device='auto',
        is_train=False  # 应用模式
    )
    agent_eval.load_model(model_path)
    print("模型加载完成，已切换到应用模式")
    
    # 6. 应用模式推理
    print("\n[步骤6] 应用模式推理")
    state = np.random.randn(25).astype(np.float32)
    candidate_actions = np.random.randn(5, 5).astype(np.float32)
    
    # 使用应用模式选择动作（Argmax）
    optimal_action = agent_eval.select_action(
        state=state,
        candidate_actions=candidate_actions,
        mode='eval'
    )
    print(f"应用模式选择的动作: {optimal_action}")
    
    print("\n" + "=" * 60)
    print("训练流程示例完成")
    print("=" * 60)


def example_state_action_definition():
    """
    状态向量和动作向量定义示例
    
    展示如何构建符合规范的25维状态向量和5维动作向量。
    """
    print("\n" + "=" * 60)
    print("状态向量和动作向量定义示例")
    print("=" * 60)
    
    # 状态向量（25维）
    print("\n[状态向量定义] (25维)")
    state = np.zeros(25, dtype=np.float32)
    
    # 维度 0-1: 骑手当前经度、纬度
    state[0] = 116.397  # 经度（归一化）
    state[1] = 39.909   # 纬度（归一化）
    print(f"维度 0-1 (骑手位置): 经度={state[0]}, 纬度={state[1]}")
    
    # 维度 2: 当前时间步的时间戳（归一化）
    state[2] = 0.5  # 一天中的比例 0~1
    print(f"维度 2 (时间戳): {state[2]}")
    
    # 维度 3-18: 路径点1~8的经纬度（共16维）
    for i in range(8):
        state[3 + i*2] = 116.397 + i * 0.001      # 经度
        state[4 + i*2] = 39.909 + i * 0.001      # 纬度
    print(f"维度 3-18 (路径点经纬度): 已填充8个路径点")
    
    # 维度 19-24: 路径点1~8对应的"距离超时的剩余时间"（归一化）
    for i in range(8):
        state[19 + i] = max(0, 1.0 - i * 0.1)  # 剩余时间，负数设为0
    print(f"维度 19-24 (剩余时间): {state[19:25]}")
    
    # 动作向量（5维）
    print("\n[动作向量定义] (5维)")
    action = np.zeros(5, dtype=np.float32)
    
    # 维度 0-1: 新订单取餐点经度、纬度
    action[0] = 116.398  # 取餐点经度
    action[1] = 39.910   # 取餐点纬度
    print(f"维度 0-1 (取餐点): 经度={action[0]}, 纬度={action[1]}")
    
    # 维度 2-3: 新订单送餐点经度、纬度
    action[2] = 116.400  # 送餐点经度
    action[3] = 39.912   # 送餐点纬度
    print(f"维度 2-3 (送餐点): 经度={action[2]}, 纬度={action[3]}")
    
    # 维度 4: 该订单的可用配送时长（归一化）
    action[4] = 0.8  # 可用配送时长
    print(f"维度 4 (可用配送时长): {action[4]}")
    
    print("\n" + "=" * 60)


def example_reward_calculation():
    """
    奖励函数计算示例
    
    展示如何计算边际收益作为即时奖励。
    """
    print("\n" + "=" * 60)
    print("奖励函数计算示例")
    print("=" * 60)
    
    # 假设的订单配送距离
    delivery_distance = 3.5  # 公里
    
    # 单个订单基础收益
    base_reward = 3 + delivery_distance
    print(f"\n基础收益: 3 + {delivery_distance} = {base_reward}")
    
    # 超时惩罚示例
    overtime_minutes = 15  # 超时15分钟
    
    if overtime_minutes <= 10:
        # 超时10分钟内：收益打8折
        final_reward = base_reward * 0.8
        print(f"超时{overtime_minutes}分钟（<=10分钟）: 收益打8折 = {final_reward}")
    elif overtime_minutes <= 20:
        # 超时10-20分钟：收益打5折
        final_reward = base_reward * 0.5
        print(f"超时{overtime_minutes}分钟（10-20分钟）: 收益打5折 = {final_reward}")
    else:
        # 超时20分钟以上：收益为0
        final_reward = 0
        print(f"超时{overtime_minutes}分钟（>20分钟）: 收益为0")
    
    # 边际奖励计算
    print("\n边际奖励计算:")
    R_old = 10.0  # 接单前所有已有订单总收益
    R_new = 15.0  # 接单后所有订单总收益
    marginal_reward = R_new - R_old
    print(f"接单前总收益: {R_old}")
    print(f"接单后总收益: {R_new}")
    print(f"边际奖励: R_new - R_old = {marginal_reward}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行所有示例
    example_training_workflow()
    example_state_action_definition()
    example_reward_calculation()
