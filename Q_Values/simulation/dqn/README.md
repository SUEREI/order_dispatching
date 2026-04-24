# Actor-Critic强化学习系统 - 即时配送订单派发

## 概述

本模块实现了基于PyTorch的Actor-Critic强化学习网络，用于即时配送订单派发仿真系统。系统采用离线交替训练机制，支持训练模式和应用模式的切换。

## 模块结构

```
Q_Values/simulation/dqn/
├── actor_network.py      # Actor网络（策略网络/DMN）
├── critic_network.py     # Critic网络（价值网络）
├── replay_buffer.py      # 经验回放缓冲区
├── agent.py              # Agent集成类
├── test_networks.py      # 测试脚本
├── usage_example.py      # 使用示例
└── checkpoints/          # 模型保存目录
```

## 核心组件

### 1. ActorNetwork（策略网络）

**功能**：评估状态-动作对的Q值 Q(s,a)

**输入**：状态向量（25维）+ 动作向量（5维）= 30维

**输出**：Q值标量（1维）

**网络结构**：
- 输入层：Linear(30, 128)
- 隐藏层1：Linear(128, 64) + ReLU
- 输出层：Linear(64, 1)

**关键方法**：
- `forward(state, action)`：前向传播计算Q值
- `freeze()`：冻结网络参数
- `unfreeze()`：解冻网络参数

### 2. CriticNetwork（价值网络）

**功能**：评估状态价值 V(s)

**输入**：状态向量（25维）

**输出**：状态价值标量（1维）

**网络结构**：
- 输入层：Linear(25, 128)
- 隐藏层1：Linear(128, 64) + ReLU
- 输出层：Linear(64, 1)

**关键方法**：
- `forward(state)`：前向传播计算状态价值
- `freeze()`：冻结网络参数
- `unfreeze()`：解冻网络参数

### 3. ReplayBuffer（经验回放缓冲区）

**功能**：存储仿真经验数据，支持随机采样和FIFO策略

**存储结构**：`(state, action, reward, next_state, done)`

**关键方法**：
- `push(state, action, reward, next_state, done)`：添加单条经验
- `push_batch(...)`：批量添加经验
- `sample(batch_size)`：随机采样
- `sample_all()`：获取全量数据

### 4. Agent（集成类）

**功能**：封装Actor-Critic网络和ReplayBuffer，提供统一接口

**关键方法**：
- `select_action(state, candidate_actions, mode)`：动作选择
  - 训练模式：Softmax概率采样
  - 应用模式：Argmax确定性决策
- `update_critic_offline(batch_size)`：离线训练Critic
- `update_actor_offline(batch_size)`：离线训练Actor
- `train_step(batch_size)`：交替训练（先Critic后Actor）
- `save_model(filepath)`：保存模型
- `load_model(filepath)`：加载模型

## 状态向量定义（25维）

| 维度范围 | 含义 | 说明 |
|---------|------|------|
| 0-1 | 骑手当前经度、纬度 | 归一化处理 |
| 2 | 当前时间步的时间戳 | 归一化到0~1 |
| 3-18 | 路径点1~8的经纬度 | 共16维，不足用0填充 |
| 19-24 | 路径点1~8的剩余时间 | 归一化到0~1，负数设为0 |

## 动作向量定义（5维）

| 维度范围 | 含义 | 说明 |
|---------|------|------|
| 0-1 | 新订单取餐点经度、纬度 | 归一化处理 |
| 2-3 | 新订单送餐点经度、纬度 | 归一化处理 |
| 4 | 订单可用配送时长 | 归一化处理 |

## 训练流程

### 离线交替训练机制

```
Step 1: 数据收集
├── 完成一整天仿真
├── 收集所有转移元组 (s_t, a_t, r_t, s_{t+1})
└── 存入ReplayBuffer

Step 2: 冻结Actor，训练Critic
├── 冻结Actor网络参数
├── 计算TD目标: y = r_t + γ·V(s_{t+1})·(1-done)
├── 计算MSE损失: Loss = mean((V(s_t) - y)²)
├── 反向传播更新Critic
└── 解冻Actor网络

Step 3: 冻结Critic，训练Actor
├── 冻结Critic网络参数
├── 计算Q值: Q(s_t, a_t)
├── 计算损失: Loss = -mean(Q(s_t, a_t))
├── 反向传播更新Actor
└── 解冻Critic网络

循环: 进入下一天仿真，重复Step 1-3
```

## 使用示例

### 基本使用

```python
from Q_Values.simulation.dqn.agent import Agent
import numpy as np

# 1. 初始化Agent
agent = Agent(
    state_dim=25,
    action_dim=5,
    hidden_dims=[128, 64],
    gamma=0.99,
    is_train=True
)

# 2. 训练模式：选择动作
state = np.random.randn(25).astype(np.float32)
candidate_actions = np.random.randn(10, 5).astype(np.float32)
action = agent.select_action(state, candidate_actions, mode='train')

# 3. 存储经验
next_state = np.random.randn(25).astype(np.float32)
reward = 1.0
done = False
agent.replay_buffer.push(state, action, reward, next_state, done)

# 4. 离线训练
critic_loss, actor_loss = agent.train_step(batch_size=128)

# 5. 保存模型
agent.save_model("model.pt")
```

### 应用模式

```python
# 加载模型并切换到应用模式
agent_eval = Agent(state_dim=25, action_dim=5, is_train=False)
agent_eval.load_model("model.pt")

# 应用模式：确定性决策
optimal_action = agent_eval.select_action(state, candidate_actions, mode='eval')
```

## 奖励函数

### 边际收益计算

```
单个订单基础收益 = 3 + 配送距离(KM)

超时惩罚：
- 超时 ≤ 10分钟：收益 × 0.8
- 超时 10-20分钟：收益 × 0.5
- 超时 > 20分钟：收益 = 0

边际奖励：
r = R_new - R_old
其中：
- R_old：接单前所有已有订单总收益
- R_new：接单后所有订单总收益
```

## 测试验证

运行测试脚本验证所有功能：

```bash
python Q_Values/simulation/dqn/test_networks.py
```

测试内容包括：
- ActorNetwork前向传播和冻结/解冻
- CriticNetwork前向传播和冻结/解冻
- ReplayBuffer数据存储和采样
- Agent动作选择和离线训练
- 模型保存和加载

## 技术特性

1. **类型提示**：所有方法均包含完整的Type Hints
2. **文档字符串**：使用Google风格docstring
3. **设备自适应**：自动选择GPU或CPU
4. **批量处理**：支持批量数据训练
5. **模式切换**：训练模式和应用模式灵活切换
6. **模型持久化**：支持模型保存和加载

## 依赖项

- Python >= 3.7
- PyTorch >= 1.8
- NumPy >= 1.19

## 注意事项

1. 输入网络的连续特征（经纬度、时间等）需在外部完成归一化
2. 动作空间为连续空间，通过候选动作集进行离散化选择
3. 训练时使用Softmax采样增加探索性，应用时使用Argmax进行确定性决策
4. 建议在GPU环境下训练以提升效率

## 版本信息

- 版本：v1.0
- 创建日期：2025-04-24
- 最后更新：2025-04-24
