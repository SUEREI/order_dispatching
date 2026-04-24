# Actor-Critic强化学习系统集成指南

## 快速开始

### 1. 文件结构

```
Q_Values/simulation/dqn/
├── actor_network.py          # Actor网络
├── critic_network.py         # Critic网络
├── replay_buffer.py          # 经验回放缓冲区
├── agent.py                  # Agent集成类
├── state_builder.py          # 状态向量构建器（新增）
├── action_builder.py         # 动作向量构建器（新增）
├── reward_calculator.py      # 奖励计算器（新增）
├── integrated_env.py         # 集成仿真环境（新增）
├── train.py                  # 完整训练脚本（新增）
├── test_networks.py          # 测试脚本
├── usage_example.py          # 使用示例
└── README.md                 # 说明文档
```

### 2. 一键训练

直接运行训练脚本即可开始训练：

```bash
cd "d:\Sys\Desktop\毕业论文\2 Python"
python Q_Values/simulation/dqn/train.py
```

训练脚本会自动：
1. 加载订单数据（如果数据文件不存在，会使用模拟数据）
2. 初始化所有组件
3. 运行仿真收集经验
4. 执行离线交替训练
5. 保存最佳模型和最终模型

### 3. 自定义训练

如果需要自定义训练流程，可以参考以下代码：

```python
from Q_Values.simulation.dqn.agent import Agent
from Q_Values.simulation.dqn.state_builder import StateBuilder
from Q_Values.simulation.dqn.action_builder import ActionBuilder
from Q_Values.simulation.dqn.reward_calculator import RewardCalculator
from Q_Values.simulation.dqn.integrated_env import IntegratedSimulationEnv
import pandas as pd

# 1. 加载订单数据
order_df = pd.read_csv("your_order_data.csv")

# 2. 初始化组件
state_builder = StateBuilder(
    lng_min=116.0, lng_max=117.0,
    lat_min=39.0, lat_max=40.0,
    time_start=order_df['dispatch_time'].min(),
    time_end=order_df['dispatch_time'].max()
)

action_builder = ActionBuilder(
    lng_min=116.0, lng_max=117.0,
    lat_min=39.0, lat_max=40.0
)

reward_calculator = RewardCalculator(base_reward=3.0)

agent = Agent(
    state_dim=25,
    action_dim=5,
    hidden_dims=[128, 64],
    gamma=0.99,
    is_train=True
)

# 3. 创建仿真环境
env = IntegratedSimulationEnv(
    order_df=order_df,
    agent=agent,
    state_builder=state_builder,
    action_builder=action_builder,
    reward_calculator=reward_calculator
)

# 4. 运行仿真收集经验
experiences = env.run_simulation()

# 5. 存储经验
for state, action, reward, next_state in experiences:
    agent.replay_buffer.push(state, action, reward, next_state, False)

# 6. 训练
for epoch in range(10):
    critic_loss, actor_loss = agent.train_step(batch_size=128)
    print(f"Epoch {epoch}: Critic Loss={critic_loss}, Actor Loss={actor_loss}")

# 7. 保存模型
agent.save_model("my_model.pt")
```

### 4. 应用模式（推理）

训练完成后，切换到应用模式进行推理：

```python
# 加载训练好的模型
agent = Agent(state_dim=25, action_dim=5, is_train=False)
agent.load_model("my_model.pt")

# 在仿真中使用
state = state_builder.build_state(courier, route, current_time, orders)
candidate_actions = action_builder.build_candidate_actions(candidate_orders, current_time)

# 选择最优动作（Argmax，确定性决策）
optimal_action = agent.select_action(state, candidate_actions, mode='eval')
```

## 核心组件说明

### StateBuilder（状态向量构建器）

将骑手和订单信息转换为25维状态向量：

```python
state = state_builder.build_state(
    courier=courier,          # 骑手信息
    route=route,              # 路径规划结果
    current_time=current_time,# 当前时间
    orders_dict=orders        # 订单字典
)
# 输出: np.ndarray, shape=(25,)
```

### ActionBuilder（动作向量构建器）

将订单信息转换为5维动作向量：

```python
# 单个订单
action = action_builder.build_action(order, current_time)

# 候选订单集
candidate_actions = action_builder.build_candidate_actions(candidate_orders, current_time)
# 输出: np.ndarray, shape=(N, 5), N为候选订单数量

# 拒单动作
reject_action = action_builder.build_reject_action()
# 输出: np.ndarray, shape=(5,), 全0向量
```

### RewardCalculator（奖励计算器）

计算边际收益作为即时奖励：

```python
# 单个订单收益
reward = reward_calculator.calculate_order_reward(
    delivery_distance=3.5,    # 配送距离（KM）
    overtime_minutes=15       # 超时时间（分钟）
)

# 边际奖励
marginal_reward = reward_calculator.calculate_marginal_reward(
    orders_before=orders_before,  # 接单前订单列表
    orders_after=orders_after,    # 接单后订单列表
    current_time=current_time
)
```

### IntegratedSimulationEnv（集成仿真环境）

完整的仿真环境，集成所有组件：

```python
env = IntegratedSimulationEnv(
    order_df=order_df,
    agent=agent,
    state_builder=state_builder,
    action_builder=action_builder,
    reward_calculator=reward_calculator,
    max_orders_per_courier=4,      # 骑手最大持单量
    idle_timeout_minutes=30        # 骑手空闲超时时间
)

# 运行仿真
experiences = env.run_simulation(
    start_time=start_time,
    end_time=end_time,
    time_step=30  # 时间步长（秒）
)
```

## 数据格式要求

订单数据需要包含以下字段：

| 字段名 | 类型 | 说明 |
|--------|------|------|
| order_id | str | 订单ID |
| courier_id | str | 骑手ID |
| sender_lng | float | 取餐点经度（×10^6） |
| sender_lat | float | 取餐点纬度（×10^6） |
| recipient_lng | float | 送餐点经度（×10^6） |
| recipient_lat | float | 送餐点纬度（×10^6） |
| grab_lng | float | 骑手接单经度（×10^6） |
| grab_lat | float | 骑手接单纬度（×10^6） |
| dispatch_time | datetime | 派单时间 |
| arrive_time | datetime | 到达时间 |
| estimate_meal_prepare_time | datetime | 预计备餐完成时间 |
| estimate_arrived_time | datetime | 预计送达时间 |

## 训练流程

```
┌─────────────────────────────────────────┐
│  Step 1: 数据收集                        │
│  - 运行仿真环境                          │
│  - 收集所有经验 (s, a, r, s')            │
│  - 存入ReplayBuffer                      │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Step 2: 冻结Actor，训练Critic           │
│  - 计算TD目标: y = r + γ·V(s')           │
│  - 计算MSE损失                           │
│  - 更新Critic参数                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│  Step 3: 冻结Critic，训练Actor           │
│  - 计算Q值: Q(s, a)                      │
│  - 计算损失: -mean(Q)                    │
│  - 更新Actor参数                         │
└─────────────────────────────────────────┘
              ↓
         循环至收敛
```

## 常见问题

### Q1: 如何调整网络结构？

修改Agent初始化时的`hidden_dims`参数：

```python
agent = Agent(
    state_dim=25,
    action_dim=5,
    hidden_dims=[256, 128, 64],  # 三层隐藏层
    activation='leaky_relu'       # 使用LeakyReLU
)
```

### Q2: 如何调整探索策略？

在训练模式下，通过`temperature`参数控制探索程度：

```python
# 高探索性
action = agent.select_action(state, actions, mode='train', temperature=2.0)

# 低探索性（更接近贪婪策略）
action = agent.select_action(state, actions, mode='train', temperature=0.5)
```

### Q3: 如何处理不同的地理范围？

修改StateBuilder和ActionBuilder的归一化参数：

```python
state_builder = StateBuilder(
    lng_min=your_min_lng,
    lng_max=your_max_lng,
    lat_min=your_min_lat,
    lat_max=your_max_lat
)
```

### Q4: 如何监控训练过程？

训练脚本已经包含日志输出，可以查看：
- 每个epoch的损失值
- 接单/拒单统计
- 模型保存信息

也可以添加TensorBoard等可视化工具进行监控。

## 性能优化建议

1. **使用GPU**：确保PyTorch能正确识别GPU，Agent会自动使用
2. **调整批次大小**：根据内存大小调整`batch_size`
3. **调整缓冲区大小**：根据数据量调整`buffer_capacity`
4. **并行化**：可以考虑多进程并行运行仿真

## 下一步

1. 准备实际的订单数据
2. 运行`train.py`开始训练
3. 监控训练过程和损失值
4. 保存最佳模型
5. 切换到应用模式进行推理

祝训练顺利！🚀
