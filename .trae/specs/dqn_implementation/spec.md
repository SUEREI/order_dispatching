# 即时配送订单派发仿真系统 - 强化学习网络实现

## 概述
- **Summary**: 实现一个基于 Actor-Critic 架构的强化学习网络，用于即时配送订单派发仿真系统，支持离线数据交替训练模式。
- **Purpose**: 解决即时配送场景中的订单派发优化问题，通过强化学习算法提高配送效率和收益。
- **Target Users**: 算法工程师、仿真系统开发者。

## 目标
- 实现 Actor 网络和 Critic 网络的核心结构
- 实现基于离线数据的交替训练逻辑
- 支持训练模式与应用模式的切换
- 提供模型保存与加载功能
- 确保网络输入输出符合规格要求

## 非目标（范围外）
- 不实现在线实时更新逻辑
- 不处理状态向量的归一化（假设外部已处理）
- 不实现仿真环境的核心逻辑（如骑手上下线、订单生成等）

## 背景与上下文
- 系统已有的模块：courier.py（骑手模拟器）、order_g.py（订单生成器）、route.py（路径规划器）、state.py（状态更新器）
- 状态向量固定为 25 维，包含骑手位置、时间、路径点信息和超时剩余时间
- 动作向量固定为 5 维，代表新订单的取餐点、送餐点和可用配送时长
- 奖励函数基于边际收益计算，由环境侧提供

## 功能需求
- **FR-1**: 实现 ActorNetwork 类，包含前向传播方法
- **FR-2**: 实现 CriticNetwork 类，包含前向传播方法
- **FR-3**: 实现 ReplayBuffer 类，用于存储训练数据
- **FR-4**: 实现 Agent 类，封装网络和训练逻辑
- **FR-5**: 实现 select_action 方法，支持训练和应用模式
- **FR-6**: 实现 update_critic_offline 方法，执行离线 Critic 训练
- **FR-7**: 实现 update_actor_offline 方法，执行离线 Actor 训练
- **FR-8**: 实现 save_model 和 load_model 方法，支持模型持久化

## 非功能需求
- **NFR-1**: 代码需包含详尽的类型提示和 Docstring
- **NFR-2**: 网络结构应使用 PyTorch 实现
- **NFR-3**: 训练逻辑应支持基于全天仿真数据的交替训练
- **NFR-4**: 模型应能在训练模式和应用模式间切换

## 约束
- **技术**: 使用 PyTorch 框架，Python 3.6+
- **依赖**: 仅依赖 PyTorch 核心库
- **性能**: 网络结构应简洁高效，适合批量处理

## 假设
- 输入网络的连续特征已做好归一化处理
- 仿真环境会提供合法的候选订单动作
- 奖励函数由环境侧计算并直接传入

## 验收标准

### AC-1: Actor 网络实现
- **Given**: 输入状态向量和动作向量
- **When**: 调用 Actor 网络前向传播
- **Then**: 输出正确形状的 Q 值张量
- **Verification**: `programmatic`

### AC-2: Critic 网络实现
- **Given**: 输入状态向量
- **When**: 调用 Critic 网络前向传播
- **Then**: 输出正确形状的状态价值张量
- **Verification**: `programmatic`

### AC-3: ReplayBuffer 实现
- **Given**: 转移元组 (s_t, a_t, r_t, s_{t+1})
- **When**: 存储到 ReplayBuffer 并采样
- **Then**: 能正确存储和采样数据
- **Verification**: `programmatic`

### AC-4: Agent 类实现
- **Given**: 初始化 Agent 实例
- **When**: 调用各种方法
- **Then**: 所有方法能正常执行
- **Verification**: `programmatic`

### AC-5: 训练模式与应用模式切换
- **Given**: 设置不同的 is_train 标志
- **When**: 调用 select_action 方法
- **Then**: 训练模式使用 Softmax 采样，应用模式使用 Argmax
- **Verification**: `programmatic`

### AC-6: 离线训练逻辑
- **Given**: 收集的仿真数据
- **When**: 执行 update_critic_offline 和 update_actor_offline
- **Then**: 网络参数能正确更新
- **Verification**: `programmatic`

### AC-7: 模型持久化
- **Given**: 训练好的模型
- **When**: 调用 save_model 和 load_model
- **Then**: 模型能正确保存和加载
- **Verification**: `programmatic`

## 未解决问题
- [ ] 网络隐层维度的具体选择（建议 128, 64）
- [ ] 训练时的批量大小和 epoch 数
- [ ] 模型保存的具体路径格式