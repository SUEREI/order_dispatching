# 即时配送订单派发仿真系统 - 实现计划

## [ ] 任务 1: 实现 ActorNetwork 类
- **优先级**: P0
- **依赖**: 无
- **描述**:
  - 实现 Actor 网络的前向传播方法
  - 网络输入为状态向量和动作向量的拼接
  - 网络结构为几层全连接层 + ReLU
  - 输出为每个动作的 Q 值
- **验收标准**: AC-1
- **测试要求**:
  - `programmatic` TR-1.1: 验证网络输入输出形状正确
  - `programmatic` TR-1.2: 验证网络能正确处理批量数据
- **注意**: 网络结构建议使用 128, 64 的隐层维度

## [ ] 任务 2: 实现 CriticNetwork 类
- **优先级**: P0
- **依赖**: 无
- **描述**:
  - 实现 Critic 网络的前向传播方法
  - 网络输入为状态向量
  - 网络结构为几层全连接层 + ReLU
  - 输出为状态价值 V(s)
- **验收标准**: AC-2
- **测试要求**:
  - `programmatic` TR-2.1: 验证网络输入输出形状正确
  - `programmatic` TR-2.2: 验证网络能正确处理批量数据
- **注意**: 网络结构建议使用 128, 64 的隐层维度

## [ ] 任务 3: 实现 ReplayBuffer 类
- **优先级**: P0
- **依赖**: 无
- **描述**:
  - 实现 ReplayBuffer 类，用于存储训练数据
  - 能存储状态、动作向量、奖励、下一状态
  - 支持随机采样批量数据
- **验收标准**: AC-3
- **测试要求**:
  - `programmatic` TR-3.1: 验证能正确存储数据
  - `programmatic` TR-3.2: 验证能正确采样批量数据
- **注意**: 不需要存储 done 标志

## [ ] 任务 4: 实现 Agent 类初始化和 select_action 方法
- **优先级**: P0
- **依赖**: 任务 1, 任务 2
- **描述**:
  - 初始化 Actor 和 Critic 网络
  - 初始化优化器
  - 设置 gamma 和训练/评估模式标志位
  - 实现 select_action 方法，支持训练和应用模式
- **验收标准**: AC-4, AC-5
- **测试要求**:
  - `programmatic` TR-4.1: 验证初始化参数正确
  - `programmatic` TR-4.2: 验证训练模式下使用 Softmax 采样
  - `programmatic` TR-4.3: 验证应用模式下使用 Argmax
- **注意**: 处理输入张量维度

## [ ] 任务 5: 实现 update_critic_offline 方法
- **优先级**: P0
- **依赖**: 任务 2, 任务 3, 任务 4
- **描述**:
  - 实现离线训练 Critic 的逻辑
  - 冻结 Actor 网络参数
  - 计算 TD Target
  - 计算 Critic Loss 并更新
- **验收标准**: AC-6
- **测试要求**:
  - `programmatic` TR-5.1: 验证 Critic 网络参数能正确更新
  - `programmatic` TR-5.2: 验证 Actor 网络参数在训练过程中保持不变
- **注意**: 在整个 Buffer 上循环采样多个 Epoch

## [ ] 任务 6: 实现 update_actor_offline 方法
- **优先级**: P0
- **依赖**: 任务 1, 任务 3, 任务 4, 任务 5
- **描述**:
  - 实现离线训练 Actor 的逻辑
  - 冻结 Critic 网络参数
  - 计算 Advantage
  - 计算 Actor Loss 并更新
- **验收标准**: AC-6
- **测试要求**:
  - `programmatic` TR-6.1: 验证 Actor 网络参数能正确更新
  - `programmatic` TR-6.2: 验证 Critic 网络参数在训练过程中保持不变
- **注意**: 在整个 Buffer 上循环采样多个 Epoch

## [ ] 任务 7: 实现 save_model 和 load_model 方法
- **优先级**: P1
- **依赖**: 任务 4
- **描述**:
  - 实现 save_model 方法，保存网络权重
  - 实现 load_model 方法，加载网络权重
- **验收标准**: AC-7
- **测试要求**:
  - `programmatic` TR-7.1: 验证模型能正确保存
  - `programmatic` TR-7.2: 验证模型能正确加载
- **注意**: 支持只加载 Actor 网络的情况

## [ ] 任务 8: 整合和测试
- **优先级**: P1
- **依赖**: 任务 1-7
- **描述**:
  - 整合所有组件
  - 进行完整的功能测试
  - 确保代码符合类型提示和文档要求
- **验收标准**: AC-1-7
- **测试要求**:
  - `programmatic` TR-8.1: 验证所有组件能正常工作
  - `human-judgment` TR-8.2: 检查代码风格和文档完整性
- **注意**: 确保所有功能符合规格要求