# 编码任务清单

基于spec.md和design.md，将DQN派单器的实现拆解为以下编码任务，按依赖顺序排列。

---

## 任务1：创建DQNConfig配置类和ReplayBuffer经验回放缓冲区

**文件**: `simulation/core/dqn_network.py`

**内容**:
1. 创建 `DQNConfig` 数据类，包含所有超参数默认值（state_dim=21, hidden_dim=128, lr=1e-4, gamma=0.99, buffer_capacity=10000, batch_size=32, tau=0.01, epsilon_start=0.5, epsilon_decay=0.995, epsilon_min=0.01, max_orders_per_courier=4, candidate_count=20, base_revenue=3.0, distance_revenue_coeff=1.0, max_delivery_time=3600.0, lng_center=0.0, lng_range=1.0, lat_center=0.0, lat_range=1.0）
2. 创建 `ReplayBuffer` 类，使用collections.deque实现固定容量缓冲区
   - `push(state, action, reward, next_state)`: 存储经验
   - `sample(batch_size)`: 随机采样返回(state_batch, action_batch, reward_batch, next_state_batch)
   - `__len__()`: 返回当前缓冲区大小

**依赖**: 无

---

## 任务2：创建DQNNetwork神经网络和DQNAgent智能体

**文件**: `simulation/core/dqn_network.py`（在任务1同一文件中追加）

**内容**:
1. 创建 `DQNNetwork(nn.Module)` 类
   - `__init__(state_dim, hidden_dim, output_dim=1)`: 定义三层全连接网络（input→hidden→hidden→output），使用ReLU激活
   - `forward(x)`: 前向传播，输出Q值
2. 创建 `DQNAgent` 类
   - `__init__(state_dim, config=None)`: 初始化policy_net、target_net（参数相同）、optimizer（Adam）、replay_buffer、epsilon
   - `batch_predict(states)`: 接收np.ndarray形状(N, state_dim)，批量前向推理返回Q值列表[float]
   - `update(batch)`: 接收采样的batch，计算当前Q值和目标Q值（target_net），计算MSE损失，反向传播，软更新target_net，返回损失值
   - `soft_update_target()`: θ_target = τ×θ_policy + (1-τ)×θ_target
   - `save(path)`: 保存policy_net和target_net的state_dict到文件
   - `load(path)`: 从文件加载state_dict到policy_net和target_net

**依赖**: 任务1

---

## 任务3：创建OrderDispatcherByDQN主类框架

**文件**: `simulation/core/order_d_dqn.py`

**内容**:
1. 创建 `OrderDispatcherByDQN` 类
   - `__init__(self, route_planner, config=None)`: 初始化route_planner、config（默认DQNConfig）、dqn_agent（DQNAgent实例）、临时骑手订单计数字典`temp_order_counts`
   - `dispatch_orders_by_dqn(self, orders, couriers, current_time) -> Dict[str, str]`: 主派单方法框架，遍历每个订单，调用候选筛选→状态构建→Q值计算→动作选择→奖励计算→经验存储→训练，返回派单结果映射
   - `save_model(self, path)`: 调用dqn_agent.save
   - `load_model(self, path)`: 调用dqn_agent.load

**依赖**: 任务2

---

## 任务4：实现候选骑手筛选方法_select_candidates

**文件**: `simulation/core/order_d_dqn.py`（在OrderDispatcherByDQN类中）

**内容**:
1. `_select_candidates(self, order, couriers) -> List[Tuple[str, Dict]]`
   - 过滤：保留status=='online'且len(current_orders)+temp_order_counts.get(cid,0) < max_orders_per_courier的骑手
   - 计算距离：对每个过滤后骑手，调用haversine_distance(courier['position'], (order['sender_lng'], order['sender_lat']))
   - 排序：按距离升序
   - 截取：取前candidate_count名
   - 返回：[(courier_id, courier_dict), ...]
   - 异常处理：骑手position为None时跳过

**依赖**: 任务3

---

## 任务5：实现状态向量构建方法_build_state

**文件**: `simulation/core/order_d_dqn.py`（在OrderDispatcherByDQN类中）

**内容**:
1. `_build_state(self, courier, order, current_time) -> np.ndarray`
   - 从courier['route']提取路径点坐标，按sequence顺序排列
   - 最多取8个路径点，展平为[lng1,lat1,...,lng8,lat8]（16维），不足零填充
   - 编码新订单特征：[sender_lng, sender_lat, recipient_lng, recipient_lat]（4维）
   - 计算可用配送时长：(order['estimate_arrived_time'] - current_time).total_seconds()，若NaT则用max_delivery_time
   - 归一化：经纬度(value-center)/range，可用时长value/max_delivery_time
   - 拼接为21维float32向量返回

**依赖**: 任务3

---

## 任务6：实现收益计算和奖励计算方法

**文件**: `simulation/core/order_d_dqn.py`（在OrderDispatcherByDQN类中）

**内容**:
1. `_calculate_courier_revenue(self, courier, current_time) -> float`
   - 遍历courier['current_orders']，对每个订单：
     - 基础收益 = base_revenue (3元)
     - 配送距离 = haversine_distance(取餐点, 送餐点) / 1000 (km)
     - 订单收益 = 基础收益 + 配送距离 × distance_revenue_coeff
   - 基于路径估算送达时间，若超时施加惩罚（≤10min×0.9, ≤20min×0.6, >20min×0.5）
   - 返回总收益
2. `_calculate_reward(self, courier, order, current_time) -> float`
   - 计算当前收益：_calculate_courier_revenue(courier, current_time)
   - 深拷贝courier的current_orders，加入新订单（status='pending'）
   - 构造临时骑手字典，调用route_planner.plan_route获取新路径
   - 计算调整后收益：_calculate_courier_revenue(temp_courier, current_time)
   - 奖励 = 调整后收益 - 当前收益
   - 路径规划失败时奖励设为-1.0

**依赖**: 任务3, 任务5

---

## 任务7：实现ε-贪婪动作选择和完整派单流程

**文件**: `simulation/core/order_d_dqn.py`（在OrderDispatcherByDQN类中）

**内容**:
1. `_choose_action(self, q_values, candidate_ids) -> str`
   - 生成随机数r，若r<epsilon则随机选择，否则选Q值最大的
   - 派单后衰减epsilon：epsilon = max(epsilon_min, epsilon * epsilon_decay)
2. 完善 `dispatch_orders_by_dqn` 方法：
   - 初始化temp_order_counts字典
   - 遍历每个订单：
     a. 调用_select_candidates筛选候选骑手
     b. 若候选集为空，跳过该订单
     c. 对每个候选骑手调用_build_state构建状态向量
     d. 调用dqn_agent.batch_predict批量计算Q值
     e. 调用_choose_action选择骑手
     f. 调用_calculate_reward计算奖励
     g. 构建next_state（骑手加入订单后的状态）
     h. 存储经验到replay_buffer
     i. 若缓冲区足够，采样训练dqn_agent
     j. 更新temp_order_counts和派单结果
   - 返回dispatch_results

**依赖**: 任务4, 任务5, 任务6

---

## 任务8：修改core/__init__.py导出和main.py集成

**文件**: `simulation/core/__init__.py`, `simulation/main.py`

**内容**:
1. 修改 `core/__init__.py`：
   - 新增 `from core.order_d_dqn import OrderDispatcherByDQN`
   - 新增 `from core.dqn_network import DQNConfig`
   - 在 `__all__` 中添加 `'OrderDispatcherByDQN'` 和 `'DQNConfig'`
2. 修改 `main.py`：
   - 在导入区新增 `from core import OrderDispatcherByDQN, DQNConfig`
   - 在 `DeliverySimulation.__init__` 中新增 `self.dispatcher_dqn = OrderDispatcherByDQN(self.route_planner)`
   - 在 `step()` 方法中，将第136行的 `self.dispatcher_real.dispatch_orders_by_real(self.pending_orders, online_couriers)` 替换为 `self.dispatcher_dqn.dispatch_orders_by_dqn(self.pending_orders, online_couriers, current_time)`

**依赖**: 任务7

---

## 任务9：坐标归一化参数自动统计

**文件**: `simulation/main.py`

**内容**:
1. 在 `DeliverySimulation.__init__` 中，数据加载后统计经纬度的中心值和范围：
   - lng_center = df['sender_lng'].mean()（综合sender和recipient）
   - lat_center = df['sender_lat'].mean()
   - lng_range = max(df['sender_lng'].max()-lng_center, lng_center-df['sender_lng'].min()) 等
2. 创建DQNConfig实例，设置归一化参数
3. 将config传入OrderDispatcherByDQN初始化

**依赖**: 任务8
