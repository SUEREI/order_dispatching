# **1. 实现模型**

## **1.1 上下文视图**

DQN派单器作为仿真系统中的订单分派策略组件，替换现有的基于距离/真实匹配的分派器，嵌入仿真主循环的派单阶段。

```
┌─────────────────────────────────────────────────────────┐
│                  DeliverySimulation                      │
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │CourierSim│  │OrderGen  │  │StateRef  │  │RoutePlan│ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │            OrderDispatcherByDQN                     │  │
│  │  ┌────────────┐  ┌──────────┐  ┌───────────────┐  │  │
│  │  │DQNNetwork  │  │ReplayBuf │  │StateBuilder   │  │  │
│  │  └────────────┘  └──────────┘  └───────────────┘  │  │
│  │  ┌────────────┐  ┌──────────┐                      │  │
│  │  │RewardCalc  │  │Candidate │                      │  │
│  │  │            │  │Selector  │                      │  │
│  │  └────────────┘  └──────────┘                      │  │
│  └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## **1.2 服务/组件总体架构**

DQN派单器由5个内部子模块组成：

| 子模块 | 职责 | 对应文件 |
|--------|------|----------|
| DQNNetwork | DQN神经网络（主网络+目标网络）、前向推理、训练更新 | `simulation/core/dqn_network.py` |
| ReplayBuffer | 经验回放缓冲区，存储和采样经验 | `simulation/core/dqn_network.py`（内嵌类） |
| StateBuilder | 状态向量构建，编码骑手路径+新订单特征，归一化 | `simulation/core/order_d_dqn.py`（内嵌方法） |
| RewardCalculator | 边际收益奖励计算，调用RoutePlanner评估 | `simulation/core/order_d_dqn.py`（内嵌方法） |
| CandidateSelector | 候选骑手筛选，距离排序取Top-K | `simulation/core/order_d_dqn.py`（内嵌方法） |

对外暴露的主类为 `OrderDispatcherByDQN`，位于 `simulation/core/order_d_dqn.py`。

## **1.3 实现设计文档**

### 1.3.1 OrderDispatcherByDQN（主类）

**文件**: `simulation/core/order_d_dqn.py`

**类结构**:
```python
class OrderDispatcherByDQN:
    def __init__(self, route_planner, config=None)
    def dispatch_orders_by_dqn(self, orders, couriers, current_time) -> Dict[str, str]
    def _select_candidates(self, order, couriers) -> List[Tuple[str, Dict]]
    def _build_state(self, courier, order, current_time) -> np.ndarray
    def _calculate_reward(self, courier, order, current_time) -> float
    def _calculate_courier_revenue(self, courier, current_time) -> float
    def _choose_action(self, q_values, candidate_ids) -> str
    def save_model(self, path)
    def load_model(self, path)
```

**初始化参数**:
- `route_planner`: RoutePlanner实例，用于奖励计算中的路径重规划
- `config`: DQNConfig配置对象，包含所有超参数

**核心方法 `dispatch_orders_by_dqn`**:
1. 遍历每个待分配订单
2. 调用 `_select_candidates` 筛选候选骑手
3. 对每个候选骑手调用 `_build_state` 构建状态向量
4. 批量输入DQN网络计算Q值
5. 调用 `_choose_action` 执行ε-贪婪选择
6. 调用 `_calculate_reward` 计算奖励
7. 存储经验到ReplayBuffer，尝试训练
8. 更新骑手临时订单计数（用于后续订单的容量过滤）
9. 返回派单结果映射

### 1.3.2 DQNNetwork（神经网络）

**文件**: `simulation/core/dqn_network.py`

**类结构**:
```python
class DQNNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, output_dim)
    def forward(self, x) -> Tensor

class ReplayBuffer:
    def __init__(self, capacity)
    def push(self, state, action, reward, next_state)
    def sample(self, batch_size) -> Tuple
    def __len__(self) -> int

class DQNAgent:
    def __init__(self, state_dim, config=None)
    def select_action(self, state) -> int
    def batch_predict(self, states) -> List[float]
    def update(self, batch) -> float
    def soft_update_target(self)
    def save(self, path)
    def load(self, path)
```

**网络结构**:
- 输入层: state_dim (21)
- 隐藏层1: 128 neurons + ReLU
- 隐藏层2: 128 neurons + ReLU
- 输出层: 1 (Q值)

**DQNAgent包含**:
- `policy_net`: 主网络（DQNNetwork实例）
- `target_net`: 目标网络（DQNNetwork实例，参数从policy_net软更新）
- `optimizer`: Adam优化器
- `replay_buffer`: ReplayBuffer实例
- `epsilon`: 当前探索率

**batch_predict方法**: 接收多个状态向量组成的batch，一次性前向推理返回所有Q值列表，避免逐个推理的开销。

### 1.3.3 DQNConfig（配置类）

**文件**: `simulation/core/dqn_network.py`

```python
class DQNConfig:
    state_dim: int = 21
    hidden_dim: int = 128
    lr: float = 1e-4
    gamma: float = 0.99
    buffer_capacity: int = 10000
    batch_size: int = 32
    tau: float = 0.01          # 目标网络软更新系数
    epsilon_start: float = 0.5
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.01
    max_orders_per_courier: int = 4
    candidate_count: int = 20
    base_revenue: float = 3.0
    distance_revenue_coeff: float = 1.0
    max_delivery_time: float = 3600.0  # 归一化用最大配送时长(秒)
    # 坐标归一化参数（需根据数据集统计设置）
    lng_center: float = 0.0
    lng_range: float = 1.0
    lat_center: float = 0.0
    lat_range: float = 1.0
```

### 1.3.4 状态向量构建设计

**`_build_state(courier, order, current_time) -> np.ndarray`**

构建流程:
1. 从骑手的 `route` 中提取路径点坐标，按sequence顺序排列
2. 最多取8个路径点（4订单×2点），不足则零填充
3. 将路径点坐标展平为 [lng1, lat1, lng2, lat2, ..., lng8, lat8]（16维）
4. 编码新订单特征: [sender_lng, sender_lat, recipient_lng, recipient_lat]（4维）
5. 计算可用配送时长: `(order['estimate_arrived_time'] - current_time).total_seconds()`
6. 归一化处理:
   - 经纬度: `(value - center) / range`
   - 可用配送时长: `value / max_delivery_time`
7. 拼接为21维向量

### 1.3.5 候选骑手筛选设计

**`_select_candidates(order, couriers) -> List[Tuple[str, Dict]]`**

筛选流程:
1. 过滤: 保留 `status == 'online'` 且 `len(current_orders) < max_orders_per_courier` 的骑手
2. 计算距离: 对每个过滤后骑手，计算 `haversine_distance(courier['position'], (order['sender_lng'], order['sender_lat']))`
3. 排序: 按距离升序排序
4. 截取: 取前 `candidate_count`(20) 名
5. 返回: `[(courier_id, courier_dict), ...]`

### 1.3.6 奖励计算设计

**`_calculate_reward(courier, order, current_time) -> float`**

计算流程:
1. 计算当前收益: `_calculate_courier_revenue(courier, current_time)`
2. 临时加入新订单:
   - 深拷贝骑手的 `current_orders`
   - 将新订单（status='pending'）加入
   - 构造临时骑手字典用于路径规划
3. 调用 `route_planner.plan_route(temp_courier)` 获取新路径
4. 基于新路径计算调整后收益: `_calculate_courier_revenue(temp_courier, current_time)`
5. 奖励 = 调整后收益 - 当前收益
6. 恢复骑手原有状态（深拷贝已保证原状态不变）

**`_calculate_courier_revenue(courier, current_time) -> float`**

收益计算逻辑:
1. 基于骑手路径，模拟骑手从当前位置依次到达各路径点
2. 对每个订单:
   - 基础收益 = `base_revenue` (3元)
   - 配送距离 = `haversine_distance(取餐点, 送餐点)` / 1000 (km)
   - 订单收益 = 基础收益 + 配送距离 × `distance_revenue_coeff`
3. 估算送达时间，若超时则施加惩罚:
   - 超时 ≤ 10分钟: 收益 × 0.9
   - 超时 ≤ 20分钟: 收益 × 0.6
   - 超时 > 20分钟: 收益 × 0.5
4. 总收益 = 所有订单收益之和

### 1.3.7 ε-贪婪动作选择设计

**`_choose_action(q_values, candidate_ids) -> str`**

选择逻辑:
1. 生成随机数 r ~ Uniform(0, 1)
2. 若 r < epsilon: 随机选择一个候选骑手
3. 否则: 选择Q值最大的候选骑手
4. 派单完成后: `epsilon = max(epsilon_min, epsilon * epsilon_decay)`

# **2. 接口设计**

## **2.1 总体设计**

DQN派单器对外接口与现有分派器保持一致风格，但增加 `current_time` 参数（奖励计算需要）。

## **2.2 接口清单**

### OrderDispatcherByDQN 对外接口

| 方法 | 签名 | 说明 |
|------|------|------|
| `__init__` | `(self, route_planner: RoutePlanner, config: DQNConfig = None)` | 初始化DQN派单器 |
| `dispatch_orders_by_dqn` | `(self, orders: List[Dict], couriers: Dict[str, Dict], current_time: datetime) -> Dict[str, str]` | 批量DQN派单 |
| `save_model` | `(self, path: str)` | 保存模型参数到文件 |
| `load_model` | `(self, path: str)` | 从文件加载模型参数 |

### DQNAgent 对外接口

| 方法 | 签名 | 说明 |
|------|------|------|
| `batch_predict` | `(self, states: np.ndarray) -> List[float]` | 批量Q值预测 |
| `update` | `(self, batch: Tuple) -> float` | 网络参数更新，返回损失值 |
| `save` | `(self, path: str)` | 保存网络参数 |
| `load` | `(self, path: str)` | 加载网络参数 |

### DeliverySimulation 修改点

| 修改位置 | 修改内容 |
|----------|----------|
| `__init__` | 新增 `self.dispatcher_dqn = OrderDispatcherByDQN(self.route_planner)` |
| `step()` 第136行 | 将 `self.dispatcher_real.dispatch_orders_by_real(...)` 替换为 `self.dispatcher_dqn.dispatch_orders_by_dqn(self.pending_orders, online_couriers, current_time)` |
| `core/__init__.py` | 新增导出 `OrderDispatcherByDQN` 和 `DQNConfig` |

# **4. 数据模型**

## **4.1 设计目标**

状态向量和经验回放数据是DQN派单器的核心数据结构，需要明确其内存布局和序列化方式。

## **4.2 模型实现**

### 状态向量 (State Vector)

| 字段 | 偏移 | 维度 | 类型 | 说明 |
|------|------|------|------|------|
| route_points | 0-15 | 16 | float32 | 路径点坐标[lng1,lat1,...,lng8,lat8]，归一化后 |
| new_order_coords | 16-19 | 4 | float32 | 新订单[sender_lng, sender_lat, recipient_lng, recipient_lat]，归一化后 |
| available_time | 20 | 1 | float32 | 可用配送时长，归一化后 |

总维度: 21

### 经验回放 (Experience)

| 字段 | 类型 | 说明 |
|------|------|------|
| state | np.ndarray (21,) | 当前状态向量 |
| action | int | 选择的候选骑手索引（在候选集中的位置） |
| reward | float | 边际收益奖励 |
| next_state | np.ndarray (21,) | 下一状态向量（派单后的状态） |

### 派单结果 (Dispatch Result)

| 字段 | 类型 | 说明 |
|------|------|------|
| key | str | order_id |
| value | str | courier_id |

与现有分派器输出格式一致: `Dict[str, str]`
