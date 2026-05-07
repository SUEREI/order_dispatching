# 强化学习订单派单Agent

基于深度Q网络(DQN)的订单智能派单系统，通过强化学习优化配送效率、准时率与成本。

## 项目结构

```
Q_Values/
├── main.py                          # 训练主程序
├── config.py                        # 配置文件
├── requirements.txt                 # 依赖包列表
├── dqn/
│   ├── __init__.py
│   ├── dqn_network.py               # 网络定义(Actor和Critic)
│   ├── agent.py                     # 智能体(动作选择、训练、模型管理)
│   └── checkpoints/                 # 模型和训练曲线保存目录
│       ├── actor_best.pth
│       ├── critic_best.pth
│       ├── loss_curve.png
│       └── reward_curve.png
└── simulation/
    ├── __init__.py
    ├── main_dqn.py                  # 应用模式派单
    └── core/                        # 仿真核心模块
        ├── courier.py               # 骑手模拟器
        ├── order_g.py               # 订单生成器
        ├── route.py                 # 路径规划器
        └── state.py                 # 状态管理器
```

## 核心设计

### 状态向量 (27维)

| 维度范围 | 内容 | 说明 |
|---------|------|------|
| 0-1 | 骑手位置 | 经度、纬度 |
| 2 | 时间戳 | 归一化到[0,1] |
| 3-18 | 路径点经纬度 | 最多8个点，每个2维 |
| 19-26 | 距离超时剩余时间 | 归一化，每个路径点1维 |

### 动作向量 (5维)

| 维度范围 | 内容 | 说明 |
|---------|------|------|
| 0-1 | 取餐点位置 | 经度、纬度 |
| 2-3 | 送餐点位置 | 经度、纬度 |
| 4 | 可用时长 | 归一化 |

### 网络结构

#### Actor网络 (Q值计算)
```
输入: 状态(27维) + 动作(5维) → 拼接为32维
Dense(256) → BatchNorm → ReLU
Dense(256) → BatchNorm → ReLU
Dense(128) → ReLU
Dense(1) → Q值输出
```

#### Critic网络 (V值计算)
```
输入: 状态(27维)
Dense(256) → BatchNorm → ReLU
Dense(256) → BatchNorm → ReLU
Dense(128) → ReLU
Dense(1) → V值输出
```

### 训练策略

**交替训练**：
1. 冻结Actor，训练Critic直至收敛
2. 冻结Critic，训练Actor直至收敛
3. 更新Critic目标网络

**收敛判定**：连续10个batch损失无下降则判定收敛

**动作选择**：
- 训练模式：Softmax概率采样（探索）
- 应用模式：Argmax直接选择（利用）

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 使用方法

### 训练模式

```bash
# 使用默认参数训练
python main.py

# 自定义参数训练
python main.py --lr_actor 1e-4 --lr_critic 1e-3 --gamma 0.99 --batch_size 64 --max_days 10
```

**训练参数说明**：
- `--lr_actor`: Actor网络学习率（默认1e-4）
- `--lr_critic`: Critic网络学习率（默认1e-3）
- `--gamma`: 折扣因子（默认0.99）
- `--batch_size`: 批量大小（默认64）
- `--max_days`: 最大训练天数（默认10）
- `--temperature`: Softmax温度参数（默认1.0）
- `--checkpoint_dir`: 模型保存目录（默认Q_Values/dqn/checkpoints）

### 应用模式

```bash
# 使用训练好的模型进行派单
python simulation/main_dqn.py --model_path Q_Values/dqn/checkpoints --data_path data/orders.csv

# 指定输出路径
python simulation/main_dqn.py --model_path Q_Values/dqn/checkpoints --data_path data/orders.csv --output_path results/dispatch_results.csv
```

**应用参数说明**：
- `--model_path`: 模型路径（默认Q_Values/dqn/checkpoints）
- `--data_path`: 订单数据路径（默认data/orders.csv）
- `--output_path`: 结果输出路径（默认results/dispatch_results.csv）

## 配置说明

修改`config.py`文件可以调整所有训练参数：

```python
# 网络参数
STATE_DIM = 27
ACTION_DIM = 5

# 学习率
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3

# 折扣因子
GAMMA = 0.99

# 经验池容量
BUFFER_CAPACITY = 100000

# 训练参数
BATCH_SIZE = 64
MAX_EPOCHS = 100
CONVERGENCE_THRESHOLD = 10

# Softmax温度参数
TEMPERATURE = 1.0
```

## 奖励设计

| 事件 | 奖励计算 |
|------|---------|
| 准时送达 | +1.0 × 取送距离 |
| 超时送达 | -0.1 × 超时分钟数 |
| 行驶成本 | -0.1 × 行驶公里数 |
| 取餐完成 | +0.1 |

## 性能指标

| 指标 | 要求 |
|------|------|
| 单步决策延迟 | < 100ms |
| 批量推理吞吐量 | 50个候选订单 < 50ms |
| 经验池容量 | ≥ 10万条 |

## 输出文件

训练完成后会在`checkpoints`目录生成：

1. **模型文件**：
   - `actor_YYYYMMDD_N.pth`: Actor网络参数
   - `critic_YYYYMMDD_N.pth`: Critic网络参数
   - `actor_best.pth`: 最优Actor模型
   - `critic_best.pth`: 最优Critic模型

2. **训练曲线**：
   - `loss_curve.png`: 损失曲线图
   - `reward_curve.png`: 奖励曲线图

3. **日志文件**：
   - `training.log`: 训练日志

## 依赖说明

- **PyTorch >= 1.10**: 深度学习框架
- **NumPy**: 数值计算
- **Pandas**: 数据处理
- **Matplotlib**: 可视化
- **tqdm**: 进度条

## 注意事项

1. **数据准备**：训练前需要准备订单数据，格式见`order_g.py`中的数据结构
2. **GPU加速**：代码自动检测CUDA，如有GPU会自动使用
3. **模型版本**：建议保存多个版本的模型，便于回退
4. **收敛监控**：通过日志和曲线图监控训练过程，及时调整参数

## 扩展接口

### 单次派单决策

```python
from dqn.agent import DQNAgent

# 加载模型
agent = DQNAgent()
agent.load_model('Q_Values/dqn/checkpoints')

# 单次派单
courier_id, q_value = app.dispatch_single_order(order, couriers_state)
```

### 自定义状态构建

继承`DQNAgent`类并重写`build_state_vector`方法：

```python
class CustomAgent(DQNAgent):
    def build_state_vector(self, ...):
        # 自定义状态构建逻辑
        pass
```

## 开发者

华为云码道（CodeArts）代码智能体

## 许可证

内部项目，仅供学术研究使用
