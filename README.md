# 即时配送智能派单系统

基于机器学习与强化学习的即时配送智能派单算法研究——毕业论文项目。

## 项目结构

```
2 Python/
├── Q_Values/                    # 配送仿真系统 + DQN 派单
│   ├── Data/
│   │   ├── Data_clean.py              # 原始数据清洗（时间转换、空值处理）
│   │   └── Data_clean_behavior.py     # 行为特征工程（疲劳度、在手订单、骑手位置估计）
│   └── simulation/
│       ├── core/
│       │   ├── courier.py             # 骑手模拟器（上下线事件生成）
│       │   ├── order_g.py             # 订单生成器（按时序派发订单）
│       │   ├── order_d_dqn.py         # DQN 派单器（Actor-Critic 强化学习）
│       │   ├── order_d_real.py        # 真实匹配派单器（基线方法）
│       │   ├── route.py               # 路径规划器（贪婪算法）
│       │   ├── state.py               # 状态刷新器（骑手移动、取送餐状态更新）
│       │   └── behavior.py            # 抢单预测 MLP 模型训练
│       ├── dqn/
│       │   ├── dqn_network.py         # Actor-Critic 网络（Q网络、V网络、回放缓冲区）
│       │   └── checkpoints/           # 模型保存目录
│       ├── utils/
│       │   └── geo_utils.py           # 地理工具（Haversine 距离计算等）
│       ├── main_dqn.py                # DQN 仿真主程序入口
│       └── main_real.py               # 真实匹配仿真主程序入口（基线对比）
├── SPOT/                        # 骑手接单行为预测模型
│   ├── Data/
│   │   └── Data_clean.py              # 行为数据清洗与特征工程
│   ├── core/
│   │   ├── decision_tree.py           # 标准决策树分类器（预测是否接单）
│   │   ├── smart_decision_tree.py     # 自定义相似集决策树（MSE + 决策损失联合优化）
│   │   ├── sets_build/
│   │   │   ├── sample_set.py          # 基于仿真引擎的样本集生成
│   │   │   └── similarity_set.py      # 相似集构建（特征加权距离）
│   │   └── pkl_s/                     # 训练好的模型存储
│   └── Results/                       # 评估结果与可视化
├── requirements.txt
└── .gitignore
```

## 项目概述

本项目研究即时配送场景下的智能派单算法，核心思路分为两层：

### 1. 行为预测层 (SPOT)

预测骑手是否会接某个订单，为派单决策提供依据。

- **标准决策树** (`decision_tree.py`)：以 4 个核心特征（疲劳度、配送距离、在手订单数等）训练分类树，预测 `is_courier_grabbed`，带类别权重处理样本不平衡。
- **相似集决策树** (`smart_decision_tree.py`)：自定义决策树回归器，损失函数为 `a × MSE + (1-a) × 决策损失`。决策损失衡量：在同一相似集内，被选中的 Top-N 订单的真实价值与最优选择的差距。支持并行化特征搜索。
- **样本集生成** (`sample_set.py`)：驱动仿真引擎，在目标订单派单时刻提取在线骑手特征，以决策树预测的接单概率作为目标值。
- **相似集构建** (`similarity_set.py`)：基于特征加权距离，将相似订单聚为同一集合，用于决策损失计算。

### 2. 配送仿真系统 (Q_Values)

基于真实数据驱动的时序仿真平台，模拟骑手上下线、订单生成、路径规划与派单全过程。

- **仿真引擎**：以 30 秒为步长，按真实时间线驱动事件（骑手上/下线、订单到达）。
- **路径规划**：贪婪算法，满足先取后送约束，实时刷新骑手位置与订单状态。
- **派单策略**：
  - **DQN (Actor-Critic)**：27 维状态 + 5 维动作，边际收益奖励，离线交替训练。
  - **真实匹配**：按历史数据中的真实订单-骑手配对派单（基线）。
  - **距离优先**：派给距离最近的骑手（基线）。
- **抢单行为 MLP** (`behavior.py`)：训练 MLP 模型预测骑手接单概率，用于 DQN 的候选骑手筛选。

## 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0
- 依赖见 `requirements.txt`

## 快速开始

### 数据准备

1. 将美团原始运单数据（CSV 格式）放入 `Q_Values/Data/0MeiTuan_Data/` 目录。
2. 运行数据清洗：

```bash
python Q_Values/Data/Data_clean.py
python Q_Values/Data/Data_clean_behavior.py
```

### 训练行为预测模型

```bash
python SPOT/core/decision_tree.py
```

### 运行仿真

**DQN 派单（训练/评估）**：修改 `main_dqn.py` 中的 `IS_TRAIN` 和 `DATE_FILTER` 参数后运行：

```bash
python Q_Values/simulation/main_dqn.py
```

**真实匹配派单（基线对比）**：

```bash
python Q_Values/simulation/main_real.py
```

## 关键结果

仿真结束后输出指标包括：
- 总订单数、送达数、拒单数
- 拒单率、平均配送时长
- 超时率、平均超时时长
- 骑手平均订单收益

模型检查点与损失曲线自动保存至 `Q_Values/simulation/dqn/checkpoints/`。
