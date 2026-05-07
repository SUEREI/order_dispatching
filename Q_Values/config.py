"""
训练配置文件
"""

class Config:
    """训练和应用配置"""
    
    # 网络参数
    STATE_DIM = 27  # 状态向量维度
    ACTION_DIM = 5   # 动作向量维度
    
    # 网络结构
    HIDDEN_DIMS = [256, 256, 128]  # 隐藏层维度
    
    # 学习率
    LR_ACTOR = 1e-4     # Actor学习率
    LR_CRITIC = 1e-3    # Critic学习率
    
    # 折扣因子
    GAMMA = 0.99
    
    # 经验池
    BUFFER_CAPACITY = 500000  # 经验池容量
    
    # 训练参数
    BATCH_SIZE = 64              # 批量大小
    MAX_EPOCHS = 100             # 每轮最大训练epoch
    CONVERGENCE_THRESHOLD = 10   # 收敛判定阈值（连续N次损失无下降）
    
    # 探索参数
    TEMPERATURE = 1.0  # Softmax温度参数（训练模式）
    
    # 仿真参数
    TIME_STEP = 30      # 仿真时间步长（秒）
    MAX_DAYS = 10       # 最大训练天数
    
    # 性能要求
    MAX_DECISION_TIME = 100  # 最大决策延迟（ms）
    MAX_ORDERS_PER_COURIER = 8  # 骑手最大订单数
    
    # 奖励参数
    ON_TIME_REWARD_COEF = 1.0     # 准时送达奖励系数
    LATE_PENALTY_COEF = 0.1       # 超时惩罚系数
    TRAVEL_COST_COEF = 0.1        # 行驶成本系数
    PICKUP_REWARD = 0.1           # 取餐完成奖励
    
    # 路径配置
    CHECKPOINT_DIR = 'Q_Values/dqn/checkpoints'
    LOG_DIR = 'logs'
    DATA_DIR = 'data'
    RESULT_DIR = 'results'
    
    # 设备配置
    DEVICE = 'cuda'  # 'cuda' 或 'cpu'


# 创建全局配置实例
config = Config()
