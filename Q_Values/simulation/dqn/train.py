"""
完整训练脚本

演示如何使用集成仿真环境进行完整的训练流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from tqdm import tqdm

from agent import Agent
from state_builder import StateBuilder
from action_builder import ActionBuilder
from reward_calculator import RewardCalculator
from integrated_env import IntegratedSimulationEnv
from typing import Tuple, List

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_order_data(filepath: str) -> pd.DataFrame:
    """
    加载订单数据
    
    Args:
        filepath: 数据文件路径
    
    Returns:
        pd.DataFrame: 订单数据
    """
    logger.info(f"正在加载订单数据: {filepath}")
    
    # 这里需要根据实际数据格式进行调整
    # 示例：假设数据为CSV格式
    df = pd.read_csv(filepath)
    
    # 转换时间列
    df['dispatch_time'] = pd.to_datetime(df['dispatch_time'])
    df['arrive_time'] = pd.to_datetime(df['arrive_time'])
    df['estimate_meal_prepare_time'] = pd.to_datetime(df['estimate_meal_prepare_time'])
    df['estimate_arrived_time'] = pd.to_datetime(df['estimate_arrived_time'])
    
    logger.info(f"订单数据加载完成，共 {len(df)} 条订单")
    
    return df


def train_one_day(
    env: IntegratedSimulationEnv,
    agent: Agent,
    day_data: pd.DataFrame,
    num_epochs: int = 10,
    batch_size: int = 128
) -> Tuple[float, float]:
    """
    训练一天的数据
    
    Args:
        env: 仿真环境
        agent: Agent
        day_data: 一天的订单数据
        num_epochs: 训练轮数
        batch_size: 批次大小
    
    Returns:
        Tuple[float, float]: 平均Critic损失和Actor损失
    """
    logger.info("开始一天的数据收集...")
    
    # 1. 数据收集：运行仿真
    experiences = env.run_simulation()
    
    if len(experiences) == 0:
        logger.warning("未收集到任何经验数据")
        return 0.0, 0.0
    
    # 2. 存储经验到ReplayBuffer
    for state, action, reward, next_state in experiences:
        done = False  # 这里简化处理，实际应根据是否到达终止状态设置
        agent.replay_buffer.push(state, action, reward, next_state, done)
    
    logger.info(f"数据收集完成，共 {len(experiences)} 条经验")
    
    # 3. 离线交替训练
    logger.info("开始离线交替训练...")
    total_critic_loss = 0.0
    total_actor_loss = 0.0
    
    for epoch in tqdm(range(num_epochs), desc="训练进度"):
        critic_loss, actor_loss = agent.train_step(batch_size=batch_size)
        total_critic_loss += critic_loss
        total_actor_loss += actor_loss
        
        if (epoch + 1) % 2 == 0:
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Critic Loss = {critic_loss:.6f}, "
                       f"Actor Loss = {actor_loss:.6f}")
    
    avg_critic_loss = total_critic_loss / num_epochs
    avg_actor_loss = total_actor_loss / num_epochs
    
    logger.info(f"训练完成，平均Critic Loss: {avg_critic_loss:.6f}, "
               f"平均Actor Loss: {avg_actor_loss:.6f}")
    
    return avg_critic_loss, avg_actor_loss


def main():
    """主训练流程"""
    logger.info("=" * 60)
    logger.info("开始Actor-Critic强化学习训练")
    logger.info("=" * 60)
    
    # 1. 加载订单数据
    # 这里需要替换为实际的数据路径
    data_path = "Q_Values/Data_Process/processed_orders.csv"
    
    try:
        order_df = load_order_data(data_path)
    except FileNotFoundError:
        logger.warning(f"数据文件不存在: {data_path}")
        logger.info("使用模拟数据进行演示...")
        
        # 创建模拟数据
        order_df = create_mock_data()
    
    # 2. 初始化强化学习组件
    logger.info("初始化强化学习组件...")
    
    # 获取时间范围
    time_start = order_df['dispatch_time'].min()
    time_end = order_df['dispatch_time'].max()
    
    # 状态构建器
    state_builder = StateBuilder(
        lng_min=116.0,
        lng_max=117.0,
        lat_min=39.0,
        lat_max=40.0,
        time_start=time_start,
        time_end=time_end,
        max_time_remaining=3600.0
    )
    
    # 动作构建器
    action_builder = ActionBuilder(
        lng_min=116.0,
        lng_max=117.0,
        lat_min=39.0,
        lat_max=40.0,
        max_delivery_time=3600.0
    )
    
    # 奖励计算器
    reward_calculator = RewardCalculator(base_reward=3.0)
    
    # Agent
    agent = Agent(
        state_dim=27,
        action_dim=5,
        hidden_dims=[128, 64],
        activation='relu',
        learning_rate=1e-3,
        gamma=0.99,
        buffer_capacity=100000,
        device='auto',
        is_train=True
    )
    
    logger.info(f"Agent初始化完成，设备: {agent.device}")
    
    # 3. 初始化仿真环境
    env = IntegratedSimulationEnv(
        order_df=order_df,
        agent=agent,
        state_builder=state_builder,
        action_builder=action_builder,
        reward_calculator=reward_calculator,
        max_orders_per_courier=4,
        idle_timeout_minutes=30
    )
    
    # 4. 训练循环
    num_days = 10  # 训练天数
    best_loss = float('inf')
    
    for day in range(num_days):
        logger.info(f"\n{'='*60}")
        logger.info(f"训练第 {day+1}/{num_days} 天")
        logger.info(f"{'='*60}")
        
        # 训练一天
        avg_critic_loss, avg_actor_loss = train_one_day(
            env, agent, order_df, num_epochs=10, batch_size=128
        )
        
        # 保存最佳模型
        total_loss = avg_critic_loss + abs(avg_actor_loss)
        if total_loss < best_loss:
            best_loss = total_loss
            model_path = f"Q_Values/simulation/dqn/checkpoints/best_model_day_{day+1}.pt"
            agent.save_model(model_path)
            logger.info(f"保存最佳模型至: {model_path}")
    
    # 5. 保存最终模型
    final_model_path = "Q_Values/simulation/dqn/checkpoints/final_model.pt"
    agent.save_model(final_model_path)
    logger.info(f"最终模型已保存至: {final_model_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("训练完成！")
    logger.info("=" * 60)


def create_mock_data() -> pd.DataFrame:
    """
    创建模拟数据（用于演示）
    
    Returns:
        pd.DataFrame: 模拟订单数据
    """
    logger.info("创建模拟数据...")
    
    # 创建100个模拟订单
    num_orders = 100
    
    data = {
        'order_id': [f'order_{i}' for i in range(num_orders)],
        'courier_id': [f'courier_{i % 10}' for i in range(num_orders)],
        'sender_lng': np.random.uniform(116.0, 117.0, num_orders) * 1e6,
        'sender_lat': np.random.uniform(39.0, 40.0, num_orders) * 1e6,
        'recipient_lng': np.random.uniform(116.0, 117.0, num_orders) * 1e6,
        'recipient_lat': np.random.uniform(39.0, 40.0, num_orders) * 1e6,
        'grab_lng': np.random.uniform(116.0, 117.0, num_orders) * 1e6,
        'grab_lat': np.random.uniform(39.0, 40.0, num_orders) * 1e6,
    }
    
    # 时间列
    base_time = pd.Timestamp('2024-01-01 08:00:00')
    data['dispatch_time'] = [base_time + pd.Timedelta(minutes=i*5) for i in range(num_orders)]
    data['arrive_time'] = [t + pd.Timedelta(minutes=np.random.randint(10, 30)) for t in data['dispatch_time']]
    data['estimate_meal_prepare_time'] = [t + pd.Timedelta(minutes=np.random.randint(5, 15)) for t in data['dispatch_time']]
    data['estimate_arrived_time'] = [t + pd.Timedelta(minutes=np.random.randint(20, 40)) for t in data['dispatch_time']]
    
    df = pd.DataFrame(data)
    
    logger.info(f"模拟数据创建完成，共 {len(df)} 条订单")
    
    return df


if __name__ == "__main__":
    main()
