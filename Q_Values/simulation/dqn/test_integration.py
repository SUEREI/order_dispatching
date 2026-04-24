"""
集成测试脚本

验证所有集成组件是否正常工作
"""

import sys
import os
# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from Q_Values.simulation.dqn.agent import Agent
from Q_Values.simulation.dqn.state_builder import StateBuilder
from Q_Values.simulation.dqn.action_builder import ActionBuilder
from Q_Values.simulation.dqn.reward_calculator import RewardCalculator


def test_state_builder():
    """测试状态向量构建器"""
    print("\n" + "=" * 60)
    print("测试 StateBuilder")
    print("=" * 60)
    
    # 创建状态构建器
    time_start = datetime(2024, 1, 1, 8, 0, 0)
    time_end = datetime(2024, 1, 1, 20, 0, 0)
    
    state_builder = StateBuilder(
        lng_min=116.0, lng_max=117.0,
        lat_min=39.0, lat_max=40.0,
        time_start=time_start,
        time_end=time_end
    )
    
    # 模拟骑手信息
    courier = {
        'courier_id': 'courier_1',
        'position': (116.5, 39.5),
        'current_orders': []
    }
    
    # 模拟路径
    route = [
        {'sequence': 1, 'point_type': 'pickup', 'order_id': 'order_1', 
         'position': (116.51, 39.51)},
        {'sequence': 2, 'point_type': 'delivery', 'order_id': 'order_1', 
         'position': (116.52, 39.52)}
    ]
    
    # 模拟订单
    orders_dict = {
        'order_1': {
            'order_id': 'order_1',
            'estimate_meal_prepare_time': datetime(2024, 1, 1, 9, 0, 0),
            'estimate_arrived_time': datetime(2024, 1, 1, 9, 30, 0)
        }
    }
    
    current_time = datetime(2024, 1, 1, 8, 30, 0)
    
    # 构建状态向量
    state = state_builder.build_state(courier, route, current_time, orders_dict)
    
    print(f"[OK] 状态向量构建成功")
    print(f"  - 维度: {state.shape}")
    print(f"  - 骑手位置: ({state[0]:.4f}, {state[1]:.4f})")
    print(f"  - 时间戳: {state[2]:.4f}")
    print(f"  - 前2个路径点: ({state[3]:.4f}, {state[4]:.4f}), ({state[5]:.4f}, {state[6]:.4f})")
    
    assert state.shape == (25,), f"状态向量维度错误: {state.shape}"
    print("[OK] 状态向量维度验证通过")


def test_action_builder():
    """测试动作向量构建器"""
    print("\n" + "=" * 60)
    print("测试 ActionBuilder")
    print("=" * 60)
    
    action_builder = ActionBuilder(
        lng_min=116.0, lng_max=117.0,
        lat_min=39.0, lat_max=40.0
    )
    
    # 模拟订单
    order = {
        'order_id': 'order_1',
        'sender_lng': 116.5,
        'sender_lat': 39.5,
        'recipient_lng': 116.6,
        'recipient_lat': 39.6,
        'estimate_arrived_time': datetime(2024, 1, 1, 9, 30, 0)
    }
    
    current_time = datetime(2024, 1, 1, 8, 30, 0)
    
    # 构建动作向量
    action = action_builder.build_action(order, current_time)
    
    print(f"[OK] 动作向量构建成功")
    print(f"  - 维度: {action.shape}")
    print(f"  - 取餐点: ({action[0]:.4f}, {action[1]:.4f})")
    print(f"  - 送餐点: ({action[2]:.4f}, {action[3]:.4f})")
    print(f"  - 配送时长: {action[4]:.4f}")
    
    assert action.shape == (5,), f"动作向量维度错误: {action.shape}"
    print("[OK] 动作向量维度验证通过")
    
    # 测试候选动作集
    candidate_orders = [order, order, order]  # 3个相同订单
    candidate_actions = action_builder.build_candidate_actions(candidate_orders, current_time)
    
    print(f"[OK] 候选动作集构建成功")
    print(f"  - 维度: {candidate_actions.shape}")
    
    assert candidate_actions.shape == (3, 5), f"候选动作集维度错误: {candidate_actions.shape}"
    print("[OK] 候选动作集维度验证通过")
    
    # 测试拒单动作
    reject_action = action_builder.build_reject_action()
    print(f"[OK] 拒单动作构建成功")
    print(f"  - 维度: {reject_action.shape}")
    print(f"  - 值: {reject_action}")
    
    assert np.allclose(reject_action, 0), "拒单动作应为全0向量"
    print("[OK] 拒单动作验证通过")


def test_reward_calculator():
    """测试奖励计算器"""
    print("\n" + "=" * 60)
    print("测试 RewardCalculator")
    print("=" * 60)
    
    reward_calculator = RewardCalculator(base_reward=3.0)
    
    # 测试单个订单收益
    reward1 = reward_calculator.calculate_order_reward(
        delivery_distance=3.5,
        overtime_minutes=0
    )
    print(f"[OK] 无超时订单收益: {reward1:.2f}")
    # 基础收益 = 3.0 + 3.5 = 6.5，无超时打8折 = 5.2
    assert abs(reward1 - 5.2) < 0.01, f"无超时收益计算错误: {reward1}"
    
    reward2 = reward_calculator.calculate_order_reward(
        delivery_distance=3.5,
        overtime_minutes=5
    )
    print(f"[OK] 超时5分钟订单收益: {reward2:.2f}")
    # 超时5分钟（<=10分钟）打8折
    assert abs(reward2 - 6.5 * 0.8) < 0.01, f"超时5分钟收益计算错误: {reward2}"
    
    reward3 = reward_calculator.calculate_order_reward(
        delivery_distance=3.5,
        overtime_minutes=15
    )
    print(f"[OK] 超时15分钟订单收益: {reward3:.2f}")
    # 超时15分钟（10-20分钟）打5折
    assert abs(reward3 - 6.5 * 0.5) < 0.01, f"超时15分钟收益计算错误: {reward3}"
    
    reward4 = reward_calculator.calculate_order_reward(
        delivery_distance=3.5,
        overtime_minutes=25
    )
    print(f"[OK] 超时25分钟订单收益: {reward4:.2f}")
    # 超时25分钟（>20分钟）收益为0
    assert reward4 == 0, f"超时25分钟收益计算错误: {reward4}"
    
    print("[OK] 奖励计算验证通过")


def test_integration():
    """测试完整集成流程"""
    print("\n" + "=" * 60)
    print("测试完整集成流程")
    print("=" * 60)
    
    # 初始化所有组件
    time_start = datetime(2024, 1, 1, 8, 0, 0)
    time_end = datetime(2024, 1, 1, 20, 0, 0)
    
    state_builder = StateBuilder(
        lng_min=116.0, lng_max=117.0,
        lat_min=39.0, lat_max=40.0,
        time_start=time_start,
        time_end=time_end
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
        is_train=True,
        device='cpu'
    )
    
    print("[OK] 所有组件初始化成功")
    
    # 模拟一个完整的决策流程
    # 1. 构建状态
    courier = {
        'courier_id': 'courier_1',
        'position': (116.5, 39.5),
        'current_orders': []
    }
    route = []
    orders_dict = {}
    current_time = datetime(2024, 1, 1, 8, 30, 0)
    
    state = state_builder.build_state(courier, route, current_time, orders_dict)
    print(f"[OK] 状态构建成功: {state.shape}")
    
    # 2. 构建候选动作
    candidate_orders = [
        {
            'order_id': 'order_1',
            'sender_lng': 116.51,
            'sender_lat': 39.51,
            'recipient_lng': 116.52,
            'recipient_lat': 39.52,
            'estimate_arrived_time': datetime(2024, 1, 1, 9, 0, 0)
        },
        {
            'order_id': 'order_2',
            'sender_lng': 116.53,
            'sender_lat': 39.53,
            'recipient_lng': 116.54,
            'recipient_lat': 39.54,
            'estimate_arrived_time': datetime(2024, 1, 1, 9, 10, 0)
        }
    ]
    
    candidate_actions = action_builder.build_candidate_actions(candidate_orders, current_time)
    reject_action = action_builder.build_reject_action()
    candidate_actions_with_reject = np.vstack([candidate_actions, reject_action])
    
    print(f"[OK] 候选动作构建成功: {candidate_actions_with_reject.shape}")
    
    # 3. 选择动作
    selected_action = agent.select_action(
        state, candidate_actions_with_reject, mode='train'
    )
    print(f"[OK] 动作选择成功: {selected_action.shape}")
    
    # 4. 计算奖励
    reward = reward_calculator.calculate_order_reward(
        delivery_distance=3.5,
        overtime_minutes=0
    )
    print(f"[OK] 奖励计算成功: {reward:.2f}")
    
    # 5. 存储经验
    next_state = state  # 简化处理
    agent.replay_buffer.push(state, selected_action, reward, next_state, False)
    print(f"[OK] 经验存储成功，缓冲区大小: {len(agent.replay_buffer)}")
    
    # 6. 训练
    critic_loss, actor_loss = agent.train_step(batch_size=1)
    print(f"[OK] 训练成功")
    print(f"  - Critic Loss: {critic_loss:.6f}")
    print(f"  - Actor Loss: {actor_loss:.6f}")
    
    print("\n[OK] 完整集成流程测试通过！")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("开始集成测试")
    print("=" * 60)
    
    try:
        test_state_builder()
        test_action_builder()
        test_reward_calculator()
        test_integration()
        
        print("\n" + "=" * 60)
        print("[OK] 所有集成测试通过！")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"[ERROR] 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
