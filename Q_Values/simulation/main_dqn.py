"""
应用模式派单
使用训练好的模型进行实时派单决策
"""

import os
import logging
import argparse
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch

# 导入DQN模块
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dqn.agent import DQNAgent

# 导入仿真模块
from core.courier import CourierSimulator
from core.order_g import OrderGenerator
from core.route import RoutePlanner
from core.state import StateManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ApplicationConfig:
    """应用配置"""
    def __init__(self):
        # 网络参数
        self.state_dim = 27
        self.action_dim = 5
        
        # 设备
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # 仿真参数
        self.time_step = 30  # 秒


class OrderDispatchApplication:
    """订单派单应用"""
    
    def __init__(self, config, model_path):
        """
        初始化应用
        
        Args:
            config: 应用配置
            model_path: 模型路径
        """
        self.config = config
        
        # 初始化智能体并加载模型
        self.agent = DQNAgent(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            device=config.device
        )
        
        # 加载训练好的模型
        if not self.agent.load_model(model_path):
            raise RuntimeError(f"无法加载模型: {model_path}")
        
        logger.info(f"模型加载成功: {model_path}")
    
    def run(self, order_df):
        """
        运行应用模式
        
        Args:
            order_df: 订单数据DataFrame
        
        Returns:
            results: 派单结果
        """
        logger.info("=" * 50)
        logger.info("开始应用模式派单")
        logger.info("=" * 50)
        
        # 初始化仿真模块
        courier_sim = CourierSimulator(order_df)
        order_gen = OrderGenerator(order_df)
        route_planner = RoutePlanner()
        state_manager = StateManager()
        
        # 时间管理
        timestamps = order_gen.get_all_timestamps()
        if not timestamps:
            raise ValueError("订单数据为空")
        
        start_time = timestamps[0]
        end_time = timestamps[-1]
        current_time = start_time
        
        # 骑手状态
        couriers = {}
        
        # 待派单订单池
        pending_orders = []
        
        # 派单结果
        dispatch_results = []
        
        # 统计信息
        total_orders = 0
        dispatched_orders = 0
        total_decision_time = 0
        decision_count = 0
        
        logger.info(f"仿真时间范围: {start_time} -> {end_time}")
        
        # 仿真循环
        while current_time <= end_time:
            # 处理骑手上下线事件
            courier_events = courier_sim.get_events_at_time(current_time)
            for courier_id, event_type, position in courier_events:
                if event_type == 'online':
                    couriers[courier_id] = {
                        'status': 'online',
                        'position': position,
                        'orders': []
                    }
                elif event_type == 'offline':
                    if courier_id in couriers:
                        couriers[courier_id]['status'] = 'offline'
            
            # 获取新订单
            new_orders = order_gen.get_orders_at_time(current_time)
            pending_orders.extend(new_orders)
            total_orders += len(new_orders)
            
            # 派单决策
            if len(pending_orders) > 0:
                # 记录决策开始时间
                decision_start = datetime.now()
                
                # 构建状态向量（简化版）
                state = np.zeros(self.config.state_dim, dtype=np.float32)
                state[0] = 116.4  # 骑手经度（示例）
                state[1] = 39.9   # 骑手纬度（示例）
                state[2] = (current_time - start_time).total_seconds() / (24 * 3600)
                
                # 构建候选订单列表
                candidate_orders = []
                for order in pending_orders:
                    # 检查是否有可用骑手
                    for courier_id, courier_data in couriers.items():
                        if courier_data['status'] == 'online' and len(courier_data['orders']) < 8:
                            candidate_orders.append({
                                'order': order,
                                'courier_id': courier_id,
                                'sender_lng': order['sender_lng'],
                                'sender_lat': order['sender_lat'],
                                'receiver_lng': order['recipient_lng'],
                                'receiver_lat': order['recipient_lat'],
                                'available_time': 30
                            })
                            break
                
                # 选择动作
                if len(candidate_orders) > 0:
                    selected_order, q_value = self.agent.select_action(
                        state, candidate_orders, mode='application'
                    )
                    
                    # 记录决策时间
                    decision_time = (datetime.now() - decision_start).total_seconds() * 1000
                    total_decision_time += decision_time
                    decision_count += 1
                    
                    # 执行派单
                    if selected_order:
                        order = selected_order['order']
                        courier_id = selected_order['courier_id']
                        
                        # 分配订单给骑手
                        couriers[courier_id]['orders'].append(order)
                        order['courier_id'] = courier_id
                        pending_orders.remove(order)
                        dispatched_orders += 1
                        
                        # 记录结果
                        dispatch_results.append({
                            'order_id': order['order_id'],
                            'courier_id': courier_id,
                            'dispatch_time': current_time,
                            'q_value': q_value,
                            'decision_time_ms': decision_time
                        })
                        
                        logger.debug(f"派单: 订单{order['order_id']} -> 骑手{courier_id}, Q值={q_value:.4f}")
                        
                        # 检查决策延迟
                        if decision_time > 100:
                            logger.warning(f"决策延迟过高: {decision_time:.2f}ms (要求<100ms)")
            
            # 推进时间
            time_idx = timestamps.index(current_time) if current_time in timestamps else -1
            if time_idx + 1 < len(timestamps):
                current_time = timestamps[time_idx + 1]
            else:
                current_time += timedelta(seconds=self.config.time_step)
        
        # 统计结果
        avg_decision_time = total_decision_time / max(decision_count, 1)
        dispatch_rate = dispatched_orders / max(total_orders, 1)
        
        logger.info("\n" + "=" * 50)
        logger.info("应用模式派单完成")
        logger.info("=" * 50)
        logger.info(f"总订单数: {total_orders}")
        logger.info(f"已派单数: {dispatched_orders}")
        logger.info(f"派单率: {dispatch_rate:.2%}")
        logger.info(f"平均决策延迟: {avg_decision_time:.2f}ms")
        logger.info(f"最大决策延迟: {total_decision_time:.2f}ms" if decision_count > 0 else "无决策")
        
        return dispatch_results
    
    def dispatch_single_order(self, order, couriers_state):
        """
        单次派单决策（用于实时调度）
        
        Args:
            order: 待派单订单
            couriers_state: 骑手状态字典
        
        Returns:
            courier_id: 选择的骑手ID
            q_value: Q值
        """
        # 构建状态向量
        state = np.zeros(self.config.state_dim, dtype=np.float32)
        state[0] = 116.4  # 示例
        state[1] = 39.9
        state[2] = 0.5  # 示例
        
        # 构建候选订单列表
        candidate_orders = []
        for courier_id, courier_data in couriers_state.items():
            if courier_data['status'] == 'online' and len(courier_data['orders']) < 8:
                candidate_orders.append({
                    'order': order,
                    'courier_id': courier_id,
                    'sender_lng': order['sender_lng'],
                    'sender_lat': order['sender_lat'],
                    'receiver_lng': order['recipient_lng'],
                    'receiver_lat': order['recipient_lat'],
                    'available_time': 30
                })
        
        # 选择动作
        if len(candidate_orders) > 0:
            selected_order, q_value = self.agent.select_action(
                state, candidate_orders, mode='application'
            )
            if selected_order:
                return selected_order['courier_id'], q_value
        
        return None, None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='应用模式订单派单')
    parser.add_argument('--model_path', type=str, default='Q_Values/dqn/checkpoints',
                       help='模型路径')
    parser.add_argument('--data_path', type=str, default='data/orders.csv',
                       help='订单数据路径')
    parser.add_argument('--output_path', type=str, default='results/dispatch_results.csv',
                       help='结果输出路径')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ApplicationConfig()
    
    # 初始化应用
    app = OrderDispatchApplication(config, args.model_path)
    
    # 加载订单数据
    try:
        order_df = pd.read_csv(args.data_path)
        order_df['dispatch_time'] = pd.to_datetime(order_df['dispatch_time'])
        order_df['arrive_time'] = pd.to_datetime(order_df['arrive_time'])
        logger.info(f"加载订单数据: {len(order_df)} 条")
    except Exception as e:
        logger.error(f"无法加载订单数据: {e}")
        return
    
    # 运行应用
    results = app.run(order_df)
    
    # 保存结果
    if results:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_df.to_csv(args.output_path, index=False)
        logger.info(f"派单结果已保存至: {args.output_path}")


if __name__ == "__main__":
    main()
