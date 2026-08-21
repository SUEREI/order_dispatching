#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
仿真主程序 - 距离贪心派单（基线策略）

以 30 秒为步长驱动时序仿真：骑手上下线、订单生成、路径规划，
派单策略为距离贪心：每个待分配订单派给距离其取餐点最近的在线骑手。

输出指标（拒单率、超时率、平均配送时长、平均收益等）与 main_dqn.py
保持一致，便于与 DQN/Actor-Critic 派单、真实匹配派单进行对比。
"""

import logging
import os
import random
from datetime import timedelta
from itertools import count
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from core import CourierSimulator
from core import OrderGenerator
from core import OrderDispatcherByDistance
from core import RoutePlanner
from core import StateRefresher
from utils import haversine_distance

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


class TqdmLoggingHandler(logging.Handler):
    """自定义 logging handler，通过 tqdm.write() 输出日志，
    使进度条始终保持在终端最底部。"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


class DeliverySimulation:
    def __init__(self, order_data_path, idle_timeout_minutes=30, date_filter=None):
        # 加载数据
        self.df = pd.read_csv(order_data_path)
        self.df = self.df.dropna(subset=['order_id'])
        self.df = self.df.dropna(subset=['courier_id'])
        self.df['order_id'] = self.df['order_id'].astype(int).astype(str)
        self.df["courier_id"] = self.df["courier_id"].astype(int).astype(str)
        if not pd.api.types.is_datetime64_any_dtype(self.df['dispatch_time']):
            self.df['dispatch_time'] = pd.to_datetime(self.df['dispatch_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['arrive_time']):
            self.df['arrive_time'] = pd.to_datetime(self.df['arrive_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['estimate_meal_prepare_time']):
            self.df['estimate_meal_prepare_time'] = pd.to_datetime(self.df['estimate_meal_prepare_time'])
        if not pd.api.types.is_datetime64_any_dtype(self.df['estimate_arrived_time']):
            self.df['estimate_arrived_time'] = pd.to_datetime(self.df['estimate_arrived_time'])

        # 筛选特定日期（在时间列转换之后，在生成器初始化之前）
        if date_filter is not None:
            self.df = self.df[self.df['dt'].isin(date_filter)]
            logger.info(f"按日期筛选: {date_filter}，剩余 {len(self.df)} 条订单")

        # 初始化生成器
        self.order_gen = OrderGenerator(self.df)
        self.courier_sim = CourierSimulator(self.df, idle_timeout_minutes)
        # 初始化派单器、路由规划器、状态刷新器
        self.state_refresher = StateRefresher()
        self.route_planner = RoutePlanner()
        self.dispatcher_dis = OrderDispatcherByDistance(max_orders_per_courier=4)

        # 构建骑手池
        self.couriers = {}
        # 初始化骑手池
        self._init_couriers()
        # 构建订单池
        self.orders = {order['order_id']: order for orders in self.order_gen.orders_by_time.values() for order in orders}

        # 时间轴管理
        # 获取最早的骑手上线时间作为起始时间
        earliest_event_time = self.courier_sim.get_earliest_event_time()
        self.current_time = earliest_event_time

        # 待分配订单池
        self.order_pool = 0
        self.pending_orders = []

    def _init_couriers(self) -> None:
        """初始化所有骑手状态，默认全部离线"""
        all_couriers = self.courier_sim.get_all_courier_ids()
        for cid in all_couriers:
            self.couriers[cid] = {
                'courier_id': cid,
                'status': 'offline',
                'position': None,
                'rejected_rate': None,
                'courier_prefer_reject': None,
                'fatigue': None,
                'order_increased': False,
                'current_orders': [],
                'route': [],
                'has_had_orders': False
            }

    def step(self) -> None:
        """执行一步仿真"""
        # 保存当前时间（用于返回）
        current_time = self.current_time
        # 重置当前时间步的新订单数量
        self.order_pool = 0

        for courier_id, courier in self.couriers.items():
            # 1. 刷新骑手状态(如果当前有订单)
            if courier.get('current_orders'):
                updates = self.state_refresher.refresh_courier(
                    courier,
                    current_time,
                )
                if updates.get('courier_updates'):
                    self.couriers[courier_id].update(updates['courier_updates'])

                # 记录已完成订单
                completed_orders = []
                for order in courier['current_orders']:
                    if order['status'] == 'delivered':
                        completed_orders.append(order)
                # 如果订单有对应的骑手记录，可以在这里更新订单状态
                for order in completed_orders:
                    if order['order_id'] in self.orders:
                        self.orders[order['order_id']]['actual_pickup_time'] = pd.to_datetime(order['actual_pickup_time'])
                        self.orders[order['order_id']]['actual_delivery_time'] = pd.to_datetime(order['actual_delivery_time'])
                    # 从骑手当前订单中移除已完成的订单
                    courier['current_orders'].remove(order)

        # 2. 处理当前时间点的骑手上下线事件
        courier_events = self.courier_sim.get_events_at_time(self.current_time)
        for cid, etype, pos, rejected_rate, courier_prefer_reject in courier_events:
            self.couriers[cid]['status'] = etype
            if etype == 'online':
                self.couriers[cid]['position'] = pos
                self.couriers[cid]['rejected_rate'] = rejected_rate
                self.couriers[cid]['courier_prefer_reject'] = courier_prefer_reject
                self.couriers[cid]['fatigue'] = 0
                self.couriers[cid]['has_had_orders'] = False
        online_couriers = {cid: info for cid, info in self.couriers.items() if info['status'] == 'online'}

        # 3. 处理当前时间点的订单生成事件
        new_orders = self.order_gen.get_orders_at_time(self.current_time)
        if new_orders:
            self.pending_orders.extend(new_orders)
            self.order_pool = len(new_orders)

        # 4. 如果 pending_orders 不为空，使用距离贪心派单器派单
        if self.pending_orders:
            # 派单给距离取餐点最近的骑手
            dispatch_results = self.dispatcher_dis.dispatch_orders_by_distance(
                self.pending_orders, online_couriers
            )
            # 处理分派结果
            dispatched_orders = []
            for order in self.pending_orders:
                if order['order_id'] in dispatch_results:
                    courier_id = dispatch_results[order['order_id']]
                    self.couriers[courier_id]['has_had_orders'] = True
                    self.couriers[courier_id]['order_increased'] = True
                    self.couriers[courier_id]['current_orders'].append(order)
                    self.orders[order['order_id']]['courier_id'] = courier_id
                    self.orders[order['order_id']]['grab_time'] = current_time
                    dispatched_orders.append(order)
            # 从待分配订单池中移除已分派的订单
            for order in dispatched_orders:
                self.pending_orders.remove(order)

        # 5. 规划路径
        for courier_id, courier in self.couriers.items():
            # 检查是否有订单增加
            if courier['order_increased']:
                route = self.route_planner.plan_route(courier)
                courier['route'] = route
                courier['order_increased'] = False

        done = self.courier_sim.get_next_event_time() is None

        # 6. 推进 current_time
        if not done:
            self.current_time += timedelta(seconds=30)

        return done, current_time


# ===================== 辅助函数 =====================


def _compute_revenue(row):
    """计算订单基础收入（起步价3元 + 里程费）"""
    distance_km = haversine_distance(
        (row['sender_lng'], row['sender_lat']),
        (row['recipient_lng'], row['recipient_lat'])
    ) / 1000
    return 3 + distance_km


def _apply_overtime_penalty(row):
    """根据超时时长对订单收入进行惩罚"""
    overtime = row['overtime_minutes']
    if overtime <= 10:
        return row['revenue'] * 0.9
    elif overtime <= 20:
        return row['revenue'] * 0.6
    else:
        return row['revenue'] * 0.5


def _orders_to_dataframe(orders_dict, delivered_only=False, current_time=None):
    """将订单字典转为DataFrame并计算基础收入"""
    records = [
        o for o in orders_dict.values()
        if pd.notna(o.get('order_id'))
    ]
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    if delivered_only and 'actual_delivery_time' in df.columns:
        df = df[df['actual_delivery_time'].notna()]
        if current_time is not None:
            df = df[df['actual_delivery_time'] <= current_time]

    if 'revenue' not in df.columns:
        df['revenue'] = df.apply(_compute_revenue, axis=1)

    return df


def _compute_delivery_metrics(orders_df):
    """从订单DataFrame计算完整的配送指标"""
    if orders_df.empty:
        return None

    delivered = orders_df[orders_df['actual_delivery_time'].notna()].copy()
    if delivered.empty:
        return None

    delivered['overtime_minutes'] = (
        delivered['actual_delivery_time'] - delivered['estimate_arrived_time']
    ).dt.total_seconds() / 60
    delivered['overtime_minutes'] = delivered['overtime_minutes'].clip(lower=0)

    delivered['delivery_duration'] = (
        delivered['actual_delivery_time'] - delivered['dispatch_time']
    ).dt.total_seconds() / 60

    delivered['actual_revenue'] = delivered.apply(_apply_overtime_penalty, axis=1)

    timeout_mask = delivered['actual_delivery_time'] > delivered['estimate_arrived_time']
    timeout_orders = delivered[timeout_mask]

    rejected_count = int(orders_df['status'].eq('rejected').sum()) if 'status' in orders_df.columns else 0
    rejection_rate = rejected_count / len(orders_df) if len(orders_df) > 0 else 0.0

    # 非重复拒单统计
    rejected_unique_count = int(orders_df.loc[orders_df['status'] == 'rejected', 'order_id'].nunique()) if 'status' in orders_df.columns else 0
    total_unique_count = int(orders_df['order_id'].nunique())
    rejection_rate_unique = rejected_unique_count / total_unique_count if total_unique_count > 0 else 0.0

    return {
        'total': len(orders_df),
        'delivered_count': len(delivered),
        'rejected_count': rejected_count,
        'rejection_rate': rejection_rate,
        'rejected_unique_count': rejected_unique_count,
        'rejection_rate_unique': rejection_rate_unique,
        'timeout_count': len(timeout_orders),
        'avg_delivery_duration': delivered['delivery_duration'].mean(),
        'overtime_rate': len(timeout_orders) / len(delivered),
        'avg_overtime': timeout_orders['overtime_minutes'].mean() if len(timeout_orders) > 0 else 0.0,
        'avg_revenue': delivered['actual_revenue'].mean(),
    }


def _log_final_summary(metrics):
    """记录仿真结束时的最终指标汇总"""
    if metrics is None:
        logger.info("没有订单数据，无法计算超时率")
        return
    logger.info(
        f"总订单数: {metrics['total']}\n"
        f"已送达订单数: {metrics['delivered_count']}\n"
        f"拒单数: {metrics['rejected_count']}\n"
        f"拒单率: {metrics['rejection_rate']:.2%}\n"
        f"拒单率-非重复计数: {metrics['rejection_rate_unique']:.2%}\n"
        f"平均配送时长: {metrics['avg_delivery_duration']:.2f}分钟\n"
        f"超时订单数: {metrics['timeout_count']}\n"
        f"超时率: {metrics['overtime_rate']:.2%}\n"
        f"平均超时时长: {metrics['avg_overtime']:.2f}分钟\n"
        f"骑手平均订单收益: {metrics['avg_revenue']:.2f}"
    )


# ===================== 主流程 =====================


def main():
    import yaml
    config_path = Path(__file__).resolve().parent / 'config' / 'config.yaml'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    seed = config.get('random_seed', 42)

    # 构建日志文件名：sim_{种子}_dis.log
    log_filename = f"sim_{seed}_dis.log"
    log_dir = Path(__file__).resolve().parent / 'logs'
    log_dir.mkdir(exist_ok=True)

    formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

    tqdm_handler = TqdmLoggingHandler()
    tqdm_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_dir / log_filename, encoding='utf-8')
    file_handler.setFormatter(formatter)

    logger.root.handlers = [tqdm_handler, file_handler]

    logger.info(f"使用随机种子: {seed}")

    random.seed(seed)
    np.random.seed(seed)

    data_path = r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv'

    TIME_STEP = 30

    # 按日期筛选：设为 None 使用全部数据，或传入整数列表
    # DATE_FILTER = None
    DATE_FILTER = [20221023, 20221024]

    sim = DeliverySimulation(data_path, date_filter=DATE_FILTER)

    start_time, end_time = sim.courier_sim.get_earliest_event_time(), sim.courier_sim.get_latest_event_time()
    total_steps = int((end_time - start_time).total_seconds() / TIME_STEP) if start_time and end_time else None

    step_iter = range(1, total_steps + 1) if total_steps else count(1)

    for step_num in tqdm(step_iter, desc="仿真进度"):
        done, current_time = sim.step()

        if done:
            logger.info(f"仿真结束，总耗时: {current_time - sim.df['dispatch_time'].min()}")
            break

    # ==================== 后处理 ====================
    all_orders_df = _orders_to_dataframe(sim.orders)
    final_metrics = _compute_delivery_metrics(all_orders_df)
    _log_final_summary(final_metrics)

    result_dir = r'Q_Values\simulation\simulation_result'
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, f'dis_dispatch_{DATE_FILTER[0]}.csv')

    orders_df = all_orders_df.copy()
    orders_df['order_id'] = orders_df['order_id'].astype(str).str.split('_').str[0]
    orders_df['overtime_minutes'] = (
        orders_df['actual_delivery_time'] - orders_df['estimate_arrived_time']
    ).dt.total_seconds() / 60
    orders_df['overtime_minutes'] = orders_df['overtime_minutes'].clip(lower=0)
    orders_df['delivery_duration'] = (
        orders_df['actual_delivery_time'] - orders_df['dispatch_time']
    ).dt.total_seconds() / 60
    orders_df['actual_revenue'] = orders_df.apply(_apply_overtime_penalty, axis=1)
    orders_df.to_csv(result_path, index=False)
    logger.info(f"订单结果已保存到 {result_path}")


if __name__ == '__main__':
    main()
