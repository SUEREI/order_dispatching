#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于仿真引擎的样本集生成

对于一个目标订单，驱动仿真运行到该订单的派单时间，
提取所有在线骑手与该订单配对的特征，构建训练样本。

目标值使用决策树预测的接单概率，
而不是硬分类 0/1。

输出：
  - 样本 DataFrame：每行 = (订单, 骑手) 配对 + 7维特征 + 接单概率
  - 保存为 CSV
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent / 'Q_Values' / 'simulation'))
from utils import haversine_distance

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ============================================================
#  特征列名（与 decision_tree.py 保持一致）
# ============================================================

# FEATURE_NAMES = [
#     'fatigue',
#     'dispatch_to_sender_delivery_distance',
#     'sender_to_recipient_delivery_distance',
#     'ongoing_order_count',
#     'max_ongoing_sender_to_recipient_distance',
#     'courier_id_daily_avg_rejected_rate',
#     'poi_id_daily_avg_rejected_rate',
# ]

TARGET = 'is_courier_grabbed'


# ============================================================
#  决策树预测器（懒加载）
# ============================================================

_DT_MODEL = None

# 决策树训练时使用的特征（与 prepare_features_target 一致）
_DT_FEATURES = [
    'fatigue',
    'dispatch_to_sender_delivery_distance',
    'sender_to_recipient_delivery_distance',
    'ongoing_order_count',
    'courier_prefer_reject',
    'poi_prefer_reject',
]


def _ensure_dt():
    """懒加载决策树模型"""
    global _DT_MODEL

    if _DT_MODEL is not None:
        return

    model_path = (
        Path(__file__).resolve().parent.parent
        / 'pkl_s' / 'decision_tree_model_grab.pkl'
    )
    _DT_MODEL = joblib.load(str(model_path))
    print(f'决策树模型已加载: {model_path}')


def predict_grab_probability(order: dict, courier: dict, current_time) -> float:
    """使用决策树预测骑手接单概率

    从 (order, courier) 中提取 4 维特征，用决策树 predict_proba 推理。

    Args:
        order: 订单字典
        courier: 骑手字典
        current_time: 当前仿真时间（未使用，保留接口兼容）

    Returns:
        grab_prob: 接单概率 (0~1)
    """
    _ensure_dt()

    features = build_courier_features(order, courier)
    x = pd.DataFrame([[features[f] for f in _DT_FEATURES]], columns=_DT_FEATURES, dtype=np.float64)
    prob = _DT_MODEL.predict_proba(x)[0, 1]
    return float(prob)


# ============================================================
#  特征构建
# ============================================================

def build_courier_features(order: dict, courier: dict) -> dict:
    """为 (订单, 骑手) 配对构建特征字典"""
    # 骑手到发件点距离
    courier_pos = courier.get('position')
    if courier_pos is not None:
        dispatch_distance = haversine_distance(
            courier_pos,
            (order['sender_lng'], order['sender_lat'])
        )
    else:
        dispatch_distance = np.nan

    # 当前正在配送的订单数量
    ongoing_orders = courier.get('current_orders', [])

    # 已有订单中最大 sender_to_recipient 距离
    max_dist = 0.0
    for o in ongoing_orders:
        d = haversine_distance(
            (o['sender_lng'], o['sender_lat']),
            (o['recipient_lng'], o['recipient_lat'])
        )
        if d > max_dist:
            max_dist = d

    return {
        'fatigue': float(courier.get('fatigue', 0.0)),
        'dispatch_to_sender_delivery_distance': dispatch_distance / 1000.0 if not np.isnan(dispatch_distance) else np.nan,
        'sender_to_recipient_delivery_distance': haversine_distance(
            (order['sender_lng'], order['sender_lat']),
            (order['recipient_lng'], order['recipient_lat'])
        ) / 1000.0,
        'ongoing_order_count': float(len(ongoing_orders)),
        # 'max_ongoing_sender_to_recipient_distance': max_dist / 1000.0,
        'courier_prefer_reject': float(courier.get('prefer_reject', 0.0)),
        'poi_prefer_reject': float(order.get('poi_prefer_reject', 0.0)),
        # 'courier_id_daily_avg_rejected_rate': float(courier.get('rejected_rate', 0.0)),
        # 'poi_id_daily_avg_rejected_rate': float(order.get('poi_id_daily_avg_rejected_rate', 0.0)),
    }


# ============================================================
#  仿真推进到指定时间点
# ============================================================

def run_simulation_to_time(sim, target_time):
    """步进仿真到 target_time，返回该时刻的在线骑手字典"""
    while sim.current_time < target_time:
        done, current_time = sim.step()
        if done:
            break
    # 最后一次 step 后 current_time 可能刚好等于或略过 target_time
    # 取此刻的在线骑手
    online = {
        cid: c for cid, c in sim.couriers.items()
        if c['status'] == 'online' and c.get('position') is not None
    }
    return online


# ============================================================
#  构建样本集
# ============================================================

def build_sample_set(target_order_id, date_filters=None) -> pd.DataFrame:
    """
    构建一个订单的样本集，目标值为决策树预测的接单概率

    Args:
        target_order_id: 目标订单 ID（整数或字符串均可）
        date_filters: 日期筛选列表，如 [20221023]。为 None 时自动使用订单日期。

    Returns:
        samples: DataFrame，每行 = (订单, 骑手) 配对 + 7维特征 + grab_probability
    """
    # 统一转为字符串（sim.df 中 order_id 是 string 类型）
    target_order_id = str(int(target_order_id))

    # 先查原始数据获取订单日期，确保 date_filters 包含它
    raw_df = pd.read_csv(r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv')
    order_row_raw = raw_df[raw_df['order_id'] == int(target_order_id)]
    if order_row_raw.empty:
        raise ValueError(f'原始数据中不存在 order_id = {target_order_id}')
    order_date = order_row_raw.iloc[0]['dt']

    # 如果 date_filters 未指定，自动使用订单所属日期
    if date_filters is None:
        date_filters = [order_date]
        print(f'自动设定 date_filters = {date_filters}')

    from main_dqn import DeliverySimulation

    data_path = r'Q_Values/Data/Data_Processed/cleaned_waybill_info.csv'
    sim = DeliverySimulation(data_path, date_filter=date_filters, is_train=False)

    # 从原始数据中找到目标订单及其派单时间
    order_row = sim.df[sim.df['order_id'] == target_order_id]
    if order_row.empty:
        raise ValueError(f'数据中不存在 order_id = {target_order_id}')
    dispatch_time = order_row.iloc[0]['dispatch_time']
    print(f'目标订单 {target_order_id}（日期 {order_date}），派单时间: {dispatch_time}')

    # 找出真实接单骑手（用于参考）
    actual_courier_id = order_row_raw.iloc[0]['courier_id']
    print(f'真实接单骑手: {actual_courier_id}')

    # 步进仿真到派单时间
    print('推进仿真...')
    online_couriers = run_simulation_to_time(sim, dispatch_time)
    print(f'到达派单时间，在线骑手数: {len(online_couriers)}')

    # 获取订单对象
    order_obj = sim.orders.get(target_order_id)
    if order_obj is None:
        raise ValueError(f'sim.orders 中不存在 order_id = {target_order_id}')

    # 预加载决策树
    _ensure_dt()

    # 计算每个骑手到发件点的距离，按距离升序排序取前500
    courier_with_dist = []
    for cid, courier in online_couriers.items():
        courier_pos = courier.get('position')
        if courier_pos is not None:
            dist = haversine_distance(courier_pos, (order_obj['sender_lng'], order_obj['sender_lat']))
        else:
            dist = float('inf')
        courier_with_dist.append((cid, courier, dist))

    courier_with_dist.sort(key=lambda x: x[2])
    top_couriers = courier_with_dist[:500]
    print(f'距离排序后候选骑手数: {len(top_couriers)}（原始在线: {len(online_couriers)}）')

    # 为每个骑手构建特征，并用决策树预测接单概率
    records = []
    for cid, courier, _ in top_couriers:
        features = build_courier_features(order_obj, courier)
        features['courier_id'] = cid
        features['order_id'] = target_order_id
        features['dispatch_time'] = dispatch_time
        features[TARGET] = predict_grab_probability(order_obj, courier, dispatch_time)
        features['is_actual_grabbed'] = 1 if cid == actual_courier_id else 0
        records.append(features)

    samples = pd.DataFrame(records)
    samples = samples.sort_values(TARGET, ascending=False).reset_index(drop=True)

    # 打印统计
    actual_match = int(samples['is_actual_grabbed'].sum())
    top_k = min(5, len(samples))
    print(f'样本数: {len(samples)}，真实接单骑手匹配: {actual_match} / 1')
    print(f'Top-{top_k} 预测概率最高的骑手:')
    for i in range(top_k):
        row = samples.iloc[i]
        mark = ' ← 真实接单' if row['is_actual_grabbed'] else ''
        print(f'  {i+1}. 骑手 {row["courier_id"]} 概率={row[TARGET]:.4f}{mark}')

    return samples


# ============================================================
#  主入口
# ============================================================

def main():
    target_order_id = '522093'

    samples = build_sample_set(target_order_id)

    # 查看特征概况
    # print(f'\n特征概况:')
    # print(samples[FEATURE_NAMES].describe())

    # 保存
    output_dir = Path(__file__).resolve().parent / 'sample_sets'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'sample_{target_order_id}.csv'
    samples.to_csv(output_path, index=False)
    print(f'\n样本集已保存到: {output_path}')

    return samples


if __name__ == '__main__':
    result = main()
