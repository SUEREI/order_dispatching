import pandas as pd
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from datetime import datetime, timezone, timedelta
from tqdm import tqdm

# 设置pandas显示选项，显示所有列
pd.set_option('display.max_columns', None)


def _weighted_fill_series(series, time_series):
    """
    用上下值的加权求和填充空值，权重为时间差的倒数
    对series中的每个空值，找到同组内最近上下非空值
    如果时间差小于180秒，则直接取小于一侧的值
    否则，以时间差倒数为权重做加权平均。
    """
    result = series.copy()
    mask = result.isna()
    if not mask.any():
        return result

    n = len(result)
    # 预先缓存所有非空值的索引，提升查找效率
    valid_idx = np.where(~mask.values)[0]

    if len(valid_idx) == 0:
        return result

    na_idx = np.where(mask.values)[0]

    for idx in na_idx:
        # 找到 idx 之前最近的 valid 索引
        left_candidates = valid_idx[valid_idx < idx]
        above_idx = left_candidates[-1] if len(left_candidates) > 0 else None

        # 找到 idx 之后最近的 valid 索引
        right_candidates = valid_idx[valid_idx > idx]
        below_idx = right_candidates[0] if len(right_candidates) > 0 else None

        if above_idx is not None and below_idx is not None:
            t_curr = time_series.iloc[idx]
            t_above = time_series.iloc[above_idx]
            t_below = time_series.iloc[below_idx]

            dt_above = abs(t_curr - t_above)
            dt_below = abs(t_curr - t_below)

            if dt_above < 180:
                result.iloc[idx] = series.iloc[above_idx]
            elif dt_below < 180:
                result.iloc[idx] = series.iloc[below_idx]
            else:
                w_above = 1.0 / dt_above
                w_below = 1.0 / dt_below
                result.iloc[idx] = (
                    series.iloc[above_idx] * w_above +
                    series.iloc[below_idx] * w_below
                ) / (w_above + w_below)
        elif above_idx is not None:
            result.iloc[idx] = series.iloc[above_idx]
        elif below_idx is not None:
            result.iloc[idx] = series.iloc[below_idx]

    return result


def position_extend(df):
    """估计派单时骑手的位置"""
    df_copy = df.copy()
    df_extend = df.copy()
    df_copy.loc[df_copy['grab_time'] == 0, 'grab_time'] = np.nan
    mask = df_copy['grab_time'].notna()
    # 用.loc给副本赋值（仅修改mask为True的行）
    df_copy.loc[mask, 'dispatch_time'] = df_copy.loc[mask, 'grab_time']
    type_mapping = {
        "grab": ("dispatch_time", "grab_lng", "grab_lat"),
        "fetch": ("fetch_time", "sender_lng", "sender_lat"),
        "recipient": ("arrive_time", "recipient_lng", "recipient_lat")
    }
    long_data_list = []
    for type_name, (time_col, lng_col, lat_col) in type_mapping.items():
        # 提取当前type所需的列：waybill_id + 对应time/lng/lat
        temp_df = df_copy[["waybill_id", "dt", "courier_id", time_col, lng_col, lat_col]].copy()
        # 重命名为统一列名：time/lng/lat
        temp_df.rename(
            columns={
                time_col: "time",
                lng_col: "lng",
                lat_col: "lat"
            },
            inplace=True
        )
        # 添加type列标识当前类型
        temp_df["type"] = type_name
        long_data_list.append(temp_df)
    long_df = pd.concat(long_data_list, ignore_index=True)
    # 将time列的0值替换为空值
    long_df.loc[long_df['time'] == 0, 'time'] = np.nan
    # 去除time列空值的行
    long_df = long_df.dropna(axis=0, subset=["time"])
    # 将经纬度为0的值转换为空值，便于后续空值填充操作
    long_df.loc[long_df['lng'] == 0, ['lng', 'lat']] = np.nan
    # 按时间排序
    long_df = long_df.sort_values(by=['dt', 'courier_id', 'time'], ignore_index=True)
    # 分组后按时间差倒数加权填充经纬度空值
    long_df[['lng', 'lat']] = (
        long_df.groupby(['dt', 'courier_id'], group_keys=False)
        .apply(lambda g: g[['lng', 'lat']].apply(
            lambda col: _weighted_fill_series(col, g['time'])
        ), include_groups=False)
    )
    long_df_grab = long_df[long_df['type'] == 'grab']
    # 合并到原数据表
    grab_merge_data = long_df_grab[['waybill_id', 'lng', 'lat']].rename(
        columns={
            'lng': 'dispatch_lng',  # 重命名为目标列名
            'lat': 'dispatch_lat'
        }
    )
    df_extend = df_extend.merge(
        grab_merge_data,
        on='waybill_id',  # 关联键
        how='left'  # 左合并：原df所有行保留，匹配不到则dispatch_lng/lat为NaN
    )
    print(f'骑手位置估计完成')
    return df_extend


def time_convert(df, time_columns):
    """将时间戳转换为东八区datetime格式"""
    df_cleaned = df.copy()
    # 东八区时区
    tz_east8 = timezone(timedelta(hours=8))
    # 转换时间戳为东八区datetime
    for col in time_columns:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col], utc=True)
        df_cleaned[col] = df_cleaned[col].dt.tz_convert(tz_east8)
        # 去除时区信息，只保留datetime
        df_cleaned[col] = df_cleaned[col].dt.tz_localize(None)
    print(f"\n已转换 {len(time_columns)} 个时间字段为东八区datetime格式")
    return df_cleaned


def add_ongoing_set(df):
    """计算派单时骑手正在处理的其他订单集合"""
    df_result = df.copy()
    df_result['ongoing_order_set'] = None
    
    for cid, group in df_result.groupby('courier_id', sort=False):
        group_sorted = group.sort_values('dispatch_time')

        ongoing_set = set()
        events = []

        for idx, row in group_sorted.iterrows():
            events.append((row['dispatch_time'], 0, idx, row['waybill_id']))
            if pd.notna(row['grab_time']):
                events.append((row['grab_time'], 1, idx, row['waybill_id']))
            if pd.notna(row['arrive_time']):
                events.append((row['arrive_time'], 2, idx, row['waybill_id']))

        events.sort(key=lambda x: (x[0], x[1]))

        for event_time, event_type, idx, waybill_id in events:
            if event_type == 1:
                ongoing_set.add(waybill_id)
            elif event_type == 2:
                ongoing_set.discard(waybill_id)
            else:
                df_result.at[idx, 'ongoing_order_set'] = ongoing_set.copy() if ongoing_set else {}

    print(f'\n骑手在手订单集合计算完成')
    return df_result


def drop_na_rows(df, column):
    """针对某列去除空值行，输出去除前和去除后的行数，以及去除数量"""
    rows_before = len(df)
    df_cleaned = df.dropna(axis=0, subset=[column])
    rows_after = len(df_cleaned)
    rows_dropped = rows_before - rows_after
    print(f"\n列 '{column}' 空值行去除统计:")
    print(f"  去除前行数: {rows_before}")
    print(f"  去除后行数: {rows_after}")
    print(f"  去除数量: {rows_dropped}")
    return df_cleaned


def courier_fatigue(df, idle_timeout_minutes=30, buffer_time=5):
    """
    计算骑手疲劳度：每单 dispatch_time 时，骑手当前连续在线时长（分钟）
    逻辑：对每个骑手构建上线时间点序列，对于每单派单时间，找到最近一次上线时间，
          dispatch_time - 最近上线时间 = 当前连续在线时长
    """
    df_result = df.copy()
    df_result['fatigue'] = 0.0

    idle_timeout = timedelta(minutes=idle_timeout_minutes)
    buffer = timedelta(minutes=buffer_time)

    df_result = df_result.sort_values(['courier_id', 'dispatch_time'])

    # 计算疲劳度
    courier_ids = df_result['courier_id'].nunique()
    for courier_id, group in tqdm(df_result.groupby('courier_id'),
                                  total=courier_ids, desc='计算疲劳度'):
        indices = group.index.tolist()
        dts = group['dispatch_time'].values
        ats = group['arrive_time'].values
        n = len(group)

        # 构建该骑手的所有上线时间点（有序）
        epoch = pd.Timestamp('1970-01-01 08:00:00')
        online_times = [pd.Timestamp(dts[0]) - buffer]

        for i in range(1, n):
            cur_dispatch = pd.Timestamp(dts[i])
            prev_arrives = [pd.Timestamp(ats[j])
                            if pd.Timestamp(ats[j]) != epoch
                            else pd.Timestamp(dts[j])
                            for j in range(i) ]
            if not prev_arrives:
                online_times.append(cur_dispatch - buffer)
                continue
            last_arrive = max(prev_arrives)

            if cur_dispatch - last_arrive > idle_timeout:
                online_times.append(cur_dispatch - buffer)

        # 对每单，找到最近一次上线时间，计算连续在线时长
        ptr = 0
        for i in range(n):
            dt = pd.Timestamp(dts[i])
            while ptr + 1 < len(online_times) and online_times[ptr + 1] <= dt:
                ptr += 1
            online_so_far = (dt - online_times[ptr]).total_seconds() / 60
            df_result.loc[indices[i], 'fatigue'] = online_so_far

    print(f'\n已完成疲劳度的计算')

    return df_result


def _haversine_np(lng1, lat1, lng2, lat2):
    """numpy 向量化版 haversine 距离计算（米）"""
    lng1, lat1, lng2, lat2 = map(np.radians, [lng1, lat1, lng2, lat2])
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371000 * c * 1.4 / 1000


def delivery_distance(df, type_1, type_2):
    """
    计算两个位置类型间的配送距离（向量化）
    """
    COORD_SCALE = 10 ** 6
    df_result = df.copy()
    col_name = f'{type_1}_to_{type_2}_delivery_distance'
    df_result[col_name] = 0.0

    lng1 = df_result[f'{type_1}_lng']
    lat1 = df_result[f'{type_1}_lat']
    lng2 = df_result[f'{type_2}_lng']
    lat2 = df_result[f'{type_2}_lat']
    mask = lng1.notna() & lat1.notna() & lng2.notna() & lat2.notna()

    if mask.any():
        df_result.loc[mask, col_name] = _haversine_np(
            lng1[mask] / COORD_SCALE,
            lat1[mask] / COORD_SCALE,
            lng2[mask] / COORD_SCALE,
            lat2[mask] / COORD_SCALE,
        )

    return df_result


def filter_by_speed(df, speed_threshold):
    """
    根据配送速度过滤行数据
    去除 dispatch_to_sender_delivery_distance/(fetch_time-dispatch_time) 大于阈值的行
    speed_threshold: 速度阈值（单位：公里/分钟）
    """
    df_result = df.copy()
    rows_before = len(df_result)
    
    # 计算时间差（小时）
    time_diff = (df_result['fetch_time'] - df_result['dispatch_time']).dt.total_seconds() / 3600
    
    # 计算速度（公里/小时）
    speed = df_result['dispatch_to_sender_delivery_distance'] / time_diff
    
    # 过滤条件：速度不超过阈值
    mask = (speed <= speed_threshold)
    df_result = df_result[mask].copy()
    
    rows_after = len(df_result)
    rows_dropped = rows_before - rows_after
    
    print(f"\n速度过滤统计 (阈值: {speed_threshold} 公里/分钟):")
    print(f"  过滤前行数: {rows_before}")
    print(f"  过滤后行数: {rows_after}")
    print(f"  过滤数量: {rows_dropped}")
    
    return df_result


def parse_set_string(x):
    """将字符串形式的集合转换为真正的集合"""
    if isinstance(x, set):
        return x
    if pd.isna(x) or x is None:
        return set()
    if isinstance(x, str):
        # 去除空格，处理 {} 和 {1,2} 这样的形式
        x = x.strip()
        if x == '{}' or x == 'set()':
            return set()
        # 去除花括号
        x = x.strip('{}')
        if x == '':
            return set()
        # 分割并转换为整数集合
        try:
            return set(int(i.strip()) for i in x.split(','))
        except ValueError:
            return set()
    return set()


def count_ongoing_orders(df):
    """
    计算字段ongoing_order_set集合中的元素个数
    新增列'ongoing_order_count'表示集合中的订单数量
    列里面的数据是{}, {1,2}这样的形式
    """
    df_result = df.copy()
    
    # 先把数据转换为集合
    df_result['ongoing_order_set'] = df_result['ongoing_order_set'].apply(parse_set_string)
    df_result['ongoing_order_count'] = df_result['ongoing_order_set'].apply(len)
    print(f'\n已完成ongoing_order_set集合元素个数的计算')
    return df_result


def max_ongoing_distance(df):
    """
    计算 ongoing_order_set 中所有订单的 sender_to_recipient_delivery_distance 的最大值
    结果存入新列 max_ongoing_sender_to_recipient_distance
    """
    df_result = df.copy()

    if 'ongoing_order_set' not in df_result.columns:
        print('警告: 数据中不存在 ongoing_order_set 列')
        df_result['max_ongoing_sender_to_recipient_distance'] = 0.0
        return df_result

    df_result['ongoing_order_set'] = df_result['ongoing_order_set'].apply(parse_set_string)

    distance_map = df_result.set_index('waybill_id')['sender_to_recipient_delivery_distance'].to_dict()

    def max_distance(waybill_set):
        if not waybill_set:
            return 0.0
        distances = [distance_map.get(wid, 0.0) for wid in waybill_set]
        return max(distances) if distances else 0.0

    df_result['max_ongoing_sender_to_recipient_distance'] = (
        df_result['ongoing_order_set'].apply(max_distance)
    )

    print(f'\n已完成在手订单最大配送距离的计算')
    return df_result


def daily_avg_rejection(df, group_col, dt_col='dt', target_col='is_courier_grabbed'):
    """
    计算不同实体（骑手/商家）的日均拒单数量和日均拒单率，合并到原表

    拒单定义: target_col == 0（骑手未抢单，即系统派单）

    Parameters
    ----------
    df : DataFrame
        输入数据，需包含 group_col、dt_col、target_col 三列
    group_col : str
        分组列名，如 'courier_id'（骑手）或 'poi_id'（商家）
    dt_col : str
        日期列名，默认 'dt'
    target_col : str
        目标列名（是否抢单），默认 'is_courier_grabbed'

    Returns
    -------
    DataFrame
        原表基础上新增两列:
        {group_col}_daily_avg_rejected_count
        {group_col}_daily_avg_rejected_rate
    """
    df_result = df.copy()

    daily_stats = (
        df_result.groupby([group_col, dt_col])
        .agg(
            total_orders=(target_col, 'count'),
            rejected_orders=(target_col, lambda x: (x == 0).sum()),
        )
        .reset_index()
    )

    summary = (
        daily_stats.groupby(group_col)
        .agg(
            total_orders=('total_orders', 'sum'),
            rejected_orders=('rejected_orders', 'sum'),
            active_days=(dt_col, 'nunique'),
        )
        .reset_index()
    )

    summary[f'{group_col}_daily_avg_rejected_count'] = summary['rejected_orders'] / summary['active_days']
    summary[f'{group_col}_daily_avg_rejected_rate'] = summary['rejected_orders'] / summary['total_orders']

    df_result = df_result.merge(
        summary[[group_col, f'{group_col}_daily_avg_rejected_count', f'{group_col}_daily_avg_rejected_rate']],
        on=group_col,
        how='left'
    )

    print(f'\n[{group_col}] 日均拒单统计已合并到原表')
    return df_result


def calculate_grab_cof(df):
    """计算接单系数"""
    df_result = df.copy()
    
    # 计算时间差（小时）
    time_diff_hours = (df_result['arrive_time'] - df_result['grab_time']).dt.total_seconds() / 3600
    
    # 计算速度（小时倒数），避免除以零
    grab_cof = np.where(time_diff_hours > 0, 1.0 / time_diff_hours, 0.0)
    
    # 若 is_courier_grabbed 为 0，则结果为 0
    df_result['grab_cof'] = np.where(
        df_result['is_courier_grabbed'] == 1,
        grab_cof,
        0.0
    )
    
    print(f'\n已完成 grab_cof 的计算')
    return df_result


def save_csv(df, output_dir, df_name):
    """保存数据"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存数据
    output_path = os.path.join(output_dir, df_name)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n数据已保存至: {output_path}")


def main():
    # 读取数据
    df = pd.read_csv(r'SPOT/Data/Data_Processed/cleaned_waybill_info.csv')
    
    # 扩展位置信息
    if 'dispatch_lat' not in df.columns or 'dispatch_lng' not in df.columns:
        df = position_extend(df)

    # 转换时间格式
    time_columns = [
        'estimate_arrived_time',
        'dispatch_time',
        'grab_time',
        'fetch_time',
        'arrive_time',
        'estimate_meal_prepare_time',
        'order_push_time',
        'platform_order_time'
    ]
    df = time_convert(df, time_columns)

    # 删除包含缺失值的行
    df = drop_na_rows(df, 'dispatch_lat')

    # 计算派单时骑手正在处理的其他订单集合
    if 'ongoing_order_set' not in df.columns:
        df = add_ongoing_set(df)

    # 计算骑手疲劳度
    if 'fatigue' not in df.columns:
        df = courier_fatigue(df)

    # 计算距离
    if 'sender_to_recipient_delivery_distance' not in df.columns:
        df = delivery_distance(df, 'dispatch', 'sender')
    if 'recipient_to_sender_delivery_distance' not in df.columns:
        df = delivery_distance(df, 'sender', 'recipient')

    # 过滤配送速度
    # if 'delivery_speed' not in df.columns:
    #     df = filter_by_speed(df, speed_threshold=30)

    # 计算ongoing_order_set集合中的元素个数
    if 'ongoing_order_count' not in df.columns:
        df = count_ongoing_orders(df)

    # 计算在手订单最大配送距离
    if 'max_ongoing_sender_to_recipient_distance' not in df.columns:
        df = max_ongoing_distance(df)

    # 计算骑手接单到送达的速度
    if 'grab_cof' not in df.columns:
        df = calculate_grab_cof(df)

    # 计算骑手日均拒单数量和日均拒单率
    if 'courier_daily_avg_rejected_count' not in df.columns or 'courier_daily_avg_rejected_rate' not in df.columns:
        df = daily_avg_rejection(df, group_col='courier_id')

    # 计算商家日均拒单数量和日均拒单率
    if 'poi_daily_avg_rejected_count' not in df.columns or 'poi_daily_avg_rejected_rate' not in df.columns:
        df = daily_avg_rejection(df, group_col='poi_id')

    print(df.head())
    
    # 保存数据
    output_dir = r'SPOT/Data/Data_Processed'
    save_csv(df, output_dir, 'cleaned_waybill_info.csv')



if __name__ == '__main__':
    main()
