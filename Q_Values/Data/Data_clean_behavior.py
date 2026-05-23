import pandas as pd
import os
import numpy as np
from datetime import datetime, timezone, timedelta
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


def time_convert(df, time_columns):
    """将时间戳转换为东八区datetime格式"""
    df_cleaned = df.copy()
    # 东八区时区
    tz_east8 = timezone(timedelta(hours=8))
    # 转换时间戳为东八区datetime
    for col in time_columns:
        df_cleaned[col] = pd.to_datetime(df_cleaned[col], unit='s', utc=True)
        df_cleaned[col] = df_cleaned[col].dt.tz_convert(tz_east8)
        # 去除时区信息，只保留datetime
        df_cleaned[col] = df_cleaned[col].dt.tz_localize(None)
    print(f"\n已转换 {len(time_columns)} 个时间字段为东八区datetime格式")
    return df_cleaned


def save_csv(df, output_dir, df_name):
    """保存数据"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    # 保存数据
    output_path = os.path.join(output_dir, df_name)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n数据已保存至: {output_path}")

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


def main():
    """主函数"""
    # 读取原始数据
    df = pd.read_csv(r'0MeiTuan_Data/all_waybill_info_meituan_0322.csv')

    # 估计派单时骑手的位置
    df_extend = position_extend(df)

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
    df_cleaned = time_convert(df_extend, time_columns)

    # 去除空值行
    df_cleaned = drop_na_rows(df_cleaned, 'dispatch_lng')
    df_cleaned = drop_na_rows(df_cleaned, 'dispatch_lat')

    # 计算派单时骑手正在处理的其他订单集合
    df_result = add_ongoing_set(df_cleaned)

    # 保存数据
    output_dir = r'Q_values/Data/Data_Processed'
    save_csv(df_result, output_dir, 'cleaned_waybill_info_behavior.csv')



if __name__ == '__main__':
    main()
