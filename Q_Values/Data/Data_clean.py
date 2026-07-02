import pandas as pd
import os
from datetime import datetime, timezone, timedelta
# 设置pandas显示选项，显示所有列
pd.set_option('display.max_columns', None)

# 数据清洗脚本

# 读取原始数据
print("正在读取原始数据...")
input_path = '0MeiTuan_Data/all_waybill_info_meituan_0322.csv'
df = pd.read_csv(input_path)
print(f"原始数据形状: {df.shape}")
print(f"原始数据记录数: {len(df)}")


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


def add_prefer_reject_label(df):
    """
    添加骑手和商家是否偏好拒单的标签（1/0）

    判断逻辑：对骑手和商家分别计算各自的实体平均拒单率，
              若该实体拒单率高于实体平均拒单率，则为1，否则为0
    """
    df_result = df.copy()

    # 骑手偏好拒单标签
    if 'courier_id_daily_avg_rejected_rate' in df_result.columns:
        courier_rates = df_result[['courier_id', 'courier_id_daily_avg_rejected_rate']].drop_duplicates()
        courier_avg_rate = courier_rates['courier_id_daily_avg_rejected_rate'].mean()
        courier_rates['courier_prefer_reject'] = (
            courier_rates['courier_id_daily_avg_rejected_rate'] > 3 * courier_avg_rate
        ).astype(int)
        df_result = df_result.merge(
            courier_rates[['courier_id', 'courier_prefer_reject']],
            on='courier_id',
            how='left'
        )
        prefer_count = courier_rates['courier_prefer_reject'].sum()
        total_couriers = len(courier_rates)
        print(f'骑手平均拒单率: {courier_avg_rate:.4f}')
        print(f'骑手偏好拒单标签: {prefer_count}/{total_couriers} 个骑手标记为1')

    # 商家偏好拒单标签
    if 'poi_id_daily_avg_rejected_rate' in df_result.columns:
        poi_rates = df_result[['poi_id', 'poi_id_daily_avg_rejected_rate']].drop_duplicates()
        poi_avg_rate = poi_rates['poi_id_daily_avg_rejected_rate'].mean()
        poi_rates['poi_prefer_reject'] = (
            poi_rates['poi_id_daily_avg_rejected_rate'] > 3 * poi_avg_rate
        ).astype(int)
        df_result = df_result.merge(
            poi_rates[['poi_id', 'poi_prefer_reject']],
            on='poi_id',
            how='left'
        )
        prefer_count = poi_rates['poi_prefer_reject'].sum()
        total_pois = len(poi_rates)
        print(f'商家平均拒单率: {poi_avg_rate:.4f}')
        print(f'商家偏好拒单标签: {prefer_count}/{total_pois} 个商家标记为1')

    return df_result


# 计算日均拒单统计
df = daily_avg_rejection(df, 'courier_id')
df = daily_avg_rejection(df, 'poi_id')
df = add_prefer_reject_label(df)

# 1. 去除is_courier_grabbed为0的字段
print("\n正在进行数据清洗...")
before_count = len(df)
df_cleaned = df[df['is_courier_grabbed'] != 0].copy()
after_grabbed = len(df_cleaned)
print(f"去除is_courier_grabbed为0的记录: {before_count - after_grabbed} 条")

# 1. 去除arrive_time小于等于dispatch_time的数据
before_count = len(df_cleaned)
df_cleaned = df_cleaned[df_cleaned['arrive_time'] > df_cleaned['dispatch_time']].copy()
after_time_filter = len(df_cleaned)
print(f"去除arrive_time <= dispatch_time的记录: {before_count - after_time_filter} 条")

# 2. 去除空值字段
before_count = len(df_cleaned)
df_cleaned = df_cleaned.dropna()
after_dropna = len(df_cleaned)
print(f"去除空值记录: {before_count - after_dropna} 条")

# 3. 将时间戳转换为东八区datetime格式
print("\n正在转换时间格式...")
# 定义时间字段
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

# 东八区时区
tz_east8 = timezone(timedelta(hours=8))

# 转换时间戳为东八区datetime
for col in time_columns:
    df_cleaned[col] = pd.to_datetime(df_cleaned[col], unit='s', utc=True)
    df_cleaned[col] = df_cleaned[col].dt.tz_convert(tz_east8)
    # 去除时区信息，只保留datetime
    df_cleaned[col] = df_cleaned[col].dt.tz_localize(None)

print(f"已转换 {len(time_columns)} 个时间字段为东八区datetime格式")


def round_dispatch_time(dt):
    """
    将dispatch_time转换为以30s或00s结尾的时间
    向后推到最近的30秒或00秒时刻
    
    Args:
        dt: datetime对象
        
    Returns:
        datetime: 转换后的时间
        
    Examples:
        0:12 -> 0:30
        0:45 -> 1:00
        0:30 -> 0:30 (不变)
        0:00 -> 0:00 (不变)
    """
    second = dt.second
    
    if second == 0 or second == 30:
        # 已经是00s或30s结尾，不变
        return dt
    elif second < 30:
        # 向后推到30s
        return dt.replace(second=30, microsecond=0)
    else:
        # 向后推到下一分钟的00s
        return (dt + timedelta(minutes=1)).replace(second=0, microsecond=0)


# 处理dispatch_time字段，应用round_dispatch_time函数
print("\n正在处理dispatch_time,arrive_time字段...")
df_cleaned['dispatch_time'] = df_cleaned['dispatch_time'].apply(round_dispatch_time)
df_cleaned['arrive_time'] = df_cleaned['arrive_time'].apply(round_dispatch_time)
print("dispatch_time,arrive_time字段处理完成")

# 输出清洗结果
print("\n" + "="*50)
print("数据清洗结果:")
print("="*50)
print(f"原始数据记录数: {len(df)}")
print(f"清洗后数据记录数: {len(df_cleaned)}")
print(f"总共删除记录数: {len(df) - len(df_cleaned)}")
print(f"数据保留比例: {len(df_cleaned)/len(df)*100:.2f}%")
print(f"清洗后数据形状: {df_cleaned.shape}")

# 确保输出目录存在
output_dir = 'Q_values/Data/Data_Processed'
os.makedirs(output_dir, exist_ok=True)

# 保存清洗后的数据
output_path = os.path.join(output_dir, 'cleaned_waybill_info.csv')
df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n清洗后的数据已保存至: {output_path}")

# # 显示数据统计信息
# print("\n数据统计信息:")
# print(df_cleaned.describe())