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
output_dir = 'Q-values/Data/Data_Processed'
os.makedirs(output_dir, exist_ok=True)

# 保存清洗后的数据
output_path = os.path.join(output_dir, 'cleaned_waybill_info.csv')
df_cleaned.to_csv(output_path, index=False, encoding='utf-8')
print(f"\n清洗后的数据已保存至: {output_path}")

# # 显示数据统计信息
# print("\n数据统计信息:")
# print(df_cleaned.describe())