import pandas as pd
from tqdm import tqdm


def load_behavior_data(csv_path):
    """加载骑手行为数据"""
    df = pd.read_csv(csv_path)
    return df


def process_courier_ongoing_orders(df):
    """处理骑手数据，记录每个查询时刻的现有订单id"""
    
    results = []
    
    # 按骑手分组处理
    for courier_id, group in df.groupby('courier_id'):
        group_sorted = group.sort_values('dispatch_time').copy()
        
        ongoing_set = set()  # 当前进行中的订单集合
        events = []  # 事件列表: (时间, 事件类型优先级, 订单索引, 事件类型)
        
        # 构建事件列表
        for idx, row in group_sorted.iterrows():
            # 查询事件（派单时）
            events.append((row['dispatch_time'], 1, idx, 'query'))
            # 抢单事件
            if pd.notna(row['grab_time']):
                events.append((row['grab_time'], 0, idx, 'grab'))
            # 送达事件
            if pd.notna(row['arrive_time']):
                events.append((row['arrive_time'], -1, idx, 'arrive'))
        
        # 按时间排序，时间相同按事件类型优先级排序（grab在query前，arrive在grab前）
        events.sort(key=lambda x: (x[0], -x[1]))
        
        # 创建订单索引到原始数据的映射
        order_lookup = group_sorted.reset_index().to_dict('index')
        index_to_original_idx = {i: idx for i, idx in enumerate(group_sorted.index)}
        
        # 处理事件
        for event_time, event_priority, order_idx, event_kind in tqdm(events, desc=f"处理骑手 {courier_id} 的事件"):
            original_order_idx = index_to_original_idx.get(order_idx, order_idx)
            
            if event_kind == 'grab':
                # 抢单：加入进行中的订单集合
                ongoing_set.add(original_order_idx)
            elif event_kind == 'arrive':
                # 送达：从进行中的订单集合移除
                ongoing_set.discard(original_order_idx)
            else:
                # 查询事件：记录当前进行中的订单id列表
                results.append({
                    'courier_id': courier_id,
                    'order_idx': original_order_idx,
                    'dispatch_time': event_time,
                    'ongoing_order_ids': list(ongoing_set).copy()
                })
    
    # 构建结果DataFrame
    result_df = pd.DataFrame(results)
    return result_df


def main():
    csv_path = r'Q_Values\Data\Data_Processed\cleaned_waybill_info_behavior.csv'
    df = load_behavior_data(csv_path)
    # print(df.head())
    


if __name__ == '__main__':
    main()
