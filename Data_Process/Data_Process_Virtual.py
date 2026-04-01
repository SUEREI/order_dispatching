import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import tqdm
import ast
from scipy.spatial import ConvexHull
from pyproj import Transformer
import datetime
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_columns', None)


def load_data(file_path, dtype=None):
    """
    加载数据
    :param file_path: 数据文件路径
    :param dtype: 数据类型字典，用于优化内存使用
    :return: 数据框
    """
    df = pd.read_csv(file_path, dtype=dtype)
    print(f'加载数据完成，共 {len(df)} 条记录，数据形状为 {df.shape}')
    print('-----------------')
    return df


class MergeOrderInfo:
    """
    步骤1: 根据order_id合并数据，从all_waybill_info复制字段到dispatch_waybill
    """
    def __init__(self, df_waybill, df_all_info):
        """
        初始化合并类
        :param df_waybill: dispatch_waybill_meituan数据框
        :param df_all_info: all_waybill_info_meituan_0322数据框
        """
        self.df_waybill = df_waybill
        self.df_all_info = df_all_info
        self.utm_transformer = Transformer.from_crs('epsg:4326', 'epsg:32660')

    def merge_order_info(self):
        """
        根据order_id合并数据
        对于all_waybill_info中order_id不唯一的情况，取dispatch_time更早的记录
        :return: 合并后的数据框
        """
        print('开始合并订单信息...')
        copy_cols = [
            'da_id', 'is_courier_grabbed', 'is_weekend', 'estimate_arrived_time',
            'is_prebook', 'poi_id', 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat',
            'dispatch_time', 'estimate_meal_prepare_time', 'order_push_time', 'platform_order_time',
            'waybill_id', 'courier_id', 'grab_time', 'fetch_time', 'arrive_time', 'grab_lng', 'grab_lat'
        ]
        df_all_sorted = self.df_all_info.sort_values(['order_id', 'dispatch_time'])
        df_all_unique = df_all_sorted.drop_duplicates(subset=['order_id'], keep='first')
        df_all_subset = df_all_unique[['order_id'] + [col for col in copy_cols if col in df_all_unique.columns]]
        df_merged = self.df_waybill.merge(df_all_subset, on='order_id', how='left', suffixes=('', '_info'))
        
        df_merged['dispatch_to_estimateArrived_tm'] = (
            df_merged['estimate_arrived_time'] - df_merged['dispatch_time']
        ) / 3600  # 直接相减（都是秒时间戳），转换为小时
        
        # 检查字段是否存在
        # print('字段检查:')
        # print('estimate_arrived_time 存在:', 'estimate_arrived_time' in df_merged.columns)
        # print('dispatch_time 存在:', 'dispatch_time' in df_merged.columns)
        # print('dispatch_to_estimateArrived_tm 存在:', 'dispatch_to_estimateArrived_tm' in df_merged.columns)
        # if 'dispatch_to_estimateArrived_tm' in df_merged.columns:
        #     print('dispatch_to_estimateArrived_tm 前5行:', df_merged['dispatch_to_estimateArrived_tm'].head())
        
        print(f'订单信息合并完成，数据形状为 {df_merged.shape}')
        print('-----------------')
        return df_merged


class CalculateRiderOngoingFeatures:
    """
    步骤2: 计算dispatch_rider表中骑手现有订单的统计信息
    """
    COORDINATE_SCALE = 10**6
    EARTH_RADIUS_KM = 6371

    def __init__(self, df_rider, df_all_info, df_all_info_with_stats):
        """
        初始化骑手现有订单特征计算类
        :param df_rider: dispatch_rider_meituan数据框
        :param df_all_info: all_waybill_info_meituan_0322数据框
        :param df_all_info_with_stats: 包含累计统计的订单信息数据框
        """
        self.df_rider = df_rider
        self.df_all_info = df_all_info
        self.df_all_info_with_stats = df_all_info_with_stats
        self.waybill_lookup = self._build_waybill_lookup()
        self.utm_transformer = Transformer.from_crs('epsg:4326', 'epsg:32660')

    def _build_waybill_lookup(self):
        """
        构建waybill_id到订单信息的查找字典
        :return: 字典
        """
        print('构建waybill_id查找字典...')
        lookup_cols = ['waybill_id', 'da_id', 'sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat']
        df_subset = self.df_all_info[lookup_cols].drop_duplicates(subset=['waybill_id'])
        return df_subset.set_index('waybill_id').to_dict('index')

    def _parse_waybills(self, waybills_str):
        """
        解析courier_waybills字符串为列表
        :param waybills_str: 字符串格式的列表
        :return: 列表
        """
        try:
            if pd.isna(waybills_str):
                return []
            if isinstance(waybills_str, str):
                return ast.literal_eval(waybills_str)
            return list(waybills_str)
        except:
            return []

    def _cal_distance_km(self, lng1, lat1, lng2, lat2):
        """
        计算两点之间的距离（公里）
        :param lng1, lat1: 点1经纬度
        :param lng2, lat2: 点2经纬度
        :return: 距离（公里）
        """
        if pd.isna(lng1) or pd.isna(lat1) or pd.isna(lng2) or pd.isna(lat2):
            return np.nan
        lng1_rad, lat1_rad = np.radians(lng1 / self.COORDINATE_SCALE), np.radians(lat1 / self.COORDINATE_SCALE)
        lng2_rad, lat2_rad = np.radians(lng2 / self.COORDINATE_SCALE), np.radians(lat2 / self.COORDINATE_SCALE)
        d_lon, d_lat = lng2_rad - lng1_rad, lat2_rad - lat1_rad
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
        return 2 * self.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def _cal_area_km2(self, points):
        """
        计算经纬度点集围成的区域面积（平方公里）- 使用UTM投影+最小凸包
        :param points: 点集数组，每个点为 (经度, 纬度)
        :return: 面积（平方公里）
        """
        if len(points) < 3:
            return 0.0
        
        def lnglat_to_utm(lng, lat):
            x, y = self.utm_transformer.transform(lat, lng)
            return x, y
        
        points_array = np.array(points)
        lng = points_array[:, 0] / self.COORDINATE_SCALE
        lat = points_array[:, 1] / self.COORDINATE_SCALE
        
        utm_coords = np.array([lnglat_to_utm(l, b) for l, b in zip(lng, lat)])
        
        try:
            hull = ConvexHull(utm_coords)
            area = hull.volume
        except Exception:
            return 0.0
        
        return round(area / 1e6, 4)

    def calculate_rider_ongoing_features(self):
        """
        计算骑手现有订单的统计信息
        :return: 包含新特征的数据框
        """
        print('开始计算骑手现有订单统计信息...')
        
        # 向量化解析waybills
        self.df_rider['waybills_list'] = self.df_rider['courier_waybills'].apply(self._parse_waybills)
        
        # 计算订单数量
        self.df_rider['ongoing_order_count'] = self.df_rider['waybills_list'].apply(len)
        
        # 计算其他特征
        def calculate_features(waybills):
            da_ids = set()
            max_distance = 0.0
            points = []
            for wid in waybills:
                info = self.waybill_lookup.get(wid)
                if info:
                    da_id = info.get('da_id')
                    if pd.notna(da_id):
                        da_ids.add(da_id)
                    sender_lng, sender_lat = info.get('sender_lng'), info.get('sender_lat')
                    recipient_lng, recipient_lat = info.get('recipient_lng'), info.get('recipient_lat')
                    if pd.notna(sender_lng) and pd.notna(sender_lat) and pd.notna(recipient_lng) and pd.notna(recipient_lat):
                        dist = self._cal_distance_km(sender_lng, sender_lat, recipient_lng, recipient_lat)
                        if pd.notna(dist):
                            max_distance = max(max_distance, dist)
                        points.append([sender_lng, sender_lat])
                        points.append([recipient_lng, recipient_lat])
            cross_area_count = len(da_ids)
            area_km2 = self._cal_area_km2(points)
            return cross_area_count, round(max_distance, 2), area_km2, points
        
        # 应用计算函数
        features = self.df_rider['waybills_list'].apply(calculate_features)
        features_df = pd.DataFrame(features.tolist(), columns=['ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2', 'ongoing_points_list'])
        
        # 合并结果
        for col in features_df.columns:
            self.df_rider[col] = features_df[col]
        
        # 清理临时列
        self.df_rider.drop('waybills_list', axis=1, inplace=True)
        
        # 计算骑手的累计统计特征
        self._calculate_courier_cumulative_stats()
        
        print(f'骑手现有订单特征计算完成，数据形状为 {self.df_rider.shape}')
        print('-----------------')
        return self.df_rider
    
    def _calculate_courier_cumulative_stats(self):
        """
        计算骑手的累计统计特征并添加到骑手表中（使用searchsorted优化）
        """
        print('开始计算骑手累计统计特征...')
        
        # 初始化累计统计字段
        self.df_rider['courier_daily_completed_count'] = 0
        self.df_rider['courier_daily_completed_distance_km'] = 0.0
        self.df_rider['courier_session_completed_count'] = 0
        self.df_rider['courier_session_completed_distance_km'] = 0.0
        self.df_rider['area_familiarity'] = 0.0
        
        # 按骑手分组处理
        for courier_id, courier_rider_group in tqdm.tqdm(
            self.df_rider.groupby('courier_id', sort=False),
            desc="计算骑手累计统计特征",
            unit="骑手"
        ):
            # 获取该骑手的订单信息
            courier_orders = self.df_all_info_with_stats[
                self.df_all_info_with_stats['courier_id'] == courier_id
            ].copy()
            
            if courier_orders.empty:
                continue
            
            # 按日期分组
            for dt, dt_orders in courier_orders.groupby('dt'):
                # 按dispatch_time_ts排序
                dt_orders = dt_orders.sort_values('dispatch_time_ts')
                
                # 获取该骑手在该日期的记录索引
                rider_indices = self.df_rider[
                    (self.df_rider['courier_id'] == courier_id) & 
                    (self.df_rider['dt'] == dt)
                ].index
                
                if rider_indices.empty:
                    continue
                
                # 获取骑手的dispatch_time
                rider_dispatch_times = self.df_rider.loc[rider_indices, 'dispatch_time'].values
                
                # 使用searchsorted快速找到对应的订单索引
                order_dispatch_times = dt_orders['dispatch_time_ts'].values
                order_indices = np.searchsorted(order_dispatch_times, rider_dispatch_times, side='right') - 1
                
                # 为每个骑手记录获取对应的累计统计值
                for i, (rider_idx, order_idx) in enumerate(zip(rider_indices, order_indices)):
                    if order_idx >= 0:
                        latest_order = dt_orders.iloc[order_idx]
                        self.df_rider.loc[rider_idx, 'courier_daily_completed_count'] = latest_order.get('courier_daily_completed_count', 0)
                        self.df_rider.loc[rider_idx, 'courier_daily_completed_distance_km'] = latest_order.get('courier_daily_completed_distance_km', 0.0)
                        self.df_rider.loc[rider_idx, 'courier_session_completed_count'] = latest_order.get('courier_session_completed_count', 0)
                        self.df_rider.loc[rider_idx, 'courier_session_completed_distance_km'] = latest_order.get('courier_session_completed_distance_km', 0.0)
                        self.df_rider.loc[rider_idx, 'area_familiarity'] = latest_order.get('area_familiarity', 0.0)
        
        print('骑手累计统计特征计算完成')


class CalculateCourierCumulativeStats:
    """
    步骤3: 计算骑手累计骑行距离和工作窗口内累计统计、区域熟悉度
    """
    TIME_ZONE_OFFSET = 8
    COORDINATE_SCALE = 10**6
    EARTH_RADIUS_KM = 6371

    def __init__(self, df_all_info):
        """
        初始化累计统计计算类
        :param df_all_info: all_waybill_info_meituan_0322数据框
        """
        self.df = df_all_info.copy()

    def _time_convert(self):
        """
        转换时间列
        """
        # 保留原始时间戳
        if 'dispatch_time_ts' not in self.df.columns:
            self.df['dispatch_time_ts'] = self.df['dispatch_time']
        if 'arrive_time' in self.df.columns and 'arrive_time_ts' not in self.df.columns:
            self.df['arrive_time_ts'] = self.df['arrive_time']
        if 'grab_time' in self.df.columns and 'grab_time_ts' not in self.df.columns:
            self.df['grab_time_ts'] = self.df['grab_time']
        # 转换为datetime类型
        time_cols = ['dispatch_time', 'grab_time', 'fetch_time', 'arrive_time', 'estimate_arrived_time']
        for col in time_cols:
            if col in self.df.columns:
                # 只将0值替换为NaN，保留其他值
                # 注意：这里不替换0值，因为arrive_time为0可能表示订单未完成
                # self.df[col] = self.df[col].replace(0, np.nan)
                self.df[col] = pd.to_datetime(self.df[col], unit='s', errors='coerce') + pd.Timedelta(hours=self.TIME_ZONE_OFFSET)

    def _cal_distance_km(self, lng1, lat1, lng2, lat2):
        """
        计算两点之间的距离（公里）
        """
        if pd.isna(lng1) or pd.isna(lat1) or pd.isna(lng2) or pd.isna(lat2):
            return np.nan
        lng1_rad, lat1_rad = np.radians(lng1 / self.COORDINATE_SCALE), np.radians(lat1 / self.COORDINATE_SCALE)
        lng2_rad, lat2_rad = np.radians(lng2 / self.COORDINATE_SCALE), np.radians(lat2 / self.COORDINATE_SCALE)
        d_lon, d_lat = lng2_rad - lng1_rad, lat2_rad - lat1_rad
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
        return 2 * self.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def _calculate_sender_to_recipient_km(self):
        """
        计算订单配送距离
        """
        print('计算订单配送距离...')
        self.df['sender_to_recipient_km'] = self.df.apply(
            lambda row: self._cal_distance_km(
                row['sender_lng'], row['sender_lat'],
                row['recipient_lng'], row['recipient_lat']
            ), axis=1
        )
        self.df['sender_to_recipient_km'] = self.df['sender_to_recipient_km'].round(2)

    def _cal_cumulative_stats(self, group):
        """
        计算每个运单在派单时间点，骑手组内累计完成的订单数量、累计骑行距离
        """
        # 使用原始时间戳进行计算，而不是datetime对象
        group_sorted = group.sort_values('dispatch_time_ts')
        # 筛选arrive_time不为0的订单（0表示未完成）
        completed_orders = group_sorted[group_sorted['arrive_time_ts'] > 0]
        completed_arrive_times = completed_orders[['arrive_time_ts', 'sender_to_recipient_km']].sort_values('arrive_time_ts')
        counts = np.zeros(len(group_sorted), dtype=int)
        distances = np.zeros(len(group_sorted), dtype=float)
        if len(completed_arrive_times) > 0:
            arrive_times_list = completed_arrive_times['arrive_time_ts'].values
            distance_list = completed_arrive_times['sender_to_recipient_km'].fillna(0).cumsum().values
            counts = np.searchsorted(arrive_times_list, group_sorted['dispatch_time_ts'].values, side='right')
            mask = counts > 0
            positions = counts[mask] - 1
            distances[mask] = distance_list[positions]
        return pd.Series(counts, index=group_sorted.index), pd.Series(distances, index=group_sorted.index)

    def calculate_cumulative_stats(self):
        """
        计算骑手今日累计骑行距离和工作窗口内累计骑行距离
        """
        print('开始计算骑手累计统计...')
        self._time_convert()
        self._calculate_sender_to_recipient_km()
        df_copy = self.df.copy()
        # 使用原始时间戳进行排序
        df_copy = df_copy.sort_values(['courier_id', 'dt', 'dispatch_time_ts'])
        df_copy['session_id'] = df_copy.groupby(['courier_id', 'dt'])['is_courier_grabbed'].transform(
            lambda x: x.eq(0).shift(1).fillna(0).infer_objects(copy=False).cumsum().astype(int)
        )
        results_daily_count, results_daily_dist = [], []
        results_session_count, results_session_dist = [], []
        for name, group in tqdm.tqdm(df_copy.groupby(['courier_id', 'dt'], sort=False),
                                     desc="计算骑手今日累计统计", unit="组"):
            counts, dists = self._cal_cumulative_stats(group)
            results_daily_count.append(counts)
            results_daily_dist.append(dists)
        for name, group in tqdm.tqdm(df_copy.groupby(['courier_id', 'dt', 'session_id'], sort=False),
                                     desc="计算骑手工作窗口累计统计", unit="组"):
            counts, dists = self._cal_cumulative_stats(group)
            results_session_count.append(counts)
            results_session_dist.append(dists)
        self.df['courier_daily_completed_count'] = pd.concat(results_daily_count)
        self.df['courier_daily_completed_distance_km'] = pd.concat(results_daily_dist)
        self.df['courier_session_completed_count'] = pd.concat(results_session_count)
        self.df['courier_session_completed_distance_km'] = pd.concat(results_session_dist)
        print(f'骑手累计统计计算完成')
        print('-----------------')
        return self.df

    def calculate_area_familiarity(self):
        """
        计算骑手的区域熟悉度
        """
        print('开始计算骑手区域熟悉度...')
        df_copy = self.df.copy()
        # 使用原始时间戳进行排序
        df_copy = df_copy.sort_values(['courier_id', 'dispatch_time_ts'])
        results_total, results_da = [], []
        for name, group in tqdm.tqdm(df_copy.groupby(['courier_id'], sort=False),
                                     desc="计算骑手累计完成总订单", unit="组"):
            # 筛选arrive_time_ts大于0的订单（0表示未完成）
            completed = group[group['arrive_time_ts'] > 0].sort_values('arrive_time_ts')
            if len(completed) > 0:
                counts = np.searchsorted(
                    completed['arrive_time_ts'].values,
                    group.sort_values('dispatch_time_ts')['dispatch_time_ts'].values,
                    side='right'
                )
            else:
                counts = np.zeros(len(group), dtype=int)
            results_total.append(pd.Series(counts, index=group.sort_values('dispatch_time_ts').index))
        for name, group in tqdm.tqdm(df_copy.groupby(['courier_id', 'da_id'], sort=False),
                                     desc="计算骑手区域累计完成订单", unit="组"):
            # 筛选arrive_time_ts大于0的订单（0表示未完成）
            completed = group[group['arrive_time_ts'] > 0].sort_values('arrive_time_ts')
            if len(completed) > 0:
                counts = np.searchsorted(
                    completed['arrive_time_ts'].values,
                    group.sort_values('dispatch_time_ts')['dispatch_time_ts'].values,
                    side='right'
                )
            else:
                counts = np.zeros(len(group), dtype=int)
            results_da.append(pd.Series(counts, index=group.sort_values('dispatch_time_ts').index))
        self.df['completed_orders_total'] = pd.concat(results_total)
        self.df['completed_orders_da'] = pd.concat(results_da)
        self.df['area_familiarity'] = self.df['completed_orders_da'] / self.df['completed_orders_total'].replace(0, np.nan)
        self.df['area_familiarity'] = self.df['area_familiarity'].fillna(0)
        print(f'骑手区域熟悉度计算完成')
        print('-----------------')
        return self.df


class ExtractTimeFeatures:
    """
    步骤4: 获取dispatch_time的小时特征
    """
    TIME_ZONE_OFFSET = 8

    def __init__(self, df):
        """
        初始化时间特征提取类
        :param df: 数据框
        """
        self.df = df

    def extract_time_features(self):
        """
        提取时间特征：是否周末、时间节点（小时）
        """
        print('开始提取时间特征...')
        if 'dispatch_time' in self.df.columns:
            self.df['dispatch_time_dt'] = pd.to_datetime(self.df['dispatch_time'], unit='s', errors='coerce') + pd.Timedelta(hours=self.TIME_ZONE_OFFSET)
            self.df['dispatch_hour'] = self.df['dispatch_time_dt'].dt.hour
        if 'is_weekend' not in self.df.columns:
            self.df['is_weekend'] = self.df['dispatch_time_dt'].dt.dayofweek.isin([5, 6]).astype(int)
        print(f'时间特征提取完成')
        print('-----------------')
        return self.df


class VirtualDispatch:
    """
    步骤5&6: 虚拟订单分配，计算订单与骑手的匹配特征
    """
    COORDINATE_SCALE = 10**6
    EARTH_RADIUS_KM = 6371
    DEFAULT_NEARBY_COUNT = 10

    def __init__(self, df_waybill, df_rider, df_all_info_with_stats, nearby_count=None):
        """
        初始化虚拟分配类
        :param df_waybill: 订单数据框（已合并信息）
        :param df_rider: 骑手数据框（已计算现有订单特征）
        :param df_all_info_with_stats: 包含累计统计的订单信息数据框
        :param nearby_count: 筛选的附近骑手数量，默认10
        """
        self.df_waybill = df_waybill
        self.df_rider = df_rider
        self.df_all_info_with_stats = df_all_info_with_stats
        self.nearby_count = nearby_count or self.DEFAULT_NEARBY_COUNT
        self.utm_transformer = Transformer.from_crs('epsg:4326', 'epsg:32660')
        self._build_rider_spatial_index()

    def _build_rider_spatial_index(self):
        """
        构建骑手空间索引，按时间分组
        """
        print('构建骑手空间索引...')
        self.rider_by_time = {}
        # 按时间和日期分组
        for (dt, dispatch_time), group in self.df_rider.groupby(['dt', 'dispatch_time']):
            # 提取经纬度并转换为numpy数组
            coords = group[['rider_lng', 'rider_lat']].values / self.COORDINATE_SCALE
            # 保存骑手信息
            self.rider_by_time[(dt, dispatch_time)] = {
                'coords': coords,
                'riders': group.to_dict('records')
            }
        print(f'骑手空间索引构建完成，共 {len(self.rider_by_time)} 个时间点')

    def _cal_distance_km(self, lng1, lat1, lng2, lat2):
        """
        计算两点之间的距离（公里）
        """
        if pd.isna(lng1) or pd.isna(lat1) or pd.isna(lng2) or pd.isna(lat2):
            return np.nan
        lng1_rad, lat1_rad = np.radians(lng1 / self.COORDINATE_SCALE), np.radians(lat1 / self.COORDINATE_SCALE)
        lng2_rad, lat2_rad = np.radians(lng2 / self.COORDINATE_SCALE), np.radians(lat2 / self.COORDINATE_SCALE)
        d_lon, d_lat = lng2_rad - lng1_rad, lat2_rad - lat1_rad
        a = np.sin(d_lat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(d_lon / 2) ** 2
        return 2 * self.EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    def _cal_distance_batch(self, order_lng, order_lat, coords):
        """
        批量计算订单到多个骑手的距离
        :param order_lng, order_lat: 订单经纬度
        :param coords: 骑手坐标数组 (n, 2)
        :return: 距离数组 (n,)
        """
        order_lng_rad = np.radians(order_lng / self.COORDINATE_SCALE)
        order_lat_rad = np.radians(order_lat / self.COORDINATE_SCALE)
        
        lng_rad = np.radians(coords[:, 0])
        lat_rad = np.radians(coords[:, 1])
        
        d_lon = lng_rad - order_lng_rad
        d_lat = lat_rad - order_lat_rad
        
        a = np.sin(d_lat / 2) ** 2 + np.cos(order_lat_rad) * np.cos(lat_rad) * np.sin(d_lon / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return c * self.EARTH_RADIUS_KM

    def _lnglat_to_utm(self, lng, lat):
        """
        将经纬度转换为UTM平面坐标（米）- 使用pyproj国际标准库
        :param lng: 经度（度）
        :param lat: 纬度（度）
        :return: (x, y) - UTM东向和北向坐标（米）
        """
        x, y = self.utm_transformer.transform(lat, lng)
        return x, y

    def _cal_area_km2(self, points):
        """
        计算经纬度点集围成的区域面积（平方公里）- 使用UTM投影+最小凸包
        :param points: 点集数组，每个点为 (经度, 纬度)，已缩放（需除以COORDINATE_SCALE）
        :return: 面积（平方公里）
        """
        if len(points) < 3:
            return 0.0
        
        points_array = np.array(points)
        lng = points_array[:, 0] / self.COORDINATE_SCALE
        lat = points_array[:, 1] / self.COORDINATE_SCALE
        
        utm_coords = np.array([self._lnglat_to_utm(l, b) for l, b in zip(lng, lat)])
        
        try:
            hull = ConvexHull(utm_coords)
            area = hull.volume
        except Exception:
            return 0.0
        
        return round(area / 1e6, 4)

    def _cal_point_vector(self, start_point, end_point):
        """
        计算两点构成的向量（使用UTM投影坐标）
        """
        start_lng, start_lat = start_point[0] / self.COORDINATE_SCALE, start_point[1] / self.COORDINATE_SCALE
        end_lng, end_lat = end_point[0] / self.COORDINATE_SCALE, end_point[1] / self.COORDINATE_SCALE
        
        start_x, start_y = self._lnglat_to_utm(start_lng, start_lat)
        end_x, end_y = self._lnglat_to_utm(end_lng, end_lat)
        
        return np.array([end_x - start_x, end_y - start_y])

    def _cal_vector_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        """
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _find_nearby_riders(self, order_lng, order_lat, dispatch_time, dt):
        """
        找到订单周围一定范围内的骑手
        :param order_lng, order_lat: 订单取餐点经纬度
        :param dispatch_time: 派单时间
        :param dt: 日期
        :return: 附近骑手列表（按距离排序）
        """
        key = (dt, dispatch_time)
        if key not in self.rider_by_time:
            return []
        
        rider_data = self.rider_by_time[key]
        coords = rider_data['coords']
        riders = rider_data['riders']
        
        # 批量计算距离
        distances = self._cal_distance_batch(order_lng, order_lat, coords)
        
        # 按距离排序并取前N个
        sorted_indices = np.argsort(distances)[:self.nearby_count]
        
        # 构建结果
        result = []
        for idx in sorted_indices:
            rider = riders[idx].copy()
            rider['distance_to_order'] = distances[idx]
            result.append(rider)
        
        return result

    def virtual_dispatch(self):
        """
        执行虚拟订单分配，计算所有特征
        :return: 虚拟分配结果数据框
        """
        print('开始虚拟订单分配...')
        results = []
        
        # 预处理订单数据
        valid_orders = self.df_waybill.dropna(subset=['sender_lng', 'sender_lat', 'recipient_lng', 'recipient_lat'])
        
        for idx, order in tqdm.tqdm(valid_orders.iterrows(), total=len(valid_orders), desc="虚拟分配订单", unit="订单"):
            order_lng, order_lat = order.get('sender_lng'), order.get('sender_lat')
            dispatch_time = order.get('dispatch_time')
            dt = order.get('dt')
            
            nearby_riders = self._find_nearby_riders(order_lng, order_lat, dispatch_time, dt)
            
            recipient_lng, recipient_lat = order.get('recipient_lng'), order.get('recipient_lat')
            order_distance = self._cal_distance_km(order_lng, order_lat, recipient_lng, recipient_lat)
            new_order_vector = self._cal_point_vector([order_lng, order_lat], [recipient_lng, recipient_lat])
            
            for rider in nearby_riders:
                rider_to_order_distance = rider.get('distance_to_order', np.nan)
                ongoing_points = rider.get('ongoing_points_list', [])
                ongoing_area = rider.get('ongoing_area_km2', 0)
                
                new_points = ongoing_points + [[order_lng, order_lat], [recipient_lng, recipient_lat]]
                new_area = self._cal_area_km2(new_points)
                area_change = new_area - ongoing_area
                
                if ongoing_area > 0:
                    area_change_ratio = area_change / ongoing_area
                else:
                    area_change_ratio = 0.0 if new_area == 0 else 1.0
                
                # 计算余弦相似度：计算所有在途订单的平均配送向量
                if len(ongoing_points) >= 2:  # 至少需要1个订单（2个点）
                    # 计算所有在途订单的取餐点平均坐标
                    pickup_points = ongoing_points[0::2]  # 取餐点（偶数索引）
                    avg_pickup = np.mean(pickup_points, axis=0)
                    
                    # 计算所有在途订单的送餐点平均坐标
                    delivery_points = ongoing_points[1::2]  # 送餐点（奇数索引）
                    avg_delivery = np.mean(delivery_points, axis=0)
                    
                    # 计算平均配送向量
                    avg_vector = self._cal_point_vector(avg_pickup, avg_delivery)
                    
                    # 计算新订单的配送向量
                    new_order_vector = self._cal_point_vector([order_lng, order_lat], [recipient_lng, recipient_lat])
                    
                    # 计算两个向量的余弦相似度
                    direction_similarity = self._cal_vector_similarity(new_order_vector, avg_vector)
                else:
                    direction_similarity = 1.0
                
                result_row = {
                    'order_uid': order.get('uid'),
                    'order_id': order.get('order_id'),
                    'courier_id': rider.get('courier_id'),
                    'dt': dt,
                    'dispatch_time': dispatch_time,
                    'da_id': order.get('da_id'),
                    'order_distance_km': round(order_distance, 2) if pd.notna(order_distance) else np.nan,
                    'rider_to_order_km': round(rider_to_order_distance, 2) if pd.notna(rider_to_order_distance) else np.nan,
                    'ongoing_order_count': rider.get('ongoing_order_count', 0),
                    'ongoing_cross_area_count': rider.get('ongoing_cross_area_count', 0),
                    'ongoing_max_distance_km': rider.get('ongoing_max_distance_km', 0),
                    'ongoing_area_km2': ongoing_area,
                    'new_area_km2': new_area,
                    'area_change_km2': round(area_change, 4),
                    'area_change_ratio': round(area_change_ratio, 4),
                    'direction_similarity': round(direction_similarity, 4),
                    'courier_daily_completed_count': rider.get('courier_daily_completed_count', 0),
                    'courier_daily_completed_distance_km': rider.get('courier_daily_completed_distance_km', 0.0),
                    'courier_session_completed_count': rider.get('courier_session_completed_count', 0),
                    'courier_session_completed_distance_km': rider.get('courier_session_completed_distance_km', 0.0),
                    'area_familiarity': rider.get('area_familiarity', 0.0),
                    'is_weekend': order.get('is_weekend', 0),
                    'dispatch_hour': order.get('dispatch_hour', 0),
                    'dispatch_to_estimateArrived_tm': order.get('dispatch_to_estimateArrived_tm', np.nan)
                }
                
                results.append(result_row)
        
        df_result = pd.DataFrame(results)
        print(f'虚拟订单分配完成，共产生 {len(df_result)} 条分配记录')
        print('-----------------')
        return df_result


class DataProcessVirtual:
    """
    数据处理主类
    """
    def __init__(self, waybill_path, rider_path, all_info_path, output_path, nearby_count=10):
        """
        初始化数据处理类
        :param waybill_path: dispatch_waybill_meituan文件路径
        :param rider_path: dispatch_rider_meituan文件路径
        :param all_info_path: all_waybill_info_meituan_0322文件路径
        :param output_path: 输出文件路径
        :param nearby_count: 筛选的附近骑手数量
        """
        self.waybill_path = waybill_path
        self.rider_path = rider_path
        self.all_info_path = all_info_path
        self.output_path = output_path
        self.nearby_count = nearby_count
        # 优化数据类型，减少内存使用
        self.waybill_dtype = {
            'uid': 'int32',
            'dt': 'int32',
            'dispatch_time': 'int64',
            'order_id': 'int32'
        }
        self.rider_dtype = {
            'uid': 'int32',
            'dt': 'int32',
            'rider_lat': 'int32',
            'rider_lng': 'int32',
            'dispatch_time': 'int64',
            'courier_id': 'int32'
        }
        self.all_info_dtype = {
            'uid': 'int32',
            'dt': 'int32',
            'order_id': 'int32',
            'waybill_id': 'int32',
            'courier_id': 'int32',
            'da_id': 'int32',
            'is_courier_grabbed': 'int8',
            'is_weekend': 'int8',
            'estimate_arrived_time': 'int64',
            'is_prebook': 'int8',
            'poi_id': 'int32',
            'sender_lng': 'int32',
            'sender_lat': 'int32',
            'recipient_lng': 'int32',
            'recipient_lat': 'int32',
            'grab_lng': 'int32',
            'grab_lat': 'int32',
            'dispatch_time': 'int64',
            'grab_time': 'int64',
            'fetch_time': 'int64',
            'arrive_time': 'int64',
            'estimate_meal_prepare_time': 'int64',
            'order_push_time': 'int64',
            'platform_order_time': 'int64'
        }

    def process(self):
        """
        执行完整的数据处理流程
        """
        df_waybill = load_data(self.waybill_path, dtype=self.waybill_dtype)
        df_rider = load_data(self.rider_path, dtype=self.rider_dtype)
        df_all_info = load_data(self.all_info_path, dtype=self.all_info_dtype)
        merger = MergeOrderInfo(df_waybill, df_all_info)
        df_waybill_merged = merger.merge_order_info()
        time_extractor = ExtractTimeFeatures(df_waybill_merged)
        df_waybill_merged = time_extractor.extract_time_features()
        
        # 先计算累计统计特征
        cumulative_calc = CalculateCourierCumulativeStats(df_all_info)
        df_all_info_with_stats = cumulative_calc.calculate_cumulative_stats()
        df_all_info_with_stats = cumulative_calc.calculate_area_familiarity()
        
        # 再计算骑手现有订单特征（包含累计统计特征）
        rider_feature_calc = CalculateRiderOngoingFeatures(df_rider, df_all_info, df_all_info_with_stats)
        df_rider_with_features = rider_feature_calc.calculate_rider_ongoing_features()
        
        virtual_dispatcher = VirtualDispatch(df_waybill_merged, df_rider_with_features, df_all_info_with_stats, self.nearby_count)
        df_result = virtual_dispatcher.virtual_dispatch()
        
        # 将dispatch_time从时间戳转换为东八区的时间显示格式
        print('转换dispatch_time为东八区时间格式...')
        # 时间戳转换为东八区的datetime对象
        df_result['dispatch_time'] = pd.to_datetime(df_result['dispatch_time'], unit='s') + pd.Timedelta(hours=8)
        # 转换为指定格式
        df_result['dispatch_time'] = df_result['dispatch_time'].dt.strftime('%Y/%m/%d %H:%M:%S')
        
        print(f'数据处理完成，结果保存到 {self.output_path}')
        df_result.to_csv(self.output_path, index=False)
        return df_result


if __name__ == '__main__':
    # 优化数据类型，减少内存使用
    waybill_dtype = {
        'uid': 'int32',
        'dt': 'int32',
        'dispatch_time': 'int64',
        'order_id': 'int32'
    }
    rider_dtype = {
        'uid': 'int32',
        'dt': 'int32',
        'rider_lat': 'int32',
        'rider_lng': 'int32',
        'dispatch_time': 'int64',
        'courier_id': 'int32'
    }
    all_info_dtype = {
        'uid': 'int32',
        'dt': 'int32',
        'order_id': 'int32',
        'waybill_id': 'int32',
        'courier_id': 'int32',
        'da_id': 'int32',
        'is_courier_grabbed': 'int8',
        'is_weekend': 'int8',
        'estimate_arrived_time': 'int64',
        'is_prebook': 'int8',
        'poi_id': 'int32',
        'sender_lng': 'int32',
        'sender_lat': 'int32',
        'recipient_lng': 'int32',
        'recipient_lat': 'int32',
        'grab_lng': 'int32',
        'grab_lat': 'int32',
        'dispatch_time': 'int64',
        'grab_time': 'int64',
        'fetch_time': 'int64',
        'arrive_time': 'int64',
        'estimate_meal_prepare_time': 'int64',
        'order_push_time': 'int64',
        'platform_order_time': 'int64'
    }
    processor = DataProcessVirtual(
        waybill_path=r'0MeiTuan_Data/dispatch_waybill_meituan.csv',
        rider_path=r'0MeiTuan_Data/dispatch_rider_meituan.csv',
        all_info_path=r'0MeiTuan_Data/all_waybill_info_meituan_0322.csv',
        output_path=r'.\Data_Process\Data_Processed\virtual_dispatch_result_v2.csv',
        nearby_count=10
    )
    df_result = processor.process()
    print(df_result.head())
    print(df_result.columns.tolist())
