import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import tqdm
from scipy.spatial import ConvexHull
from pyproj import Transformer
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
pd.set_option('future.no_silent_downcasting', True)
pd.set_option('display.max_columns', None)


def load_data(file_path):
    """
    加载数据
    :param file_path: 数据文件路径
    :return: 数据框
    """
    df = pd.read_csv(file_path)
    # df = df[df['is_prebook'] == 0]
    print(f'加载数据完成，共 {len(df)} 条记录，数据形状为 {df.shape}')
    print('-----------------')
    # print(df.head())
    return df


class ConvertandExtend:
    """
    转换并扩展数据
    """
    # 东八区偏移小时数
    TIME_ZONE_OFFSET = 8
    # 秒转分钟系数
    SECONDS_TO_MINUTES = 60
    # 坐标放大系数
    COORDINATE_SCALE = 10**6
    # 地球半径（公里）
    EARTH_RADIUS_KM = 6371
    def __init__(self, df):
        """
        初始化转换并扩展数据类
        :param df: 数据框
        """
        self.df = df

    def time_convert(self, time_cols = [
            'estimate_arrived_time', 'dispatch_time', 'grab_time', 'fetch_time',
            'arrive_time', 'estimate_meal_prepare_time', 'order_push_time', 'platform_order_time'
        ]):
        """
        转换时间列
        :param time_cols: 时间列名列表
        :return: 数据框
        """
        print(f'开始转换时间列：{time_cols}')
        for col in time_cols:
            if col in self.df.columns:
                # 时间戳转换为东八区的时间格式
                self.df[col] = self.df[col].replace(0, np.nan)
                self.df[col] = pd.to_datetime(self.df[col], unit='s', errors='coerce') + pd.Timedelta(hours=self.TIME_ZONE_OFFSET)
        # 提取dispatch_time的小时数
        if 'dispatch_time' in self.df.columns:
            self.df['dispatch_hour'] = self.df['dispatch_time'].dt.hour
        print(f'时间列转换完成')
        print('-----------------')
        # print(self.df.head())
        return self.df

    def time_extend(self, differ_set = {
            # 商家备餐时长
            'dispatch_to_estimateMeal_tm': ('estimate_meal_prepare_time', 'dispatch_time'),
            # 骑手取货时长
            'grab_to_fetch_tm': ('fetch_time', 'grab_time'),
            # 骑手送货时长
            'fetch_to_arrive_tm': ('arrive_time', 'fetch_time'),
            # 骑手总配送时长
            'grab_to_arrive_tm': ('arrive_time', 'grab_time'),
            # 可用配送时长
            'dispatch_to_estimateArrived_tm': ('estimate_arrived_time', 'dispatch_time'),
            # 剩余配送时长  
            'arrive_to_estimateArrived_tm': ('estimate_arrived_time', 'arrive_time')
        }):
        """
        扩展时间列
        :return: 数据框
        """
        print(f'开始扩展时间列：{differ_set.keys()}')
        for col_name, (end_col, start_col) in differ_set.items():
            if end_col in self.df.columns and start_col in self.df.columns:
                self.df[col_name] = (self.df[end_col] - self.df[start_col]).dt.total_seconds() / self.SECONDS_TO_MINUTES
        
        print(f'时间列扩展完成，数据形状为 {self.df.shape}')
        print('-----------------')
        # print(self.df.head())
        return self.df

    def position_extend(self):
        """
        估计派单时骑手的位置
        :return: 数据框
        """
        print(f'开始估计派单时骑手的位置')
        df_copy = self.df.copy(deep=True)
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
        # 去除time列空值
        long_df = long_df.dropna(axis=0, subset=["time"])
        # 将经纬度为0的值转换为空值，便于后续空值填充操作
        long_df.loc[long_df['lng'] == 0, ['lng', 'lat']] = np.nan
        # 按时间排序
        long_df = long_df.sort_values(by=['dt', 'courier_id', 'time'], ignore_index=True)
        # 分组填充（先向下填充，后向上填充）
        long_df[['lng', 'lat']] = long_df.groupby(['dt', 'courier_id'])[['lng', 'lat']].ffill().bfill().infer_objects(copy=False)
        long_df_grab = long_df[long_df['type'] == 'grab']
        # 合并到原数据表
        grab_merge_data = long_df_grab[['waybill_id', 'lng', 'lat']].rename(
            columns={
                'lng': 'dispatch_lng',  # 重命名为目标列名
                'lat': 'dispatch_lat'
            }
        )
        self.df = self.df.merge(
            grab_merge_data,
            on='waybill_id',  # 关联键
            how='left'  # 左合并：原df所有行保留，匹配不到则dispatch_lng/lat为NaN
        )
        print(f'骑手位置估计完成')
        print('-----------------')
        return self.df

    def distance_extend(self, point_set = {
        # 骑手取货距离
        'dispatch_to_sender_km': [('dispatch_lng', 'dispatch_lat'), ('sender_lng', 'sender_lat')],
        # 骑手送货距离
        'sender_to_recipient_km': [('sender_lng', 'sender_lat'), ('recipient_lng', 'recipient_lat')]
    }):
        """
        扩展距离列
        :return: 数据框
        """
        print(f'开始扩展距离列：{point_set.keys()}')
        def distance_cal(col_name, point_pairs):
            """
            计算距离
            :param col_name: 距离列名
            :param point_pairs: 点对列表，每个点对为 (经度, 纬度)
            :return: 数据框
            """
            # 将坐标列中的0值替换为NaN
            for col in [col for pair in point_pairs for col in pair]:
                self.df[col] = self.df[col].replace(0, np.nan)
            # 提取两组坐标的列名
            (lng1_col, lat1_col) = point_pairs[0]
            (lng2_col, lat2_col) = point_pairs[1]
            # 还原放大前的坐标
            lng1 = self.df[lng1_col] / self.COORDINATE_SCALE
            lat1 = self.df[lat1_col] / self.COORDINATE_SCALE
            lng2 = self.df[lng2_col] / self.COORDINATE_SCALE
            lat2 = self.df[lat2_col] / self.COORDINATE_SCALE
            # 检查无效点（任一坐标为NaN）
            point1_invalid = lng1.isna() | lat1.isna()
            point2_invalid = lng2.isna() | lat2.isna()
            invalid_mask = point1_invalid | point2_invalid
            # 初始化距离数组（全为NaN）
            distances = np.full(len(self.df), np.nan)
            # 只计算有效点的距离
            valid_mask = ~invalid_mask
            if valid_mask.any():
                # 提取有效点坐标
                valid_lng1 = lng1[valid_mask].values
                valid_lat1 = lat1[valid_mask].values
                valid_lng2 = lng2[valid_mask].values
                valid_lat2 = lat2[valid_mask].values
                # 转换为弧度
                valid_lng1, valid_lat1, valid_lng2, valid_lat2 = map(
                    np.radians, [valid_lng1, valid_lat1, valid_lng2, valid_lat2]
                )
                # Haversine公式向量化计算
                d_lon = valid_lng2 - valid_lng1
                d_lat = valid_lat2 - valid_lat1
                a = np.sin(d_lat / 2) ** 2 + np.cos(valid_lat1) * np.cos(valid_lat2) * np.sin(d_lon / 2) ** 2
                c = 2 * np.arcsin(np.sqrt(a))
                valid_distances = c * self.EARTH_RADIUS_KM
                # 将有效距离放入结果数组
                distances[valid_mask] = valid_distances
            self.df[col_name] = distances.round(2)
        for col_name, point_pairs in point_set.items():
            distance_cal(col_name, point_pairs)
        print(f'距离列扩展完成，数据形状为 {self.df.shape}')
        print('-----------------')
        # print(self.df.head())
        return self.df

    def convert_and_extend(self):
        """
        转换并扩展数据
        :return: 数据框
        """
        self.time_convert()
        self.time_extend()
        self.position_extend()
        self.distance_extend()
        return self.df

    
        

class FillMissingValues:
    """
    填充缺失值
    """
    def __init__(self, df):
        """
        初始化填充缺失值类
        :param df: 数据框
        """
        self.df = df
        self.missing_values = self.df.isnull().sum()  # 计算缺失值数量
        # print(f'缺失值数量：\n{self.missing_values}')
    
    def fill_meal_prepare_time(self):
        """
        按商圈中位数填充餐厅备餐完成时间空值（抗离群值）
        :return: 填充后的DataFrame
        """
        print(f'开始填充餐厅备餐时间缺失值')
        # 计算店铺备餐时间中位数、区域备餐时间中位数以及全局中位数（用中位数抗离群值）
        poi_median = self.df.groupby('poi_id')['dispatch_to_estimateMeal_tm'].median()
        da_median = self.df.groupby('da_id')['dispatch_to_estimateMeal_tm'].median()
        global_median = self.df['dispatch_to_estimateMeal_tm'].median()
        # 向量化填充
        na_mask = self.df['dispatch_to_estimateMeal_tm'].isna()
        # 匹配店铺中位数，无则用区域，再无则用全局
        self.df.loc[na_mask, 'dispatch_to_estimateMeal_tm'] = (self.df.loc[na_mask, 'da_id'].map(poi_median).fillna(da_median).fillna(global_median))
        self.missing_values = self.df.isnull().sum()  # 计算缺失值数量
        # print(f'填充后缺失值数量：\n{self.missing_values}')
        print(f'餐厅备餐时间缺失值填充完成')
        print('-----------------')
        return self.df

    def fill_missing_value(self):
        """
        填充所有缺失值
        :return: 数据框
        """
        self.fill_meal_prepare_time()
        return self.df


class RemoveOutliers:
    """
    去除异常值
    """
    IQR_FACTOR = 1.5
    def __init__(self, df):
        """
        初始化去除异常值类
        :param df: 数据框
        """
        self.df = df

    def negative_outliers(self):
        """
        去除负值异常值
        :return: 数据框
        """
        self.df['dispatch_to_estimateMeal_tm'] = self.df['dispatch_to_estimateMeal_tm'].clip(lower=0)
    
    def remove_outliers_iqr(self, cols = ['dispatch_to_estimateArrived_tm'], iqr_factor = IQR_FACTOR):
        """
        去除异常值（IQR方法）
        :param cols: 异常值列名列表
        :param iqr_factor: IQR因子，默认1.5
        :return: 去除后的DataFrame
        """
        print(f'开始去除异常值：{cols}')
        for col in cols:
            if col in self.df.columns:
                # 计算IQR
                q1 = self.df[col].quantile(0.25)
                q3 = self.df[col].quantile(0.75)
                iqr = q3 - q1
                # 计算异常值上下限
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr
                print(f'异常值上下限：{lower_bound:.2f} | {upper_bound:.2f}')
                # 过滤异常值
                self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
        print(f'异常值去除完成，数据形状为 {self.df.shape}')
        print('-----------------')
        return self.df

    def remove_outliers(self):
        """
        去除所有异常值
        :return: 数据框
        """
        self.negative_outliers()
        self.remove_outliers_iqr()
        return self.df


class ExtractCourierOngoingFeatures:
    """
    提取骑手在手订单数列
    """
    def __init__(self, df):
        """
        初始化提取骑手在手订单数列类
        :param df: 数据框
        """
        self.df = df
        self.utm_transformer = Transformer.from_crs('epsg:4326', 'epsg:32660')
        # 缓存UTM转换结果，避免重复计算
        self.coord_cache = {}

    def cal_point_vector(self, start_point, end_point):
        """
        计算两点构成的向量（使用UTM投影坐标）
        :param start_point: 起始点坐标
        :param end_point: 结束点坐标
        :return: 向量数组
        """
        # 缓存键
        start_key = (start_point[0], start_point[1])
        end_key = (end_point[0], end_point[1])
        
        # 从缓存获取转换结果
        if start_key not in self.coord_cache:
            start_lng, start_lat = start_point[0] / 1e6, start_point[1] / 1e6
            self.coord_cache[start_key] = np.array(self.utm_transformer.transform([start_lat], [start_lng])).T[0]
        
        if end_key not in self.coord_cache:
            end_lng, end_lat = end_point[0] / 1e6, end_point[1] / 1e6
            self.coord_cache[end_key] = np.array(self.utm_transformer.transform([end_lat], [end_lng])).T[0]
        
        return self.coord_cache[end_key] - self.coord_cache[start_key]

    def cal_vector_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度
        :param vec1: 向量1
        :param vec2: 向量2
        :return: 余弦相似度
        """
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 1.0
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def cal_area_km2(self, points):
        """
        计算经纬度点集围成的区域面积（平方公里）- 使用UTM投影+最小凸包
        :param points: 点集数组，每个点为 (经度, 纬度)，已缩放（需除以10^6）
        :return: 面积（平方公里）
        """
        if len(points) < 3:
            return 0.0
        
        # 缓存键：使用点集的元组作为键
        points_key = tuple(tuple(point) for point in points)
        
        # 从缓存获取结果
        if points_key in self.coord_cache:
            return self.coord_cache[points_key]
        
        points_array = np.array(points)
        lng = points_array[:, 0] / 1e6
        lat = points_array[:, 1] / 1e6
        
        # 向量化转换坐标，避免循环
        utm_coords = np.array(self.utm_transformer.transform(lat, lng)).T
        
        try:
            hull = ConvexHull(utm_coords)
            area = hull.volume
            area_km2 = round(area / 1e6, 4)
        except Exception:
            area_km2 = 0.0
        
        # 缓存结果
        self.coord_cache[points_key] = area_km2
        
        return area_km2

    def process_single_courier(self, courier_data):
        """
        处理单个骑手的数据（用于并行处理）
        :param courier_data: (courier_id, group) 元组
        :return: 骑手结果DataFrame
        """
        courier_id, group = courier_data
        group_sorted = group.sort_values('dispatch_time').copy()
        
        ongoing_set = set()
        area_counter = Counter()
        events = []
        
        for idx, row in group_sorted.iterrows():
            events.append((row['dispatch_time'], 1, idx, row['da_id'], 'query'))
            if pd.notna(row['grab_time']):
                events.append((row['grab_time'], 0, idx, row['da_id'], 'grab'))
            if pd.notna(row['arrive_time']):
                events.append((row['arrive_time'], -1, idx, row['da_id'], 'arrive'))
        
        events.sort(key=lambda x: (x[0], -x[1]))
        rider_results = defaultdict(list)
        
        order_lookup = group_sorted.to_dict('index')
        
        for event_time, event_type, order_idx, da_id, event_kind in events:
            if event_kind == 'grab':
                ongoing_set.add(order_idx)
                area_counter[da_id] += 1
            elif event_kind == 'arrive':
                if order_idx in ongoing_set:
                    ongoing_set.remove(order_idx)
                    area_counter[da_id] -= 1
                    if area_counter[da_id] == 0:
                        del area_counter[da_id]
            else:
                other_count = len(ongoing_set)
                rider_results['ongoing_other_count'].append((order_idx, other_count))
                
                if other_count == 0:
                    rider_results['ongoing_cross_area_count'].append((order_idx, 0))
                    rider_results['ongoing_max_distance_km'].append((order_idx, 0.0))
                    rider_results['ongoing_area_km2'].append((order_idx, 0.0))
                    rider_results['ongoing_area_km2_with_new'].append((order_idx, 0.0))
                    rider_results['direction_similarity'].append((order_idx, 1.0))
                else:
                    cross_area_count = len(area_counter)
                    rider_results['ongoing_cross_area_count'].append((order_idx, cross_area_count))
                    
                    other_km = []
                    points = []
                    sender_points = []
                    recipient_points = []
                    
                    new_row = order_lookup[order_idx]
                    sender_point_new = [new_row['sender_lng'], new_row['sender_lat']]
                    recipient_point_new = [new_row['recipient_lng'], new_row['recipient_lat']]
                    new_vector = self.cal_point_vector(sender_point_new, recipient_point_new)
                    
                    for other_idx in ongoing_set:
                        other_row = order_lookup[other_idx]
                        other_km.append(other_row['sender_to_recipient_km'])
                        sender_point = [other_row['sender_lng'], other_row['sender_lat']]
                        recipient_point = [other_row['recipient_lng'], other_row['recipient_lat']]
                        sender_points.append(sender_point)
                        recipient_points.append(recipient_point)
                        points.append(sender_point)
                        points.append(recipient_point)
                    
                    max_km = float(np.max(other_km))
                    rider_results['ongoing_max_distance_km'].append((order_idx, max_km))
                    
                    points_array = np.array(points)
                    points_km2 = self.cal_area_km2(points_array)
                    rider_results['ongoing_area_km2'].append((order_idx, points_km2))
                    
                    points_with_new = points + [sender_point_new, recipient_point_new]
                    points_with_new_array = np.array(points_with_new)
                    points_km2_with_new = self.cal_area_km2(points_with_new_array)
                    rider_results['ongoing_area_km2_with_new'].append((order_idx, points_km2_with_new))
                    
                    sender_center = np.array(sender_points).mean(axis=0)
                    recipient_center = np.array(recipient_points).mean(axis=0)
                    now_vector_all = self.cal_point_vector(sender_center, recipient_center)
                    vector_similarity_all = self.cal_vector_similarity(now_vector_all, new_vector)
                    rider_results['direction_similarity'].append((order_idx, vector_similarity_all))
        
        rider_df = pd.DataFrame({
            'order_idx': [idx for idx, _ in rider_results['ongoing_other_count']],
            'ongoing_other_count': [val for _, val in rider_results['ongoing_other_count']],
            'ongoing_cross_area_count': [val for _, val in rider_results['ongoing_cross_area_count']],
            'ongoing_max_distance_km': [val for _, val in rider_results['ongoing_max_distance_km']],
            'ongoing_area_km2': [val for _, val in rider_results['ongoing_area_km2']],
            'ongoing_area_km2_with_new': [val for _, val in rider_results['ongoing_area_km2_with_new']],
            'direction_similarity': [val for _, val in rider_results['direction_similarity']],
        }).set_index('order_idx')
        
        return rider_df

    def extract_courier_ongoing_features(self):
        """
        提取骑手在手订单数列（使用多线程并行）
        :return: 数据框,添加特征列后的数据
        """
        print('开始提取骑手现有订单特征...')
        df_copy = self.df.copy(deep=True)
        
        courier_groups = list(df_copy.groupby('courier_id', sort=False))
        
        n_workers = min(8, multiprocessing.cpu_count())
        print(f'使用 {n_workers} 个线程并行处理 {len(courier_groups)} 个骑手...')
        
        results = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results = list(tqdm.tqdm(
                executor.map(self.process_single_courier, courier_groups),
                total=len(courier_groups),
                desc="统计骑手现有订单的情况",
                unit="骑手"
            ))
        
        features_df = pd.concat(results)
        self.df = self.df.join(features_df)
        
        ongoing_area = self.df['ongoing_area_km2']
        new_area = self.df['ongoing_area_km2_with_new']
        area_change = new_area - ongoing_area
        
        self.df['area_increase_ratio'] = np.where(
            ongoing_area > 0,
            area_change / ongoing_area,
            np.where(new_area > 0, 1.0, 0.0)
        )
        
        return self.df


class ExtractCourierCompletedFeatures:
    """
    提取骑手已完成订单数列
    """
    def __init__(self, df):
        """
        初始化提取骑手已完成订单数列类
        :param df: 数据框
        """
        self.df = df

    def cal_cumulative_stats(self, group):
        """
        计算每个运单在派单时间点，骑手组内累计完成的订单数量、累计骑行距离以及累计骑行时间
        :param group: 骑手组
        :return: 计算结果的Series
        """
        # 按派单时间排序，确保计算顺序
        group_sorted = group.sort_values('dispatch_time')
        # 提取已完成的订单送达时间并排序
        completed_arrive_times = group_sorted.loc[
            group_sorted['arrive_time'].notna(),
            ['arrive_time', 'grab_to_arrive_tm', 'sender_to_recipient_km']
        ].sort_values('arrive_time')
        # 初始化数值
        counts = np.zeros(len(group_sorted), dtype=int)
        durations = np.zeros(len(group_sorted), dtype=float)
        distances = np.zeros(len(group_sorted), dtype=float)
        # 如果没有已完成的订单，直接返回0
        if len(completed_arrive_times) > 0:
            arrive_times_list = completed_arrive_times['arrive_time'].values
            duration_list = completed_arrive_times['grab_to_arrive_tm'].cumsum().values
            distance_list = completed_arrive_times['sender_to_recipient_km'].cumsum().values
            # 核心优化：使用searchsorted在O(log n)时间内完成每个查询
            # 计算每个dispatch_time之前有多少个arrive_time
            # side='right'表示严格小于（不包含等于）
            counts = np.searchsorted(
                arrive_times_list,
                group_sorted['dispatch_time'].values,
                side='right'
            )
            # 找到有效位置
            mask = counts > 0
            positions = counts[mask] - 1
            # 填充数据
            durations[mask] = duration_list[positions]
            distances[mask] = distance_list[positions]
        # 保持原始索引，方便后续合并
        counts_series = pd.Series(counts, index=group_sorted.index)
        durations_series = pd.Series(durations, index=group_sorted.index)
        distances_series = pd.Series(distances, index=group_sorted.index)
        return counts_series, durations_series, distances_series

    def cal_daily_completely_orders(self):
        """
        计算骑手每天的已完成订单数
        :return: 数据框
        """
        df_copy = self.df.copy(deep=True)
        # 定义结果列名
        result_col_1 = 'courier_daily_completed_before_dispatch_count'
        result_col_2 = 'courier_daily_completed_before_dispatch_duration'
        result_col_3 = 'courier_daily_completed_before_dispatch_distance'
        # 分组键
        group_keys = ['courier_id', 'dt']
        # 应用分组计算，使用进度条显示进度
        grouped = df_copy.groupby(group_keys, sort=False)
        # 收集所有分组的结果
        results_1, results_2, results_3 = [], [], []
        for name, group in tqdm.tqdm(grouped, desc="计算骑手当日累计完成订单量", unit="组"):
            counts_series, durations_series, distances_series = self.cal_cumulative_stats(group)
            results_1.append(counts_series)
            results_2.append(durations_series)
            results_3.append(distances_series)
        # 将结果合并到原始DataFrame
        df[result_col_1] = pd.concat(results_1)
        df[result_col_2] = pd.concat(results_2)
        df[result_col_3] = pd.concat(results_3)
        self.df = df
        return self.df

    def cal_daily_completely_orders_with_re(self):
        """
        计算计算骑手每天的已完成订单数——拒单重置计数
        :return: 数据框
        """
        df_copy = self.df.copy(deep=True)
        # 按骑手、日期、派单时间排序
        df_copy = df_copy.sort_values(['courier_id', 'dt', 'dispatch_time'])
        # 计算会话ID：拒单后的订单属于新会话
        # 拒单订单本身仍属于上一个会话（因为它发生时已有累计值）
        # 公式：is_accepted.eq(0).shift(1) 检测前一个订单是否拒单
        df_copy['session_id_courier'] = df_copy.groupby(['courier_id', 'dt'])['is_courier_grabbed'].transform(
            lambda x: x.eq(0).shift(1).fillna(0).infer_objects(copy=False).cumsum().astype(int)
        )
        result_col_1 = 'courier_daily_completed_before_dispatch_with_reset_count'
        result_col_2 = 'courier_daily_completed_before_dispatch_with_reset_duration'
        result_col_3 = 'courier_daily_completed_before_dispatch_with_reset_distance'
        # 按骑手+日期+会话分组计算
        group_keys = ['courier_id', 'dt', 'session_id_courier']
        results_1, results_2, results_3 = [], [], []
        for name, group in tqdm.tqdm(df_copy.groupby(group_keys, sort=False),
                                desc="计算骑手当日会话窗口内累计完成订单量", unit="组"):
            counts_series, durations_series, distances_series = self.cal_cumulative_stats(group)
            results_1.append(counts_series)
            results_2.append(durations_series)
            results_3.append(distances_series)
        # 合并结果并清理临时列
        self.df[result_col_1] = pd.concat(results_1)
        self.df[result_col_2] = pd.concat(results_2)
        self.df[result_col_3] = pd.concat(results_3)
        return self.df

    def cal_area_familiarity(self):
        """
        计算骑手的区域熟悉度
        :return: 数据框
        """
        df_copy = self.df.copy(deep=True)
        result_col = 'completed_orders_count'
        result_col_da = 'completed_orders_count_da'
        # 按骑手和区域分组计算
        group_key = ['courier_id']
        group_keys_da = ['courier_id', 'da_id']
        # 应用分组计算并显示进度
        results = []
        results_da = []
        for name, group in tqdm.tqdm(df_copy.groupby(group_key, sort=False),
                                desc="计算骑手累计完成总订单数量", unit="组"):
            results.append(self.cal_cumulative_stats(group)[0])
        for name, group in tqdm.tqdm(df_copy.groupby(group_keys_da, sort=False),
                                desc="计算骑手区域累计完成总订单数量", unit="组"):
            results_da.append(self.cal_cumulative_stats(group)[0])
        # 合并结果
        df[result_col] = pd.concat(results)
        df[result_col_da] = pd.concat(results_da)
        df['area_familiarity'] = df[result_col_da] / df[result_col]
        df['area_familiarity'] = df['area_familiarity'].fillna(0)
        self.df = df
        return self.df


    def extract_courier_completed_features(self):
        """
        提取骑手已完成订单数列
        :return: 数据框,添加特征列后的数据
        """
        self.cal_daily_completely_orders()
        self.cal_daily_completely_orders_with_re()
        self.cal_area_familiarity()
        return self.df


class TargetFeature:
    """
    目标特征
    """
    def __init__(self, df):
        """
        初始化目标特征类
        :param df: 数据框
        """
        self.df = df

    def extract_target_feature(self):
        """
        提取目标特征
        :return: 数据框,添加特征列后的数据
        """
        # 构建目标特征列：拒单惩罚系数
        # 如果骑手拒单（is_courier_grab==0），则惩罚系数为 1/订单可用配送时长（小时），否则为0
        # 根据is_courier_grab生成拒单惩罚系数
        self.df['reject_penalty_ratio'] = np.where(
            self.df['is_courier_grabbed'] == 0,
            1 / (self.df['dispatch_to_estimateArrived_tm'] / 60),
            0
        )

        # 处理可用时长为0或负数时的极端情况，避免除零
        self.df['reject_penalty_ratio'] = self.df['reject_penalty_ratio'].clip(lower=0).replace(np.inf, 0)
        return self.df

    def target_feature(self):
        """
        目标特征
        :return: 数据框,添加特征列后的数据
        """
        self.extract_target_feature()
        return self.df


if __name__ == '__main__':
    # 加载数据
    df = load_data(r'0MeiTuan_Data/all_waybill_info_meituan_0322.csv')

    # 转换并扩展数据
    converter_and_extender = ConvertandExtend(df)
    df = converter_and_extender.convert_and_extend()

    # 填充缺失值
    filler = FillMissingValues(df)
    df = filler.fill_missing_value()

    # 去除异常值
    remover = RemoveOutliers(df)
    df = remover.remove_outliers()
    # print(df.head())

    # --- 特征工程 --- #
    print('\n-----------------')
    print('特征工程处理开始')
    print('-----------------\n')

    # 提取骑手在手订单数列
    extractor = ExtractCourierOngoingFeatures(df)
    df = extractor.extract_courier_ongoing_features()

    # 提取骑手已完成订单数列
    extractor = ExtractCourierCompletedFeatures(df)
    df = extractor.extract_courier_completed_features()

    # 提取目标特征
    target_feature = TargetFeature(df)
    df = target_feature.target_feature()

    # 保存最终的df为csv文件
    df.to_csv(r'.\Data_Process\Data_Processed\all_waybill_info_meituan_v2.csv', index=False, encoding='utf-8')
    print('数据已保存为 all_waybill_info_meituan_v2.csv')

