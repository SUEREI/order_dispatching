import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
pd.set_option('display.max_columns', None)


def load_data(file_path, sample=False, sample_num=500, random_state=42, columns=[
    # 距离相关特征
    'sender_to_recipient_km', 'dispatch_to_sender_km', 
    # 工作量相关特征
    'ongoing_other_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2', 
    # 新订单与现有订单适配性相关特征
    'area_increase_ratio', 'direction_similarity', 
    # 疲惫程度相关特征
    'courier_daily_completed_before_dispatch_distance', 'courier_daily_completed_before_dispatch_with_reset_distance',
    # 区域熟悉度相关特征
    'area_familiarity', 
    # 时间相关特征
    'is_weekend', 'dispatch_hour',
    # 目标变量
    'reject_penalty_ratio']):
    """
    加载数据
    :param file_path: 数据文件路径
    :param columns: 要选择的列名列表，None表示选择所有列
    :return: 数据框
    """
    df = pd.read_csv(file_path)
    if sample:
        # 随机抽取sample_num行数据
        sampled_df = df.sample(n=sample_num, random_state=random_state).reset_index(drop=True)
    else:
        sampled_df = df
    # 选择指定的列
    if columns is not None:
        # 确保所有指定的列都存在于数据框中
        existing_columns = [col for col in columns if col in sampled_df.columns]
        sampled_df = sampled_df[existing_columns]
    return sampled_df


class SimilarSetBuilder:
    def __init__(self, df, df_example, target_col='reject_penalty_ratio', feature_weight=0.7):
        self.df = df.copy()
        self.df_example = df_example.copy()
        self.target_col = target_col
        self.feature_weight = feature_weight
        self.target_weight = 1 - feature_weight
        self.scaler = None
        self.standardized_df = None
        self.standardized_df_example = None

    def standardize(self):
        """
        标准化数据（排除目标变量）
        :return: 标准化后的数据框
        """
        # 获取特征列（排除目标变量）
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        # 初始化标准化器
        self.scaler = StandardScaler()
        # 对训练集进行标准化
        self.standardized_df = self.df.copy()
        self.standardized_df[feature_cols] = self.scaler.fit_transform(self.df[feature_cols])
        # 对样本集进行标准化（使用训练集的标准化参数）
        self.standardized_df_example = self.df_example.copy()
        self.standardized_df_example[feature_cols] = self.scaler.transform(self.df_example[feature_cols])
        return self.standardized_df, self.standardized_df_example

    def cal_weighted_distance(self, row1, row2):
        """
        计算加权平方欧几里得距离
        :param row1: 第一行数据
        :param row2: 第二行数据
        :return: 加权距离
        """
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        # 计算特征的加权距离
        feature_distance = 0
        for col in feature_cols:
            feature_distance += (row1[col] - row2[col]) ** 2
        feature_distance = self.feature_weight * feature_distance
        # 计算目标变量的加权距离
        target_distance = self.target_weight * (row1[self.target_col] - row2[self.target_col]) ** 2
        return feature_distance + target_distance

    def calculate_weights(self, result_df):
        """
        计算权重和归一化权重
        :param result_df: 结果数据框
        """
        # 按相似集编号分组，计算每组距离之和
        group_distances = result_df.groupby('similar_set_id')['similar_sample_distance'].sum()
        # 计算权重（距离之和的倒数）
        result_df['similar_set_weight'] = result_df['similar_set_id'].map(lambda x: 1 / group_distances[x] if x in group_distances.index else None)
        # 归一化权重：当前值/各set_id权重之和（每个set_id只计算一次）
        # 获取每个set_id对应的权重（去重）
        unique_weights = result_df.drop_duplicates(subset=['similar_set_id'])['similar_set_weight'].dropna()
        total_weight = unique_weights.sum()
        if total_weight > 0:
            result_df['normalized_weight'] = result_df['similar_set_weight'] / total_weight
        else:
            result_df['normalized_weight'] = None

    def build_similarity_sets(self):
        """
        构建相似集
        :return: 添加了相似集信息的训练集数据框
        """
        # 标准化数据
        self.standardize()
        # 初始化结果数据框
        result_df = self.df.copy()
        result_df['similar_set_id'] = None
        result_df['similar_sample_distance'] = None
        result_df['similar_set_weight'] = None
        result_df['normalized_weight'] = None
        
        # 预计算样本集与训练集的距离矩阵
        distance_matrix = {}
        for example_idx, example_row in tqdm(self.standardized_df_example.iterrows(), 
                                             total=len(self.standardized_df_example),
                                             desc="计算距离矩阵"):
            distances = []
            for train_idx, train_row in self.standardized_df.iterrows():
                distance = self.cal_weighted_distance(example_row, train_row)
                distances.append((train_idx, distance))
            distance_matrix[example_idx] = distances
        
        # 剩余训练集索引集合
        remaining_train_indices = set(self.standardized_df.index)
        # 相似集编号
        set_id = 0
        # 计算最大可能的相似集数量
        max_sets = len(self.standardized_df) // len(self.standardized_df_example)
        
        # 重复构建相似集，直到剩余训练集数量不足以构建一个完整的相似集
        with tqdm(total=max_sets, desc="构建相似集") as pbar:
            while len(remaining_train_indices) >= len(self.standardized_df_example):
                # 本次相似集的样本索引
                similar_indices = []
                similar_distances = []
                
                # 遍历样本集的每一行
                for example_idx, example_row in self.standardized_df_example.iterrows():
                    # 从预计算的距离矩阵中筛选剩余训练集的距离
                    remaining_distances = [(idx, dist) for idx, dist in distance_matrix[example_idx] 
                                          if idx in remaining_train_indices]
                    # 找到距离最小的行
                    min_distance_pair = min(remaining_distances, key=lambda x: x[1])
                    best_train_idx, min_distance = min_distance_pair
                    # 记录相似样本信息
                    similar_indices.append(best_train_idx)
                    similar_distances.append(min_distance)
                    # 从剩余训练集中移除已匹配的行
                    remaining_train_indices.remove(best_train_idx)
                
                # 为本次相似集的样本标记信息
                for idx, dist in zip(similar_indices, similar_distances):
                    result_df.loc[idx, 'similar_set_id'] = set_id
                    result_df.loc[idx, 'similar_sample_distance'] = dist
                
                # 更新相似集编号
                set_id += 1
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"剩余训练集": len(remaining_train_indices)})
        
        # 删除未被分配到相似集的多余行
        result_df = result_df[result_df['similar_set_id'].notna()].reset_index(drop=True)
        
        # 计算权重和归一化权重
        self.calculate_weights(result_df)
        return result_df
        

if __name__ == '__main__':
    # 加载数据
    df = load_data(r'all_waybill_info_meituan.csv')
    df_example = load_data(r'all_waybill_info_meituan.csv', sample=True, sample_num=50, random_state=32)
    
    # 创建相似集构建器
    builder = SimilarSetBuilder(df, df_example, target_col='reject_penalty_ratio', feature_weight=0.7)
    
    # 构建相似集
    result_df = builder.build_similarity_sets()
    
    # 输出结果
    print("相似集构建完成！")
    print(f"训练集数据量: {len(df)}")
    print(f"样本集数据量: {len(df_example)}")
    print(f"相似集数量: {result_df['similar_set_id'].nunique()}")
    print("\n结果数据框形状:", result_df.shape)
    print("\n结果数据框列名:", result_df.columns.tolist())
    print("\n前10行数据:")
    print(result_df.head(10))
    print("\n相似集统计:")
    print(result_df.groupby('similar_set_id').agg({
        'similar_sample_distance': ['sum', 'mean'],
        'similar_set_weight': 'first',
        'normalized_weight': 'first'
    }).head(10))
    
    # 保存结果为CSV文件
    result_df.to_csv('similarity_sets_result.csv', index=False, encoding='utf-8-sig')
    print("\n结果已保存到 similarity_sets_result.csv")
