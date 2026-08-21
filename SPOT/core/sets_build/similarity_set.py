import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
pd.set_option('display.max_columns', None)


def load_data(file_path, sample=False, sample_num=500, random_state=42, columns=[
    # 特征列
    'fatigue',
    'dispatch_to_sender_delivery_distance',
    'sender_to_recipient_delivery_distance',
    'ongoing_order_count',
    'courier_prefer_reject',
    'poi_prefer_reject',
    # 目标变量
    'is_courier_grabbed']):
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

    def compute_distance_matrix(self):
        """
        逐样本行向量化计算加权距离矩阵，避免 (M, N, F) 三维中间数组导致内存爆炸
        :return: (M x N) 距离矩阵 (float32), M=样本集行数, N=训练集行数
        """
        feature_cols = [col for col in self.df.columns if col != self.target_col]
        
        # 转为 float32 减半内存
        X_train = self.standardized_df[feature_cols].values.astype(np.float32)
        X_example = self.standardized_df_example[feature_cols].values.astype(np.float32)
        y_train = self.standardized_df[self.target_col].values.astype(np.float32)
        y_example = self.standardized_df_example[self.target_col].values.astype(np.float32)
        
        M, N = X_example.shape[0], X_train.shape[0]
        dist_matrix = np.empty((M, N), dtype=np.float32)
        
        for i in tqdm(range(M), desc="计算距离矩阵"):
            # (N, F) - (1, F) -> (N, F) -> sum(axis=1) -> (N,)
            diff = X_train - X_example[i:i+1]
            feature_dist = self.feature_weight * (diff ** 2).sum(axis=1)
            target_dist = self.target_weight * (y_train - y_example[i]) ** 2
            dist_matrix[i] = feature_dist + target_dist
        
        return dist_matrix

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
        
        # 预计算样本集与训练集的距离矩阵 (向量化加速)
        distance_matrix = self.compute_distance_matrix()  # (M, N)
        M, N = distance_matrix.shape
        train_index_list = list(self.standardized_df.index)
        
        # 可用训练集标记 (布尔数组，比 Python set 查找快得多)
        available = np.ones(N, dtype=bool)
        # 相似集编号
        set_id = 0
        # 计算最大可能的相似集数量
        max_sets = len(self.standardized_df) // len(self.standardized_df_example)
        
        # 重复构建相似集，直到剩余训练集数量不足以构建一个完整的相似集
        with tqdm(total=max_sets, desc="构建相似集") as pbar:
            while available.sum() >= M:
                # 本次相似集的样本索引和距离
                similar_indices = []
                similar_distances = []
                
                # 遍历样本集的每一行
                for example_idx in range(M):
                    dist_row = distance_matrix[example_idx]
                    # 已匹配行的距离设无穷大，argmin 取最近可用行
                    valid_distances = np.where(available, dist_row, np.inf)
                    best_global_col = np.argmin(valid_distances)
                    
                    best_train_idx = train_index_list[best_global_col]
                    min_distance = valid_distances[best_global_col]
                    
                    similar_indices.append(best_train_idx)
                    similar_distances.append(min_distance)
                    available[best_global_col] = False
                
                # 为本次相似集的样本标记信息
                for idx, dist in zip(similar_indices, similar_distances):
                    result_df.loc[idx, 'similar_set_id'] = set_id
                    result_df.loc[idx, 'similar_sample_distance'] = dist
                
                # 更新相似集编号
                set_id += 1
                # 更新进度条
                pbar.update(1)
                pbar.set_postfix({"剩余训练集": int(available.sum())})
        
        # 删除未被分配到相似集的多余行
        result_df = result_df[result_df['similar_set_id'].notna()].reset_index(drop=True)
        
        # 计算权重和归一化权重
        self.calculate_weights(result_df)
        return result_df
        

if __name__ == '__main__':
    # 加载数据
    df = load_data(r'SPOT\Data\Data_Processed\cleaned_waybill_info.csv')
    df_example = load_data(r'SPOT\core\sets_build\sample_sets\sample_522093.csv', sample=False)
    
    # 按时序划分训练集和测试集 (7:3)
    # 按派单时间升序排序后按比例切分，避免随机划分造成的时间泄漏
    time_col = pd.read_csv(r'SPOT\Data\Data_Processed\cleaned_waybill_info.csv', usecols=['dispatch_time'])
    time_col['dispatch_time'] = pd.to_datetime(time_col['dispatch_time'])
    sorted_idx = time_col['dispatch_time'].sort_values().index
    split_idx = int(len(df) * 0.7)
    df_train = df.loc[sorted_idx[:split_idx]].reset_index(drop=True)
    df_test = df.loc[sorted_idx[split_idx:]].reset_index(drop=True)
    print(f"原始数据量: {len(df)}")
    print(f"训练集数据量: {len(df_train)}")
    print(f"测试集数据量: {len(df_test)}")
    print(f"样本集数据量: {len(df_example)}")
    
    # ========== 训练集构建相似集 ==========
    print("\n===== 训练集相似集构建 =====")
    builder_train = SimilarSetBuilder(df_train, df_example, target_col='is_courier_grabbed', feature_weight=0.7)
    result_train_df = builder_train.build_similarity_sets()
    print(f"训练集相似集数量: {result_train_df['similar_set_id'].nunique()}")
    
    # ========== 测试集构建相似集 ==========
    print("\n===== 测试集相似集构建 =====")
    builder_test = SimilarSetBuilder(df_test, df_example, target_col='is_courier_grabbed', feature_weight=0.7)
    result_test_df = builder_test.build_similarity_sets()
    print(f"测试集相似集数量: {result_test_df['similar_set_id'].nunique()}")
    
    # 输出结果概览
    print("\n========== 结果概览 ==========")
    print("训练集结果形状:", result_train_df.shape)
    print("测试集结果形状:", result_test_df.shape)
    print("\n训练集前5行:")
    print(result_train_df.head(5))
    print("\n测试集前5行:")
    print(result_test_df.head(5))
    
    # 保存结果为CSV文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.join(project_root, 'Data', 'Data_Processed')
    os.makedirs(output_dir, exist_ok=True)
    
    train_output_path = os.path.join(output_dir, 'similarity_sets_train.csv')
    test_output_path = os.path.join(output_dir, 'similarity_sets_test.csv')
    result_train_df.to_csv(train_output_path, index=False, encoding='utf-8-sig')
    result_test_df.to_csv(test_output_path, index=False, encoding='utf-8-sig')
    print(f"\n训练集结果已保存到 {train_output_path}")
    print(f"测试集结果已保存到 {test_output_path}")
    
    # 汇总信息
    print("\n========== 汇总 ==========")
    print(f"原始数据: {len(df)} 条")
    print(f"训练集: {len(df_train)} 条, 相似集 {result_train_df['similar_set_id'].nunique()} 个")
    print(f"测试集: {len(df_test)} 条, 相似集 {result_test_df['similar_set_id'].nunique()} 个")
