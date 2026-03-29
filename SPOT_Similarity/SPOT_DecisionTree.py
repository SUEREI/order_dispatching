import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


class SimilaritySetDecisionTree(BaseEstimator, RegressorMixin):
    def __init__(self, a=0.5, max_depth=5, min_samples_split=10, min_samples_leaf=5, top_n=20, n_thresholds=20, n_jobs=-1):
        """
        自定义决策树，结合MSE损失和相似集决策损失
        :param a: MSE损失权重，最终损失 = a * MSE + (1-a) * 决策损失
        :param max_depth: 最大深度
        :param min_samples_split: 最小分裂样本数
        :param min_samples_leaf: 叶节点最小样本数
        :param top_n: 决策损失计算中取前N个样本
        :param n_thresholds: 每个特征采样的阈值数量（使用分位数）
        :param n_jobs: 并行工作数，-1表示使用所有CPU核心
        """
        self.a = a
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.top_n = top_n
        self.n_thresholds = n_thresholds
        self.n_jobs = n_jobs if n_jobs != -1 else multiprocessing.cpu_count()
        self.tree_ = None
        self.n_features_in_ = None
        
    def fit(self, X, y, similar_set_info=None):
        """
        训练模型
        :param X: 特征数据
        :param y: 目标变量
        :param similar_set_info: 相似集信息数据框，必须包含similar_set_id和normalized_weight列
        """
        if similar_set_info is None:
            raise ValueError("必须提供similar_set_info参数")
        
        self.n_features_in_ = X.shape[1]
        
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(similar_set_info, pd.DataFrame):
            similar_set_info = similar_set_info.copy()
        
        self.similar_set_info_ = similar_set_info.copy()
        
        print(f"开始训练决策树 (a={self.a}, 最大深度={self.max_depth}, 特征数={X.shape[1]}, 并行数={self.n_jobs})")
        # 构建决策树
        self.tree_ = self._build_tree(X, y, similar_set_info, depth=0)
        print("训练完成!")
        
        return self
    
    def _calculate_decision_loss(self, y_subset, y_pred_subset, similar_set_ids_subset, normalized_weights_subset):
        """
        计算决策损失（纯Numpy实现，优化性能）
        :param y_subset: 真实目标变量子集
        :param y_pred_subset: 预测目标变量子集
        :param similar_set_ids_subset: 相似集ID子集
        :param normalized_weights_subset: 归一化权重子集
        :return: 决策损失值
        """
        # 预计算唯一的相似集ID和对应的掩码
        unique_set_ids, inverse_indices = np.unique(similar_set_ids_subset, return_inverse=True)
        decision_loss = 0.0
        
        # 按相似集分组处理
        for i, set_id in enumerate(unique_set_ids):
            # 使用inverse_indices快速获取该相似集的所有索引
            group_mask = (inverse_indices == i)
            set_y = y_subset[group_mask]
            set_y_pred = y_pred_subset[group_mask]
            
            # 按预测值排序
            sorted_indices = np.argsort(set_y_pred)
            sorted_y = set_y[sorted_indices]
            
            # 取前top_n个样本的真实值求和
            top_n_sum = np.sum(sorted_y[:self.top_n])
            
            # 获取权重（该相似集的所有样本权重相同）
            weight = normalized_weights_subset[group_mask][0]
            decision_loss += weight * top_n_sum
        
        return decision_loss
    
    def _calculate_mse(self, y_subset):
        """计算MSE损失"""
        if len(y_subset) == 0:
            return 0.0
        return np.mean((y_subset - np.mean(y_subset)) ** 2)
    
    def _calculate_combined_loss(self, y_subset, similar_set_ids_subset, normalized_weights_subset):
        """
        计算组合损失
        :param y_subset: 真实目标变量子集
        :param similar_set_ids_subset: 相似集ID子集
        :param normalized_weights_subset: 归一化权重子集
        """
        if len(y_subset) == 0:
            return float('inf')
        
        if self.a == 1.0:
            # 只计算MSE损失
            return self._calculate_mse(y_subset)
        elif self.a == 0.0:
            # 只计算决策损失
            y_pred_subset = np.full_like(y_subset, np.mean(y_subset))
            return self._calculate_decision_loss(y_subset, y_pred_subset, similar_set_ids_subset, normalized_weights_subset)
        else:
            # 计算组合损失
            mse_loss = self._calculate_mse(y_subset)
            y_pred_subset = np.full_like(y_subset, np.mean(y_subset))
            decision_loss = self._calculate_decision_loss(y_subset, y_pred_subset, similar_set_ids_subset, normalized_weights_subset)
            return self.a * mse_loss + (1 - self.a) * decision_loss
    
    def _find_best_split(self, X, y, similar_set_ids, normalized_weights, feature_idx):
        """
        找到最佳分裂点（使用分位数采样优化阈值遍历）
        """
        best_gain = -float('inf')
        best_threshold = None
        
        feature_values = X[:, feature_idx]
        unique_values = np.unique(feature_values)
        
        if len(unique_values) <= 1:
            return None, None
        
        # 使用分位数采样减少阈值遍历次数
        n_samples = len(unique_values)
        if n_samples <= self.n_thresholds:
            # 如果唯一值数量小于等于采样数量，使用所有值
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
        else:
            # 否则使用分位数采样（优化版本）
            quantile_indices = np.linspace(0, n_samples - 1, self.n_thresholds, dtype=int)
            thresholds = (unique_values[quantile_indices[:-1]] + unique_values[quantile_indices[1:]]) / 2
        
        # 遍历所有阈值
        for threshold in thresholds:
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                continue
            
            left_y = y[left_mask]
            right_y = y[right_mask]
            left_set_ids = similar_set_ids[left_mask]
            right_set_ids = similar_set_ids[right_mask]
            left_weights = normalized_weights[left_mask]
            right_weights = normalized_weights[right_mask]
            
            current_loss = self._calculate_combined_loss(y, similar_set_ids, normalized_weights)
            left_loss = self._calculate_combined_loss(left_y, left_set_ids, left_weights)
            right_loss = self._calculate_combined_loss(right_y, right_set_ids, right_weights)
            
            n_left = len(left_y)
            n_right = len(right_y)
            n_total = len(y)
            
            weighted_child_loss = (n_left / n_total) * left_loss + (n_right / n_total) * right_loss
            gain = current_loss - weighted_child_loss
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def _find_best_split_for_feature(self, X, y, similar_set_ids, normalized_weights, feature_idx):
        """
        为单个特征找到最佳分裂点（用于并行化）
        """
        return self._find_best_split(X, y, similar_set_ids, normalized_weights, feature_idx)
    
    def _build_tree(self, X, y, similar_set_info, depth):
        """
        递归构建决策树
        """
        n_samples = len(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf):
            return {
                'leaf': True,
                'value': np.mean(y),
                'similar_set_distribution': self._get_similar_set_distribution(similar_set_info)
            }
        
        # 预计算相似集ID和权重数组（避免重复转换）
        similar_set_ids = similar_set_info['similar_set_id'].values
        normalized_weights = similar_set_info['normalized_weight'].values
        
        best_feature = None
        best_threshold = None
        best_gain = -float('inf')
        
        # 并行化特征遍历
        if self.n_jobs > 1 and X.shape[1] > 1:
            # 优化：根据特征数量动态调整并行度
            actual_workers = min(self.n_jobs, X.shape[1])
            if actual_workers <= 1:
                # 特征数量少于并行数，直接单线程处理
                for feature_idx in tqdm(range(X.shape[1]), desc=f"分裂节点 (深度={depth})"):
                    threshold, gain = self._find_best_split(
                        X, y, similar_set_ids, normalized_weights, feature_idx
                    )
                    if threshold is not None and gain > best_gain:
                        best_gain = gain
                        best_feature = feature_idx
                        best_threshold = threshold
            else:
                # 使用多进程并行处理特征
                with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                    # 批量提交任务
                    futures = [
                        executor.submit(
                            self._find_best_split_for_feature,
                            X, y, similar_set_ids, normalized_weights, feature_idx
                        )
                        for feature_idx in range(X.shape[1])
                    ]
                    
                    # 收集结果
                    for feature_idx, future in tqdm(enumerate(futures), 
                                                   total=len(futures),
                                                   desc=f"分裂节点 (深度={depth})"):
                        threshold, gain = future.result()
                        if threshold is not None and gain > best_gain:
                            best_gain = gain
                            best_feature = feature_idx
                            best_threshold = threshold
        else:
            # 单线程处理
            for feature_idx in tqdm(range(X.shape[1]), desc=f"分裂节点 (深度={depth})"):
                threshold, gain = self._find_best_split(
                    X, y, similar_set_ids, normalized_weights, feature_idx
                )
                
                if threshold is not None and gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return {
                'leaf': True,
                'value': np.mean(y),
                'similar_set_distribution': self._get_similar_set_distribution(similar_set_info)
            }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_tree = self._build_tree(
            X[left_mask], y[left_mask], 
            similar_set_info[left_mask], depth + 1
        )
        right_tree = self._build_tree(
            X[right_mask], y[right_mask], 
            similar_set_info[right_mask], depth + 1
        )
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def _get_similar_set_distribution(self, similar_set_info):
        """获取相似集分布"""
        if len(similar_set_info) == 0:
            return {}
        return similar_set_info['similar_set_id'].value_counts().to_dict()
    
    def predict(self, X):
        """预测"""
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        predictions = np.array([self._predict_single(x, self.tree_) for x in X])
        return predictions
    
    def _predict_single(self, x, node):
        """预测单个样本"""
        if node['leaf']:
            return node['value']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])
    
    def get_depth(self):
        """获取树的深度"""
        return self._get_depth(self.tree_)
    
    def _get_depth(self, node):
        if node['leaf']:
            return 0
        return 1 + max(self._get_depth(node['left']), self._get_depth(node['right']))
    
    def get_n_leaves(self):
        """获取叶节点数量"""
        return self._get_n_leaves(self.tree_)
    
    def _get_n_leaves(self, node):
        if node['leaf']:
            return 1
        return self._get_n_leaves(node['left']) + self._get_n_leaves(node['right'])


def calculate_decision_loss(y, y_pred, similar_set_ids, normalized_weights, top_n=20, verbose=False):
    """
    独立的决策损失计算函数（用于验证和调试）
    :param y: 真实目标变量
    :param y_pred: 预测目标变量
    :param similar_set_ids: 相似集ID
    :param normalized_weights: 归一化权重
    :param top_n: 取前N个最小值求和
    :param verbose: 是否打印详细信息
    :return: 决策损失值
    """
    df = pd.DataFrame({
        'y': y,
        'y_pred': y_pred,
        'similar_set_id': similar_set_ids,
        'normalized_weight': normalized_weights
    })
    
    total_decision_loss = 0.0
    set_details = []
    
    for set_id in df['similar_set_id'].unique():
        set_data = df[df['similar_set_id'] == set_id]
        # 按预测值排序
        sorted_data = set_data.sort_values('y_pred')
        # 取前top_n个样本的真实值求和
        top_n_sum = sorted_data.head(top_n)['y'].sum()
        
        weight = set_data['normalized_weight'].iloc[0]
        set_loss = weight * top_n_sum
        total_decision_loss += set_loss
        
        set_details.append({
            'set_id': set_id,
            'samples': len(set_data),
            'top_n_sum': top_n_sum,
            'weight': weight,
            'set_loss': set_loss,
            'avg_y': set_data['y'].mean(),
            'avg_y_pred': set_data['y_pred'].mean(),
            'min_y_pred': set_data['y_pred'].min(),
            'max_y_pred': set_data['y_pred'].max()
        })
    
    if verbose:
        print("\n=== 决策损失详细计算 ===")
        print(f"总决策损失: {total_decision_loss:.6f}")
        print(f"相似集数量: {len(set_details)}")
        for detail in set_details:
            print(f"Set {detail['set_id']}: 样本数={detail['samples']}, 前{top_n}真实值和={detail['top_n_sum']:.6f}, 权重={detail['weight']:.6f}, 加权损失={detail['set_loss']:.6f}, 平均真实值={detail['avg_y']:.6f}, 平均预测值={detail['avg_y_pred']:.6f}, 预测值范围=[{detail['min_y_pred']:.6f}, {detail['max_y_pred']:.6f}]")
        
        # 分析预测值分布
        # print("\n=== 预测值分布 ===")
        # print(f"预测值均值: {df['y_pred'].mean():.6f}")
        # print(f"预测值标准差: {df['y_pred'].std():.6f}")
        # print(f"预测值最小值: {df['y_pred'].min():.6f}")
        # print(f"预测值最大值: {df['y_pred'].max():.6f}")
    
    return total_decision_loss


if __name__ == '__main__':
    import Similarity_Set
    import os
    from sklearn.metrics import mean_squared_error
    
    top_n_num = 10
    sample_set_size = 20
    test_num = 6000    
    max_depth = 3

    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'all_waybill_info_meituan.csv')
    print(f"数据文件路径: {data_path}")
    
    # 加载原始数据
    df_full = Similarity_Set.load_data(data_path, sample=True, sample_num=test_num, random_state=42)
    df_example = Similarity_Set.load_data(data_path, sample=True, sample_num=sample_set_size, random_state=2)
    
    # 按6:4比例划分为训练集和测试集
    df_train, df_test = train_test_split(df_full, test_size=0.4, random_state=42)
    print(f"\n数据划分:")
    print(f"训练集样本数: {len(df_train)}")
    print(f"测试集样本数: {len(df_test)}")
    
    # 为训练集构建相似集
    print("\n为训练集构建相似集...")
    builder_train = Similarity_Set.SimilarSetBuilder(df_train, df_example, target_col='reject_penalty_ratio', feature_weight=0.7)
    result_df_train = builder_train.build_similarity_sets()
    
    # 为测试集构建相似集
    print("\n为测试集构建相似集...")
    builder_test = Similarity_Set.SimilarSetBuilder(df_test, df_example, target_col='reject_penalty_ratio', feature_weight=0.7)
    result_df_test = builder_test.build_similarity_sets()
    
    # 准备训练集数据
    feature_cols = [col for col in result_df_train.columns if col not in 
                   ['similar_set_id', 'similar_sample_distance', 'similar_set_weight', 'normalized_weight', 'reject_penalty_ratio']]
    X_train = result_df_train[feature_cols]
    y_train = result_df_train['reject_penalty_ratio']
    similar_set_info_train = result_df_train[['similar_set_id', 'normalized_weight']].copy()
    
    # 准备测试集数据
    X_test = result_df_test[feature_cols]
    y_test = result_df_test['reject_penalty_ratio']
    similar_set_info_test = result_df_test[['similar_set_id', 'normalized_weight']].copy()
    
    print("\n" + "="*60)
    print("训练模型并评估测试集效果")
    print("="*60)
    
    # 测试不同a值的决策树
    for a in [0, 1.0]:
        print(f"\n{'='*60}")
        print(f"测试 a={a} 的决策树")
        print(f"{'='*60}")
        
        tree = SimilaritySetDecisionTree(a=a, max_depth=max_depth, min_samples_split=20, min_samples_leaf=10, top_n=top_n_num)
        tree.fit(X_train, y_train, similar_set_info_train)
        
        print(f"决策树深度: {tree.get_depth()}")
        print(f"叶节点数量: {tree.get_n_leaves()}")
        
        # 训练集预测效果
        y_pred_train = tree.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        # print(f"\n训练集预测值统计:")
        # print(f"  预测值均值: {y_pred_train.mean():.6f}")
        # print(f"  预测值标准差: {y_pred_train.std():.6f}")
        # print(f"  预测值最小值: {y_pred_train.min():.6f}")
        # print(f"  预测值最大值: {y_pred_train.max():.6f}")
        # print(f"  预测值唯一值数量: {len(np.unique(y_pred_train))}")
        
        decision_loss_train = calculate_decision_loss(y_train, y_pred_train, similar_set_info_train['similar_set_id'].values, similar_set_info_train['normalized_weight'].values, top_n=tree.top_n, verbose=False)
        
        print(f"\n训练集效果:")
        print(f"  MSE: {mse_train:.6f}")
        print(f"  决策损失: {decision_loss_train:.6f}")
        
        # 测试集预测效果
        y_pred_test = tree.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        decision_loss_test = calculate_decision_loss(y_test, y_pred_test, similar_set_info_test['similar_set_id'].values, similar_set_info_test['normalized_weight'].values, top_n=tree.top_n)
        
        print(f"\n测试集效果:")
        print(f"  MSE: {mse_test:.6f}")
        print(f"  决策损失: {decision_loss_test:.6f}")
        
        print(f"\n对比总结 (a={a}):")
        print(f"  训练集 MSE: {mse_train:.6f} -> 测试集 MSE: {mse_test:.6f} (变化: {mse_test-mse_train:+.6f})")
        print(f"  训练集 决策损失: {decision_loss_train:.6f} -> 测试集 决策损失: {decision_loss_test:.6f} (变化: {decision_loss_test-decision_loss_train:+.6f})")
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
