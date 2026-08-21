import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import BaseEstimator, RegressorMixin
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
        
    def fit(self, X, y, similar_set_info=None, sample_weight=None):
        """
        训练模型
        :param X: 特征数据
        :param y: 目标变量
        :param similar_set_info: 相似集信息数据框，必须包含similar_set_id和normalized_weight列
        :param sample_weight: 样本权重（用于处理类别不平衡），形状为(n_samples,)
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
        
        # 处理样本权重
        sample_weight_provided = sample_weight is not None
        if sample_weight_provided:
            if isinstance(sample_weight, pd.Series):
                sample_weight = sample_weight.values
            sample_weight = np.asarray(sample_weight, dtype=float)
            # 归一化使权重均值为1，不影响整体尺度
            sample_weight = sample_weight / np.mean(sample_weight)
        else:
            sample_weight = np.ones(len(y))
        
        self.similar_set_info_ = similar_set_info.copy()
        
        print(f"开始训练决策树 (a={self.a}, 最大深度={self.max_depth}, 特征数={X.shape[1]}, 并行数={self.n_jobs})")
        if sample_weight_provided:
            print(f"样本权重已启用: min={sample_weight.min():.4f}, max={sample_weight.max():.4f}")
        # 构建决策树
        self.tree_ = self._build_tree(X, y, similar_set_info, sample_weight, depth=0)
        print("训练完成!")
        
        return self
    
    def _calculate_decision_loss(self, y_subset, y_pred_subset, similar_set_ids_subset, normalized_weights_subset):
        """
        计算决策损失（纯Numpy实现，优化性能）
        损失 = 真实值降序top-n之和 - 预测值降序top-n对应的真实值之和
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
            
            # 按真实值降序排列，取前top_n个真实值之和（oracle最优）
            sorted_by_true = np.argsort(-set_y)
            top_n_true_sum = np.sum(set_y[sorted_by_true[:self.top_n]])
            
            # 按预测值降序排列，取前top_n对应的真实值之和（模型实际选到的）
            sorted_by_pred = np.argsort(-set_y_pred)
            top_n_pred_sum = np.sum(set_y[sorted_by_pred[:self.top_n]])
            
            # 获取权重（该相似集的所有样本权重相同）
            weight = normalized_weights_subset[group_mask][0]
            decision_loss += weight * (top_n_true_sum - top_n_pred_sum)
        
        return decision_loss
    
    def _calculate_mse(self, y_subset, sample_weight=None):
        """计算MSE损失（支持加权）"""
        if len(y_subset) == 0:
            return 0.0
        if sample_weight is not None:
            weighted_mean = np.average(y_subset, weights=sample_weight)
            return np.average((y_subset - weighted_mean) ** 2, weights=sample_weight)
        return np.mean((y_subset - np.mean(y_subset)) ** 2)
    
    def _calculate_combined_loss(self, y_subset, similar_set_ids_subset, normalized_weights_subset, sample_weight=None):
        """
        计算组合损失
        :param y_subset: 真实目标变量子集
        :param similar_set_ids_subset: 相似集ID子集
        :param normalized_weights_subset: 归一化权重子集
        :param sample_weight: 样本权重子集
        """
        if len(y_subset) == 0:
            return float('inf')
        
        if self.a == 1.0:
            # 只计算MSE损失
            return self._calculate_mse(y_subset, sample_weight)
        elif self.a == 0.0:
            # 只计算决策损失
            y_pred_subset = np.full_like(y_subset, np.mean(y_subset))
            return self._calculate_decision_loss(y_subset, y_pred_subset, similar_set_ids_subset, normalized_weights_subset)
        else:
            # 计算组合损失
            mse_loss = self._calculate_mse(y_subset, sample_weight)
            y_pred_subset = np.full_like(y_subset, np.mean(y_subset))
            decision_loss = self._calculate_decision_loss(y_subset, y_pred_subset, similar_set_ids_subset, normalized_weights_subset)
            return self.a * mse_loss + (1 - self.a) * decision_loss
    
    def _find_best_split(self, X, y, similar_set_ids, normalized_weights, sample_weight, feature_idx):
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
            left_sample_weight = sample_weight[left_mask] if sample_weight is not None else None
            right_sample_weight = sample_weight[right_mask] if sample_weight is not None else None
            
            n_left = len(left_y)
            n_right = len(right_y)
            n_total = len(y)
            
            # 父节点损失
            current_loss = self._calculate_combined_loss(y, similar_set_ids, normalized_weights, sample_weight)
            
            # MSE：左右子节点分别计算，加权平均
            mse_left = self._calculate_mse(left_y, left_sample_weight)
            mse_right = self._calculate_mse(right_y, right_sample_weight)
            weighted_mse = (n_left / n_total) * mse_left + (n_right / n_total) * mse_right
            
            # 决策损失：在左右合并集上用子节点均值作为预测值计算
            # 这样同一个相似集跨左右节点的样本之间就有了排序信号
            if self.a < 1.0:
                left_mean_val = np.mean(left_y)
                right_mean_val = np.mean(right_y)
                y_combined = np.concatenate([left_y, right_y])
                y_pred_combined = np.concatenate([
                    np.full(n_left, left_mean_val),
                    np.full(n_right, right_mean_val)
                ])
                set_ids_combined = np.concatenate([left_set_ids, right_set_ids])
                weights_combined = np.concatenate([left_weights, right_weights])
                decision_loss_after = self._calculate_decision_loss(
                    y_combined, y_pred_combined, set_ids_combined, weights_combined
                )
            else:
                decision_loss_after = 0.0
            
            child_loss = self.a * weighted_mse + (1 - self.a) * decision_loss_after
            gain = current_loss - child_loss
            
            if gain > best_gain:
                best_gain = gain
                best_threshold = threshold
        
        return best_threshold, best_gain
    
    def _find_best_split_for_feature(self, X, y, similar_set_ids, normalized_weights, sample_weight, feature_idx):
        """
        为单个特征找到最佳分裂点（用于并行化）
        """
        return self._find_best_split(X, y, similar_set_ids, normalized_weights, sample_weight, feature_idx)
    
    def _build_tree(self, X, y, similar_set_info, sample_weight, depth):
        """
        递归构建决策树
        """
        n_samples = len(y)
        
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf):
            return {
                'leaf': True,
                'value': np.average(y, weights=sample_weight) if sample_weight is not None else np.mean(y),
                # 'value': np.mean(y) if self.a == 0.0 else (np.average(y, weights=sample_weight) if sample_weight is not None else np.mean(y)),
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
                        X, y, similar_set_ids, normalized_weights, sample_weight, feature_idx
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
                            X, y, similar_set_ids, normalized_weights, sample_weight, feature_idx
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
                    X, y, similar_set_ids, normalized_weights, sample_weight, feature_idx
                )
                
                if threshold is not None and gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        if best_feature is None:
            return {
                'leaf': True,
                'value': np.average(y, weights=sample_weight) if sample_weight is not None else np.mean(y),
                # 'value': np.mean(y) if self.a == 0.0 else (np.average(y, weights=sample_weight) if sample_weight is not None else np.mean(y)),
                'similar_set_distribution': self._get_similar_set_distribution(similar_set_info)
            }
        
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        left_sample_weight = sample_weight[left_mask] if sample_weight is not None else None
        right_sample_weight = sample_weight[right_mask] if sample_weight is not None else None
        
        left_tree = self._build_tree(
            X[left_mask], y[left_mask], 
            similar_set_info[left_mask], left_sample_weight, depth + 1
        )
        right_tree = self._build_tree(
            X[right_mask], y[right_mask], 
            similar_set_info[right_mask], right_sample_weight, depth + 1
        )
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def save_model(self, filepath):
        """保存模型到文件"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"模型已保存至: {filepath}")

    @staticmethod
    def load_model(filepath):
        """从文件加载模型"""
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"模型已加载: {filepath}")
        return model
    
    def plot_tree(self, feature_names=None, filepath=None, figsize=(20, 10)):
        """
        绘制决策树可视化图
        :param feature_names: 特征名称列表
        :param filepath: 图片保存路径，为None则显示
        :param figsize: 画布大小
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        
        if feature_names is None:
            feature_names = [f"X[{i}]" for i in range(self.n_features_in_)]
        
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # 递归计算树的布局
        def _layout_tree(node, depth=0, x_offset=0.0, width=1.0):
            """返回节点列表，每个节点包含位置信息"""
            if node['leaf']:
                return [{
                    'type': 'leaf',
                    'value': node['value'],
                    'depth': depth,
                    'x': x_offset + width / 2,
                    'n_samples': node.get('n_samples', '?'),
                    'similar_set_distribution': node.get('similar_set_distribution', {})
                }]
            
            # 非叶节点，递归计算左右子树的布局
            left_nodes = _layout_tree(node['left'], depth + 1, x_offset, width / 2)
            right_nodes = _layout_tree(node['right'], depth + 1, x_offset + width / 2, width / 2)
            
            feature_name = feature_names[node['feature']] if node['feature'] < len(feature_names) else f"X[{node['feature']}]"
            threshold = node['threshold']
            
            current_node = {
                'type': 'split',
                'feature_name': feature_name,
                'threshold': threshold,
                'depth': depth,
                'x': x_offset + width / 2,
                'n_samples': node.get('n_samples', '?')
            }
            
            return [current_node] + left_nodes + right_nodes
        
        # 先为节点补充样本数信息后计算布局（在_build_tree时未保存，只能从已知结构推断）
        # 简化：用递归标注n_samples
        def _annotate_n_samples(node):
            if node['leaf']:
                node['n_samples'] = 1  # 占位，实际无法从树结构反推
                return
            _annotate_n_samples(node['left'])
            _annotate_n_samples(node['right'])
        
        # 重新用更精确的方法计算布局时带上n_samples
        # 由于树结构中不保存n_samples，这里直接标注0
        nodes = _layout_tree(self.tree_)
        
        # 计算画布参数
        max_depth = max(n['depth'] for n in nodes)
        depth_height = 1.0 / (max_depth + 2)
        
        # 节点大小
        node_width = 0.08
        node_height = 0.04
        
        # 绘制节点和边
        depth_nodes = {}
        for n in nodes:
            d = n['depth']
            if d not in depth_nodes:
                depth_nodes[d] = []
            depth_nodes[d].append(n)
        
        # 先画边
        def _draw_edges(node, ax, max_depth, depth_height):
            if node.get('leaf'):
                return
            parent_x = node.get('_plot_x', 0.5)
            parent_y = 1 - (node.get('_layout_depth', 0) + 0.5) * depth_height
            
            for child_key in ['left', 'right']:
                child = node[child_key]
                child_x = child.get('_plot_x', 0.5)
                child_y = 1 - (child.get('_layout_depth', 0) + 0.5) * depth_height
                
                ax.plot([parent_x, child_x], [parent_y, child_y], 
                       'k-', lw=1.5, alpha=0.5)
                _draw_edges(child, ax, max_depth, depth_height)
        
        # 标注布局深度和x坐标到节点上
        def _annotate_plot_positions(node, depth=0, x_offset=0.0, width=1.0):
            node['_layout_depth'] = depth
            node['_plot_x'] = x_offset + width / 2
            if not node.get('leaf'):
                _annotate_plot_positions(node['left'], depth + 1, x_offset, width / 2)
                _annotate_plot_positions(node['right'], depth + 1, x_offset + width / 2, width / 2)
        
        _annotate_plot_positions(self.tree_)
        _draw_edges(self.tree_, ax, max_depth, depth_height)
        
        # 再画节点
        def _draw_nodes(node, ax, max_depth, depth_height, feature_names):
            d = node['_layout_depth']
            x = node['_plot_x']
            y = 1 - (d + 0.5) * depth_height
            
            if node.get('leaf'):
                # 叶节点
                rect = patches.FancyBboxPatch(
                    (x - node_width/2, y - node_height/2),
                    node_width, node_height,
                    boxstyle="round,pad=0.02",
                    facecolor='#90EE90', edgecolor='black', lw=1.5
                )
                ax.add_patch(rect)
                text = f"value={node['value']:.4f}"
                ax.text(x, y, text, ha='center', va='center', fontsize=8, fontweight='bold')
            else:
                # 分裂节点
                rect = patches.FancyBboxPatch(
                    (x - node_width/2, y - node_height/2),
                    node_width, node_height,
                    boxstyle="round,pad=0.02",
                    facecolor='#FFB6C1', edgecolor='black', lw=1.5
                )
                ax.add_patch(rect)
                
                feat_name = feature_names[node['feature']] if node['feature'] < len(feature_names) else f"X[{node['feature']}]"
                text = f"{feat_name}\n≤ {node['threshold']:.4f}"
                ax.text(x, y, text, ha='center', va='center', fontsize=7, fontweight='bold')
                
                # 递归画子节点
                _draw_nodes(node['left'], ax, max_depth, depth_height, feature_names)
                _draw_nodes(node['right'], ax, max_depth, depth_height, feature_names)
        
        _draw_nodes(self.tree_, ax, max_depth, depth_height, feature_names)
        
        plt.title(f"SimilaritySetDecisionTree (a={self.a}, max_depth={self.get_depth()}, leaves={self.get_n_leaves()})", 
                 fontsize=14, fontweight='bold')
        
        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"决策树可视化已保存至: {filepath}")
            plt.close()
        else:
            plt.show()
    
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
    损失 = 真实值降序top-n之和 - 预测值降序top-n对应的真实值之和
    :param y: 真实目标变量
    :param y_pred: 预测目标变量
    :param similar_set_ids: 相似集ID
    :param normalized_weights: 归一化权重
    :param top_n: 取前N个
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
        # 按真实值降序排列，取前top_n真实值之和（oracle最优）
        top_n_true_sum = set_data.sort_values('y', ascending=False).head(top_n)['y'].sum()
        # 按预测值降序排列，取前top_n对应的真实值之和（模型实际选到的）
        top_n_pred_sum = set_data.sort_values('y_pred', ascending=False).head(top_n)['y'].sum()
        
        weight = set_data['normalized_weight'].iloc[0]
        set_loss = weight * (top_n_true_sum - top_n_pred_sum)
        total_decision_loss += set_loss
        
        set_details.append({
            'set_id': set_id,
            'samples': len(set_data),
            'top_n_true_sum': top_n_true_sum,
            'top_n_pred_sum': top_n_pred_sum,
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
            print(f"Set {detail['set_id']}: 样本数={detail['samples']}, 真实top-{top_n}和={detail['top_n_true_sum']:.6f}, 预测top-{top_n}真实和={detail['top_n_pred_sum']:.6f}, 权重={detail['weight']:.6f}, 加权损失={detail['set_loss']:.6f}, 平均真实值={detail['avg_y']:.6f}, 平均预测值={detail['avg_y_pred']:.6f}, 预测值范围=[{detail['min_y_pred']:.6f}, {detail['max_y_pred']:.6f}]")
        
        # 分析预测值分布
        # print("\n=== 预测值分布 ===")
        # print(f"预测值均值: {df['y_pred'].mean():.6f}")
        # print(f"预测值标准差: {df['y_pred'].std():.6f}")
        # print(f"预测值最小值: {df['y_pred'].min():.6f}")
        # print(f"预测值最大值: {df['y_pred'].max():.6f}")
    
    return total_decision_loss


if __name__ == '__main__':
    from sklearn.metrics import mean_squared_error
    import os

    top_n_num = 300
    max_depth = 3

    # ====== 加载预切分的训练集和测试集 ======
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(project_root, r'SPOT\Data\Data_Processed')
    
    train_path = os.path.join(data_dir, 'similarity_sets_train.csv')
    test_path = os.path.join(data_dir, 'similarity_sets_test.csv')
    
    print(f"加载训练集: {train_path}")
    df_train = pd.read_csv(train_path)
    print(f"加载测试集: {test_path}")
    df_test = pd.read_csv(test_path)
    print(f"训练集样本数: {len(df_train)}, 相似集数量: {df_train['similar_set_id'].nunique()}")
    print(f"测试集样本数: {len(df_test)}, 相似集数量: {df_test['similar_set_id'].nunique()}")

    # 特征列（排除相似集相关列和目标列）
    feature_cols = [col for col in df_train.columns if col not in
                   ['similar_set_id', 'similar_sample_distance', 'similar_set_weight',
                    'normalized_weight', 'is_courier_grabbed']]
    target_col = 'is_courier_grabbed'

    # 准备训练集数据
    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    similar_set_info_train = df_train[['similar_set_id', 'normalized_weight']].copy()

    # 准备测试集数据
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]
    similar_set_info_test = df_test[['similar_set_id', 'normalized_weight']].copy()
    
    # 处理样本权重：负样本太少，使用类别平衡权重
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weight = compute_sample_weight(None, y_train)
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    print(f"\n类别分布: 负样本={neg_count}, 正样本={pos_count}, 比例={neg_count/pos_count:.2f}:1")
    print(f"样本权重范围: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
    
    print("\n" + "="*60)
    print("训练模型并评估测试集效果")
    print("="*60)
    
    # 测试不同a值的决策树
    print("\n测试不同a值的决策树...，a=0为SPOT，a=1.0为传统决策树")
    for a in [0, 1.0]:
        print(f"\n{'='*60}")
        print(f"测试 a={a} 的决策树")
        print(f"{'='*60}")
        
        tree = SimilaritySetDecisionTree(a=a, max_depth=max_depth, min_samples_split=20, min_samples_leaf=10, top_n=top_n_num)
        tree.fit(X_train, y_train, similar_set_info_train, sample_weight=sample_weight)
        
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
        
        # ====== 保存 a=0 的模型 ======
        if a == 0:
            pkl_dir = os.path.join(project_root, r'SPOT\core\pkl_s')
            os.makedirs(pkl_dir, exist_ok=True)
            model_path = os.path.join(pkl_dir, 'smart_decision_tree_model.pkl')
            tree.save_model(model_path)
            
            # 可视化图片保存到 Results
            img_dir = os.path.join(project_root, r'SPOT\Results')
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, 'smart_decision_tree_structure.png')
            tree.plot_tree(feature_names=feature_cols, filepath=img_path, figsize=(24, 12))
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
