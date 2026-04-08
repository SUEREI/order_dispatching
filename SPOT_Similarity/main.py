import pandas as pd
from scipy.sparse import data
from SPOT_DecisionTree import SimilaritySetDecisionTree
from Sample_Set import Sample_SetProcessor
from Similarity_Set import SimilarSetBuilder
from Similarity_Set import load_data
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from SPOT_DecisionTree import calculate_decision_loss


def load_data_sample(data_path, order_id=522093):
    # 加载样本集
    sample_processor = Sample_SetProcessor(
        data_path=data_path,
        model_path=r'Models\pkl_s\xgboost\xgboost_model.pkl',
        scaler_path=r'Models\pkl_s\xgboost\scaler.pkl',
        feature_columns_path=r'Models\pkl_s\xgboost\feature_columns.txt'
    )
    sample_processor_viture = Sample_SetProcessor(
        data_path=data_path,
        model_path=r'Models\pkl_s\xgboost\xgboost_model.pkl',
        scaler_path=r'Models\pkl_s\xgboost\scaler.pkl',
        feature_columns_path=r'Models\pkl_s\xgboost\feature_columns.txt'
    )
    df_sample = sample_processor.get_order_data(order_id = order_id)
    df_viture = sample_processor_viture.process_viture()
    return df_sample, df_viture


def load_data_all(data_path, data_path_sample, order_id):
    # 加载训练数据
    df_full = load_data(data_path, sample=sample, sample_num=sample_num, random_state=random_state)
    print("训练数据加载成功！")
    print("数据形状：", df_full.shape)
    # 加载样本集
    df_sample, df_viture = load_data_sample(data_path_sample, order_id=order_id)
    print("样本集加载成功！")
    print("数据形状：", df_sample.shape)
    print("Viture数据形状：", df_viture.shape)
    print("---------")
    # 检查样本集是否与训练数据一致
    if check_file_consistency(df_full, df_sample, df_viture):
        print("数据列一致！")
        print("---------")
    return df_full, df_sample, df_viture


def check_file_consistency(df_full, df_sample, df_viture):
    # 检查样本集是否与训练数据一致
    df_full_cols = df_full.columns.tolist()
    df_sample_cols = df_sample.columns.tolist()
    df_full_cols_no_target = df_full_cols[:-1]  # 不考虑最后一列
    df_viture_cols = [col for col in df_viture.columns if col not in ['order_id', 'courier_id']]
    if df_full_cols != df_sample_cols:
        raise ValueError("样本集与训练数据列不一致")
    if df_full_cols_no_target != df_viture_cols:
        raise ValueError("待预测样本与训练数据列不一致")
    return True


def train_evaluate_model(X_train, y_train, X_test, y_test, similar_set_info_train, similar_set_info_test, max_depth, top_n_num):
    # 测试不同a值的决策树
    print("\n测试不同a值的决策树...，a=0为SPO决策树，a=1.0为传统决策树")
    trees = {}
    for a in [1.0, 0]:
        print(f"\n{'='*60}")
        print(f"测试 a={a} 的决策树")
        print(f"{'='*60}")
        
        tree = SimilaritySetDecisionTree(a=a, max_depth=max_depth, min_samples_split=20, min_samples_leaf=10, top_n=top_n_num)
        tree.fit(X_train, y_train, similar_set_info_train)
        trees[a] = tree
        print(f"决策树深度: {tree.get_depth()}")
        print(f"叶节点数量: {tree.get_n_leaves()}")
        
        # 训练集预测效果
        y_pred_train = tree.predict(X_train)
        mse_train = mean_squared_error(y_train, y_pred_train)
        decision_loss_train = calculate_decision_loss(y_train, y_pred_train, similar_set_info_train['similar_set_id'].values, similar_set_info_train['normalized_weight'].values, top_n=tree.top_n, verbose=False)
        print(f"\n训练集效果:")
        print(f"  MSE: {mse_train:.6f}")
        print(f"  决策损失: {decision_loss_train:.6f}")
        # 测试集预测效果
        y_pred_test = tree.predict(X_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        decision_loss_test = calculate_decision_loss(y_test, y_pred_test, similar_set_info_test['similar_set_id'].values, similar_set_info_test['normalized_weight'].values, top_n=tree.top_n, verbose=False)
        print(f"\n测试集效果:")
        print(f"  MSE: {mse_test:.6f}")
        print(f"  决策损失: {decision_loss_test:.6f}")
        
        print(f"\n对比总结 (a={a}):")
        print(f"  训练集 MSE: {mse_train:.6f} -> 测试集 MSE: {mse_test:.6f} (变化: {mse_test-mse_train:+.6f})")
        print(f"  训练集 决策损失: {decision_loss_train:.6f} -> 测试集 决策损失: {decision_loss_test:.6f} (变化: {decision_loss_test-decision_loss_train:+.6f})")
    
    return trees


def candidate_rider(df_viture, trees, top_n_num):
    # 用a=0的决策树预测
    print("\n使用a=0的决策树预测df_viture...")
    tree_spo = trees[0]
    feature_columns = [col for col in df_viture.columns if col not in ['order_id', 'courier_id']]
    y_pred_viture = tree_spo.predict(df_viture[feature_columns])
    
    result_df_viture = df_viture.copy()
    # 将预测结果添加到结果DataFrame
    result_df_viture['predicted_reject_penalty_ratio'] = y_pred_viture
    
    # 按order_id分组，取预测值最小的top_n_num个
    print(f"\n取每个order_id下预测值最小的top_n_num={top_n_num}个...")
    result_df_viture_sorted = result_df_viture.sort_values(['order_id', 'predicted_reject_penalty_ratio'])
    top_n_result = result_df_viture_sorted.groupby('order_id').head(top_n_num)
    
    return top_n_result


def print_train_test_split(df_train, df_test):
    # 打印训练集和测试集的样本数
    print(f"训练集样本数: {len(df_train)}")
    print(f"测试集样本数: {len(df_test)}")
    print("---------")
    print(f"数据划分:")
    print(f"训练集样本数: {len(df_train)}")
    print(f"测试集样本数: {len(df_test)}")
    print("---------")


if __name__ == '__main__':
    # 筛选排名前top_n_num的骑手
    top_n_num = 6
    # 设置最大树深度max_depth
    max_depth = 3
    # 设置训练数据是否抽样
    sample, sample_num, random_state = True, 30000, 42
    # 样本订单id
    order_id = 522093


    # 加载训练数据和样本集
    data_path = r'.\Data_Process\Data_Processed\all_waybill_info_meituan_v2.csv'
    data_path_sample = r'.\Data_Process\Data_Processed\virtual_dispatch_result_v2.csv'
    df_full, df_sample, df_viture = load_data_all(data_path, data_path_sample, order_id=order_id)
    # 划分训练集和测试集合
    df_train, df_test = train_test_split(df_full, test_size=0.4, random_state=42)
    print_train_test_split(df_train, df_test)

    # 为训练集构建相似集
    print("为训练集构建相似集...")
    builder_train = SimilarSetBuilder(df_train, df_sample, target_col='reject_penalty_ratio', feature_weight=0.7)
    result_df_train = builder_train.build_similarity_sets()
    # 为测试集构建相似集
    print("为测试集构建相似集...")
    builder_test = SimilarSetBuilder(df_test, df_sample, target_col='reject_penalty_ratio', feature_weight=0.7)
    result_df_test = builder_test.build_similarity_sets()
    print("相似集构建完成！")
    print("---------")

    # 定义特征列和目标列
    feature_cols = [col for col in result_df_train.columns if col not in 
                   ['similar_set_id', 'similar_sample_distance', 'similar_set_weight', 'normalized_weight', 'reject_penalty_ratio']]
    # 准备训练集数据
    X_train = result_df_train[feature_cols]
    y_train = result_df_train['reject_penalty_ratio']
    similar_set_info_train = result_df_train[['similar_set_id', 'normalized_weight']].copy()
    # 准备测试集数据
    X_test = result_df_test[feature_cols]
    y_test = result_df_test['reject_penalty_ratio']
    similar_set_info_test = result_df_test[['similar_set_id', 'normalized_weight']].copy()

    # 训练和评估模型
    trees = train_evaluate_model(X_train, y_train, X_test, y_test, similar_set_info_train, similar_set_info_test, max_depth, top_n_num)

    # 候选骑手筛选
    top_n_result = candidate_rider(df_viture, trees, top_n_num)
    # 保存结果
    output_path = r'.\SPOT_Similarity\Data_Processed\candidate_courier_result.csv'
    top_n_result.to_csv(output_path, index=False)
    print(f"\n候选骑手结果已保存到: {output_path}")
    print(f"结果形状: {top_n_result.shape}")



