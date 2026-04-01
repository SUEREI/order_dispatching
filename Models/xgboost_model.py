import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, log_loss
from sklearn.metrics import matthews_corrcoef
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import pickle
import os

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f'加载数据完成，共 {len(df)} 条记录，数据形状为 {df.shape}')
    return df

def main():
    data_path = r'D:\Sys\Desktop\毕业论文\2 Python\all_waybill_info_meituan_v2.csv'
    
    if not os.path.exists(data_path):
        print(f'错误：数据文件 {data_path} 不存在')
        return
    
    df = load_data(data_path)
    
    print('\n当前字段名称：')
    print(df.columns.tolist())
    
    old_names = [
        # 距离相关特征
        'order_distance_km', 'rider_to_order_km', 
        # 工作量相关特征
        'ongoing_order_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2', 
        # 新订单与现有订单适配性相关特征
        'area_change_ratio', 'direction_similarity', 
        # 疲惫程度相关特征
        'courier_daily_completed_distance_km', 'courier_session_completed_distance_km',
        # 区域熟悉度相关特征
        'area_familiarity', 
        # 时间相关特征
        'is_weekend', 'dispatch_hour'
    ]
    
    new_names = [
        'sender_to_recipient_km', 'dispatch_to_sender_km',
        'ongoing_other_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2',
        'area_increase_ratio', 'direction_similarity',
        'courier_daily_completed_before_dispatch_distance', 'courier_daily_completed_before_dispatch_with_reset_distance',
        'area_familiarity',
        'is_weekend', 'dispatch_hour'
    ]
    
    name_mapping = dict(zip(old_names, new_names))
    
    df = df.rename(columns=name_mapping)
    
    feature_columns = [
        'sender_to_recipient_km', 'dispatch_to_sender_km',
        'ongoing_other_count', 'ongoing_cross_area_count', 'ongoing_max_distance_km', 'ongoing_area_km2',
        'area_increase_ratio', 'direction_similarity',
        'courier_daily_completed_before_dispatch_distance', 'courier_daily_completed_before_dispatch_with_reset_distance',
        'area_familiarity',
        'is_weekend', 'dispatch_hour'
    ]
    
    target_column = 'is_courier_grabbed'
    
    available_features = [col for col in feature_columns if col in df.columns]
    print(f'\n可用特征 ({len(available_features)}个)：{available_features}')
    
    X = df[available_features].copy()
    y = df[target_column].copy()
    
    print(f'\n缺失值统计：')
    print(X.isnull().sum())
    print(f'\n目标变量分布：')
    print(y.value_counts())
    
    X = X.fillna(X.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f'\n原始训练集分布：')
    print(y_train.value_counts().sort_index())
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    
    print(f'\nSMOTE过采样后训练集分布：')
    print(y_train_resampled.value_counts().sort_index())
    
    model = XGBClassifier(
        random_state=42,
        scale_pos_weight=(y_train_resampled.value_counts()[0] / y_train_resampled.value_counts()[1]),
        max_depth=6,
        learning_rate=0.1,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8
    )
    model.fit(X_train_resampled, y_train_resampled)
    
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    best_threshold = 0.5
    # best_f1 = 0
    # for threshold in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    #     y_pred_temp = (y_pred_proba >= threshold).astype(int)
    #     f1_temp = f1_score(y_test, y_pred_temp, zero_division=0)
    #     if f1_temp > best_f1:
    #         best_f1 = f1_temp
    #         best_threshold = threshold
    
    # print(f'\n最佳分类阈值: {best_threshold}')
    
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    try:
        auc = roc_auc_score(y_test, y_pred_proba)
    except:
        auc = 0.5
    
    logloss = log_loss(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = cm.ravel()
    tpr = recall
    tnr = TN / (TN + FP)
    prevalence = (TP + FN) / len(y_test)
    gini = 2 * auc - 1
    mcc = matthews_corrcoef(y_test, y_pred)
    
    print('\n' + '='*50)
    print('模型评价指标')
    print('='*50)
    print(f'准确率 (Accuracy):               {accuracy:.4f}')
    print(f'True Positive Rate (Sensitivity): {tpr:.4f}')
    print(f'True Negative Rate (Specificity): {tnr:.4f}')
    print(f'精确率 (Precision):             {precision:.4f}')
    print(f'F1分数 (F1-Score):              {f1:.4f}')
    print(f'ROC AUC:                        {auc:.4f}')
    print(f'Log Loss (Cross-Entropy):       {logloss:.4f}')
    print(f'Gini Index:                     {gini:.4f}')
    print(f'Prevalence:                     {prevalence:.4f}')
    print(f'Matthews Correlation Coefficient: {mcc:.4f}')
    
    print('\n分类报告：')
    print(classification_report(y_test, y_pred, zero_division=0))
    
    print('混淆矩阵：')
    print(cm)
    
    feature_importance = pd.DataFrame({
        '特征': available_features,
        '重要性': model.feature_importances_
    }).sort_values('重要性', ascending=False)
    
    print('\n特征重要性：')
    print(feature_importance)
    
    model_dir = r'D:\Sys\Desktop\毕业论文\2 Python\models\pkl_s\xgboost'
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'xgboost_model.pkl')
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    features_path = os.path.join(model_dir, 'feature_columns.txt')
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f'\n模型已保存到：{model_path}')
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'标准化器已保存到：{scaler_path}')
    
    with open(features_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(available_features))
    print(f'特征列表已保存到：{features_path}')
    
    params_path = os.path.join(model_dir, 'model_params.txt')
    with open(params_path, 'w', encoding='utf-8') as f:
        f.write('XGBoost模型参数\n')
        f.write('='*50 + '\n\n')
        f.write(f'准确率 (Accuracy):               {accuracy:.4f}\n')
        f.write(f'True Positive Rate (Sensitivity): {tpr:.4f}\n')
        f.write(f'True Negative Rate (Specificity): {tnr:.4f}\n')
        f.write(f'精确率 (Precision):             {precision:.4f}\n')
        f.write(f'F1分数 (F1-Score):              {f1:.4f}\n')
        f.write(f'ROC AUC:                        {auc:.4f}\n')
        f.write(f'Log Loss (Cross-Entropy):       {logloss:.4f}\n')
        f.write(f'Gini Index:                     {gini:.4f}\n')
        f.write(f'Prevalence:                     {prevalence:.4f}\n')
        f.write(f'Matthews Correlation Coefficient: {mcc:.4f}\n\n')
        f.write('混淆矩阵：\n')
        f.write(str(cm) + '\n\n')
        f.write('特征重要性：\n')
        f.write(feature_importance.to_string(index=False))
    print(f'模型参数已保存到：{params_path}')

if __name__ == '__main__':
    main()
