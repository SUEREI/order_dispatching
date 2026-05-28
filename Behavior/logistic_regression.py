import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, confusion_matrix,
)
import warnings
warnings.filterwarnings('ignore')


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df


def prepare_features_target(df):
    target = 'is_courier_grabbed'
    continuous_features = [
        'fatigue',
        'dispatch_to_sender_delivery_distance',
        'sender_to_recipient_delivery_distance',
        'ongoing_order_count',
        'max_ongoing_sender_to_recipient_distance',
    ]
    binary_features = [
        'is_prebook',
        'is_weekend',
    ]
    features = continuous_features + binary_features

    X = df[features].copy()
    y = df[target].copy()

    feature_info = {
        'continuous': continuous_features,
        'binary': binary_features,
        'all': features,
    }
    return X, y, feature_info


def standardize_features(X_train, X_test, feature_info):
    continuous_cols = feature_info['continuous']
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test_scaled[continuous_cols] = scaler.transform(X_test[continuous_cols])
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_info):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_test_prob = model.predict_proba(X_test)[:, 1]

    tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
    tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()

    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'tpr': tp_train / (tp_train + fn_train) if (tp_train + fn_train) > 0 else 0.0,
            'tnr': tn_train / (tn_train + fp_train) if (tn_train + fp_train) > 0 else 0.0,
            'f1': f1_score(y_train, y_train_pred),
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'tpr': tp_test / (tp_test + fn_test) if (tp_test + fn_test) > 0 else 0.0,
            'tnr': tn_test / (tn_test + fp_test) if (tn_test + fp_test) > 0 else 0.0,
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_prob),
        },
    }

    coeff_df = pd.DataFrame({
        'feature': feature_info['all'],
        'coefficient': model.coef_[0],
        'odds_ratio': np.exp(model.coef_[0]),
    })
    coeff_df = coeff_df.sort_values('coefficient', ascending=False).reset_index(drop=True)

    conf_matrix = confusion_matrix(y_test, y_test_pred)

    return metrics, coeff_df, conf_matrix, y_test_prob


def print_results(metrics, coeff_df, conf_matrix, feature_info):
    lines = []
    lines.append('=' * 60)
    lines.append('多元逻辑回归 - 二分类预测结果')
    lines.append(f'特征数量: {len(feature_info["all"])}')
    lines.append(f'连续特征: {feature_info["continuous"]}')
    lines.append(f'二值特征: {feature_info["binary"]}')
    lines.append('=' * 60)

    label_map = {'accuracy': 'Accuracy', 'tpr': 'True positive rate', 'tnr': 'True negative rate', 'f1': 'F1 score', 'roc_auc': 'ROC-AUC'}

    lines.append('\n[训练集评估]')
    for k in ['accuracy', 'tpr', 'tnr', 'f1']:
        lines.append(f'  {label_map[k]}: {metrics["train"][k]:.4f}')

    lines.append('\n[测试集评估]')
    for k in ['accuracy', 'tpr', 'tnr', 'f1', 'roc_auc']:
        lines.append(f'  {label_map[k]}: {metrics["test"][k]:.4f}')

    lines.append('\n[混淆矩阵 (测试集)]')
    lines.append(f'         预测 0    预测 1')
    lines.append(f'实际 0   {conf_matrix[0,0]:>6}   {conf_matrix[0,1]:>6}')
    lines.append(f'实际 1   {conf_matrix[1,0]:>6}   {conf_matrix[1,1]:>6}')

    tn, fp, fn, tp = conf_matrix.ravel()
    total = tn + fp + fn + tp
    lines.append(f'\n  真负(TN): {tn} ({tn/total*100:.1f}%)')
    lines.append(f'  假正(FP): {fp} ({fp/total*100:.1f}%)')
    lines.append(f'  假负(FN): {fn} ({fn/total*100:.1f}%)')
    lines.append(f'  真正(TP): {tp} ({tp/total*100:.1f}%)')

    lines.append('\n[特征系数 (按影响大小排序)]')
    lines.append(f'{"特征":<45} {"系数":>10} {"优势比(OR)":>10}')
    lines.append('-' * 67)
    for _, row in coeff_df.iterrows():
        lines.append(f'{row["feature"]:<45} {row["coefficient"]:>10.4f} {row["odds_ratio"]:>10.4f}')

    output = '\n'.join(lines)
    print(output)

    return output


def save_results(output, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'logic_regression_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f'\n结果已保存至: {path}')


def main():
    data_path = r'Behavior/Data/Data_Processed/cleaned_waybill_info.csv'
    output_dir = r'Behavior/Results'

    print('加载数据...', flush=True)
    df = load_data(data_path)
    print(f'数据形状: {df.shape}', flush=True)

    print('准备特征和目标变量...', flush=True)
    X, y, feature_info = prepare_features_target(df)
    print(f'样本数: {len(X)}, 特征数: {X.shape[1]}', flush=True)
    print(f'目标分布: 1={y.sum()}, 0={(1-y).sum()}', flush=True)

    print('划分训练集和测试集...', flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'训练集: {len(X_train)}, 测试集: {len(X_test)}', flush=True)

    print('标准化连续特征...', flush=True)
    X_train_scaled, X_test_scaled, scaler = standardize_features(
        X_train, X_test, feature_info
    )

    print('训练逻辑回归模型...', flush=True)
    model = train_logistic_regression(X_train_scaled, y_train)

    print('评估模型...', flush=True)
    metrics, coeff_df, conf_matrix, y_test_prob = evaluate_model(
        model, X_train_scaled, y_train, X_test_scaled, y_test, feature_info
    )

    output = print_results(metrics, coeff_df, conf_matrix, feature_info)
    save_results(output, output_dir)


if __name__ == '__main__':
    main()
