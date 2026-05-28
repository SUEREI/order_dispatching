import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
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
        # 'is_prebook',
        # 'is_weekend',
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


def train_decision_tree(X_train, y_train, X_val, y_val):
    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        ccp_alpha=0.001,
        class_weight='balanced',
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_test, y_test, feature_info):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]

    train_cm = confusion_matrix(y_train, y_train_pred)
    test_cm = confusion_matrix(y_test, y_test_pred)

    metrics = {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred),
            'recall': recall_score(y_train, y_train_pred),
            'specificity': train_cm[0, 0] / (train_cm[0, 0] + train_cm[0, 1]),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, y_train_prob),
        },
        'test': {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred),
            'recall': recall_score(y_test, y_test_pred),
            'specificity': test_cm[0, 0] / (test_cm[0, 0] + test_cm[0, 1]),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_prob),
        },
    }

    importance_df = pd.DataFrame({
        'feature': feature_info['all'],
        'importance': model.feature_importances_,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return metrics, importance_df, y_test_pred, y_test_prob, train_cm, test_cm


def print_results(metrics, importance_df, feature_info, model, train_cm, test_cm):
    depth = model.get_depth()
    n_leaves = model.get_n_leaves()

    lines = []
    lines.append('=' * 60)
    lines.append('Decision Tree Classification - Predict is_courier_grabbed')
    lines.append('=' * 60)
    lines.append(f'Number of features: {len(feature_info["all"])}')
    lines.append(f'Tree depth: {depth}')
    lines.append(f'Number of leaves: {n_leaves}')
    lines.append(f'Continuous features: {feature_info["continuous"]}')
    lines.append(f'Binary features: {feature_info["binary"]}')

    lines.append('\n[Train Set Evaluation]')
    lines.append(f'  Accuracy:  {metrics["train"]["accuracy"]:.4f}')
    lines.append(f'  Precision: {metrics["train"]["precision"]:.4f}')
    lines.append(f'  Recall:    {metrics["train"]["recall"]:.4f}')
    lines.append(f'  Specificity:{metrics["train"]["specificity"]:.4f}')
    lines.append(f'  F1 Score:  {metrics["train"]["f1"]:.4f}')
    lines.append(f'  ROC-AUC:   {metrics["train"]["roc_auc"]:.4f}')
    lines.append(f'  Confusion Matrix:')
    lines.append(f'    TN={train_cm[0,0]:>6}  FP={train_cm[0,1]:>6}')
    lines.append(f'    FN={train_cm[1,0]:>6}  TP={train_cm[1,1]:>6}')

    lines.append('\n[Test Set Evaluation]')
    lines.append(f'  Accuracy:  {metrics["test"]["accuracy"]:.4f}')
    lines.append(f'  Precision: {metrics["test"]["precision"]:.4f}')
    lines.append(f'  Recall:    {metrics["test"]["recall"]:.4f}')
    lines.append(f'  Specificity:{metrics["test"]["specificity"]:.4f}')
    lines.append(f'  F1 Score:  {metrics["test"]["f1"]:.4f}')
    lines.append(f'  ROC-AUC:   {metrics["test"]["roc_auc"]:.4f}')
    lines.append(f'  Confusion Matrix:')
    lines.append(f'    TN={test_cm[0,0]:>6}  FP={test_cm[0,1]:>6}')
    lines.append(f'    FN={test_cm[1,0]:>6}  TP={test_cm[1,1]:>6}')

    lines.append('\n[Feature Importance]')
    lines.append(f'{"Feature":<50} {"Importance":>10}')
    lines.append('-' * 62)
    for _, row in importance_df.iterrows():
        lines.append(f'{row["feature"]:<50} {row["importance"]:>10.4f}')

    output = '\n'.join(lines)
    print(output)

    return output


def save_results(output, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, 'decision_tree_results.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(output)
    print(f'\nResults saved to: {path}')


def save_model(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, 'decision_tree_model.pkl')
    joblib.dump(model, path)
    print(f'Model saved to: {path}')


def main():
    data_path = r'SPOT/Data/Data_Processed/cleaned_waybill_info.csv'
    output_dir = r'SPOT/Results'

    print('加载数据...', flush=True)
    df = load_data(data_path)
    print(f'数据形状: {df.shape}', flush=True)

    print('准备特征和目标变量...', flush=True)
    X, y, feature_info = prepare_features_target(df)
    print(f'样本数: {len(X)}, 特征数: {X.shape[1]}', flush=True)
    print(f'目标描述:\n{y.describe()}', flush=True)

    print('划分训练集和测试集...', flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'训练集: {len(X_train)}, 测试集: {len(X_test)}', flush=True)

    print('训练决策树分类模型...', flush=True)
    model = train_decision_tree(X_train, y_train, X_test, y_test)
    print(f'树深度: {model.get_depth()}, 叶子节点数: {model.get_n_leaves()}', flush=True)

    print('评估模型...', flush=True)
    metrics, importance_df, y_test_pred, y_test_prob, train_cm, test_cm = evaluate_model(
        model, X_train, y_train, X_test, y_test, feature_info
    )

    output = print_results(metrics, importance_df, feature_info, model, train_cm, test_cm)
    save_results(output, output_dir)

    model_dir = r'SPOT/core/pkl_s'
    save_model(model, model_dir)


if __name__ == '__main__':
    main()
