import ast
from datetime import datetime
from typing import Dict, List, Set, Any, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from core.route import RoutePlanner
from core.order_d_dqn import OrderDispatcherByActorCritic
from dqn.dqn_network import ActorCriticConfig
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


pd.set_option('display.max_columns', None)

COORDINATE_SCALE = 10 ** 6


def build_order_lookup(df: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """构建订单查询字典 {waybill_id: {sender/recipient坐标, 时间信息}}"""
    lookup = {}
    for _, row in df.iterrows():
        lookup[row['waybill_id']] = {
            'sender_lng': row['sender_lng'] / COORDINATE_SCALE,
            'sender_lat': row['sender_lat'] / COORDINATE_SCALE,
            'recipient_lng': row['recipient_lng'] / COORDINATE_SCALE,
            'recipient_lat': row['recipient_lat'] / COORDINATE_SCALE,
            'estimate_arrived_time': row['estimate_arrived_time'],
            'grab_time': row['grab_time'],
            'fetch_time': row['fetch_time'],
            'arrive_time': row['arrive_time'],
        }
    logger.info(f'已构建 {len(lookup)} 条订单查询记录')
    return lookup


def parse_ongoing_set(ongoing_set_val) -> Set[int]:
    """解析ongoing_order_set列，返回waybill_id集合"""
    if isinstance(ongoing_set_val, set):
        return ongoing_set_val
    if isinstance(ongoing_set_val, str):
        parsed = ast.literal_eval(ongoing_set_val)
        if isinstance(parsed, set):
            return parsed
    return set()


def determine_order_status(
    order_detail: Dict[str, Any],
    dispatch_time: datetime
) -> str:
    """
    根据订单详情和派单时间确定订单状态
    - 'pending': 已接单但未取餐
    - 'picked_up': 已取餐但未送达
    """
    fetch_time = order_detail.get('fetch_time')
    arrive_time = order_detail.get('arrive_time')

    if pd.notna(fetch_time) and fetch_time <= dispatch_time:
        return 'picked_up'

    return 'pending'


def build_courier_for_routing(
    row: pd.Series,
    order_lookup: Dict[int, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    将数据表中的一行转换为 RoutePlanner.plan_route 需要的 courier dict
    """
    courier_id = str(row['courier_id'])

    position = (
        row['dispatch_lng'] / COORDINATE_SCALE,
        row['dispatch_lat'] / COORDINATE_SCALE
    )

    dispatch_time = row['dispatch_time']

    ongoing_waybills = parse_ongoing_set(row['ongoing_order_set'])

    current_orders = []
    for wid in ongoing_waybills:
        detail = order_lookup.get(wid)
        if detail is None:
            continue

        status = determine_order_status(detail, dispatch_time)

        estimate_arrived_time = detail.get('estimate_arrived_time')

        current_orders.append({
            'order_id': wid,
            'sender_lng': detail['sender_lng'],
            'sender_lat': detail['sender_lat'],
            'recipient_lng': detail['recipient_lng'],
            'recipient_lat': detail['recipient_lat'],
            'estimate_arrived_time': estimate_arrived_time,
            'status': status,
        })

    courier = {
        'courier_id': courier_id,
        'position': position,
        'current_orders': current_orders,
    }

    return courier


def process_dispatch_data(
    df: pd.DataFrame,
    order_lookup: Dict[int, Dict[str, Any]],
    route_planner: RoutePlanner
) -> pd.DataFrame:
    """
    批量处理派单数据，为每条记录规划骑手路径
    返回包含规划结果的DataFrame副本
    """
    df_result = df.copy()
    courier_info = []

    iter_data = df.iterrows()

    for idx, row in tqdm(iter_data, total=len(df), desc='规划路径'):
        courier = build_courier_for_routing(row, order_lookup)
        courier_info.append(courier)

    logger.info(f'构建完成 {len(courier_info)} 条骑手记录，其中 {sum(1 for r in courier_info if r["current_orders"])} 条有在手订单')

    routes = []
    for courier in tqdm(courier_info, desc='计算路径'):
        try:
            route = route_planner.plan_route(courier)
            routes.append(route)
        except Exception as e:
            logger.error(f"处理骑手 {courier['courier_id']} 时出错: {e}")
            routes.append([])

    df_result['planned_route'] = routes

    return df_result


def build_state_action_vector(
    df: pd.DataFrame,
    order_lookup: Dict[int, Dict[str, Any]],
    route_planner: RoutePlanner,
    dispatcher: OrderDispatcherByActorCritic,
) -> np.ndarray:
    """
    为每条派单记录构建状态-动作合并向量 (27+5=32维)

    状态向量（27维, _build_state）：
      0-1:   骑手当前位置经度、纬度（归一化）
      2:     当前时间步的时间戳（归一化 0~1）
      3-18:  路径点 1~8 的经纬度（归一化, 8*2=16维）
      19-26: 路径点 1~8 的超时剩余时间（归一化 0~1, 8维）

    动作向量（5维, _build_action）：
      0-1:   新订单取餐点经度、纬度（归一化）
      2-3:   新订单送餐点经度、纬度（归一化）
      4:     新订单可用配送时长（归一化 0~1）

    Args:
        df: 派单数据 DataFrame
        order_lookup: 订单查询字典 {waybill_id: detail}
        route_planner: 路径规划器
        dispatcher: Actor-Critic 分派器实例（提供 _build_state / _build_action）

    Returns:
        np.ndarray, shape (n_rows, 32), 每行 = [state_27d | action_5d]
    """
    n = len(df)
    sa_dim = dispatcher.STATE_DIM + dispatcher.ACTION_DIM
    sa_vectors = np.zeros((n, sa_dim), dtype=np.float32)

    for idx, row in tqdm(df.iterrows(), total=n, desc='构建状态-动作向量'):
        dispatch_time = row['dispatch_time']

        # ===== 状态向量 (27维) =====
        courier = build_courier_for_routing(row, order_lookup)
        courier['route'] = route_planner.plan_route(courier)
        state = dispatcher._build_state(courier, dispatch_time)

        # ===== 动作向量 (5维) =====
        eta_new_order = row['estimate_arrived_time']
        order_action = {
            'sender_lng': row['sender_lng'] / COORDINATE_SCALE,
            'sender_lat': row['sender_lat'] / COORDINATE_SCALE,
            'recipient_lng': row['recipient_lng'] / COORDINATE_SCALE,
            'recipient_lat': row['recipient_lat'] / COORDINATE_SCALE,
            'estimate_arrived_time': eta_new_order,
        }
        action = dispatcher._build_action(order_action, dispatch_time)

        # ===== 合并 =====
        sa_vectors[idx] = np.concatenate([state, action])

    logger.info(
        f'状态-动作向量构建完成: {n} 条, 维度 {sa_dim} '
        f'[{dispatcher.STATE_DIM}(state) + {dispatcher.ACTION_DIM}(action)]'
    )
    return sa_vectors


class GrabPredictionMLP(nn.Module):
    """
    MLP 二分类网络

    输入: 状态-动作合并特征向量 (feature_dim,)
    输出: 该订单被骑手接取的概率 logit (1,)

    结构: MLP + ReLU + Dropout → Sigmoid 输出
    """

    def __init__(self, feature_dim: int = 32, hidden_dims: List[int] = None, dropout: float = 0.3):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        input_dim = feature_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.net(x))


def prepare_classification_data(
    df: pd.DataFrame,
    order_lookup: Dict[int, Dict[str, Any]],
    route_planner: RoutePlanner,
    test_size: float = 0.2,
    random_state: int = 42,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray, int]:
    """准备 MLP 二分类训练/测试数据

    流程:
    1. 调用 build_state_action_vector 构建 32 维特征矩阵 X
    2. 从 df 中提取 is_courier_grabbed 作为标签 y
    3. 按比例划分训练/测试集
    4. 封装为 PyTorch DataLoader

    Args:
        df: 派单数据 DataFrame（必须包含 is_courier_grabbed 列）
        order_lookup: 订单查询字典
        route_planner: 路径规划器
        test_size: 测试集比例
        random_state: 随机种子
        batch_size: DataLoader 批次大小

    Returns:
        (train_loader, test_loader, X, y, feature_dim)
    """
    config = ActorCriticConfig()
    dispatcher = OrderDispatcherByActorCritic(route_planner, config)

    logger.info('开始构建特征向量...')
    X = build_state_action_vector(df, order_lookup, route_planner, dispatcher)

    y = df['is_courier_grabbed'].values.astype(np.float32)
    n_positive = int(y.sum())
    n_negative = len(y) - n_positive
    logger.info(
        f'标签分布: 正例={n_positive} ({n_positive/len(y):.2%}), '
        f'负例={n_negative} ({n_negative/len(y):.2%})'
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(
        f'数据集划分: 训练集 {len(X_train)} 条, 测试集 {len(X_test)} 条'
    )

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test).unsqueeze(1)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X, y, X.shape[1]


def train_grab_classifier(
    train_loader: DataLoader,
    test_loader: DataLoader,
    feature_dim: int,
    hidden_dims: List[int] = None,
    dropout: float = 0.3,
    lr: float = 1e-3,
    epochs: int = 200,
    weight_decay: float = 1e-4,
    early_stop_patience: int = 15,
    pos_weight: float = None,
    device: str = None,
) -> Tuple[GrabPredictionMLP, Dict[str, List[float]]]:
    """训练 GrabPredictionMLP 二分类器

    Args:
        train_loader: 训练集 DataLoader
        test_loader: 测试集 DataLoader
        feature_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        dropout: Dropout 概率
        lr: 学习率
        epochs: 最大训练轮数
        weight_decay: L2 正则化系数
        early_stop_patience: 早停耐心值（监控验证损失）
        pos_weight: 正类权重（负样本数/正样本数），用于缓解类别不平衡
        device: 训练设备 (auto-detect if None)

    Returns:
        (trained_model, history)
        history = {'train_loss': [...], 'val_loss': [...], 'val_auc': [...]}
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model = GrabPredictionMLP(feature_dim, hidden_dims, dropout).to(device)
    pos_weight_t = torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc': [],
    }

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    logger.info(f'训练设备: {device}')
    logger.info(f'MLP 结构: {feature_dim} → {hidden_dims} → 1, Dropout={dropout}')
    logger.info(f'优化器: AdamW, lr={lr}, weight_decay={weight_decay}')

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_loss = 0.0
        train_batches = 0

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{epochs}', leave=False)
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            train_batches += 1
            train_pbar.set_postfix(loss=loss.item())

        avg_train_loss = epoch_train_loss / max(train_batches, 1)

        model.eval()
        epoch_val_loss = 0.0
        val_batches = 0
        all_val_probs = []
        all_val_labels = []

        with torch.no_grad():
            val_pbar = tqdm(test_loader, desc=f'验证    {epoch:3d}/{epochs}', leave=False)
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                loss = criterion(logits, batch_y)
                epoch_val_loss += loss.item()
                val_batches += 1

                probs = torch.sigmoid(logits)
                all_val_probs.append(probs.cpu().numpy())
                all_val_labels.append(batch_y.cpu().numpy())

        avg_val_loss = epoch_val_loss / max(val_batches, 1)

        val_probs = np.concatenate(all_val_probs).ravel()
        val_labels = np.concatenate(all_val_labels).ravel()
        val_auc = roc_auc_score(val_labels, val_probs) if len(np.unique(val_labels)) > 1 else 0.5

        scheduler.step(avg_val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_auc'].append(val_auc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch == 1 or epoch % 20 == 0 or patience_counter == 0:
            logger.info(
                f'Epoch {epoch:3d}/{epochs} | '
                f'Train Loss: {avg_train_loss:.4f} | '
                f'Val Loss: {avg_val_loss:.4f} | '
                f'Val AUC: {val_auc:.4f} | '
                f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
            )

        if patience_counter >= early_stop_patience:
            logger.info(f'早停触发: 验证损失 {early_stop_patience} 轮未下降')
            break

    model.load_state_dict(best_model_state)
    logger.info(f'训练完成, 最佳验证损失: {best_val_loss:.4f}')

    return model, history


def evaluate_grab_classifier(
    model: GrabPredictionMLP,
    test_loader: DataLoader,
    device: str = None,
) -> Dict[str, float]:
    """全面评估二分类器性能

    Args:
        model: 训练好的 GrabPredictionMLP
        test_loader: 测试集 DataLoader
        device: 计算设备

    Returns:
        metrics: 评估指标字典
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            logits = model(batch_X)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch_y.numpy())

    y_prob = np.concatenate(all_probs).ravel()
    y_pred = np.concatenate(all_preds).ravel()
    y_true = np.concatenate(all_labels).ravel()

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
    }

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    metrics['true_negative'] = int(tn)
    metrics['false_positive'] = int(fp)
    metrics['false_negative'] = int(fn)
    metrics['true_positive'] = int(tp)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    logger.info('=' * 50)
    logger.info('二分类评估结果')
    logger.info('=' * 50)
    logger.info(f'准确率 (Accuracy):  {metrics["accuracy"]:.4f}')
    logger.info(f'精确率 (Precision): {metrics["precision"]:.4f}')
    logger.info(f'召回率 (Recall):    {metrics["recall"]:.4f}')
    logger.info(f'F1 分数:            {metrics["f1_score"]:.4f}')
    logger.info(f'AUC-ROC:            {metrics["roc_auc"]:.4f}')
    logger.info(f'特异度 (Specificity): {metrics["specificity"]:.4f}')
    logger.info('-' * 50)
    logger.info(f'混淆矩阵:')
    logger.info(f'  TN={tn}  FP={fp}')
    logger.info(f'  FN={fn}  TP={tp}')
    logger.info('=' * 50)

    return metrics


def plot_training_history(history: Dict[str, List[float]], save_path: str = None) -> None:
    """绘制训练曲线"""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        epochs_range = range(1, len(history['train_loss']) + 1)

        axes[0].plot(epochs_range, history['train_loss'], label='Train Loss', alpha=0.7)
        axes[0].plot(epochs_range, history['val_loss'], label='Val Loss', alpha=0.7)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('MLP 训练/验证损失')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(epochs_range, history['val_auc'], label='Val AUC', color='green')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('AUC')
        axes[1].set_title('验证集 AUC')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f'训练曲线已保存到 {save_path}')
        else:
            plt.show()

    except ImportError:
        logger.warning('matplotlib 未安装，跳过绘图')


def main():
    csv_path = r'D:\Sys\Desktop\毕业论文\2 Python\Q_Values\Data\Data_Processed\cleaned_waybill_info_behavior.csv'
    df = pd.read_csv(csv_path)

    time_cols = [
        'dispatch_time', 'estimate_arrived_time',
        'grab_time', 'fetch_time', 'arrive_time',
    ]
    for col in time_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    order_lookup = build_order_lookup(df)
    route_planner = RoutePlanner()

    logger.info('=' * 60)
    logger.info('开始 MLP 二分类训练 (is_courier_grabbed 预测)')
    logger.info('=' * 60)

    train_loader, test_loader, X, y, feature_dim = prepare_classification_data(
        df, order_lookup, route_planner,
        test_size=0.2, batch_size=128,
    )

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f'正类权重 pos_weight={pos_weight:.4f}（负样本 {n_neg} / 正样本 {n_pos}）')

    model, history = train_grab_classifier(
        train_loader, test_loader,
        feature_dim=feature_dim,
        hidden_dims=[128, 64, 32],
        dropout=0.3,
        lr=1e-3,
        epochs=10,
        weight_decay=1e-4,
        early_stop_patience=15,
        pos_weight=pos_weight,
    )

    metrics = evaluate_grab_classifier(model, test_loader)

    save_dir = Path(csv_path).parent.parent.parent / 'simulation' / 'behavior'
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / 'grab_prediction_mlp.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f'模型已保存到 {model_path}')

    import json
    history_path = save_dir / 'training_history.json'
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history_serializable, f, indent=2, ensure_ascii=False)
    logger.info(f'训练历史已保存到 {history_path}')

    plot_path = save_dir / 'training_history.png'
    plot_training_history(history, str(plot_path))

    logger.info(df.head())


if __name__ == '__main__':
    main()
