"""绘制实验结果相关图片"""
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from _basic_set import plt_set, save_fig, BASE_DIR, RESULT_DIR

LOSSES_JSON = os.path.join(
    BASE_DIR, "..", "Q_Values", "simulation", "dqn", "checkpoints", "dqn_model_losses.json"
)
PERIOD_METRICS_JSON = os.path.join(
    BASE_DIR, "..", "Q_Values", "simulation", "dqn", "checkpoints", "period_metrics.json"
)

colors = [plt.cm.Set2(i) for i in range(4)]


def comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_1(save_name='None'):
    """绘制预测效用（不同深度下 CART 与 SPO 的 MSE 损失对比条形图）"""
    depth = [1, 2, 3, 4]
    MSE_Loss_CART = [0.1031, 0.0988, 0.0976, 0.0961]
    MSE_Loss_SPO = [0.1127, 0.1094, 0.1090, 0.1087]

    x = np.arange(len(depth))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 5))
    bars1 = ax.bar(x - width / 2, MSE_Loss_CART, width, label="传统决策树",
                   color=colors[0], edgecolor="white", hatch="///")
    bars2 = ax.bar(x + width / 2, MSE_Loss_SPO, width, label="SPO决策树",
                   color=colors[1], edgecolor="white")

    # 在柱子上方标注数值
    top = max(MSE_Loss_CART + MSE_Loss_SPO)
    for bar, val in zip(bars1, MSE_Loss_CART):
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.06, bar.get_height() + top * 0.01,
                f"{val:.4f}", ha="center", va="bottom")
    for bar, val in zip(bars2, MSE_Loss_SPO):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01,
                f"{val:.4f}", ha="center", va="bottom")

    ax.set_xlabel("深度")
    ax.set_xticks(x)
    ax.set_xticklabels(depth)
    ax.set_ylabel("MSE损失")
    ax.set_ylim(bottom=0, top=max(MSE_Loss_SPO) * 1.3)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(-0.005, 0.95, "(a)", fontsize=12, va='top', ha='left')

    # 额外保存一份高分辨率 JPG（600 dpi）供论文插图使用
    if save_name != 'None':
        jpg_path = os.path.join(
            RESULT_DIR, os.path.splitext(save_name)[0] + '_hires.jpg'
        )
        fig.savefig(jpg_path, dpi=600, bbox_inches='tight',
                    facecolor='white', pad_inches=0.02)
        print(f"高清图片保存：{jpg_path}")

    save_fig(save_name)


def comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_2(save_name='None'):
    """绘制决策效用（不同深度下 CART 与 SPO 的 SPO 损失对比条形图）"""
    depth = [1, 2, 3, 4]
    SPO_Loss_CART = [22.84, 20.11, 18.23, 17.57]
    SPO_Loss_SPO = [20.39, 18.44, 17.86, 17.36]

    x = np.arange(len(depth))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 5))
    bars1 = ax.bar(x - width / 2, SPO_Loss_CART, width, label="传统决策树",
                   color=colors[0], edgecolor="white", hatch="///")
    bars2 = ax.bar(x + width / 2, SPO_Loss_SPO, width, label="SPO决策树",
                   color=colors[1], edgecolor="white")

    # 在柱子上方标注数值
    top = max(SPO_Loss_CART + SPO_Loss_SPO)
    for bar, val in zip(bars1, SPO_Loss_CART):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
                f"{val:.2f}", ha="center", va="bottom")
    for bar, val in zip(bars2, SPO_Loss_SPO):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.00,
                f"{val:.2f}", ha="center", va="bottom")

    ax.set_xlabel("深度")
    ax.set_xticks(x)
    ax.set_xticklabels(depth)
    ax.set_ylabel("SPO损失")
    ax.set_ylim(bottom=0, top=max(SPO_Loss_CART) * 1.08)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(0.02, 0.95, "(b)", fontsize=12, va='top', ha='left')

    # 额外保存一份高分辨率 JPG（600 dpi）供论文插图使用
    if save_name != 'None':
        jpg_path = os.path.join(
            RESULT_DIR, os.path.splitext(save_name)[0] + '_hires.jpg'
        )
        fig.savefig(jpg_path, dpi=600, bbox_inches='tight',
                    facecolor='white', pad_inches=0.02)
        print(f"高清图片保存：{jpg_path}")

    save_fig(save_name)


def comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_1_en(save_name='None'):
    """英文版：预测效用（不同深度下 CART 与 SPO 的 MSE 损失对比条形图）"""
    depth = [1, 2, 3, 4]
    MSE_Loss_CART = [0.1031, 0.0988, 0.0976, 0.0961]
    MSE_Loss_SPO = [0.1127, 0.1094, 0.1090, 0.1087]

    x = np.arange(len(depth))
    width = 0.35
    fig, ax = plt.subplots(figsize=(4, 5))
    bars1 = ax.bar(x - width / 2, MSE_Loss_CART, width, label="CART",
                   color=colors[0], edgecolor="white", hatch="///")
    bars2 = ax.bar(x + width / 2, MSE_Loss_SPO, width, label="SPOT",
                   color=colors[1], edgecolor="white")

    # 在柱子上方标注数值
    top = max(MSE_Loss_CART + MSE_Loss_SPO)
    for bar, val in zip(bars1, MSE_Loss_CART):
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.06, bar.get_height() + top * 0.01,
                f"{val:.4f}", ha="center", va="bottom")
    for bar, val in zip(bars2, MSE_Loss_SPO):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.01,
                f"{val:.4f}", ha="center", va="bottom")

    ax.set_xlabel("Depth")
    ax.set_xticks(x)
    ax.set_xticklabels(depth)
    ax.set_ylabel("MSE Value")
    ax.set_ylim(bottom=0, top=max(MSE_Loss_SPO) * 1.3)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(-0.005, 0.95, "(a)", fontsize=12, va='top', ha='left')

    # 额外保存一份高分辨率 JPG（600 dpi）供论文插图使用
    if save_name != 'None':
        jpg_path = os.path.join(
            RESULT_DIR, os.path.splitext(save_name)[0] + '_hires.jpg'
        )
        fig.savefig(jpg_path, dpi=600, bbox_inches='tight',
                    facecolor='white', pad_inches=0.02)
        print(f"高清图片保存：{jpg_path}")

    save_fig(save_name)


def comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_2_en(save_name='None'):
    """英文版：决策效用（不同深度下 CART 与 SPO 的 SPO 损失对比条形图）"""
    depth = [1, 2, 3, 4]
    SPO_Loss_CART = [22.84, 20.11, 18.23, 17.57]
    SPO_Loss_SPO = [20.39, 18.44, 17.86, 17.36]

    x = np.arange(len(depth))
    width = 0.35

    fig, ax = plt.subplots(figsize=(4, 5))
    bars1 = ax.bar(x - width / 2, SPO_Loss_CART, width, label="CART",
                   color=colors[0], edgecolor="white", hatch="///")
    bars2 = ax.bar(x + width / 2, SPO_Loss_SPO, width, label="SPOT",
                   color=colors[1], edgecolor="white")

    # 在柱子上方标注数值
    top = max(SPO_Loss_CART + SPO_Loss_SPO)
    for bar, val in zip(bars1, SPO_Loss_CART):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02,
                f"{val:.2f}", ha="center", va="bottom")
    for bar, val in zip(bars2, SPO_Loss_SPO):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.00,
                f"{val:.2f}", ha="center", va="bottom")

    ax.set_xlabel("Depth")
    ax.set_xticks(x)
    ax.set_xticklabels(depth)
    ax.set_ylabel("SPO Loss")
    ax.set_ylim(bottom=0, top=max(SPO_Loss_CART) * 1.08)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(0.02, 0.95, "(b)", fontsize=12, va='top', ha='left')

    # 额外保存一份高分辨率 JPG（600 dpi）供论文插图使用
    if save_name != 'None':
        jpg_path = os.path.join(
            RESULT_DIR, os.path.splitext(save_name)[0] + '_hires.jpg'
        )
        fig.savefig(jpg_path, dpi=600, bbox_inches='tight',
                    facecolor='white', pad_inches=0.02)
        print(f"高清图片保存：{jpg_path}")

    save_fig(save_name)


def _load_dqn_losses():
    """加载 DQN 训练损失数据"""
    with open(LOSSES_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def _smooth_losses(losses):
    """高斯滤波平滑（sigma 按数据长度比例动态调整）"""
    sigma = max(1.5, len(losses) / 20)
    return gaussian_filter1d(losses, sigma=sigma)


def dqn_actor_training_loss_curve(save_name='None'):
    """绘制 DQN Actor 网络的 Epoch 平均损失曲线（含平滑线）"""
    actor_losses = _load_dqn_losses()['epoch_actor_losses']
    epochs = range(1, len(actor_losses) + 1)
    smooth = _smooth_losses(actor_losses)

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(epochs, actor_losses, label="Q网络损失", color=colors[0],
            linewidth=1.5)
    ax.plot(epochs, smooth, label="Q网络损失 (平滑)", color=colors[1],
            linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("平均损失")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(0.02, 0.95, "(a)", fontsize=12, va='top', ha='left')

    save_fig(save_name)


def dqn_critic_training_loss_curve(save_name='None'):
    """绘制 DQN Critic 网络的 Epoch 平均损失曲线（含平滑线）"""
    critic_losses = _load_dqn_losses()['epoch_critic_losses']
    epochs = range(1, len(critic_losses) + 1)
    smooth = _smooth_losses(critic_losses)

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.plot(epochs, critic_losses, label="V网络损失", color=colors[2],
            linewidth=1.5)
    ax.plot(epochs, smooth, label="V网络损失 (平滑)", color=colors[3],
            linewidth=1.0, linestyle="--", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("平均损失")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # 子图编号标注（图左上角外侧）
    fig.text(0.02, 0.95, "(b)", fontsize=12, va='top', ha='left')

    save_fig(save_name)


# 周期评估指标配置（key 对应 json 字段，scale 用于单位换算）
PERIOD_METRICS_CFG = [
    {"key": "overtime_rate", "name": "超时率", "unit": "%", "scale": 100, "label": "a"},
    {"key": "avg_overtime", "name": "平均超时时长", "unit": "分钟", "scale": 1, "label": "b"},
    {"key": "delivery_duration", "name": "平均配送时长", "unit": "分钟", "scale": 1, "label": "c"},
    {"key": "avg_revenue", "name": "平均收益", "unit": "元", "scale": 1, "label": "d"},
]


def dqn_period_metrics_curve(cfg, color, save_name='None'):
    """绘制 DQN 各周期评估指标曲线（单指标）"""
    with open(PERIOD_METRICS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    periods = data['period_labels']
    values = [v * cfg['scale'] for v in data['period_metrics'][cfg['key']]]

    fig, ax = plt.subplots(figsize=(4, 2.5))
    ax.plot(periods, values, marker="o", color=color, linewidth=1.5, markersize=4)
    ax.set_xlabel("周期")
    ax.set_ylabel(f"{cfg['name']} ({cfg['unit']})")
    ax.set_xticks(periods)
    # 子图编号标注（图左上角外侧）
    fig.text(0.02, 0.95, f"({cfg['label']})", fontsize=12,
             va='top', ha='left')

    save_fig(save_name)


def comparison_of_delivery_metrics_under_different_strategies(
        save_name='None'):
    """绘制不同策略下的配送指标对比曲线"""
    strategies = ['reject_rate', 'overtime_rate', 'avg_overtime', 'delivery_duration', 'avg_revenue']
    real = [0, 13.11, 10.51, 24.74, 4.66]
    dis = [31.23, 11.62, 7.64, 24.90, 4.68]
    dt = [31.19, 11.63, 7.60, 24.90, 4.68]
    spot = [31.18, 11.61, 7.59, 24.89, 4.68]
    pass



def main():
    plt_set()

    # 决策树预测 / 决策效用对比
    comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_1(
        save_name='不同深度下的预测MSE损失对比.svg'
    )
    comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_2(
        save_name='不同深度下的决策SPO损失对比.svg'
    )

    # 决策树预测 / 决策效用对比（英文版）
    comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_1_en(
        save_name='不同深度下的预测MSE损失对比_en.svg'
    )
    comparison_of_prediction_accuracy_and_decision_utility_of_decision_tree_models_2_en(
        save_name='不同深度下的决策SPO损失对比_en.svg'
    )

    # DQN 网络训练损失曲线
    dqn_actor_training_loss_curve(save_name='DQN Actor训练损失曲线.svg')
    dqn_critic_training_loss_curve(save_name='DQN Critic训练损失曲线.svg')

    # DQN 各周期评估指标
    colors = [plt.cm.Set2(i) for i in range(4)]
    for cfg, color in zip(PERIOD_METRICS_CFG, colors):
        dqn_period_metrics_curve(cfg, color, save_name=f"各周期{cfg['name']}.svg")


if __name__ == '__main__':
    main()
