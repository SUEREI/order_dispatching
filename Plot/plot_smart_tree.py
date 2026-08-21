"""读取 SPOT 智能决策树模型并绘制树结构图"""
import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from _basic_set import plt_set, save_fig, BASE_DIR

# 让 SPOT.core 中的模型类可导入
CORE_DIR = os.path.join(os.path.dirname(BASE_DIR), r'SPOT\core')
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from smart_decision_tree_all import SimilaritySetDecisionTree

# numpy 2.x pickle 兼容 shim（pkl 按 numpy._core 路径序列化，当前环境为 numpy 1.x）
import numpy as _np
if not hasattr(_np, '_core') and hasattr(_np, 'core'):
    sys.modules['numpy._core'] = _np.core
    for _sub in ['multiarray', 'umath', '_multiarray_umath', 'numeric', 'shape_base']:
        if hasattr(_np.core, _sub):
            sys.modules[f'numpy._core.{_sub}'] = getattr(_np.core, _sub)

PKL_PATH = os.path.join(CORE_DIR, r'pkl_s\smart_decision_tree_model.pkl')

# 特征名（与训练脚本 feature_cols 顺序一致）
FEATURE_NAMES = [
    '疲劳度',
    '接单距离',
    '配送距离',
    '正在进行订单数',
    '骑手是否偏好拒绝',
    '商家是否常被拒单',
]


def load_model():
    """加载 pkl 模型（模型类按 __main__ 模块序列化，需注册）"""
    import __main__
    __main__.SimilaritySetDecisionTree = SimilaritySetDecisionTree
    with open(PKL_PATH, 'rb') as f:
        return pickle.load(f)


def plot_tree_structure(model, save_name='None'):
    """绘制决策树结构图：分裂节点显示特征与阈值，叶节点显示预测值与样本数"""
    # 布局：叶节点按从左到右顺序占据整数 x 坐标，非叶节点位于左右子节点中间
    pos = {}
    leaf_count = [0]

    def assign(node, depth=0):
        if node.get('leaf'):
            node['_leaf_index'] = leaf_count[0] + 1  # 叶节点从左到右编号 1~n
            pos[id(node)] = (leaf_count[0], -depth)
            leaf_count[0] += 1
        else:
            assign(node['left'], depth + 1)
            assign(node['right'], depth + 1)
            lx = pos[id(node['left'])][0]
            rx = pos[id(node['right'])][0]
            pos[id(node)] = ((lx + rx) / 2, -depth)

    assign(model.tree_)
    n_leaves = leaf_count[0]
    max_depth = model.get_depth()

    fig, ax = plt.subplots(figsize=(max(10, n_leaves * 1.6), max(6, max_depth * 2.5)))
    ax.set_xlim(-1.0, n_leaves - 1 + 1.0)
    ax.set_ylim(-max_depth - 1.3, 0.6)
    ax.axis('off')
    ax.set_aspect('equal')

    # 先画边
    def draw_edges(node):
        if node.get('leaf'):
            return
        px, py = pos[id(node)]
        for child_key in ['left', 'right']:
            cx, cy = pos[id(node[child_key])]
            ax.plot([px, cx], [py, cy], 'k-', lw=1.2, alpha=0.7)
            # 边上标注分支方向
            mx, my = (px + cx) / 2, (py + cy) / 2
            label = '≤' if child_key == 'left' else '>'
            ax.text(mx, my, label, ha='center', va='center', fontsize=8,
                    color='gray',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.9))
            draw_edges(node[child_key])
    draw_edges(model.tree_)

    # 再画节点（配色与项目其他图统一使用 Set2 色板）
    def draw_node(node):
        x, y = pos[id(node)]
        if node.get('leaf'):
            n_samples = sum(node['similar_set_distribution'].values())
            text = f"叶节点{node['_leaf_index']}: \n预测值={node['value']:.4f}\n样本数={n_samples}"
            fc, ec = 'white', '#2F8B6E'   # 叶节点：填充纯白，边框绿色
        else:
            feat = FEATURE_NAMES[node['feature']]
            text = f"{feat}\n≤ {node['threshold']:.4f}"
            fc, ec = 'white', '#C25A2E'   # 分裂节点：填充纯白，边框橙色
        ax.text(x, y, text, ha='center', va='center', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.35', fc=fc, ec='none'))

    def draw_all(node):
        draw_node(node)
        if not node.get('leaf'):
            draw_all(node['left'])
            draw_all(node['right'])
    draw_all(model.tree_)

    # ax.set_title(f"SPOT 决策树 (a={model.a}, 最大深度={max_depth}, 叶节点={n_leaves}, 特征数={model.n_features_in_})",
    #              fontsize=13, fontweight='bold')
    save_fig(save_name)


def main():
    plt_set()
    model = load_model()
    print(f"模型已加载: a={model.a}, 深度={model.get_depth()}, 叶节点={model.get_n_leaves()}")
    plot_tree_structure(model, save_name='SPO决策树决策节点展示.svg')


if __name__ == '__main__':
    main()
