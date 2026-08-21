"""从日志中解析数据并绘制 4 子图折线图"""
import re
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "..", "Q_Values", "simulation", "logs")
RESULT_DIR = os.path.join(BASE_DIR, "result")
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def plt_set():
    """
    设置绘图相关参数
    :return:None
    """
    # 1. 优先加载matplotlib的ggplot2风格核心模板（最关键）
    plt.style.use('ggplot')

    # 2. 协调seaborn样式（补充ggplot风格未覆盖的细节，避免冲突）
    sns.set_style({
        "axes.facecolor": "#E5E5E5",  # 匹配ggplot的浅灰背景
        "grid.color": "white",  # 白色网格线（ggplot经典）
        "axes.grid": True,  # 显示网格
        "xtick.color": "black",  # x轴刻度文字黑色
        "ytick.color": "black"  # y轴刻度文字黑色
    })

    # 3. 中文显示核心配置（保留原有逻辑，解决乱码/负号方块问题）
    # 全局字体族：优先衬线（适配Times New Roman + 宋体）
    plt.rcParams['font.family'] = ['serif', 'sans-serif']
    plt.rcParams['font.serif'] = [
        'Times New Roman',  # 西文优先用Times New Roman（ggplot2默认西文）
        'SimSun',  # Windows宋体（中文核心）
        'Songti SC',  # macOS宋体（适配苹果）
        'DejaVu Serif'  # 兜底衬线字体
    ]
    plt.rcParams['font.sans-serif'] = [
        'SimSun',  # 中文优先用宋体
        'Songti SC',
        'Microsoft YaHei',
        'DejaVu Sans'
    ]
    # 解决负号显示为方块的问题
    plt.rcParams['axes.unicode_minus'] = False

    # 4. 微调字体粗细（匹配ggplot2默认的正常粗细）
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.rcParams['font.weight'] = 'normal'


def parse_log(filepath):
    """解析单个日志文件，返回 (seed, coefficient, rejection_rate, timeout_rate, avg_timeout)"""
    basename = os.path.basename(filepath)
    m = re.match(r"sim_(\d+)_dis_300of300_([\d.]+)\.log", basename)
    if not m:
        return None
    seed = int(m.group(1))
    coeff = float(m.group(2))

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # 拒单率
    m_rr = re.search(r"拒单率:\s*([\d.]+)%", text)
    # 超时率
    m_tr = re.search(r"超时率:\s*([\d.]+)%", text)
    # 平均超时时长
    m_at = re.search(r"平均超时时长:\s*([\d.]+)分钟", text)

    if not (m_rr and m_tr and m_at):
        return None

    rejection_rate = float(m_rr.group(1))
    timeout_rate = float(m_tr.group(1))
    avg_timeout = float(m_at.group(1))

    return seed, coeff, rejection_rate, timeout_rate, avg_timeout


def main():
    plt_set()
    # 解析所有日志
    log_files = glob.glob(os.path.join(LOGS_DIR, "sim_*.log"))
    records = []
    for fp in log_files:
        r = parse_log(fp)
        if r:
            records.append(r)

    # 只保留接单概率系数在 [0.8, 1.2] 之间的数据
    records = [r for r in records if 0.8 <= r[1] <= 1.2]

    if not records:
        print("未解析到任何日志数据")
        return

    # 按种子分组
    seeds = sorted(set(r[0] for r in records))
    seed_data = {s: [] for s in seeds}
    for r in records:
        seed_data[r[0]].append(r)

    # 每个种子按系数排序
    for s in seeds:
        seed_data[s].sort(key=lambda x: x[1])  # 按 coefficient 排序

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors_to = ["#F8766D", "#F8766D", "#F8766D"]       # 超时率 - ggplot2 默认红
    colors_avg = ["#619CFF", "#619CFF", "#619CFF"]       # 平均超时时长 - ggplot2 默认蓝

    # ---- 前三个子图 ----
    for idx, seed in enumerate(seeds):
        ax = axes[idx // 2][idx % 2]
        data = seed_data[seed]

        rr_vals = [d[2] for d in data]   # 拒单率
        tr_vals = [d[3] for d in data]   # 超时率
        at_vals = [d[4] for d in data]   # 平均超时时长

        ax.plot(rr_vals, tr_vals, marker="o", color=colors_to[idx],
                linestyle="-", linewidth=1.5, markersize=5, label="超时率 (%)")
        ax_twin = ax.twinx()
        ax_twin.grid(False)
        ax_twin.plot(rr_vals, at_vals, marker="s", color=colors_avg[idx],
                     linestyle="--", linewidth=1.5, markersize=5, label="平均超时时长 (分钟)")

        ax.set_xlabel("拒单率 (%)")
        ax.set_ylabel("超时率 (%)")
        ax_twin.set_ylabel("平均超时时长 (分钟)")
        ax.set_title(f"随机种子 {seed}")

        # 合并图例
        l1, _ = ax.get_legend_handles_labels()
        l2, _ = ax_twin.get_legend_handles_labels()
        ax.legend(l1 + l2, ["超时率 (%)", "平均超时时长 (分钟)"],
                  loc="upper right", fontsize=8,
                  facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # ---- 第四个子图：按拒单率均值聚合（产品：超时率均值 × 平均超时时长均值） ----
    ax4 = axes[1][1]
    coeff_map = {}
    for r in records:
        _, coeff, rr, tr, at = r
        coeff_map.setdefault(coeff, []).append((rr, tr, at))

    common_coeffs = sorted(coeff_map.keys())
    mean_rr = []
    product_vals = []
    for c in common_coeffs:
        vals = coeff_map[c]
        mean_rr.append(sum(v[0] for v in vals) / len(vals))
        mean_tr = sum(v[1] for v in vals) / len(vals)
        mean_at = sum(v[2] for v in vals) / len(vals)
        product_vals.append(mean_tr * 0.01 * mean_at)

    ax4.plot(mean_rr, product_vals, marker="^", color="#00BA38",
             linestyle="-", linewidth=1.5, markersize=5, label="超时率均值 × 平均超时时长均值")

    ax4.set_xlabel("拒单率均值 (%)")
    ax4.set_ylabel("超时率均值 × 平均超时时长均值")
    ax4.set_title("各系数下跨种子平均（乘积）")
    ax4.legend(loc="upper right", fontsize=8,
               facecolor="white", edgecolor="#CCCCCC", framealpha=1)

    # plt.suptitle("DQN 配送仿真结果分析", fontsize=14, fontweight="bold")
    plt.tight_layout()
    # 修复 subplots 间距
    plt.subplots_adjust(top=0.93)
    os.makedirs(RESULT_DIR, exist_ok=True)
    out_path = os.path.join(RESULT_DIR, "simulation_plots.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"图表已保存至: {out_path}")


if __name__ == "__main__":
    main()
