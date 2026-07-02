"""
从仿真日志目录中提取指标，按随机种子排序，构建各指标的列表，
并对同一指标在不同策略间进行配对T检验。
用法:
    python extract_metrics.py <log_dir>
    python extract_metrics.py logs__3depth_can__4depth_grab_300of500
"""

import re
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel, ttest_1samp

# 指标名 -> 正则 pattern
METRIC_PATTERNS = {
    "已送达订单数":    r"已送达订单数:\s*([\d,]+)",
    "拒单率":          r"拒单率:\s*([\d.]+)%",
    "拒单率-非重复计数": r"拒单率-非重复计数:\s*([\d.]+)%",
    "平均配送时长":    r"平均配送时长:\s*([\d.]+)分钟",
    "超时订单数":      r"超时订单数:\s*([\d,]+)",
    "超时率":          r"超时率:\s*([\d.]+)%",
    "平均超时时长":    r"平均超时时长:\s*([\d.]+)分钟",
    "骑手平均订单收益": r"骑手平均订单收益:\s*([\d.]+)",
}

# 越小越好的指标
LOWER_IS_BETTER = {"拒单率", "拒单率-非重复计数", "平均配送时长", "超时订单数", "超时率", "平均超时时长"}

# DQN 基准值
BASELINE = {
    "平均配送时长": 24.93,
    "超时率": 11.81,
    "平均超时时长": 7.78,
}

# 文件名解析: sim_{seed}_{strategy}_*.log
FILE_RE = re.compile(r"sim_(\d+)_(\w+)_\d+of\d+\.log")

STRATEGY_NAMES = {
    "dis":  "距离筛选 (dis)",
    "dt":   "决策树筛选 (dt)",
    "spot": "SPOT决策树 (spot)",
}

SIGNIFICANCE_LEVEL = 0.05


def parse_seed_and_strategy(filename: str):
    m = FILE_RE.match(filename)
    if m:
        return int(m.group(1)), m.group(2)
    return None, None


def extract_metrics_from_file(filepath: str) -> dict:
    """从单个 log 文件提取所有指标，返回 {指标名: 数值}"""
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    for name, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).replace(",", "")
            value = float(raw) if "." in raw else int(raw)
            result[name] = value
    return result


def cohens_d_paired(vals1: np.ndarray, vals2: np.ndarray) -> float:
    """配对设计 Cohen's d = mean_diff / SD_diff"""
    diffs = vals1 - vals2
    sd = np.std(diffs, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return np.mean(diffs) / sd


def cohens_d_one_sample(vals: np.ndarray, mu0: float) -> float:
    """单样本 Cohen's d = (mean - mu0) / SD"""
    sd = np.std(vals, ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return (np.mean(vals) - mu0) / sd


def print_results(grouped: dict):
    """输出各策略的均值/标准差 + 原始值列表"""
    strategies = sorted(grouped.keys())
    all_metric_names = list(METRIC_PATTERNS.keys())

    # 1. 指标原始值列表（按种子升序）
    def fmt_val(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    print(f"\n{'='*80}")
    print(f"  各指标原始值列表 (按策略分组)")
    print(f"{'='*80}")
    for metric in all_metric_names:
        print(f"\n【{metric}】")
        for strategy in strategies:
            seeds_sorted = sorted(grouped[strategy].keys())
            values = [grouped[strategy][seed][metric] for seed in seeds_sorted
                      if seed in grouped[strategy] and metric in grouped[strategy][seed]]
            vals_str = ", ".join(fmt_val(v) for v in values)
            print(f"  {strategy:>5s}: [{vals_str}]")

    # 2. 均值与标准差
    COL_W = 26
    print(f"\n\n{'='*80}")
    print(f"  各策略指标均值 ± 标准差")
    print(f"{'='*80}")

    header = "指标".ljust(22)
    for strategy in strategies:
        header += f"  {strategy:>{COL_W-2}s}"
    print(header)
    print("-" * len(header))

    for metric in all_metric_names:
        row = metric.ljust(22)
        for strategy in strategies:
            seeds_sorted = sorted(grouped[strategy].keys())
            values = [grouped[strategy][seed][metric] for seed in seeds_sorted
                      if seed in grouped[strategy] and metric in grouped[strategy][seed]]
            arr = np.array(values, dtype=float)
            if len(arr) > 0:
                cell = f"{arr.mean():.2f} ± {arr.std(ddof=1):.2f}"
            else:
                cell = "NaN"
            row += f"  {cell:>{COL_W-2}s}"
        print(row)


def print_one_sided_tests(grouped: dict):
    """单侧配对T检验 P 值汇总表"""
    all_metric_names = list(METRIC_PATTERNS.keys())
    one_sided_pairs = [("dis", "dt"), ("dis", "spot"), ("dt", "spot")]

    pair_pvalues = {}
    for s1, s2 in one_sided_pairs:
        pair_pvalues[(s1, s2)] = {}
        common_seeds = sorted(set(grouped[s1].keys()) & set(grouped[s2].keys()))
        if len(common_seeds) < 2:
            continue
        for metric in all_metric_names:
            v1 = np.array([grouped[s1][seed].get(metric, np.nan) for seed in common_seeds], dtype=float)
            v2 = np.array([grouped[s2][seed].get(metric, np.nan) for seed in common_seeds], dtype=float)

            if np.any(np.isnan(v1)) or np.any(np.isnan(v2)):
                pair_pvalues[(s1, s2)][metric] = (np.nan, np.nan, np.nan)
                continue

            t_stat_two, p_two = ttest_rel(v1, v2)
            mean_diff = v1.mean() - v2.mean()
            d_val = cohens_d_paired(v1, v2)

            if metric in LOWER_IS_BETTER:
                direction_ok = mean_diff > 0
            else:
                direction_ok = mean_diff < 0

            p_one = p_two / 2 if direction_ok else 1 - p_two / 2
            if np.isnan(p_one):
                p_one = np.nan

            pair_pvalues[(s1, s2)][metric] = (p_one, t_stat_two, d_val)

    print(f"\n\n{'='*90}")
    print(f"  单侧配对T检验 P 值汇总表 (α = {SIGNIFICANCE_LEVEL}, * 表示显著)")
    print(f"  H0: 左侧策略 ≤ 右侧策略 (越小越好) 或 左侧策略 ≥ 右侧策略 (越大越好)")
    print(f"{'='*90}")

    COL_W = 26
    header = "metric".ljust(22)
    for s1, s2 in one_sided_pairs:
        short_label = f"{s2}>better>{s1}"
        header += f"  {short_label:>{COL_W-2}s}"
    print(header)
    print("-" * len(header))

    for metric in all_metric_names:
        row_p = metric.ljust(22)
        row_t = "".ljust(22)
        row_d = "".ljust(22)
        for s1, s2 in one_sided_pairs:
            p_one, t_stat, d_val = pair_pvalues.get((s1, s2), {}).get(metric, (np.nan, np.nan, np.nan))
            if not np.isnan(p_one):
                mark = "*" if p_one < SIGNIFICANCE_LEVEL else " "
                cell_p = f"p={p_one:.4f}{mark}"
                cell_t = f"t={t_stat:+6.2f}"
                cell_d = f"d={d_val:+6.2f}" if not np.isnan(d_val) else "d=  NaN"
                row_p += f"  {cell_p:>{COL_W-2}s}"
                row_t += f"  {cell_t:>{COL_W-2}s}"
                row_d += f"  {cell_d:>{COL_W-2}s}"
            else:
                row_p += f"  {'-':>{COL_W-2}s}"
                row_t += f"  {'-':>{COL_W-2}s}"
                row_d += f"  {'-':>{COL_W-2}s}"
        print(row_p)
        print(row_t)
        print(row_d)

    seeds = sorted(set(key for strategy in grouped.values() for key in strategy.keys()))
    print(f"\n  样本数 (配对种子数): {len(seeds)}")


def print_baseline_tests(grouped: dict):
    """各策略 vs DQN 基准 — 单样本单侧T检验"""
    if not BASELINE:
        return

    strategies = sorted(grouped.keys())

    print(f"\n\n{'='*90}")
    print(f"  各策略 vs DQN 基准 — 单样本单侧T检验 (α = {SIGNIFICANCE_LEVEL}, * 表示显著)")
    print(f"  H1: 策略指标 < 基准值 (越小越好)")
    print(f"{'='*90}")

    COL_W = 26
    header = "metric".ljust(22)
    for strategy in strategies:
        header += f"  {strategy} < baseline"
    print(header)
    print("-" * len(header))

    for metric in sorted(BASELINE.keys()):
        mu0 = BASELINE[metric]
        if metric not in METRIC_PATTERNS:
            continue

        row_p = metric.ljust(22)
        row_t = "".ljust(22)
        row_d = "".ljust(22)

        for strategy in strategies:
            seeds_sorted = sorted(grouped[strategy].keys())
            vals = np.array([grouped[strategy][seed].get(metric, np.nan) for seed in seeds_sorted], dtype=float)

            if np.any(np.isnan(vals)):
                row_p += f"  {'-':>{COL_W-2}s}"
                row_t += f"  {'-':>{COL_W-2}s}"
                row_d += f"  {'-':>{COL_W-2}s}"
                continue

            # 单样本 t 检验
            t_stat, p_two = ttest_1samp(vals, mu0)
            mean_diff = vals.mean() - mu0

            if metric in LOWER_IS_BETTER:
                direction_ok = mean_diff < 0  # H1: 策略均值 < 基准
            else:
                direction_ok = mean_diff > 0  # H1: 策略均值 > 基准

            p_one = p_two / 2 if direction_ok else 1 - p_two / 2
            if np.isnan(p_one):
                p_one = np.nan

            d_val = cohens_d_one_sample(vals, mu0)

            if not np.isnan(p_one):
                mark = "*" if p_one < SIGNIFICANCE_LEVEL else " "
                cell_p = f"p={p_one:.4f}{mark}"
                cell_t = f"t={t_stat:+6.2f}"
                cell_d = f"d={d_val:+6.2f}" if not np.isnan(d_val) else "d=  NaN"
                row_p += f"  {cell_p:>{COL_W-2}s}"
                row_t += f"  {cell_t:>{COL_W-2}s}"
                row_d += f"  {cell_d:>{COL_W-2}s}"
            else:
                row_p += f"  {'-':>{COL_W-2}s}"
                row_t += f"  {'-':>{COL_W-2}s}"
                row_d += f"  {'-':>{COL_W-2}s}"

        print(row_p)
        print(row_t)
        print(row_d)

    print(f"\n  DQN基准值: {BASELINE}")


def main(log_dir: str):
    if not os.path.isdir(log_dir):
        print(f"目录不存在: {log_dir}")
        sys.exit(1)

    # 结构: grouped[strategy][seed] = {指标: 值}
    grouped = defaultdict(lambda: defaultdict(dict))

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue

        seed, strategy = parse_seed_and_strategy(fname)
        if seed is None:
            continue

        filepath = os.path.join(log_dir, fname)
        metrics = extract_metrics_from_file(filepath)
        grouped[strategy][seed] = metrics
        print(f"[OK] {fname} -> seed={seed}, strategy={strategy}")

    print_results(grouped)
    print_one_sided_tests(grouped)
    print_baseline_tests(grouped)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        log_dir = sys.argv[1]
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "logs__3depth_can__4depth_grab_300of500")
        print(f"未指定日志目录，使用默认路径: {log_dir}")
    main(log_dir)
