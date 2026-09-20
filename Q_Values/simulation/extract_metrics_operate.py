"""
从仿真日志目录中提取边际收益启发式 (operate) 日志的指标。

匹配日志文件: sim_{seed}_operate.log

提取并统计:
    - 拒单数
    - 超时订单数
    - 累计超时时长 = 超时订单数 × 平均超时时长 (分钟)
    - 累计配送时长 = 已送达订单数 × 平均配送时长 (分钟)

用法:
    python extract_metrics_operate.py [log_dir]
"""

import re
import os
import sys

import numpy as np

# 原始指标 -> 正则 pattern
METRIC_PATTERNS = {
    "已送达订单数": r"已送达订单数:\s*([\d,]+)",
    "拒单数":       r"拒单数:\s*([\d,]+)",
    "平均配送时长": r"平均配送时长:\s*([\d.]+)分钟",
    "超时订单数":   r"超时订单数:\s*([\d,]+)",
    "平均超时时长": r"平均超时时长:\s*([\d.]+)分钟",
}

# 文件名解析: sim_{seed}_operate.log
FILE_RE = re.compile(r"sim_(\d+)_operate\.log")


def parse_seed(filename: str):
    m = FILE_RE.match(filename)
    return int(m.group(1)) if m else None


def extract_metrics_from_file(filepath: str) -> dict:
    """从单个 log 文件提取原始指标，返回 {指标名: 数值}"""
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    metrics = {}
    for name, pattern in METRIC_PATTERNS.items():
        m = re.search(pattern, text)
        if m:
            raw = m.group(1).replace(",", "")
            metrics[name] = float(raw) if "." in raw else int(raw)
    return metrics


def compute_derived(metrics: dict) -> dict:
    """由原始指标计算派生指标"""
    derived = {}
    if "超时订单数" in metrics and "平均超时时长" in metrics:
        derived["累计超时时长(分钟)"] = metrics["超时订单数"] * metrics["平均超时时长"]
    if "已送达订单数" in metrics and "平均配送时长" in metrics:
        derived["累计配送时长(分钟)"] = metrics["已送达订单数"] * metrics["平均配送时长"]
    return derived


def main(log_dir: str):
    if not os.path.isdir(log_dir):
        print(f"目录不存在: {log_dir}")
        sys.exit(1)

    # grouped[seed] = {指标: 值}（含原始 + 派生指标）
    grouped = {}

    for fname in sorted(os.listdir(log_dir)):
        if not fname.endswith(".log"):
            continue
        seed = parse_seed(fname)
        if seed is None:
            continue

        filepath = os.path.join(log_dir, fname)
        metrics = extract_metrics_from_file(filepath)
        metrics.update(compute_derived(metrics))
        grouped[seed] = metrics
        print(f"[OK] {fname} -> seed={seed}")

    if not grouped:
        print("未找到匹配 sim_{seed}_operate.log 的日志文件")
        sys.exit(1)

    # 输出列: (表头, 取数key, 格式)；4 个统计指标 + 中间量（便于核对）
    columns = [
        ("拒单数",            "拒单数",         "{:.0f}"),
        ("超时订单数",        "超时订单数",     "{:.0f}"),
        ("累计超时时长(分钟)", "累计超时时长(分钟)", "{:.1f}"),
        ("累计配送时长(分钟)", "累计配送时长(分钟)", "{:.1f}"),
        ("已送达订单数",      "已送达订单数",   "{:.0f}"),
        ("平均超时时长(分钟)", "平均超时时长",   "{:.2f}"),
        ("平均配送时长(分钟)", "平均配送时长",   "{:.2f}"),
    ]

    seeds = sorted(grouped.keys())

    print(f"\n{'='*100}")
    print(f"  边际收益启发式 (operate) 各种子指标")
    print(f"{'='*100}")

    header = "seed".ljust(8)
    for head, _, _ in columns:
        header += f"{head:>{22}}"
    print(header)
    print("-" * len(header))

    for seed in seeds:
        row = str(seed).ljust(8)
        for _, key, fmt in columns:
            val = grouped[seed].get(key)
            row += f"{fmt.format(val):>{22}}" if val is not None else f"{'-':>{22}}"
        print(row)

    # 均值 ± 标准差
    print("-" * len(header))
    row = "均值±SD".ljust(8)
    for _, key, _ in columns:
        vals = np.array([grouped[s][key] for s in seeds if key in grouped[s]], dtype=float)
        if len(vals) > 0:
            cell = f"{vals.mean():,.2f} ± {vals.std(ddof=1):,.2f}"
            row += f"{cell:>{22}}"
        else:
            row += f"{'-':>{22}}"
    print(row)

    print(f"\n  样本数 (种子数): {len(seeds)}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        log_dir = sys.argv[1]
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "logs")
        print(f"未指定日志目录，使用默认路径: {log_dir}")
    main(log_dir)
