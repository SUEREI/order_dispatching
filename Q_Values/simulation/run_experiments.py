#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行实验脚本：
对每个随机数种子，依次运行所有配置组合。
每次运行前自动更新 config.yaml，然后以子进程调用 main_dqn.py。
"""

import subprocess
import sys
import time
from pathlib import Path
import yaml

# ===================== 实验参数配置 =====================

# 随机数种子列表（按需修改）
SEEDS = [
    8, 9, 10, 11, 12, 13, 14, 15
]

# 配置组合列表
# 每项为 (candidate_method, candidate_model, top_k_with_dist, top_k_with_dt, grab_cof)
CONFIG_COMBOS = [
    # 配置1: 纯距离筛选
    # ("distance",       None,                        100, 100, 1),
    # ("distance",       None,                        200, 200, 1),
    # ("distance",       None,                        300, 300, 1),
    # ("distance",       None,                        400, 400, 1),
    # ("distance",       None,                        500, 500, 1),
    # # 配置2: 决策树 + 普通模型
    ("distance",  None,    300, 300, 1),
    ("decision_tree",  "decision_tree_model_used.pkl",    500, 300, 1),
    ("decision_tree",  "smart_decision_tree_model_used.pkl",    500, 300, 1),
    # # 配置3: 决策树 + smart模型
    # ("decision_tree",  "smart_decision_tree_model.pkl", 500, 300, 1),
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # 2 Python/

# 主程序路径
MAIN_SCRIPT = PROJECT_ROOT / "Q_Values" / "simulation" / "main_dqn.py"
# 配置文件路径
CONFIG_PATH = PROJECT_ROOT / "Q_Values" / "simulation" / "config" / "config.yaml"

# 运行模式：True=串行（前一个跑完再跑下一个），False=并行（不推荐，共享config.yaml会冲突）
RUN_SERIAL = True

# ===================== 运行逻辑 =====================


def make_config(seed, method, model, top_k_dist, top_k_dt, grab_cof):
    """根据参数构造配置字典"""
    config = {
        "random_seed": seed,
        "candidate_method": method,
        "top_k_with_dist": top_k_dist,
        "top_k_with_dt": top_k_dt,
        "grab_cof": grab_cof,
    }
    if model is not None:
        config["candidate_model"] = model
    return config


def run_single(seed, method, model, top_k_dist, top_k_dt, grab_cof):
    """更新 config.yaml 并运行一次 main_dqn.py"""
    config = make_config(seed, method, model, top_k_dist, top_k_dt, grab_cof)

    # 写配置
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] 种子={seed}  method={method}  "
          f"model={model or 'N/A'}  dist={top_k_dist}  dt={top_k_dt}")
    print(f"  config.yaml 已更新")

    # 运行主程序（cwd 设为项目根目录，确保数据路径正确）
    result = subprocess.run(
        [sys.executable, str(MAIN_SCRIPT)],
        cwd=PROJECT_ROOT,
        capture_output=False,
    )
    return result.returncode


def main():
    print(f"批量实验启动 — {len(SEEDS)} 个种子 × {len(CONFIG_COMBOS)} 种配置 = {len(SEEDS) * len(CONFIG_COMBOS)} 次运行")
    print(f"主程序: {MAIN_SCRIPT}")
    print()

    success = 0
    failed = 0

    for seed in SEEDS:
        for method, model, top_k_dist, top_k_dt, grab_cof in CONFIG_COMBOS:
            label = f"seed={seed}_{method}_{model or 'none'}_{top_k_dt}of{top_k_dist}_{grab_cof}"
            try:
                rc = run_single(seed, method, model, top_k_dist, top_k_dt, grab_cof)
                if rc == 0:
                    success += 1
                    print(f"  ✓ {label}  完成")
                else:
                    failed += 1
                    print(f"  ✗ {label}  退出码={rc}")
            except Exception as e:
                failed += 1
                print(f"  ✗ {label}  异常: {e}")

            if RUN_SERIAL:
                print()  # 空行分隔

    print(f"\n{'='*60}")
    print(f"全部完成: 成功 {success}, 失败 {failed}, "
          f"总计 {success + failed}")


if __name__ == "__main__":
    main()
