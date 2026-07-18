#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cov_test.py —— 协变量有效性测试
====================================
作用: 验证 TimechoAI 使用"未来协变量"
原理: 传不同的 future_covs, 看预测结果是否不同

Author: Janesong
Create Date: 2026/06/29, Update on 2026/07/12.
"""

import time
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from config.constants import DEFAULT_INPUT_LENGTH, DEFAULT_OUTPUT_LENGTH
from core.timecho import forecast, calc_metrics, calc_diff
from utils.files import save_with_json_backup

# ============================================================
# 数据相关配置
# ============================================================
SCRIPT_DIR = Path(__file__).parent
CSV_PATH = SCRIPT_DIR / "cov_test_data.csv"        # 测试数据文件
RESULT_PATH = SCRIPT_DIR / "cov_test_results.csv"  # 预测结果文件

MODEL_ID = "Auto"  # 模型列表: Auto / Timer-3.5 / Timer-3.0 / Chronos-2 / AutoARIMA / Holt-Winters

# ============================================================
# 读取数据
# ============================================================

print("📦 读取数据...")
raw_df = pd.read_csv(CSV_PATH)
raw_df["time"] = pd.to_datetime(raw_df["time"])  # 把 time 列从字符串转成时间格式
print(f"   数据总行数: {len(raw_df)}")
print(f"   列名: {list(raw_df.columns)}")
print()

# 切分数据
history = raw_df.iloc[:DEFAULT_INPUT_LENGTH].copy()
future_real = raw_df.iloc[DEFAULT_INPUT_LENGTH:].copy()
ground_truth = future_real["target"].values

print(f"   历史数据: {len(history)} 行(时间范围: {history['time'].iloc[0]} ~ {history['time'].iloc[-1]})")
print(f"   未来真实值: {len(future_real)} 行(时间范围: {future_real['time'].iloc[0]} ~ {future_real['time'].iloc[-1]})")
print()

# ============================================================
# 构造 4 个测试场景
# ============================================================
#
# 场景说明: 
#   A. 传"真实"未来协变量  →  如果模型用了 cov, 预测应该最准
#   B. 传"随机噪声"协变量  →  如果模型用了 cov, 预测应该变差
#   C. 传"完全反向"协变量  →  如果模型用了 cov, 预测应该严重偏离
#   D. 完全"不传"协变量    →  对比用: 单变量预测效果
#
# 如果模型【真的用了】协变量: A 最好, B/C 变差, D 介于中间
# 如果模型【没用】协变量:   A/B/C/D 几乎一样(传什么都无所谓)
# ============================================================

print("🔧 构造 4 个测试场景...")

# 场景 A: 真实未来协变量
future_cov_real = future_real[["time", "cov"]].copy()
print(f"   A. 真实协变量  cov 均值: {future_cov_real['cov'].mean():.2f}")

# 场景 B: 纯随机噪声(均值0, 标准差100)
future_cov_noise = future_real[["time"]].copy()
future_cov_noise["cov"] = np.random.randn(DEFAULT_OUTPUT_LENGTH) * 100
print(f"   B. 噪声协变量  cov 均值: {future_cov_noise['cov'].mean():.2f}(纯随机)")

# 场景 C: 完全反向(真实值取负号)
future_cov_anti = future_real[["time"]].copy()
future_cov_anti["cov"] = -future_real["cov"].values
print(f"   C. 反向协变量  cov 均值: {future_cov_anti['cov'].mean():.2f}(完全反号)")

# 场景 D: 不传协变量
print(f"   D. 不传协变量  (future_covs=None)")
print()

# ============================================================
# 公共参数
# ============================================================

history_targets = history[["time", "target"]]
history_covs = history[["time", "cov"]]

SCENARIOS = [
    ("A-真实协变量", future_cov_real),
    ("B-噪声协变量", future_cov_noise),
    ("C-反向协变量", future_cov_anti),
    ("D-不传协变量", None),
]


# ============================================================
# 逐个场景调用预测
# ============================================================

print("🚀 开始预测(共 4 个场景, 消耗 4 次额度)...")
print("-" * 60)

results = []
for scene_name, fc in SCENARIOS:
    print(f"  ⏳ 场景 {scene_name} 预测中...", end="", flush=True)
    pred_values, elapsed_ms, error = forecast(
        targets=history_targets,
        history_covs=history_covs,
        future_covs=fc,
        model_id=MODEL_ID,
        output_length=DEFAULT_OUTPUT_LENGTH,
        time_col="time",
        auto_adapt=True,
    )

    if error:
        print(f" 失败 ({elapsed_ms:.0f}ms): {error}")
    else:
        print(f" 完成 ({elapsed_ms:.0f}ms)")

    results.append({
        "scene": scene_name,
        "pred": pred_values,
        "latency_ms": elapsed_ms,
        "error": error,
    })
    time.sleep(1)

print()

# ============================================================
# 计算误差和场景间差异
# ============================================================

metrics = {}
for r in results:
    metrics[r["scene"]] = calc_metrics(r["pred"], ground_truth)
    r["metrics"] = metrics[r["scene"]]

# 各场景之间的预测差异
diffs = {
    "AB": calc_diff(results[0]["pred"], results[1]["pred"]),  # 真实 vs 噪声
    "AC": calc_diff(results[0]["pred"], results[2]["pred"]),  # 真实 vs 反向
    "AD": calc_diff(results[0]["pred"], results[3]["pred"]),  # 真实 vs 不传
}

# ============================================================
# 打印结果汇总
# ============================================================

print("\n" + "=" * 80)
print("📊 测试结果汇总")
print("=" * 80)

print("\n【1】各场景预测精度(vs 真实值)")
print("-" * 70)
print(f"{'场景':>20s} | {'MAE':>10s} | {'RMSE':>10s} | {'MAPE(%)':>10s} | {'耗时(ms)':>10s}")
print("-" * 70)

for r in results:
    m = r["metrics"]
    mae_str = f"{m['MAE']:.4f}" if m['MAE'] is not None else "N/A"
    rmse_str = f"{m['RMSE']:.4f}" if m['RMSE'] is not None else "N/A"
    mape_str = f"{m['MAPE']:.2f}" if m['MAPE'] is not None else "N/A"
    print(f"{r['scene']:>20s} | {mae_str:>10s} | {rmse_str:>10s} | {mape_str:>10s} | {r['latency_ms']:>10.0f}")

print("\n【2】场景间预测差异(核心指标)")
print("-" * 70)
print(f"{'对比':>30s} | {'平均绝对差异':>12s} | {'判断':>20s}")
print("-" * 70)

diff_labels = [
    ("A(真实) vs B(噪声) 预测差异", diffs["AB"]),
    ("A(真实) vs C(反向) 预测差异", diffs["AC"]),
    ("A(真实) vs D(不传) 预测差异", diffs["AD"]),
]
for name, diff in diff_labels:
    if diff is None:
        print(f"{name:>30s} | {'N/A':>12s} | {'N/A':>20s}")
    elif diff < 0.5:
        print(f"{name:>30s} | {diff:>12.4f} | {'⚠️ 几乎无差异':>20s}")
    elif diff < 2.0:
        print(f"{name:>30s} | {diff:>12.4f} | {'🟡 有微弱影响':>20s}")
    else:
        print(f"{name:>30s} | {diff:>12.4f} | {'✅ 有显著影响':>20s}")

print("\n【3】自动结论")
print("-" * 70)

if all(v is not None for v in diffs.values()):
    max_diff = max(diffs["AB"], diffs["AC"], diffs["AD"])

    if max_diff < 0.5:
        print("🔴 严重问题: 协变量完全没有生效！")
        print("   传入真实协变量、随机噪声、反向值, 预测结果几乎一致. ")
        print(f"   → {MODEL_ID} 可能未在推理时加载 future_covs 权重, 或退化为单变量模式. ")
    elif max_diff < 2.0:
        print("🟡 协变量影响微弱:")
        print("   传入不同协变量时预测有轻微变化, 但影响远小于预期. ")
        print("   → 协变量权重可能被压缩, 或被 auto_adapt 的归一化抵消. ")
    else:
        ma = metrics["A-真实协变量"]
        mb = metrics["B-噪声协变量"]
        mc = metrics["C-反向协变量"]
        if ma["MAE"] is not None and mb["MAE"] is not None:
            if ma["MAE"] < mb["MAE"] and ma["MAE"] < mc["MAE"]:
                print("✅ 协变量生效且方向正确:")
                print("   传入真实协变量时精度最好, 传入噪声/反向时精度变差. ")
                print(f"   → {MODEL_ID} 确实在利用 future_covs 信息. ")
            else:
                print("🟠 协变量有影响但方向异常:")
                print("   传入不同协变量时预测有变化, 但真实协变量未带来最优精度. ")
                print("   → 可能存在协变量归一化或对齐问题, 需进一步排查. ")

# ============================================================
# 保存结果
# ============================================================

print("\n【4】保存结果")

results_data = []
for r in results:
    scene_key = r["scene"][:1] + "_" + r["scene"][2:]  # "A-真实协变量" → "A_真实协变量"
    if r["pred"] is not None:
        for i, v in enumerate(r["pred"]):
            results_data.append({
                "scene": scene_key,
                "step": i + 1,
                "pred_value": float(v),
                "ground_truth": float(ground_truth[i]),
                **{k: v for k, v in r["metrics"].items()},
                "latency_ms": r["latency_ms"],
            })

csv_path, json_path = save_with_json_backup(RESULT_PATH, results_data)
print(f"   ✅ 详细结果已保存CSV: {csv_path}")
print(f"   ✅ 预测值JSON已保存: {json_path}")


print("\n" + "=" * 80)
print("测试完成！")
print("=" * 80)
