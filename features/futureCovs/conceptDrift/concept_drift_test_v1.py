#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
concept_drift_test_v1.py —— 概念漂移测试(简约版)
====================================
工业背景:
  设备启停、负载阶跃、季节性工况切换会导致训练数据与预测目标分布不一致.
  这是工业时序预测的首要痛点.

原理: 对比不同漂移场景的预测精度差异

测试目的:
  构造训练段平稳、预测段发生分布漂移的数据, 检验模型对三种典型漂移
  模式的抵抗力, 并验证长上下文窗口在漂移下是否反而是负担.

Author: Janesong
Create Date: 2026/07/06, Update on 2026/07/21.
"""

import time
import numpy as np
import pandas as pd

from config.settings import OUTPUT_DIR
from config.constants import MODEL_LIST, FORECAST_POINT_LEN_64, TRAIN_SEQ_LEN_512
from core.timecho import forecast
from utils.files import save_to_csv

# ============================================================
# 1. Data related configuration
# ============================================================
OUTPUT_SUBDIR = OUTPUT_DIR / "features" / "futureCovs" / "conceptDrift"
OUTPUT_SUBDIR.mkdir(parents=True, exist_ok=True)
RESULT_CSV_PATH = OUTPUT_SUBDIR / "concept_drift_result_v1.csv"    # Prediction results file 预测结果文件

N_TRAIN = TRAIN_SEQ_LEN_512          # 训练段总长度(包含历史窗口及之前的数据)
N_FORECAST = FORECAST_POINT_LEN_64   # 预测长度(64)
DRIFT_LEAD = 20             # 漂移提前量: 在训练段末尾的前 DRIFT_LEAD 个点开始引入漂移
                            # 这样当 input_length > DRIFT_LEAD 时, 历史窗口会包含漂移信息

TOTAL = N_TRAIN + N_FORECAST
np.random.seed(42)

# ============================================================
# 2. 生成基础平稳信号(训练段)
# ============================================================
dates = pd.date_range("2026-07-01", periods=TOTAL, freq="1h")

# 训练段(前 N_TRAIN 点): 趋势 + 24h周期 + 小噪声
trend_base = np.linspace(50, 65, N_TRAIN)
seasonal_base = 15 * np.sin(2 * np.pi * np.arange(N_TRAIN) / 24)
noise_base = np.random.randn(N_TRAIN) * 2
train_steady = trend_base + seasonal_base + noise_base

# ============================================================
# 3. 生成不同漂移模式的完整序列(漂移从 N_TRAIN - DRIFT_LEAD 开始)
# ============================================================
def generate_full_sequence(mode):
    """
    生成完整序列(长度为 TOTAL), 漂移从训练段末尾的前 DRIFT_LEAD 个点开始.
    返回: (full_sequence, drift_start_index)
        drift_start_index = N_TRAIN - DRIFT_LEAD   (漂移开始的位置)
    """
    t_full = np.arange(TOTAL)
    # 基础趋势(整个序列: 训练段趋势 + 预测段趋势延续)
    trend_full = np.concatenate([
        np.linspace(50, 65, N_TRAIN),
        np.linspace(65, 80, N_FORECAST)
    ])
    # 基础周期(连续)
    seasonal_full = 15 * np.sin(2 * np.pi * t_full / 24)
    # 基础噪声(整体生成, 保证连续性)
    noise_full = np.random.randn(TOTAL) * 2

    # 复制一份用于叠加漂移
    signal = trend_full + seasonal_full + noise_full

    # 漂移起始索引
    drift_start = N_TRAIN - DRIFT_LEAD

    if mode == "B1-基准(无漂移)":
        # 无任何漂移
        pass

    elif mode == "B2-均值平移(+15)":
        # 均值 +15, 从 drift_start 开始
        signal[drift_start:] += 15

    elif mode == "B3-方差扩张(3x)":
        # 方差放大 3 倍, 从 drift_start 开始, 重新生成噪声(保持趋势和周期不变)
        # 为了保持连续性, 只对 drift_start 之后的噪声进行替换
        noise_new = np.random.randn(TOTAL - drift_start) * 6
        signal[drift_start:] = (trend_full + seasonal_full)[drift_start:] + noise_new

    elif mode == "B4-周期相位偏移(90度)":
        # 周期相位偏移 90°, 从 drift_start 开始
        seasonal_shifted = 15 * np.sin(2 * np.pi * t_full / 24 + np.pi / 2)
        signal[drift_start:] = trend_full[drift_start:] + seasonal_shifted[drift_start:] + noise_full[drift_start:]

    elif mode == "B5-复合漂移(均值+方差+相位)":
        # 三者叠加
        signal[drift_start:] += 15
        noise_new = np.random.randn(TOTAL - drift_start) * 6
        seasonal_shifted = 15 * np.sin(2 * np.pi * t_full / 24 + np.pi / 2)
        signal[drift_start:] = trend_full[drift_start:] + seasonal_shifted[drift_start:] + noise_new

    else:
        raise ValueError(f"未知场景: {mode}")

    return signal.round(4), drift_start

# ============================================================
# 4. 执行测试(对每个场景、输入长度、模型)
# ============================================================
SCENARIOS = [
    "B1-基准(无漂移)",
    "B2-均值平移(+15)",
    "B3-方差扩张(3x)",
    "B4-周期相位偏移(90度)",
    "B5-复合漂移(均值+方差+相位)",
]
INPUT_LENGTHS = [96, 256, 512]

total_calls = len(MODEL_LIST) * len(SCENARIOS) * len(INPUT_LENGTHS)
print("=" * 80)
print("场景: 概念漂移测试(漂移提前出现于历史窗口末尾)")
print(f"   {len(MODEL_LIST)} 模型 x {len(SCENARIOS)} 场景 x {len(INPUT_LENGTHS)} 输入长度 = {total_calls} 次调用")
print(f"   漂移提前量: {DRIFT_LEAD} 个点(历史窗口需 > {DRIFT_LEAD} 才可见)")
print("=" * 80)

all_results = []

for mode in SCENARIOS:
    print(f"\n[场景] {mode}")
    # 固定种子保证可复现, 但每个场景不同种子使噪声独立(可选)
    np.random.seed(42 + SCENARIOS.index(mode))
    full_seq, drift_start = generate_full_sequence(mode)

    # 构造完整 DataFrame
    df = pd.DataFrame({"time": dates, "target": full_seq})

    # 真实预测段(用于计算误差)
    target_forecast = full_seq[N_TRAIN:]

    for in_len in INPUT_LENGTHS:
        # 截取历史窗口: 从 drift_start - (in_len - DRIFT_LEAD) 到 N_TRAIN
        # 但为了保证窗口长度为 in_len, 起点为 N_TRAIN - in_len
        start_idx = N_TRAIN - in_len
        history = df.iloc[start_idx:N_TRAIN][["time", "target"]].copy()

        # 检查历史窗口是否包含了漂移起始点
        contains_drift = start_idx < drift_start
        print(f"   [输入长度 {in_len}] 历史窗口包含漂移起始点: {contains_drift}")

        for model_id in MODEL_LIST:
            t0 = time.perf_counter()
            try:
                pred_values, elapsed_ms, error = forecast(
                    targets=history,
                    model_id=model_id,
                    output_length=N_FORECAST,
                    time_col="time",
                    auto_adapt=True,
                )

                if error:
                    print(f"      [{model_id}] 失败: {str(error)[:60]}")
                    all_results.append({
                        "scenario": mode, "model_id": model_id, "input_length": in_len,
                        "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                        "success": False, "error": str(error),
                        "contains_drift": contains_drift,
                    })
                else:
                    mae = float(np.mean(np.abs(pred_values - target_forecast)))
                    rmse = float(np.sqrt(np.mean((pred_values - target_forecast) ** 2)))
                    print(f"      [{model_id}] MAE={mae:.4f}  RMSE={rmse:.4f}  耗时={elapsed_ms:.0f}ms")
                    all_results.append({
                        "scenario": mode, "model_id": model_id, "input_length": in_len,
                        "mae": mae, "rmse": rmse, "latency_ms": elapsed_ms,
                        "success": True, "error": None,
                        "contains_drift": contains_drift,
                    })
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"      [{model_id}] 异常: {str(e)[:60]}")
                all_results.append({
                    "scenario": mode, "model_id": model_id, "input_length": in_len,
                    "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                    "success": False, "error": str(e),
                    "contains_drift": contains_drift,
                })

            time.sleep(1)


# ============================================================
# 5. 汇总打印
# ============================================================
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

print(f"\n{'场景':>28s} | {'输入长度':>6s} | {'模型':>12s} | {'含漂移?':>8s} | {'MAE':>10s} | {'RMSE':>10s} | 状态")
print("-" * 110)

for r in all_results:
    if r["success"]:
        contains = "是" if r["contains_drift"] else "否"
        print(f"{r['scenario']:>28s} | {r['input_length']:>6d} | {r['model_id']:>12s} | {contains:>8s} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | 成功")
    else:
        print(f"{r['scenario']:>28s} | {r['input_length']:>6d} | {r['model_id']:>12s} | {'-':>8s} | {'N/A':>10s} | {'N/A':>10s} | 失败")


# ============================================================
# 6. 核心分析
# ============================================================
print("\n" + "=" * 80)
print("核心分析")
print("=" * 80)

# 分析1: 各漂移场景 vs 基准的退化倍数(仅针对 input_length=256, 且历史包含漂移的情况)
print("\n【分析1】漂移场景 vs 基准的精度退化(input_length=256, 历史包含漂移)")
print("-" * 70)

for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    # 取基准 MAE(B1场景)
    baseline_mae = None
    for r in all_results:
        if (r["model_id"] == model_id and "B1" in r["scenario"]
                and r["input_length"] == 256 and r["success"]):
            baseline_mae = r["mae"]
            break

    if baseline_mae is None:
        print(f"     基准数据缺失, 跳过")
        continue

    print(f"     基准(B1) MAE = {baseline_mae:.4f}")
    for r in all_results:
        if (r["model_id"] == model_id and r["input_length"] == 256
                and r["success"] and "B1" not in r["scenario"]
                and r["contains_drift"]):   # 只分析历史包含漂移的情况
            ratio = r["mae"] / baseline_mae
            if ratio < 1.2:
                verdict = "[正常] 无影响"
            elif ratio < 2.0:
                verdict = "[轻微] 轻微退化"
            elif ratio < 5.0:
                verdict = "[警告] 明显退化"
            else:
                verdict = "[严重] 严重退化"
            print(f"     {r['scenario']:>28s}: MAE={r['mae']:.4f} ({ratio:.1f}x) -> {verdict}")

# 分析2: 对比“历史含漂移”与“历史不含漂移”的性能差异(以B2为例)
print("\n【分析2】历史信息对漂移适应的影响(以B2-均值平移为例, input_length=256)")
print("-" * 70)
for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    # 找B2场景下, 含漂移和不含漂移的结果
    mae_with = None
    mae_without = None
    for r in all_results:
        if r["model_id"] == model_id and "B2" in r["scenario"] and r["input_length"] == 256 and r["success"]:
            if r["contains_drift"]:
                mae_with = r["mae"]
            else:
                mae_without = r["mae"]
    if mae_with is not None and mae_without is not None:
        print(f"     历史含漂移 MAE = {mae_with:.4f}")
        print(f"     历史不含漂移 MAE = {mae_without:.4f}")
        print(f"     含漂移相对不含漂移提升 = {(mae_without - mae_with) / mae_without * 100:.1f}%")
    else:
        print(f"     数据不足, 跳过")

# 分析3: 长上下文在漂移下的有效性(B5复合漂移, 仅看历史包含漂移的情况)
print("\n【分析3】长上下文在漂移下的收益(B5-复合漂移, 历史包含漂移)")
print("-" * 70)

for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    b5_results = {}
    for r in all_results:
        if (r["model_id"] == model_id and "B5" in r["scenario"]
                and r["success"] and r["contains_drift"]):
            b5_results[r["input_length"]] = r["mae"]

    if len(b5_results) < 2:
        print(f"     B5 数据不足, 跳过")
        continue

    for in_len in sorted(b5_results.keys()):
        mae = b5_results[in_len]
        print(f"     input={in_len:>3d}: MAE={mae:.4f}")

    # 判断趋势
    mae_list = [b5_results[k] for k in sorted(b5_results.keys())]
    if mae_list[-1] < mae_list[0] * 0.9:
        print(f"     [Pass] 长窗口(512)比短窗口(96) MAE 更低 -> 长上下文在漂移下有收益")
    elif mae_list[-1] > mae_list[0] * 1.1:
        print(f"     [warning][警告] 长窗口(512)比短窗口(96) MAE 更高 -> 长上下文可能引入冗余信息")
    else:
        print(f"     -> 长短窗口 MAE 接近, input_length 对漂移场景影响不大")


# ============================================================
# 7. Save Results
# ============================================================
result_path = save_to_csv(RESULT_CSV_PATH, all_results)
print(f"   Results saved to CSV: {result_path}")
print("=" * 80)
print(" Test completed!")
