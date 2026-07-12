"""
concept_drift_test.py —— 概念漂移与工况切换测试
====================================
工业背景：
  设备启停、负载阶跃、季节性工况切换会导致训练数据与预测目标分布不一致。
  这是工业时序预测的首要痛点。

测试目的：
  构造训练段平稳、预测段发生分布漂移的数据，检验模型对三种典型漂移
  模式的抵抗力，并验证长上下文窗口在漂移下是否反而是负担。
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 路径配置与导入
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODEL_LIST, DEFAULT_OUTPUT_LENGTH
from core.timecho import forecast

SCRIPT_DIR = Path(__file__).parent

# ============================================================
# 1. 生成训练段（平稳运行工况）
# ============================================================
np.random.seed(42)

N_TRAIN = 512       # 训练段长度（足够长，支持 512 点输入窗口）
N_FORECAST = DEFAULT_OUTPUT_LENGTH  # 64
TOTAL = N_TRAIN + N_FORECAST  # 576

dates = pd.date_range("2024-01-01", periods=TOTAL, freq="1h")

# 训练段信号：趋势 50->65 + 24h周期 + 小噪声（平稳工况）
trend_train = np.linspace(50, 65, N_TRAIN)
seasonal_train = 15 * np.sin(2 * np.pi * np.arange(N_TRAIN) / 24)
noise_train = np.random.randn(N_TRAIN) * 2
target_train = trend_train + seasonal_train + noise_train


# ============================================================
# 2. 构造 5 种漂移场景的预测段
# ============================================================
def make_forecast_segment(mode):
    """
    生成不同漂移模式下的预测段（64 点）。
    训练段始终保持不变，只有预测段（ground truth）发生变化。
    """
    t = np.arange(N_TRAIN, N_TRAIN + N_FORECAST)
    # 基础趋势：延续训练段的上升趋势
    trend = np.linspace(65, 80, N_FORECAST)
    # 基础周期：与训练段连续
    seasonal = 15 * np.sin(2 * np.pi * t / 24)
    # 基础噪声
    noise = np.random.randn(N_FORECAST) * 2

    if mode == "B1-基准(无漂移)":
        # 训练/测试同分布，稳态运行
        pass

    elif mode == "B2-均值平移(+15)":
        # 整体上移 15，对应设备负荷阶跃上调
        trend = trend + 15

    elif mode == "B3-方差扩张(3x)":
        # 噪声放大 3 倍，对应工况进入不稳定区
        noise = np.random.randn(N_FORECAST) * 6

    elif mode == "B4-周期相位偏移(90度)":
        # 周期相位偏移 90度，对应季节切换导致周期错位
        seasonal = 15 * np.sin(2 * np.pi * t / 24 + np.pi / 2)

    elif mode == "B5-复合漂移(均值+方差+相位)":
        # 三者叠加，对应大修后工况全面变化
        trend = trend + 15
        noise = np.random.randn(N_FORECAST) * 6
        seasonal = 15 * np.sin(2 * np.pi * t / 24 + np.pi / 2)

    else:
        raise ValueError(f"未知场景: {mode}")

    return (trend + seasonal + noise).round(4)


# ============================================================
# 3. 执行测试
# ============================================================
SCENARIOS = [
    "B1-基准(无漂移)",
    "B2-均值平移(+15)",
    "B3-方差扩张(3x)",
    "B4-周期相位偏移(90度)",
    "B5-复合漂移(均值+方差+相位)",
]
INPUT_LENGTHS = [96, 256, 512]
# 96: 短窗口（近期数据为主）
# 256: 中窗口（现有基准）
# 512: 长窗口（最大上下文，检验是否在漂移下反而有害）

total_calls = len(MODEL_LIST) * len(SCENARIOS) * len(INPUT_LENGTHS)
print("=" * 80)
print("场景：概念漂移与工况切换测试")
print(f"   {len(MODEL_LIST)} 模型 x {len(SCENARIOS)} 场景 x {len(INPUT_LENGTHS)} 输入长度 = {total_calls} 次调用")
print("=" * 80)

all_results = []

for mode in SCENARIOS:
    print(f"\n[场景] {mode}")

    # 重新设种子，保证预测段的噪声可复现
    np.random.seed(42 + SCENARIOS.index(mode))
    target_forecast = make_forecast_segment(mode)

    # 拼接完整序列
    full_target = np.concatenate([target_train, target_forecast])
    df = pd.DataFrame({"time": dates, "target": full_target.round(4)})

    for in_len in INPUT_LENGTHS:
        # 截取历史窗口：从训练段末尾取 in_len 个点
        history = df.iloc[N_TRAIN - in_len : N_TRAIN][["time", "target"]].copy()

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
                    print(f"   [{model_id}] in={in_len:>3d} | 失败: {str(error)[:60]}")
                    all_results.append({
                        "scenario": mode, "model_id": model_id, "input_length": in_len,
                        "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                        "success": False, "error": str(error),
                    })
                else:
                    mae = float(np.mean(np.abs(pred_values - target_forecast)))
                    rmse = float(np.sqrt(np.mean((pred_values - target_forecast) ** 2)))
                    print(f"   [{model_id}] in={in_len:>3d} | 成功 MAE={mae:.4f}  RMSE={rmse:.4f}  耗时={elapsed_ms:.0f}ms")
                    all_results.append({
                        "scenario": mode, "model_id": model_id, "input_length": in_len,
                        "mae": mae, "rmse": rmse, "latency_ms": elapsed_ms,
                        "success": True, "error": None,
                    })
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"   [{model_id}] in={in_len:>3d} | 异常: {str(e)[:60]}")
                all_results.append({
                    "scenario": mode, "model_id": model_id, "input_length": in_len,
                    "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                    "success": False, "error": str(e),
                })

            time.sleep(1)


# ============================================================
# 4. 汇总打印
# ============================================================
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

print(f"\n{'场景':>28s} | {'输入长度':>6s} | {'模型':>12s} | {'MAE':>10s} | {'RMSE':>10s} | 状态")
print("-" * 100)

for r in all_results:
    if r["success"]:
        print(f"{r['scenario']:>28s} | {r['input_length']:>6d} | {r['model_id']:>12s} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | 成功")
    else:
        print(f"{r['scenario']:>28s} | {r['input_length']:>6d} | {r['model_id']:>12s} | {'N/A':>10s} | {'N/A':>10s} | 失败")


# ============================================================
# 5. 核心分析
# ============================================================
print("\n" + "=" * 80)
print("核心分析")
print("=" * 80)

# 分析1：各漂移场景 vs 基准的退化倍数
print("\n【分析1】漂移场景 vs 基准的精度退化（input_length=256）")
print("-" * 70)

for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    # 取基准 MAE
    baseline_mae = None
    for r in all_results:
        if (r["model_id"] == model_id and "B1" in r["scenario"]
                and r["input_length"] == 256 and r["success"]):
            baseline_mae = r["mae"]
            break

    if baseline_mae is None:
        print(f"     基准数据缺失，跳过")
        continue

    print(f"     基准(B1) MAE = {baseline_mae:.4f}")
    for r in all_results:
        if (r["model_id"] == model_id and r["input_length"] == 256
                and r["success"] and "B1" not in r["scenario"]):
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

# 分析2：长上下文在漂移下是否有害
print("\n【分析2】长上下文窗口在漂移下是否有害？（B5-复合漂移）")
print("-" * 70)

for model_id in MODEL_LIST:
    print(f"\n  [{model_id}]")
    b5_results = {}
    for r in all_results:
        if (r["model_id"] == model_id and "B5" in r["scenario"]
                and r["success"]):
            b5_results[r["input_length"]] = r["mae"]

    if len(b5_results) < 2:
        print(f"     B5 数据不足，跳过")
        continue

    for in_len in sorted(b5_results.keys()):
        mae = b5_results[in_len]
        print(f"     input={in_len:>3d}: MAE={mae:.4f}")

    # 判断趋势
    mae_list = [b5_results[k] for k in sorted(b5_results.keys())]
    if mae_list[-1] > mae_list[0] * 1.1:
        print(f"     [警告] 长窗口(512)比短窗口(96) MAE 更高 -> 长上下文在漂移下可能是负担")
    elif mae_list[-1] < mae_list[0] * 0.9:
        print(f"     [通过] 长窗口(512)比短窗口(96) MAE 更低 -> 长上下文在漂移下仍有收益")
    else:
        print(f"     -> 长短窗口 MAE 接近，input_length 对漂移场景影响不大")


# ============================================================
# 6. 保存结果
# ============================================================
result_df = pd.DataFrame(all_results)
out_path = SCRIPT_DIR / "concept_drift_result.csv"
result_df.to_csv(out_path, index=False)
print(f"\n结果已保存: {out_path}")
print("=" * 80)
print("测试完成！")
