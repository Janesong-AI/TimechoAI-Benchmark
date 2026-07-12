"""
场景A: 变采样率与不规则时间戳测试
====================================
工业背景:
  工业现场传感器并非等间隔采样. 经网关聚合后写入时序库时, 时间戳会出现抖动、乱序、甚至重复.

测试目的:
  检验 SDK 是否真正理解 time_col 的时间戳语义, 还是仅按行序号处理.
  若 SDK 静默按序号处理, 则无论时间戳怎么变, 预测结果应完全一致.

"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# 0. 路径配置与导入(与现有脚本保持一致)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import DEFAULT_OUTPUT_LENGTH
from core.timecho import forecast

SCRIPT_DIR = Path(__file__).parent

# ============================================================
# 1. 生成底层信号(等间隔理想序列, 与脏数据测试同源)
# ============================================================
np.random.seed(42)

TOTAL = 320  # 256 history + 64 forecast
FORECAST_LEN = DEFAULT_OUTPUT_LENGTH  # 64

# 理想等间隔时间戳(1小时一个点)
ideal_dates = pd.date_range("2024-01-01", periods=TOTAL, freq="1h")

# 底层信号: 趋势 + 周期 + 噪声
trend = np.linspace(50, 80, TOTAL)
seasonal = 15 * np.sin(2 * np.pi * np.arange(TOTAL) / 24)
noise = np.random.randn(TOTAL) * 2
target_values = (trend + seasonal + noise).round(4)

# Ground truth(最后 64 点的真实值)
ground_truth = target_values[-FORECAST_LEN:]

HISTORY_LEN = TOTAL - FORECAST_LEN  # 256


# ============================================================
# 2. 构造 4 种时间戳不规则场景
# ============================================================
def make_timestamps(mode):
    """
    根据 mode 构造不同不规则程度的时间戳.
    注意: target 值序列始终保持不变, 只有时间戳列变化.
    这样如果 SDK 忽略时间戳, 4 种场景的 MAE 应该完全相同.
    """
    if mode == "A1-基准(等间隔)":
        # 严格等间隔 1h, 工业实验室理想采集
        return ideal_dates.copy()

    elif mode == "A2-轻微抖动(5%)":
        # 间隔在 1h +- 5%(+-180秒)内随机扰动
        # 对应: 网关时钟漂移
        deltas_seconds = 3600 + np.random.randn(TOTAL - 1) * 180
        deltas_seconds = np.maximum(deltas_seconds, 60)  # 最小 1 分钟
        ts = [ideal_dates[0]]
        for d in deltas_seconds:
            ts.append(ts[-1] + pd.Timedelta(seconds=int(d)))
        return pd.DatetimeIndex(ts)

    elif mode == "A3-中度漂移(20%)":
        # 间隔在 1h +- 20%(+-720秒)内扰动, 且 10% 的点出现 2-3 倍间隔
        # 对应: 多源传感器异步聚合, 偶发丢点后补传
        deltas_seconds = 3600 + np.random.randn(TOTAL - 1) * 720
        deltas_seconds = np.maximum(deltas_seconds, 60)
        # 随机选 10% 的间隔放大 2-3 倍
        mask = np.random.rand(len(deltas_seconds)) < 0.1
        deltas_seconds[mask] *= np.random.choice([2, 3], size=mask.sum())
        ts = [ideal_dates[0]]
        for d in deltas_seconds:
            ts.append(ts[-1] + pd.Timedelta(seconds=int(d)))
        return pd.DatetimeIndex(ts)

    elif mode == "A4-严重乱序":
        # 在 A3 基础上, 随机打乱 10% 的时间戳
        # 对应: 网络重传/乱序到达, 时间戳非单调递增
        ts_list = make_timestamps("A3-中度漂移(20%)").tolist()
        n_shuffle = len(ts_list) // 10
        indices = np.random.choice(len(ts_list), size=n_shuffle, replace=False)
        shuffled_vals = [ts_list[i] for i in indices]
        np.random.shuffle(shuffled_vals)
        for i, idx in enumerate(indices):
            ts_list[idx] = shuffled_vals[i]
        return pd.DatetimeIndex(ts_list)

    else:
        raise ValueError(f"未知场景: {mode}")


# ============================================================
# 3. 执行测试
# ============================================================
MODELS = ["Timer-3.5", "Chronos-2"]
SCENARIOS = [
    "A1-基准(等间隔)",
    "A2-轻微抖动(5%)",
    "A3-中度漂移(20%)",
    "A4-严重乱序",
]

total_calls = len(MODELS) * len(SCENARIOS)
print("=" * 80)
print("场景A: 变采样率与不规则时间戳测试")
print(f"   {len(MODELS)} 模型 x {len(SCENARIOS)} 场景 = {total_calls} 次调用")
print("=" * 80)

all_results = []

for mode in SCENARIOS:
    print(f"\n[场景] {mode}")

    # 每次重新设种子, 保证不同场景的噪声扰动可复现
    np.random.seed(42)
    timestamps = make_timestamps(mode)

    # 构造 DataFrame: target 值不变, 只有 time 列变
    df = pd.DataFrame({"time": timestamps, "target": target_values})
    history = df.iloc[:HISTORY_LEN][["time", "target"]].copy()

    # 打印时间戳间隔统计
    if mode == "A4-严重乱序":
        print(f"   时间戳单调递增: {timestamps.is_monotonic_increasing}")
    deltas = np.diff(timestamps.values.astype("datetime64[s]").astype(np.int64))
    print(f"   间隔: min={deltas.min()//60}min  max={deltas.max()//3600:.1f}h  mean={deltas.mean()//3600:.1f}h")

    for model_id in MODELS:
        t0 = time.perf_counter()
        try:
            pred_values, elapsed_ms, error = forecast(
                targets=history,
                model_id=model_id,
                output_length=FORECAST_LEN,
                time_col="time",
                auto_adapt=True,
            )

            if error:
                print(f"   [{model_id}] 失败: {str(error)[:80]}")
                all_results.append({
                    "scenario": mode, "model_id": model_id,
                    "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                    "success": False, "error": str(error),
                })
            else:
                mae = float(np.mean(np.abs(pred_values - ground_truth)))
                rmse = float(np.sqrt(np.mean((pred_values - ground_truth) ** 2)))
                print(f"   [{model_id}] 成功 MAE={mae:.4f}  RMSE={rmse:.4f}  耗时={elapsed_ms:.0f}ms")
                all_results.append({
                    "scenario": mode, "model_id": model_id,
                    "mae": mae, "rmse": rmse, "latency_ms": elapsed_ms,
                    "success": True, "error": None,
                })
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"   [{model_id}] 异常: {str(e)[:80]}")
            all_results.append({
                "scenario": mode, "model_id": model_id,
                "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                "success": False, "error": str(e),
            })

        time.sleep(1)


# ============================================================
# 4. 汇总分析
# ============================================================
print("\n" + "=" * 80)
print("测试结果汇总")
print("=" * 80)

print(f"\n{'场景':>22s} | {'模型':>12s} | {'MAE':>10s} | {'RMSE':>10s} | {'耗时(ms)':>8s} | 状态")
print("-" * 90)

for r in all_results:
    if r["success"]:
        print(f"{r['scenario']:>22s} | {r['model_id']:>12s} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['latency_ms']:>8.0f} | 成功")
    else:
        print(f"{r['scenario']:>22s} | {r['model_id']:>12s} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>8s} | 失败 {r['error'][:30]}")

# ============================================================
# 5. 核心分析: SDK 是否利用了时间戳语义？
# ============================================================
print("\n" + "=" * 80)
print("核心分析: SDK 是否理解时间戳语义？")
print("=" * 80)

for model_id in MODELS:
    print(f"\n  [{model_id}]")
    model_results = [r for r in all_results if r["model_id"] == model_id and r["success"]]

    if len(model_results) < 2:
        print(f"     成功场景不足, 无法分析")
        continue

    # 取基准 MAE
    baseline = None
    for r in model_results:
        if "A1" in r["scenario"]:
            baseline = r["mae"]
            break

    if baseline is None:
        baseline = model_results[0]["mae"]

    all_same = True
    for r in model_results:
        ratio = r["mae"] / baseline if baseline > 0 else 0
        if abs(r["mae"] - baseline) > 0.01:
            all_same = False
        print(f"     {r['scenario']:>22s}: MAE={r['mae']:.4f} (基准的 {ratio:.2f}x)")

    if all_same:
        print(f"     ⚠️[警告] 所有场景 MAE 完全一致 -> SDK 可能忽略 time_col, 仅按行序号处理")
    else:
        print(f"     ✅[通过] 不同时间戳导致 MAE 变化 -> SDK 利用了时间戳语义")


# ============================================================
# 6. 保存结果
# ============================================================
result_df = pd.DataFrame(all_results)
out_path = SCRIPT_DIR / "irregular_sampling_result.csv"
result_df.to_csv(out_path, index=False)
print(f"\n结果已保存: {out_path}")
print("=" * 80)
print("测试完成！")
