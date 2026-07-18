"""
frequency_mismatch_test.py -- 频率失配测试
====================================
测试目的:
  训练段周期恒为 24h, 预测段周期分别变为 24h / 12h / 8h / 48h,
  观测模型能否自适应预测段的频率/周期突变, 量化频率失配对预测精度的影响. 

  工业场景: 设备换速、换产导致信号周期变化.

调用次数: 8 次 (2模型 × 4模式)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from config.constants import DEFAULT_OUTPUT_LENGTH
from core.timecho import forecast

FORECAST_LEN = DEFAULT_OUTPUT_LENGTH  # 64
N_TRAIN = 512
np.random.seed(42)

# ============================================================
# 1. 构造测试数据
# ============================================================
time_full = pd.date_range("2024-01-01", periods=N_TRAIN + FORECAST_LEN, freq="1h")
time_history = time_full[:N_TRAIN]
future_time = time_full[N_TRAIN:]

# 训练段: 固定 24h 周期(所有模式共用)
trend_train = np.linspace(50, 65, N_TRAIN)
seasonal_train = 15 * np.sin(2 * np.pi * np.arange(N_TRAIN) / 24)
noise_train = np.random.randn(N_TRAIN) * 2
history = (trend_train + seasonal_train + noise_train).round(4)

# 预测段: 4 种不同周期
# 趋势延续(所有模式共用, 保证趋势连续)
trend_fc = np.linspace(65, 75, FORECAST_LEN)
noise_fc = np.random.randn(FORECAST_LEN) * 2

modes = {
    "1-正常(24→24)": {"period": 24, "desc": "基准: 周期不变"},
    "2-加速2倍(24→12)": {"period": 12, "desc": "换速: 周期减半"},
    "3-加速3倍(24→8)": {"period": 8, "desc": "极端换速: 周期变1/3"},
    "4-减速2倍(24→48)": {"period": 48, "desc": "降速: 周期翻倍"},
}

futures = {}
for mode_name, cfg in modes.items():
    p = cfg["period"]
    seasonal_fc = 15 * np.sin(2 * np.pi * np.arange(N_TRAIN, N_TRAIN + FORECAST_LEN) / p)
    futures[mode_name] = (trend_fc + seasonal_fc + noise_fc).round(4)

df_history = pd.DataFrame({"time": time_history, "target": history})

print(f"训练长度: {N_TRAIN}, 预测长度: {FORECAST_LEN}")
print(f"训练段周期: 24h(固定)")
print(f"预测段周期: {[modes[m]['period'] for m in modes]}h")
print()

# ============================================================
# 2. 执行测试
# ============================================================
models = ["Chronos-2", "Timer-3.5"]
results = []

for model_id in models:
    print(f"{'='*70}")
    print(f"模型: {model_id}")
    print(f"{'='*70}")

    for mode_name, cfg in modes.items():
        gt = futures[mode_name]
        try:
            pred, _, _ = forecast(
                targets=df_history,
                model_id=model_id,
                output_length=FORECAST_LEN,
                time_col="time"
            )
            mae = float(np.mean(np.abs(pred - gt)))
            # 逐步 MAE(前16步)
            step_mae = np.abs(pred - gt)
            mae_16 = float(np.mean(step_mae[:16]))
            mae_32 = float(np.mean(step_mae[:32]))

            print(f"  {mode_name:<25s}  MAE={mae:.4f}  (前16步: {mae_16:.4f}, 前32步: {mae_32:.4f})")
            results.append((model_id, mode_name, cfg["period"], mae, mae_16, mae_32))
        except Exception as e:
            print(f"  {mode_name:<25s}  失败: {type(e).__name__}: {e}")
            results.append((model_id, mode_name, cfg["period"], None, None, None))

    print()

# ============================================================
# 3. 汇总
# ============================================================
print(f"{'='*90}")
print("C5 频率失配 - 汇总")
print(f"{'='*90}")
print(f"{'模型':<15s} | {'模式':<25s} | {'预测周期':>8s} | {'MAE':>10s} | {'前16步MAE':>10s} | {'前32步MAE':>10s}")
print("-" * 90)
for model_id, mode_name, period, mae, mae16, mae32 in results:
    mae_s = f"{mae:.4f}" if mae is not None else "N/A"
    m16_s = f"{mae16:.4f}" if mae16 is not None else "N/A"
    m32_s = f"{mae32:.4f}" if mae32 is not None else "N/A"
    print(f"{model_id:<15s} | {mode_name:<25s} | {period:>6d}h  | {mae_s:>10s} | {m16_s:>10s} | {m32_s:>10s}")

# ============================================================
# 4. 分析
# ============================================================
print(f"\n{'='*90}")
print("关键对比分析")
print(f"{'='*90}")

for model_id in models:
    model_results = [r for r in results if r[0] == model_id and r[3] is not None]
    if len(model_results) < 2:
        continue

    print(f"\n  [{model_id}]")
    base = model_results[0]  # 24→24 基准
    mae_base = base[3]

    for r in model_results[1:]:
        mode_name, period, mae = r[1], r[2], r[3]
        ratio = mae / mae_base if mae_base > 0 else 0
        print(f"    {mode_name:<25s}  MAE: {mae_base:.4f} → {mae:.4f}  ({ratio:.2f}x)")

