#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
input_length_test.py —— input_length消融测试
====================================
作用: TimechoAI 预测 API 在不同历史输入长度下的响应能力
设计目的: 验证 DEFAULT_INPUT_LENGTH 配置的实际可用性, 确保模型能正确处理
          指定长度的历史数据, 并返回预期的 output_length 个预测点. 
原理: 传不同的 input_length (96/192/256/384/512), 对比 MAE/RMSE 变化

Author: Janesong
Create Date: 2026/06/29, Updated on 2026/07/14.
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

from config.settings import DEFAULT_OUTPUT_LENGTH
from core.timecho import forecast
from utils.files import save_to_csv

SCRIPT_DIR = Path(__file__).parent
RESULT_CSV_PATH = SCRIPT_DIR / "input_length_result.csv"

# ============================================================
# 1. 生成合成数据(与脏数据测试同源, 保证可比性)
# ============================================================
print("📦 生成合成数据...")
np.random.seed(42)
TOTAL = 512 + 64  # 512(最大input) + 64(forecast) = 576
dates = pd.date_range("2024-01-01", periods=TOTAL, freq="1h")
trend = np.linspace(50, 80, TOTAL)
seasonal = 15 * np.sin(2 * np.pi * np.arange(TOTAL) / 24)
noise = np.random.randn(TOTAL) * 2
target = trend + seasonal + noise

df = pd.DataFrame({"time": dates, "target": target.round(4)})

# 真实值(最后 64 点, 作为 ground truth)
ground_truth = df.iloc[-64:]["target"].values
FORECAST_LEN = DEFAULT_OUTPUT_LENGTH

# ============================================================
# 2. 消融测试配置
# ============================================================
INPUT_LENGTHS = [96, 192, 256, 384, 512]
MODELS = ["Timer-3.5", "Chronos-2"]

# ============================================================
# 3. 逐模型 × 逐长度 测试
# ============================================================
total_calls = len(MODELS) * len(INPUT_LENGTHS)
print(f"🚀 消融测试: {len(MODELS)} 模型 × {len(INPUT_LENGTHS)} 长度 = {total_calls} 次调用")
print("=" * 80)

all_results = []

for model_id in MODELS:
    print(f"\n📋 模型: {model_id}")
    
    for in_len in INPUT_LENGTHS:
        # 截取历史数据: 取最后 in_len+FORECAST_LEN 行的前 in_len 行
        history = df.iloc[-(in_len + FORECAST_LEN):-FORECAST_LEN][["time", "target"]].copy()
        
        t0 = time.perf_counter()
        try:
            # 调用 core 层的封装方法, 它内部会调用 utils.client.get_timecho_client()
            pred_values, elapsed_ms, error = forecast(
                targets=history,
                model_id=model_id,
                output_length=FORECAST_LEN,
                time_col="time",
                auto_adapt=True,
            )
            
            if error:
                print(f"  input={in_len:>3d} | ❌ 失败: {str(error)[:80]}")
                all_results.append({
                    "model_id": model_id, "input_length": in_len,
                    "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                    "success": False, "error": str(error)
                })
            else:
                mae = float(np.mean(np.abs(pred_values - ground_truth)))
                rmse = float(np.sqrt(np.mean((pred_values - ground_truth) ** 2)))
                
                print(f"  input={in_len:>3d} | MAE={mae:.4f} | RMSE={rmse:.4f} | 耗时={elapsed_ms:.0f}ms")
                
                all_results.append({
                    "model_id": model_id, "input_length": in_len,
                    "mae": mae, "rmse": rmse, "latency_ms": elapsed_ms,
                    "success": True, "error": None
                })
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            print(f"  input={in_len:>3d} | ❌ 异常: {str(e)[:80]}")
            all_results.append({
                "model_id": model_id, "input_length": in_len,
                "mae": None, "rmse": None, "latency_ms": elapsed_ms,
                "success": False, "error": str(e)
            })
        
        time.sleep(1)

# ============================================================
# 4. 汇总打印
# ============================================================
print("\n" + "=" * 80)
print("📊 消融测试汇总")
print("=" * 80)
print(f"  {'模型':>12s} | {'input_len':>9s} | {'MAE':>10s} | {'RMSE':>10s} | {'耗时(ms)':>8s}")
print(f"  {'─'*12}─┼─{'─'*9}─┼─{'─'*10}─┼─{'─'*10}─┼─{'─'*8}")

for r in all_results:
    if r["success"]:
        print(f"  {r['model_id']:>12s} | {r['input_length']:>9d} | {r['mae']:>10.4f} | {r['rmse']:>10.4f} | {r['latency_ms']:>8.0f}")
    else:
        print(f"  {r['model_id']:>12s} | {r['input_length']:>9d} | {'N/A':>10s} | {'N/A':>10s} | {'N/A':>8s}")

# ============================================================
# 5. 保存结果
# ============================================================
result_path = save_to_csv(RESULT_CSV_PATH, all_results)
print(f"   💾 结果已保存: {result_path}")
print("=" * 80)
print("测试完成！")