#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
concept_drift_test_v2.py —— 概念漂移与工况切换测试(XYZ场景) 
====================================
工业背景:
  设备启停、负载阶跃、季节性工况切换会导致训练数据与预测目标分布不一致.
  这是工业时序预测的首要痛点.

测试目的:
  构造训练段平稳、预测段发生分布漂移的数据, 检验模型对三种典型漂移
  模式的抵抗力, 并验证长上下文窗口在漂移下是否反而是负担.

特性:
  1. 自动判断同级目录是否有 concept_drift_result_v2.csv
  2. 有 CSV: 读取记录, 跳过已完成, 走断点续跑
  3. 无 CSV: 从零开始跑全量
  4. 每完成一次调用立即追加保存
  5. 429 限流: 停止本次运行, 不扣总数, 下次可重试
  6. 422 永久失败: 扣总数, 不再重试, 不影响最终分析
  7. Timer-3.5/Timer-3.0 跑 Z 类场景: 直接跳过, 因为不支持协变量【422】

调用次数:
  主测试(XYZ): 6 模型 × 11 场景 × 3 长度 = 198 次.
  消融测试(Y4 auto_adapt): 6 模型 × 2 开关 × 3 长度 = 36 次.
  扣除不支持协变量的 12 次(NO_COV 跳过).
  扣除消融去重 18 次(Y4/adapt=True 与主测试重复).
  原始任务总数 234 次, 实际需完成 204 次.

Author: Janesong
Create Date: 2026/07/10, Update on 2026/07/12.
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
from core.resume import load_completed_results, append_result, is_rate_limited
from core.timecho import forecast

SCRIPT_DIR = Path(__file__).parent
RESULT_CSV_PATH = SCRIPT_DIR / "concept_drift_result_v2.csv"

# ============================================================
# 1. 全局参数
# ============================================================
N_CONTEXT = 512      # 上下文窗口总长度(历史段)
N_FORECAST = DEFAULT_OUTPUT_LENGTH  # 64
N_TOTAL = N_CONTEXT + N_FORECAST    # 576

NO_COV_MODELS = {"Timer-3.5", "Timer-3.0"}     # 不支持协变量的模型列表(Z类场景直接跳过)

DATES = pd.date_range("2024-01-01", periods=N_TOTAL, freq="1h")

# 基础信号参数
BASE_TREND_START = 50
BASE_TREND_END = 65
BASE_SEASONAL_AMP = 15
BASE_SEASONAL_PERIOD = 24
BASE_NOISE_STD = 2

# 漂移参数
DRIFT_MEAN_SHIFT = 15        # 均值平移幅度
DRIFT_NOISE_MULTIPLIER = 3   # 方差扩张倍数
DRIFT_PHASE_SHIFT = np.pi/2  # 相位偏移(90度)

# 漂移过渡区长度(在历史窗口末端逐步引入漂移, 模拟真实工况切换)
DRIFT_RAMP_LEN = 64          # 最后 64 个历史点逐步过渡

INPUT_LENGTHS = [96, 256, 512]

AUTO_ADAPT_ABLATION_SCENARIO = "Y4"
AUTO_ADAPT_VALUES = [True, False]

# 原始任务总数 = 主测试(11场景×6模型×3长度) + 消融(6模型×2开关×3长度)
TOTAL_RAW = 11 * 6 * 3 + 6 * 2 * 3  # = 234
# Z场景 × 不支持协变量的2个模型 × 3长度 = 12(代码层跳过, 不调API)
NO_COV_SKIP_COUNT = 2 * len(NO_COV_MODELS) * len(INPUT_LENGTHS)  # = 12
# 消融测试中 Y4/adapt=True 与主测试完全重复 = 6模型×1×3长度 = 18
DEDUP_SKIP_COUNT = 6 * 1 * len(INPUT_LENGTHS)  # = 18


# ============================================================
# 2. 信号生成与场景构造
# ============================================================
def _safe_logspace(start, end, n):
    """生成从 start 到 end 的 n 个点, 允许 start=0(用线性替代)."""
    if start <= 0:
        k = max(1, n // 3)
        part1 = np.linspace(start, 0.01, k)
        part2 = np.logspace(np.log10(0.01), np.log10(end), n - k + 1)[1:]
        return np.concatenate([part1, part2])
    return np.logspace(np.log10(start), np.log10(end), n)

def generate_base_signal(n, seed=42):
    np.random.seed(seed)
    t = np.arange(n)
    trend = np.linspace(BASE_TREND_START, BASE_TREND_END, n)
    seasonal = BASE_SEASONAL_AMP * np.sin(2 * np.pi * t / BASE_SEASONAL_PERIOD)
    noise = np.random.randn(n) * BASE_NOISE_STD
    return (trend + seasonal + noise).round(4), t

def build_scenarios():
    scenarios = []
    full_base, t = generate_base_signal(N_TOTAL, seed=42)
    base_history = full_base[:N_CONTEXT].copy()
    base_future = full_base[N_CONTEXT:].copy()

    future_mean_shift = (base_future + DRIFT_MEAN_SHIFT).copy()
    t_future = np.arange(N_CONTEXT, N_TOTAL)
    future_trend = np.linspace(BASE_TREND_END, BASE_TREND_END + 15, N_FORECAST)
    future_seasonal = BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_future / BASE_SEASONAL_PERIOD)
    
    np.random.seed(42)
    future_noise_base = np.random.randn(N_FORECAST) * BASE_NOISE_STD
    future_noise_3x = future_noise_base * DRIFT_NOISE_MULTIPLIER
    
    future_variance_expansion = (future_trend + future_seasonal + future_noise_3x).round(4)
    future_phase_shift = (future_trend + BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_future / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT) + future_noise_base).round(4)
    future_compound = (future_trend + DRIFT_MEAN_SHIFT + BASE_SEASONAL_AMP * np.sin(2 * np.pi * t_future / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT) + future_noise_3x).round(4)

    # ================================================================
    # X 类: 漂移不可见(历史段完全平稳, 未来段突变)
    # ================================================================
    for sid, label, fut, desc in [
        ("X1", "X1-基准(无漂移,不可见)", base_future, "基准"),
        ("X2", "X2-均值平移+15(不可见)", future_mean_shift, "均值突变"),
        ("X3", "X3-方差扩张3x(不可见)", future_variance_expansion, "方差突变"),
        ("X4", "X4-相位偏移90°(不可见)", future_phase_shift, "相位突变"),
        ("X5", "X5-复合漂移(不可见)", future_compound, "复合突变"),
    ]:
        scenarios.append({
            "scenario_id": sid, "category": "X", "label": label,
            "history": pd.DataFrame({"time": DATES[:N_CONTEXT], "target": base_history}),
            "future_target": fut, "future_covs": None, "description": desc
        })

    # ================================================================
    # Y 类: 漂移部分可见(历史段末端逐步过渡, 未来段延续漂移)
    # ================================================================
    def build_y_history(drift_type):
        stable_len = N_CONTEXT - DRIFT_RAMP_LEN
        history = base_history.copy()
        ramp_t = np.arange(stable_len, N_CONTEXT)
        ramp_idx = np.arange(DRIFT_RAMP_LEN)

        ramp_base_trend = np.linspace(
            BASE_TREND_START + (BASE_TREND_END - BASE_TREND_START) * stable_len / N_CONTEXT,
            BASE_TREND_END, DRIFT_RAMP_LEN)
        ramp_base_seasonal = BASE_SEASONAL_AMP * np.sin(2 * np.pi * ramp_t / BASE_SEASONAL_PERIOD)
        ramp_base_noise = base_history[stable_len:] - (
            np.linspace(BASE_TREND_START, BASE_TREND_END, N_CONTEXT)[stable_len:] +
            BASE_SEASONAL_AMP * np.sin(2 * np.pi * ramp_t / BASE_SEASONAL_PERIOD))

        weight = 1 / (1 + np.exp(-(ramp_idx - DRIFT_RAMP_LEN / 2) / 8))

        if drift_type == 'mean_shift':
            ramp_drift = ramp_base_trend + ramp_base_seasonal + ramp_base_noise + DRIFT_MEAN_SHIFT * weight
        elif drift_type == 'variance':
            ramp_drift = ramp_base_trend + ramp_base_seasonal + ramp_base_noise * (1 + (DRIFT_NOISE_MULTIPLIER - 1) * weight)
        elif drift_type == 'phase':
            ramp_drift = ramp_base_trend + BASE_SEASONAL_AMP * np.sin(2 * np.pi * ramp_t / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT * weight) + ramp_base_noise
        elif drift_type == 'compound':
            ramp_drift = (ramp_base_trend + DRIFT_MEAN_SHIFT * weight + 
                          BASE_SEASONAL_AMP * np.sin(2 * np.pi * ramp_t / BASE_SEASONAL_PERIOD + DRIFT_PHASE_SHIFT * weight) + 
                          ramp_base_noise * (1 + (DRIFT_NOISE_MULTIPLIER - 1) * weight))

        history[stable_len:] = ramp_drift.round(4)
        return history

    for sid, label, dt, fut, desc in [
        ("Y1", "Y1-均值平移+15(部分可见)", 'mean_shift', future_mean_shift, "均值外推"),
        ("Y2", "Y2-方差扩张3x(部分可见)", 'variance', future_variance_expansion, "方差外推"),
        ("Y3", "Y3-相位偏移90°(部分可见)", 'phase', future_phase_shift, "相位外推"),
        ("Y4", "Y4-复合漂移(部分可见)", 'compound', future_compound, "复合外推"),
    ]:
        scenarios.append({
            "scenario_id": sid, "category": "Y", "label": label,
            "history": pd.DataFrame({"time": DATES[:N_CONTEXT], "target": build_y_history(dt)}),
            "future_target": fut, "future_covs": None, "description": desc
        })

    # ================================================================
    # Z 类: 协变量传递(历史段平稳, 通过未来协变量传递漂移信号)
    # ================================================================
    # Z1: 历史段无协变量(全0), 未来协变量为全1
    history_cov_z1 = pd.DataFrame({
        "time": DATES[:N_CONTEXT],
        "target": base_history,
        "load_level": np.zeros(N_CONTEXT)
    })
    future_cov_z1 = pd.DataFrame({
        "time": DATES[N_CONTEXT:N_TOTAL],
        "load_level": np.ones(N_FORECAST)
    })
    scenarios.append({
        "scenario_id": "Z1", "category": "Z", "label": "Z1-均值平移(协变量传递)",
        "history": history_cov_z1,
        "future_target": future_mean_shift,
        "future_covs": future_cov_z1,
        "description": "协变量信号(历史无, 未来有)"
    })
    
    # Z2: 历史段末尾逐渐升高的协变量, 未来继续为全1
    y_history_mean = build_y_history('mean_shift')  # 复用 Y1 的历史目标值
    history_cov_z2 = pd.DataFrame({
        "time": DATES[:N_CONTEXT],
        "target": y_history_mean,
        "load_level": np.concatenate([
            np.zeros(N_CONTEXT - DRIFT_RAMP_LEN),
            _safe_logspace(0.01, 1.0, DRIFT_RAMP_LEN)
        ])
    })
    future_cov_z2 = pd.DataFrame({
        "time": DATES[N_CONTEXT:N_TOTAL],
        "load_level": np.ones(N_FORECAST)
    })
    scenarios.append({
        "scenario_id": "Z2", "category": "Z", "label": "Z2-均值平移(协变量+部分可见)",
        "history": history_cov_z2,
        "future_target": future_mean_shift,
        "future_covs": future_cov_z2,
        "description": "双重信号(历史协变量渐变 + 未来全开)"
    })

    return scenarios


# ============================================================
# 3. 执行预测与评估
# ============================================================
def compute_metrics(pred, target):
    """计算 MAE 和 RMSE."""
    return {"mae": float(np.mean(np.abs(pred - target))), "rmse": float(np.sqrt(np.mean((pred - target) ** 2)))}

def run_forecast(scenario, model_id, in_len, auto_adapt=True):
    """
    执行单次预测, 正确处理协变量的传递:
      -- 将 history 中的 time 和 target 作为 targets
      -- 如果 history 还有其它列, 则提取为 history_covs
    - future_covs 从 scenario 中获取(可能为 None 或 DataFrame)
    """
    # 提取 targets (必须包含 time 和 target)
    targets_df = scenario["history"][["time", "target"]].iloc[-in_len:].copy()
    
    # 提取历史协变量(如果存在除 time, target 之外的列)
    history_covs_df = None
    if len(scenario["history"].columns) > 2:
        # 必须包含 time 列, SDK 要求历史协变量与目标序列在时间上对齐
        cov_cols = [c for c in scenario["history"].columns if c not in ["time", "target"]]
        if cov_cols:
            history_covs_df = scenario["history"][["time"] + cov_cols].iloc[-in_len:].copy()

    # 未来协变量
    future_covs_df = scenario.get("future_covs")
    target = scenario["future_target"]

    t0 = time.perf_counter()
    try:
        kwargs = dict(
            targets=targets_df,
            model_id=model_id,
            output_length=N_FORECAST,
            time_col="time",
            auto_adapt=auto_adapt
        )
        if history_covs_df is not None:
            kwargs["history_covs"] = history_covs_df
        if future_covs_df is not None:
            kwargs["future_covs"] = future_covs_df

        pred_values, elapsed_ms, error = forecast(**kwargs)

        if error:
            return {
                "scenario_id": scenario["scenario_id"], "category": scenario["category"], "label": scenario["label"],
                "model_id": model_id, "input_length": in_len, "auto_adapt": auto_adapt, "success": False, "error": str(error),
                "mae": None, "rmse": None, "latency_ms": elapsed_ms, "ablation": False
            }

        m = compute_metrics(pred_values, target)
        return {
            "scenario_id": scenario["scenario_id"], "category": scenario["category"], "label": scenario["label"],
            "model_id": model_id, "input_length": in_len, "auto_adapt": auto_adapt, "success": True, "error": None,
            "mae": m["mae"], "rmse": m["rmse"], "latency_ms": elapsed_ms, "ablation": False
        }
    except Exception as e:
        return {
            "scenario_id": scenario["scenario_id"], "category": scenario["category"], "label": scenario["label"],
            "model_id": model_id, "input_length": in_len, "auto_adapt": auto_adapt, "success": False, "error": str(e)[:120],
            "mae": None, "rmse": None, "latency_ms": (time.perf_counter() - t0) * 1000, "ablation": False
        }


# ============================================================
# 4. 主流程
# ============================================================
scenarios = build_scenarios()
records, perm_fail_count = load_completed_results(str(RESULT_CSV_PATH))

# 构建 completed_keys: 成功 + 永久失败(非限流错误)
completed_keys = set()
for r in records:
    aa_val = str(r.get("auto_adapt", True)).strip() == "True"
    key = (str(r["scenario_id"]), str(r["model_id"]), int(r["input_length"]), aa_val)
    if r.get("success") == True:
        completed_keys.add(key)
    else:
        if not is_rate_limited(str(r.get("error", ""))):
            # 永久失败(如422), 加入 completed 跳过
            completed_keys.add(key)

# 实际需完成 = 原始234 - NO_COV跳过12 - 消融去重18 - 历史永久失败(422)
total_needed = TOTAL_RAW - NO_COV_SKIP_COUNT - DEDUP_SKIP_COUNT - perm_fail_count

# 统计已成功(去重)
all_results = list(records)
success_so_far = 0
seen_success = set()
for r in records:
    if r.get("success") == True:
        aa_val = str(r.get("auto_adapt", True)).strip() == "True"
        k = (str(r["scenario_id"]), str(r["model_id"]), int(r["input_length"]), aa_val)
        if k not in seen_success:
            seen_success.add(k)
            success_so_far += 1


print("=" * 90)
print("概念漂移与工况切换测试(429重试 / 422跳过 / NO_COV跳过 / 消融去重)")
print("=" * 90)
print(f"  原始任务总数: {TOTAL_RAW}")
print(f"  场景数: {len(scenarios)} (X=5, Y=4, Z=2)")
print(f"  模型数: {len(MODEL_LIST)} (支持协变量: {len(MODEL_LIST) - len(NO_COV_MODELS)})")
print(f"  输入长度: {INPUT_LENGTHS}")
print(f"  跳过: NO_COV={NO_COV_SKIP_COUNT}, 消融去重={DEDUP_SKIP_COUNT}, 422={perm_fail_count}")
print(f"  实际需完成: {total_needed}")
print(f"  已成功(去重): {success_so_far}")
print(f"  剩余: {total_needed - success_so_far}")
print("=" * 90)

if success_so_far >= total_needed:
    print("\n✅ 所有测试已完成, 直接输出分析结果.\n")
else:
    runned = 0
    stop_by_rate_limit = False

    print("\n[主测试] 场景 x 模型 x 输入长度")
    print("-" * 90)
    for scenario in scenarios:
        if stop_by_rate_limit: break
        for in_len in INPUT_LENGTHS:
            if stop_by_rate_limit: break
            for model_id in MODEL_LIST:
                # Z类场景跳过不支持协变量的模型
                if scenario["category"] == "Z" and model_id in NO_COV_MODELS:
                    continue
                key = (scenario["scenario_id"], model_id, in_len, True)
                if key in completed_keys:
                    continue

                r = run_forecast(scenario, model_id, in_len, auto_adapt=True)
                append_result(str(RESULT_CSV_PATH), r)
                all_results.append(r)
                runned += 1

                if r["success"]:
                    print(f"  [{r['scenario_id']}] {model_id:>14s} in={in_len:>3d} | "
                          f"MAE={r['mae']:.4f}  RMSE={r['rmse']:.4f}  耗时={r['latency_ms']:.0f}ms")
                else:
                    print(f"  [{r['scenario_id']}] {model_id:>14s} in={in_len:>3d} | 失败: {r['error'][:60]}")
                    if is_rate_limited(str(r.get("error", ""))):
                        # 429: 不加入completed, 不扣total, 停止本次运行
                        stop_by_rate_limit = True
                        print(f"     ↳ 限流失败(429), 停止本次运行.")
                        print(f"\n  ⚠️ 因限流停止.本次新增: {runned}.请获取API配额后再次运行.\n")
                        break
                    else:
                        # 422等永久失败: 加入completed, 扣total, 继续运行
                        completed_keys.add(key)
                        perm_fail_count += 1  # 累计永久失败数
                        total_needed -= 1
                        print(f"     ↳ 永久失败, 已跳过, 不再重试.")
                time.sleep(1)

    if not stop_by_rate_limit:
        print("\n[auto_adapt ablation] 场景=Y4")
        print("-" * 90)
        ablation_scenario = next(s for s in scenarios if s["scenario_id"] == AUTO_ADAPT_ABLATION_SCENARIO)
        for in_len in INPUT_LENGTHS:
            if stop_by_rate_limit: break
            for model_id in MODEL_LIST:
                if stop_by_rate_limit: break
                for aa in AUTO_ADAPT_VALUES:
                    key = (AUTO_ADAPT_ABLATION_SCENARIO, model_id, in_len, aa)
                    if key in completed_keys:
                        continue

                    r = run_forecast(ablation_scenario, model_id, in_len, auto_adapt=aa)
                    r["ablation"] = True
                    append_result(str(RESULT_CSV_PATH), r)
                    all_results.append(r)
                    runned += 1

                    aa_label = "adapt_on " if aa else "adapt_off"
                    if r["success"]:
                        print(f"  [Y4] {model_id:>14s} in={in_len:>3d} {aa_label} | MAE={r['mae']:.4f}")
                    else:
                        print(f"  [Y4] {model_id:>14s} in={in_len:>3d} {aa_label} | 失败: {r['error'][:60]}")
                        if is_rate_limited(str(r.get("error", ""))):
                            stop_by_rate_limit = True
                            print(f"     ↳ 限流失败(429), 停止本次运行.")
                            print(f"\n  ⚠️  因限流停止.本次新增: {runned}.请获取API配额后再次运行.\n")
                            break
                        else:
                            # 永久失败(如422), 加入 completed 避免下次重试
                            completed_keys.add(key)
                            perm_fail_count += 1  # 累计永久失败数
                            total_needed -= 1
                            print(f"     ↳ 永久失败, 已跳过, 不再重试.")
                    time.sleep(1)

    if not stop_by_rate_limit:
        print(f"\n  ✅ 本次运行全部完成！新增: {runned}\n")


# ============================================================
# 5. 核心分析
# ============================================================
print("=" * 90)
print("核心分析")
print("=" * 90)

# 去重: Y4/True 在主测试和消融各存了一份
seen_keys = set()
deduped_results = []
for r in all_results:
    if r.get("success") and r.get("mae") is not None:
        aa_val = str(r.get("auto_adapt", True)).strip() == "True"
        k = (str(r["scenario_id"]), str(r["model_id"]), int(r["input_length"]), aa_val)
        if k not in seen_keys:
            seen_keys.add(k)
            deduped_results.append(r)
success_results = deduped_results

print(f"  去重后成功记录: {len(success_results)} 条\n")

def get_mae(sid, mid, ilen, aa=True):
    for r in success_results:
        aa_val = str(r.get("auto_adapt", True)).strip() == "True"
        if (str(r["scenario_id"]) == sid and str(r["model_id"]) == mid and 
            int(r["input_length"]) == ilen and aa_val == aa):
            return r["mae"]
    return None

# 分析1: X类 vs Y类
print("\n【分析1】漂移可见性对比: X类(不可见) vs Y类(部分可见), in=256")
print("-" * 80)
pairs = [("X2","Y1","均值平移"), ("X3","Y2","方差扩张"), ("X4","Y3","相位偏移")]
for mid in MODEL_LIST:
    print(f"\n  [{mid}]")
    x1 = get_mae("X1", mid, 256)
    if x1: print(f"    基准(X1) MAE = {x1:.4f}")
    for xid, yid, desc in pairs:
        x = get_mae(xid, mid, 256)
        y = get_mae(yid, mid, 256)
        if x and y:
            imp = (x - y) / x * 100 if x > 0 else 0
            print(f"    {desc:8s}: 不可见={x:.4f}  部分可见={y:.4f}  改善={imp:+.1f}%")
        else:
            print(f"    {desc:8s}: 数据缺失")

# 分析2: Z类(协变量)
print("\n\n【分析2】协变量传递效果")
print("-" * 80)
for mid in MODEL_LIST:
    print(f"\n  [{mid}]")
    for il in INPUT_LENGTHS:
        x2 = get_mae("X2", mid, il)
        y1 = get_mae("Y1", mid, il)
        z1 = get_mae("Z1", mid, il)
        z2 = get_mae("Z2", mid, il)
        parts = []
        if x2 is not None: parts.append(f"不可见={x2:.4f}")
        if y1 is not None: parts.append(f"部分可见={y1:.4f}")
        if z1 is not None: parts.append(f"协变量={z1:.4f}")
        if z2 is not None: parts.append(f"协变量+可见={z2:.4f}")
        print(f"    in={il:>3d}: {'  '.join(parts) if parts else '数据缺失'}")

# 分析3: input_length 影响
print("\n\n【分析3】input_length 对漂移捕捉的影响(Y4)")
print("-" * 80)
for mid in MODEL_LIST:
    print(f"\n  [{mid}]")
    mbl = {}
    for il in INPUT_LENGTHS:
        m = get_mae("Y4", mid, il)
        if m:
            mbl[il] = m
            print(f"    in={il:>3d}: MAE={m:.4f}")
    if len(mbl) >= 2:
        sl = sorted(mbl.keys())
        ml = [mbl[k] for k in sl]
        if ml[-1] < ml[0] * 0.9: print(f"    -> 长窗口有收益")
        elif ml[-1] > ml[0] * 1.1: print(f"    -> 长上下文可能是负担")
        else: print(f"    -> 长短窗口接近")

# 分析4: auto_adapt ablation
print("\n\n【分析4】auto_adapt 开关对比(Y4)")
print("-" * 80)
for mid in MODEL_LIST:
    print(f"\n  [{mid}]")
    for il in INPUT_LENGTHS:
        on = get_mae("Y4", mid, il, True)
        off = get_mae("Y4", mid, il, False)
        if on and off:
            d = off - on
            p = d / off * 100 if off > 0 else 0
            print(f"    in={il:>3d}: on={on:.4f}  off={off:.4f}  差={d:+.4f} ({p:+.1f}%)")
        else:
            print(f"    in={il:>3d}: 数据缺失")

# 分析5: 复合漂移
print("\n\n【分析5】复合漂移: 不可见(X5) vs 部分可见(Y4), in=256")
print("-" * 80)
for mid in MODEL_LIST:
    x5 = get_mae("X5", mid, 256)
    y4 = get_mae("Y4", mid, 256)
    if x5 and y4:
        imp = (x5 - y4) / x5 * 100 if x5 > 0 else 0
        print(f"  {mid:>14s}: 不可见={x5:.4f}  部分可见={y4:.4f}  改善={imp:+.1f}%")
    else:
        print(f"  {mid:>14s}: 数据缺失")


# ============================================================
# 6. 保存结果与汇总
# ============================================================
remaining = total_needed - len(success_results)

print(f"\n{'=' * 90}")
print("最终汇总")
print("=" * 90)
print(f"  结果文件: {RESULT_CSV_PATH}")
print(f"  成功记录(去重): {len(success_results)} / 实际需完成: {total_needed}")
print("-" * 90)
print(f"  任务统计:")
print(f"    原始任务总数:           {TOTAL_RAW} 次")
print(f"    - 模型不支持协变量跳过:  {NO_COV_SKIP_COUNT} 次  (Timer-3.5/Timer-3.0 × Z场景)")
print(f"    - 消融去重(Y4/adapt=True): {DEDUP_SKIP_COUNT} 次  (与主测试重复)")
print(f"    - API永久失败(422等):    {perm_fail_count} 次")
print(f"    实际需完成:             {total_needed} 次")
print(f"    成功(去重):             {len(success_results)} 次")
if remaining > 0:
    print(f"    待完成(含429待重试):    {remaining} 次")
print("-" * 90)

if remaining == 0:
    print(f"✅ 全部完成！")
elif stop_by_rate_limit:
    print(f"⏳ 未全部完成, 请获取API额度后续继续跑.")
else:
    print(f"⏳ 未全部完成, 请检查其他错误.")
print("=" * 90)
