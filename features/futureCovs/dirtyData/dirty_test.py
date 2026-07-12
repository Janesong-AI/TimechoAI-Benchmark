# dirty_test.py —— 脏数据鲁棒性测试
# 测试目的: 验证模型对缺失值、异常尖峰的抵抗力
# 原理: 用 7 种脏数据(含 baseline)分别预测, 对比精度退化程度

import time
import json
from pathlib import Path

import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parents[3]))

from config.settings import DEFAULT_INPUT_LENGTH, DEFAULT_OUTPUT_LENGTH
from core.timecho import forecast, calc_metrics

# ============================================================
# 配置区
# ============================================================
SCRIPT_DIR = Path(__file__).parent

# 测试模型（选支持协变量的，Timer 系列会 422）
MODELS_TO_TEST = ["Timer-3.5", "Chronos-2"]

# 7 个测试场景
SCENES = [
    ("S0-干净",       "dirty_s0_clean.csv"),
    ("S1-缺失5%",     "dirty_s1_miss5.csv"),
    ("S2-缺失15%",    "dirty_s2_miss15.csv"),
    ("S3-连续缺失",   "dirty_s3_miss_block.csv"),
    ("S4-单点尖峰",   "dirty_s4_spike_single.csv"),
    ("S5-多点尖峰",   "dirty_s5_spike_multi.csv"),
    ("S6-混合脏",     "dirty_s6_mixed.csv"),
]

# ============================================================
# 读取 ground truth(用干净数据的后 64 行作为真实值)
# ============================================================
print("📦 准备 ground truth...")
clean_df = pd.read_csv(SCRIPT_DIR / "dirty_clean.csv")
clean_df["time"] = pd.to_datetime(clean_df["time"])
ground_truth = clean_df.iloc[DEFAULT_INPUT_LENGTH:]["target"].values
future_cov = clean_df.iloc[DEFAULT_INPUT_LENGTH:][["time", "cov"]].copy()
print(f"   ground_truth: {len(ground_truth)} 个点")
print(f"   ground_truth 范围: {ground_truth.min():.2f} ~ {ground_truth.max():.2f}")
print()

# ============================================================
# 逐场景 × 逐模型 测试
# ============================================================
total_calls = len(MODELS_TO_TEST) * len(SCENES) * 2
print(f"🚀 开始测试：{len(MODELS_TO_TEST)} 个模型 × {len(SCENES)} 个场景 × 2轮 = {total_calls} 次 API 调用")
print("=" * 90)

all_results = []

for model_id in MODELS_TO_TEST:
    # 判断是否为单变量模型
    is_univariate = model_id.startswith("Timer")
    print(f"\n{'─' * 90}")
    print(f"📋 模型: {model_id} (单变量模式: {is_univariate})")
    print(f"{'─' * 90}")

    for scene_name, csv_file in SCENES:
        print(f"\n  🔍 场景: {scene_name} ({csv_file})")

        # 读取脏数据
        df = pd.read_csv(SCRIPT_DIR / csv_file)
        df["time"] = pd.to_datetime(df["time"])

        history = df.iloc[:DEFAULT_INPUT_LENGTH].copy()
        # 检查脏数据概况
        nan_count = history["target"].isna().sum()
        valid_vals = history["target"].dropna()
        data_range = f"{valid_vals.min():.1f}~{valid_vals.max():.1f}" if len(valid_vals) > 0 else "全NaN"
        print(f"     历史 target: NaN={nan_count}, 范围={data_range}")

        # ===== 两轮测试: 原始 + 预处理后 =====
        for pass_name, pass_df in [("原始", history.copy()), ("预处理", history.copy())]:
            if pass_name == "预处理":
                # 前向填充 NaN, 如果前面也 NaN 就后向填充
                pass_df["target"] = pass_df["target"].ffill().bfill()
                # 不动尖峰, 让尖峰原样传入
    
            history_targets = pass_df[["time", "target"]]
            history_covs = pass_df[["time", "cov"]]
    
            label = f"{scene_name}[{pass_name}]"
            try:
                # 动态构建参数: 如果是 Timer 系列, 不传协变量
                forecast_kwargs = {
                    "targets": history_targets,
                    "model_id": model_id,
                    "output_length": DEFAULT_OUTPUT_LENGTH,
                    "time_col": "time",
                    "auto_adapt": True,
                }
                if not is_univariate:
                    forecast_kwargs["history_covs"] = history_covs
                    forecast_kwargs["future_covs"] = future_cov

                pred_values, elapsed_ms, error = forecast(**forecast_kwargs)

                if error:
                    error_msg = error
                    print(f"     [{pass_name}] ❌ 失败: {error_msg[:100]}")
                    all_results.append({
                        "model_id": model_id, "scene": label, "csv_file": csv_file,
                        "pass": pass_name, "success": False, "mae": None, "rmse": None, "mape": None,
                        "latency_ms": elapsed_ms, "pred_min": None, "pred_max": None,
                        "truth_min": float(np.min(ground_truth)), "truth_max": float(np.max(ground_truth)),
                        "is_explosion": None, "nan_count": nan_count, "error": error_msg
                    })
                else:
                    # 使用 core/timecho.py 的 calc_metrics 计算精度指标
                    metrics = calc_metrics(pred_values, ground_truth)
                    mae = metrics["MAE"]
                    rmse = metrics["RMSE"]
                    mape = metrics["MAPE"]

                    # 检测预测是否"起飞"或"雪崩"
                    pred_max = float(np.max(pred_values))
                    pred_min = float(np.min(pred_values))
                    truth_max = float(np.max(ground_truth))
                    truth_min = float(np.min(ground_truth))
                    is_explosion = pred_max > truth_max * 1.5 or pred_min < truth_min * 0.5

                    status = "💥 起飞/雪崩!" if is_explosion else "✅ 正常"
                    print(f"     [{pass_name}] {status} MAE={mae:.4f}, 范围={pred_min:.2f}~{pred_max:.2f}")

                    all_results.append({
                        "model_id": model_id, "scene": label, "csv_file": csv_file,
                        "pass": pass_name, "success": True, "mae": mae, "rmse": rmse, "mape": mape,
                        "latency_ms": elapsed_ms, "pred_min": pred_min, "pred_max": pred_max,
                        "truth_min": truth_min, "truth_max": truth_max, "is_explosion": is_explosion,
                        "nan_count": nan_count, "error": None
                    })
            except Exception as e:
                error_msg = str(e)
                print(f"     [{pass_name}] ❌ 失败: {error_msg[:100]}")
                all_results.append({
                    "model_id": model_id, "scene": label, "csv_file": csv_file,
                    "pass": pass_name, "success": False, "mae": None, "rmse": None, "mape": None,
                    "latency_ms": 0, "pred_min": None, "pred_max": None,
                    "truth_min": float(np.min(ground_truth)), "truth_max": float(np.max(ground_truth)),
                    "is_explosion": None, "nan_count": nan_count, "error": error_msg
                })

            time.sleep(1)  # 礼貌等待

# ============================================================
# 汇总表格
# ============================================================
print("\n\n" + "=" * 100)
print("📋 鲁棒性分析结论")
print("=" * 100)

def get_result(model_id, scene_prefix, pass_name="预处理"):
    """辅助函数: 精确获取某个模型、某个场景、某一轮的结果"""
    for r in all_results:
        if r["model_id"] == model_id and r["scene"].startswith(scene_prefix) and r["pass"] == pass_name:
            return r
    return None

# ============================================================
# 自动结论分析
# ============================================================
print("\n\n" + "=" * 100)
print("📋 鲁棒性分析结论")
print("=" * 100)

for model_id in MODELS_TO_TEST:
    model_results = [r for r in all_results if r["model_id"] == model_id]
    baseline = model_results[0]  # S0

    print(f"\n  【{model_id}】")
    s0_pre = get_result(model_id, "S0", "预处理")
    
    if not s0_pre or not s0_pre["success"]:
        print(f"    ⚠️ 基准场景都失败了, 无法评估鲁棒性.")
        continue

    baseline_mae = s0_pre["mae"]
    print(f"    基准(S0干净[预处理]) MAE = {baseline_mae:.4f}")

    # 分析缺失值场景
    print(f"\n    ▶ 缺失值抵抗力 (基于[预处理]结果):")
    for scene_prefix in ["S1", "S2", "S3"]:
        r_pre = get_result(model_id, scene_prefix, "预处理")
        r_raw = get_result(model_id, scene_prefix, "原始")
        if r_raw and not r_raw["success"]:
            print(f"      {r_raw['scene'].replace('[原始]',''):>14s} (原始): ❌ 报错 (不支持 NaN)")
        if r_pre and r_pre["success"]:
            ratio = r_pre["mae"] / baseline_mae if baseline_mae > 0 else float("inf")
            verdict = "✅ 无影响" if ratio < 1.5 else ("🟡 轻微退化" if ratio < 3 else "🔴 明显退化")
            print(f"      {r_pre['scene'].replace('[预处理]',''):>14s} (预处理): MAE={r_pre['mae']:.4f} (基准的 {ratio:.1f}x) → {verdict}")

    # 分析尖峰场景
    print(f"\n    ▶ 异常尖峰抵抗力 (基于[原始]结果):")
    for scene_prefix in ["S4", "S5"]:
        r_raw = get_result(model_id, scene_prefix, "原始")
        if r_raw and r_raw["success"]:
            ratio = r_raw["mae"] / baseline_mae if baseline_mae > 0 else float("inf")
            verdict = "💥 起飞/雪崩!" if r_raw["is_explosion"] else ("✅ 扛住了" if ratio < 1.5 else "⚠️ 精度退化")
            print(f"      {r_raw['scene'].replace('[原始]',''):>14s}: MAE={r_raw['mae']:.4f} (基准的 {ratio:.1f}x) → {verdict}")

    # 分析混合场景
    print(f"\n    ▶ 混合脏数据 (基于[预处理]结果):")
    r_pre = get_result(model_id, "S6", "预处理")
    if r_pre and r_pre["success"]:
        ratio = r_pre["mae"] / baseline_mae if baseline_mae > 0 else float("inf")
        verdict = "💥 起飞/雪崩!" if r_pre["is_explosion"] else ("✅ 工业可用" if ratio < 1.5 else "🔴 不可用")
        print(f"      S6-混合脏: MAE={r_pre['mae']:.4f} (基准的 {ratio:.1f}x) → {verdict}")

# 保存结果
summary_df = pd.DataFrame([{k: v for k, v in r.items()} for r in all_results])
summary_df.to_csv(SCRIPT_DIR / "dirty_test_result.csv", index=False)
with open(SCRIPT_DIR / "dirty_test_result.json", "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

print("\n" + "=" * 100)
print("测试完成！")
print("=" * 100)
