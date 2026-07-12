"""
timecho.py —— TimechoAI CRUD 封装
提供 Timecho 预测接口的通用调用层, 包括:
  - forecast(): 封装 API 调用、计时、异常处理
  - extract_pred_values(): 从 API 返回的结果 DataFrame 中提取预测值
  - calc_metrics(): 计算 MAE / RMSE / MAPE
  - calc_diff(): 计算两组预测值的平均绝对差异
"""

# ============================================================
import aiohttp

_original_aiohttp_request = aiohttp.ClientSession._request

async def _hooked_aiohttp_request(self, method, url, **kwargs):
    """拦截 aiohttp 请求, 捕获 429 的完整响应头"""
    resp = await _original_aiohttp_request(self, method, url, **kwargs)
    if resp.status == 429:
        print("\n" + "=" * 60)
        print("[429 拦截器] 捕获到 Too Many Requests (aiohttp)")
        print(f"  URL: {url}")
        print(f"  状态码: {resp.status}")
        print(f"  Retry-After: {resp.headers.get('Retry-After', '未返回')}")
        print(f"  X-RateLimit-Remaining: {resp.headers.get('X-RateLimit-Remaining', '未返回')}")
        print(f"  X-RateLimit-Reset: {resp.headers.get('X-RateLimit-Reset', '未返回')}")
        print(f"  全部响应头:")
        for k, v in resp.headers.items():
            print(f"    {k}: {v}")
        print("=" * 60 + "\n")
    return resp

aiohttp.ClientSession._request = _hooked_aiohttp_request
# ============================================================

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
import requests

from utils.client import get_timecho_client

# ============================================================
# 预测值提取
# ============================================================

def extract_pred_values(pred_df: pd.DataFrame) -> np.ndarray:
    """
    从预测结果 DataFrame 中提取数值列(排除 time 列). 

    Args:
        pred_df: API 返回的预测结果 DataFrame

    Returns:
        float 类型的 numpy 数组
    """
    if "target" in pred_df.columns:
        return pred_df["target"].values.astype(float)
    non_time_cols = [c for c in pred_df.columns if c != "time"]
    return pred_df[non_time_cols[0]].values.astype(float)


# ============================================================
# 预测调用(核心封装)
# ============================================================

def forecast(
    *,
    targets: pd.DataFrame,
    history_covs: pd.DataFrame | None = None,
    future_covs: pd.DataFrame | None = None,
    model_id: str = "Holt-Winters",
    output_length: int = 64,
    time_col: str = "time",
    auto_adapt: bool = True,
    api_key: str | None = None,
) -> tuple[np.ndarray | None, float, str | None]:
    """
    调用 TimechoAI 预测接口, 返回预测值、耗时、错误信息.

    Args:
        targets: 历史目标值 DataFrame(必须含 time 和 target 列)
        history_covs: 历史协变量 DataFrame(可选)
        future_covs: 未来协变量 DataFrame(可选, 传 None 表示不传协变量)
        model_id: 模型 ID
        output_length: 预测长度
        time_col: 时间列名
        auto_adapt: 是否自动适配
        api_key: API 密钥(可选, 默认使用全局配置)

    Returns:
        (pred_values, elapsed_ms, error_msg)
        - pred_values: 预测值数组(失败时为 None)
        - elapsed_ms: 耗时(毫秒)
        - error_msg: 错误信息(成功时为 None)
    """
    client = get_timecho_client(api_key)
    t0 = time.perf_counter()

    try:
        # 只在协变量不为 None 时才传入 API, 避免某些模型因 None 值报错
        api_kwargs: dict = {
            "targets": targets,
            "model_id": model_id,
            "output_length": output_length,
            "time_col": time_col,
            "auto_adapt": auto_adapt,
        }
        if history_covs is not None:
            api_kwargs["history_covs"] = history_covs
        if future_covs is not None:
            api_kwargs["future_covs"] = future_covs

        result = client.forecast(**api_kwargs)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        pred_values = extract_pred_values(result[0])
        return pred_values, elapsed_ms, None
    except Exception as e:
        if hasattr(e, 'response'):
            resp = e.response
            print(f"状态码: {resp.status_code}")
            print(f"Retry-After: {resp.headers.get('Retry-After', '未返回')}")
        else:
            print(f"异常类型: {type(e)}")
            # print(f"异常属性: {dir(e)}")
            print(f"异常信息: {e}")
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return None, elapsed_ms, str(e)


# ============================================================
# 精度指标
# ============================================================

def calc_metrics(
    pred: np.ndarray | None,
    truth: np.ndarray,
) -> dict[str, float | None]:
    """
    计算 MAE / RMSE / MAPE.

    Args:
        pred: 预测值
        truth: 真实值

    Returns:
        {"MAE": ..., "RMSE": ..., "MAPE": ...}
    """
    if pred is None:
        return {"MAE": None, "RMSE": None, "MAPE": None}
    mae = float(np.mean(np.abs(pred - truth)))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mape = float(np.mean(np.abs((pred - truth) / truth)) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def calc_diff(pred1: np.ndarray | None, pred2: np.ndarray | None) -> float | None:
    """
    计算两组预测值之间的平均绝对差异.

    Args:
        pred1: 第一组预测值
        pred2: 第二组预测值

    Returns:
        平均绝对差异, 任一为 None 时返回 None
    """
    if pred1 is None or pred2 is None:
        return None
    return float(np.mean(np.abs(pred1 - pred2)))
