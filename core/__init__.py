"""
core —— 核心组件层

该包封装跨业务复用的核心通用组件, 承上启下: 向下调用 utils 层服务, 向上为 features 层提供标准接口. 目前包含:

模块:
  resume.py — 断点续跑机制.
    封装检查点状态管理与文件持久化逻辑, 支持长任务中断后的状态恢复.
  timecho.py — TimechoAI API 交互逻辑.
    封装 API 请求、响应处理及底层 utils.client 调用, 对外提供统一的高级 API.

使用约定:
  业务模块(如 features/) 应通过 core.timecho 间接访问 TimechoAI 服务，
  core 层是唯一直接使用 utils.client 的模块。

  导入路径示例:
    from core.timecho import forecast  # 推荐方式
    from core.resume import load_completed_results, append_result, is_rate_limited  # 推荐方式
    # 而非:
    # from utils.client import get_timecho_client  # 业务模块应避免
"""

from core.resume import (
    load_completed_results,
    append_result,
    is_rate_limited
)

from core.timecho import (
    forecast,
    extract_pred_values,
    calc_metrics,
    calc_diff
)

__all__ = [
    "load_completed_results",
    "append_result",
    "is_rate_limited",
    "forecast",
    "extract_pred_values",
    "calc_metrics",
    "calc_diff"
]

