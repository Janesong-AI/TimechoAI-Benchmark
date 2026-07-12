"""
resume.py —— 断点续跑

提供断点续跑、结果追加、限流判断等通用工具函数, 以便供 features/ 下各测试脚本调用. 

使用方式:
  from core.resume import load_completed_results, append_result, is_rate_limited

  records, perm_fail_count = load_completed_results("path/to/result.csv")
  # 自行构建 completed_keys: set of tuples
  completed_keys = set()
  for r in records:
      if r.get("success"):
          completed_keys.add((r["scenario_id"], r["model_id"], ...))

  append_result("path/to/result.csv", new_record)
"""

import pandas as pd

RATE_LIMIT_KEYWORDS = ["429", "limit", "quota", "exceed", "rate", "too many"]


def load_completed_results(result_csv_path_file: str) -> tuple[list[dict], int]:
    """
    读取历史结果 CSV 文件, 返回记录列表和永久失败数.

    本方法只做读取和基础分类, 不假设 key 结构.
    调用方应根据自己的列结构自行构建 completed_keys.

    Args:
        result_csv_path_file: 结果 CSV 文件路径(含文件名)

    Returns:
        (all_records, perm_fail_count)
        - all_records: list[dict], CSV 中每行转为 dict
        - perm_fail_count: int, 永久失败(非限流错误)的记录数
    """
    from pathlib import Path
    path = Path(result_csv_path_file)
    if not path.exists():
        print(f"  未找到 {path.name}, 将从头开始全量测试. ")
        return [], 0

    try:
        df = pd.read_csv(result_csv_path_file)
        records = df.to_dict("records")
        perm_fail_count = 0
        retry_count = 0

        for r in records:
            if r.get("success") == True:
                continue
            if is_rate_limited(str(r.get("error", ""))):
                retry_count += 1
            else:
                perm_fail_count += 1

        msg = f"  发现 {path.name}, 成功: {len(records) - perm_fail_count - retry_count}"
        if perm_fail_count > 0:
            msg += f", 永久失败(已跳过): {perm_fail_count}"
        if retry_count > 0:
            msg += f", 待重试(429): {retry_count}"
        print(msg)
        return records, perm_fail_count
    except Exception as e:
        print(f"  读取 {path.name} 失败: {e}, 将从头开始. ")
        return [], 0


def append_result(result_csv_path_file: str, result: dict) -> None:
    """
    追加单条结果到 CSV 文件.

    Args:
        result_csv_path_file: 结果 CSV 文件路径(含文件名)
        result: 单条结果字典, key 为列名
    """
    from pathlib import Path
    path = Path(result_csv_path_file)
    row_df = pd.DataFrame([result])
    if path.exists():
        row_df.to_csv(str(path), mode="a", header=False, index=False)
    else:
        row_df.to_csv(str(path), mode="w", header=True, index=False)


def is_rate_limited(error_str: str) -> bool:
    """
    判断错误字符串是否为限流错误(429 Too Many Requests). 

    Args:
        error_str: 错误信息字符串

    Returns:
        是否为限流错误
    """
    lower = error_str.lower()
    return any(k in lower for k in RATE_LIMIT_KEYWORDS)
