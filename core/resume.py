#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
core/resume.py —— Resume from Checkpoint

Function: Provides common utility functions for checkpoint resumption, result appending, and rate limit detection, for use by business test scripts.

Usage:
  from core.resume import load_completed_results, append_result, is_rate_limited

  records, perm_fail_count = load_completed_results("path/to/result.csv")
  # Build completed_keys yourself: set of tuples
  completed_keys = set()
  for r in records:
      if r.get("success"):
          completed_keys.add((r["scenario_id"], r["model_id"], ...))

  append_result("path/to/result.csv", new_record)

Author: Janesong
Create Date: 2026/07/10.
"""

import pandas as pd

RATE_LIMIT_KEYWORDS = ["429", "limit", "quota", "exceed", "rate", "too many"]


def load_completed_results(result_csv_path_file: str) -> tuple[list[dict], int]:
    """
    Read historical result CSV file and return record list and permanent failure count.

    This method only performs reading and basic classification, does not assume key structure.
    The caller should build completed_keys based on their own column structure.

    Args:
        result_csv_path_file: Result CSV file path (including filename)

    Returns:
        (all_records, perm_fail_count)
        - all_records: list[dict], each row in CSV converted to dict
        - perm_fail_count: int, number of permanent failures (non-rate-limit errors)
    """
    from pathlib import Path
    path = Path(result_csv_path_file)
    if not path.exists():
        print(f"  {path.name} not found, will start full testing from scratch. ")
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

        msg = f"  Found {path.name}, Success: {len(records) - perm_fail_count - retry_count}"
        if perm_fail_count > 0:
            msg += f", Permanent Failures (Skipped): {perm_fail_count}"
        if retry_count > 0:
            msg += f", Pending Retry (429): {retry_count}"
        print(msg)
        return records, perm_fail_count
    except Exception as e:
        print(f"  Failed to read {path.name}: {e}, will start from scratch. ")
        return [], 0


def append_result(result_csv_path_file: str, result: dict) -> None:
    """
    Append single result to CSV file.

    Args:
        result_csv_path_file: Result CSV file path (including filename)
        result: Single result dictionary, keys are column names
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
    Determine if error string is a rate limit error (429 Too Many Requests).

    Args:
        error_str: Error message string

    Returns:
        Whether it is a rate limit error
    """
    lower = error_str.lower()
    return any(k in lower for k in RATE_LIMIT_KEYWORDS)
