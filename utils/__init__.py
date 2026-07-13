"""
utils —— 基础工具层

该包提供无状态纯函数及通用实体封装, 作为整个项目的底层支撑. 目前包含:

模块:
  client.py — TimechoAI 客户端连接实体.
    提供 get_timecho_client() 工厂函数, 统一创建和管理 TimechoAIClient 实例生命周期,
    避免各模块重复处理 API_KEY 和初始化逻辑.
  file_utils.py — 文件操作工具.
    提供 CSV/JSON 等文件的读写、追加及状态检查功能.

使用约定:
  业务模块(如 features/) 应通过 core.timecho 间接访问 TimechoAI 服务，
  core 层是唯一直接使用 utils.client 的模块。
  file_utils 作为通用工具，任何模块均可直接使用。

  导入路径示例:
    from core.timecho import forecast  # 推荐方式
    # 而非:
    # from utils.client import get_timecho_client  # 业务模块应避免
"""

from .file_utils import (
    save_to_csv,
    append_to_csv,
    save_with_json_backup,
    read_csv_to_dataframe,
    read_csv_to_list,
    csv_exists_and_not_empty,
    CSVFileError
)

__all__ = [
    "save_to_csv",
    "append_to_csv",
    "save_with_json_backup",
    "read_csv_to_dataframe",
    "read_csv_to_list",
    "csv_exists_and_not_empty",
    "CSVFileError",
    "get_timecho_client",
]
