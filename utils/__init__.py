"""
utils —— 工具层

该包提供与 TimechoAI SDK 相关的通用工具函数, 目前包含:

模块:
  client.py — TimechoAIClient 工厂函数 get_timecho_client()
    负责统一创建和管理 TimechoAIClient 实例,
    所有业务模块(core、features)均通过此工厂获取客户端,
    避免各模块重复处理 API_KEY 和初始化逻辑.

使用约定:
  业务模块(如 features/futureCovs/) 应通过 core.timecho 间接访问. core 层是唯一直接使用 utils.client 的模块.

  导入路径示例:
    from core.timecho import forecast  # 推荐方式
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
