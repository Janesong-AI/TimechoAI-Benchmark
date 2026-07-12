"""
client.py —— TimechoAIClient 工厂函数

所属层级: utils(工具层)
作用: 统一管理 TimechoAIClient 实例的创建, 屏蔽 API_KEY 的获取逻辑.

设计目标: 所有业务模块(core、features)都应通过此工厂获取客户端, 以便:
          1. API_KEY 只从 config.settings 读取, 一处修改全局生效
          2. 未来可在此处注入 mock 客户端, 方便测试
          3. SDK 构造函数签名变化时只需修改此文件
"""

from timecho_ai import TimechoAIClient
from config.settings import API_KEY


def get_timecho_client(api_key: str | None = None) -> TimechoAIClient:
    """
    获取 TimechoAIClient 实例.

    Args:
        api_key: API 密钥字符串. 传 None(默认)时自动从 config.settings.API_KEY 读取.
                 优先级: 显式传入 > 环境变量 TIMECHO_API_KEY > 默认值

    Returns:
        已初始化的 TimechoAIClient 实例

    调用链路:
        features/ 测试脚本
          → core.timecho.forecast()
            → utils.client.get_timecho_client()
              → timecho_ai.TimechoAIClient(api_key=...)
    """
    return TimechoAIClient(api_key=api_key or API_KEY)
