"""
utils/client.py —— TimechoAI客户端工厂

所属层级: utils(工具层)
作用: 统一管理 TimechoAIClient 实例的创建, 屏蔽 API_KEY 的获取逻辑.

设计目标: 所有业务模块(core、features)都应通过此工厂获取客户端, 以便:
          1. API_KEY 只从 config.settings 读取, 一处修改全局生效
          2. 未来可在此处注入 mock 客户端, 方便测试
          3. SDK 构造函数签名变化时只需修改此文件
"""

from timecho_ai import TimechoAIClient
from config.settings import API_KEY

_client: TimechoAIClient | None = None

def get_timecho_client(api_key: str | None = None) -> TimechoAIClient:
    """
    获取 TimechoAIClient 单例实例
    
    首次调用时会自动预热连接,避免首次API调用失败.
    
    Args:
        api_key: API 密钥字符串. 传 None(默认)时自动从 config.settings.API_KEY 读取.
                 优先级: 显式传入 > 环境变量 TIMECHO_API_KEY > 默认值

    Returns:
        TimechoAIClient 实例

    调用链路:
        features/ 测试脚本
          → core.timecho.forecast()
            → utils.client.get_timecho_client()
              → timecho_ai.TimechoAIClient(api_key=...)
    """
    global _client

    if _client is None:
        _client = TimechoAIClient(api_key=api_key or API_KEY)

        # 预热:触发底层初始化,避免首次调用DNS解析失败
        _warmup_client(_client)

    return _client


def _warmup_client(client: TimechoAIClient) -> None:
    """
    预热客户端连接
    
    Args:
        client: TimechoAIClient 实例
    """
    try:
        # 直接触发Session创建，不调用任何API
        client._get_session()
    except Exception as e:
        # 预热失败不阻塞流程,打印警告即可
        print(f"⚠️ SDK预热失败(可忽略): {type(e).__name__}")
        # 不抛出异常,允许后续重试


def reset_client() -> None:
    """
    重置客户端实例
    
    用于测试场景,强制下次调用重新创建客户端.
    """
    global _client
    if _client is not None:
        try:
            _client.close()
        except:
            pass
    _client = None
