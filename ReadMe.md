## 1. 核心架构 - 分层架构
- 本项目基于 Python 3.12, 核心依赖 `timecho-ai` 和 `pandas`, 是专门用于测试 TimechoAI【时序数据库大模型】.

## 2. 目录与文件规范
- `config/`: 全局配置管理模块.
- `core/`: 核心通用组件层(跨业务复用). 
- `features/`: 业务特性实现层, 存放具体业务场景逻辑.
- `utils/`: 基础工具库, 存放无状态纯函数及通用实体封装.
- `README.md`: 项目说明文档, 提供项目概述、使用方法、注意事项等.

## 3. 命令与安装
- 安装:   
   `python -m pip install timecho-ai pandas`  
- 虚拟环境:   
   `python -m venv .venv`         # 建立虚拟环境  
   `source .venv/bin/activate`    # 激活虚拟环境  
   `deactivate`                   # 退出虚拟环境  
