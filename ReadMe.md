## 1. 核心架构 - 分层架构
- 本项目基于 Python 3.12, 核心依赖 `timecho-ai` 和 `pandas`, 是专门用于测试 TimechoAI【时序数据库大模型】.

## 2. 目录与文件规范
- `config/`: 全局配置管理模块.
   - `setting.xml`: 集中管理环境变量(如 `TIMECHO_API_KEY`)及全局默认参数.
- `core/`: 核心通用组件层(跨业务复用). 
   - `resume.py`: 封装断点续跑机制, 管理检查点状态与文件持久化.
   - `timecho.py`: 封装 TimechoAI API 交互逻辑.
- `features/`: 业务特性实现层, 存放具体业务场景逻辑.
- `utils/`: 基础工具库, 存放无状态纯函数及通用实体封装.
   - `client.py`: 封装底层客户端连接实体.
   - `file_utils.py`: 文件操作工具.
- `README.md`: 项目说明文档, 提供项目概述、使用方法、注意事项等.

## 3. 测试流程
1. 初始化配置: 从 `config/setting.xml` 中读取环境变量和默认参数.

## 4. 命令与安装
- 安装:   
   `python -m pip install timecho-ai pandas`  
- 虚拟环境:   
   `python -m venv .venv`         # 建立虚拟环境  
   `source .venv/bin/activate`    # 激活虚拟环境  
   `deactivate`                   # 退出虚拟环境  
- 运行:   
   `python ./features/futureCovs/dirtyData/dirty_test.py`             # 脏数据鲁棒性  
