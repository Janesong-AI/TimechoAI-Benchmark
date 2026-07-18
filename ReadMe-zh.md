# TSFM Robustness Benchmark

[English](./README.md) | [中文](./ReadMe-zh.md)

TSFM 鲁棒性基准测试是一种系统化的测试工具, 旨在检验时间序列基础模型在边缘场景(如频率不匹配、数据污染、协变量干扰等)下的工程鲁棒性.
本次版本包含对TimechoAI作为首个靶向模型的系统性评估, 更多模型将在后续迭代中逐步整合.

## 1. 核心架构 - 分层架构
- 本项目基于 Python 3.12, 核心依赖 `timecho-ai` 和 `pandas`.

## 2. 目录与文件规范
- `config/`: 全局配置管理模块.
   - `settings.py`: 全局环境变量配置(如 `TIMECHO_API_KEY`)等.
   - `constants.py`: 全局常量定义.
- `core/`: 核心通用组件层(跨业务复用). 
   - `resume.py`: 封装断点续跑机制, 管理检查点状态与文件持久化.
   - `timecho.py`: 封装 TimechoAI API 交互逻辑.
- `features/`: 业务特性实现层, 存放具体业务场景逻辑.
- `utils/`: 基础工具库, 存放无状态纯函数及通用实体封装.
   - `client.py`: 封装底层客户端连接实体.
   - `files.py`: 文件操作工具.
- `run.py`: **项目统一入口**, 负责引导 `sys.path` 并按模块名或文件路径启动指定测试脚本.
- `README.md`: 项目说明文档, 提供项目概述、使用方法、注意事项等.

## 3. 测试流程
1. 配置初始化：从 `config/settings.py` 中读取环境变量.
2. 模型初始化：使用提供的API密钥对TimechoAI模型进行初始化.
3. 测试执行：根据提供的命令行参数执行指定的测试流程.
4. 结果输出：将测试结果输出至控制台或指定文件.

## 4. 命令与安装
- 虚拟环境:   
   `python -m venv .venv`         # 建立虚拟环境  
   `source .venv/bin/activate`    # 激活虚拟环境  
   `python -m pip install timecho-ai pandas`  # 安装依赖  
   `deactivate`                   # 退出虚拟环境  
- 运行:
   `python ./features/futureCovs/conceptDrift/concept_drift_test_v1.py`  # 概念漂移与工况切换测试  
   `python ./features/futureCovs/conceptDrift/concept_drift_test_v2.py`  # 概念漂移与工况切换测试(XYZ场景)  
   `python run.py ./features/futureCovs/covariant/cov_test.py`           # 协变量有效性  
   `python ./features/futureCovs/dirtyData/dirty_test.py`                # 脏数据鲁棒性  
   `python ./features/futureCovs/forecastHorizon/forecast_horizon_ablation.py` # C3 预测步长消融实验  
   `python ./features/futureCovs/freqMismatch/frequency_mismatch_test.py`  # C5 频率失配鲁棒性  
   `python ./features/futureCovs/inputLength/input_length_test.py`    # input_length消融测试  
   `python ./features/futureCovs/irregularSampling/irregular_sampling_test.py`  # 非规则采样鲁棒性  

## 5. 测试目标
- 边缘场景探测: 针对复杂查询、多副本不一致、时间序列乱序写入等边界条件, 系统性验证模型的工程鲁棒性.
- 防御性架构验证: 以严格的工程标准施压, 检验模型在非理想输入下的退化行为与恢复能力.

## 6. 测试范围声明
本框架的测试结果受限于模型特定版本、数据预处理策略及运行环境. 本工具旨在为时序模型的工程防御性架构设计提供客观参考视角, 而非对任何商业产品最终性能的绝对断言.


