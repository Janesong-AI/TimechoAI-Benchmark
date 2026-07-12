"""
文件操作工具模块

提供统一的CSV文件读写接口, 支持：
- 多种数据格式的保存
- 断点续跑数据追加
- 统一的错误处理

Author: Janesong
Create Date: 2026/07/12.
"""

import csv
import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd


class CSVFileError(Exception):
    """CSV文件操作异常"""
    pass


def save_to_csv(
    result_csv_path_file: Union[str, Path],
    data: Union[List[Dict[str, Any]], Dict[str, Any], pd.DataFrame],
    index: bool = False,
    mode: str = "w",
    encoding: str = "utf-8",
    **kwargs
) -> Path:
    """
    统一保存结果到CSV文件
    
    Args:
        result_csv_path_file: CSV文件路径(包含文件名), 必须参数
        data: 要保存的数据, 支持以下格式：
            - List[Dict]: 字典列表, 每个字典代表一行数据
            - Dict: 单个字典, 转换为单行数据
            - pd.DataFrame: 直接保存DataFrame
        index: 是否保存索引, 默认False
        mode: 写入模式, 'w'=覆盖写入, 'a'=追加写入, 默认'w'
        encoding: 文件编码, 默认'utf-8'
        **kwargs: 额外的pandas to_csv参数
        
    Returns:
        Path: 保存的文件路径对象
        
    Raises:
        CSVFileError: 当文件路径或数据无效时抛出
        
    Example:
        >>> # 保存字典列表
        >>> data = [
        ...     {"model": "model_a", "mae": 0.5, "rmse": 0.8},
        ...     {"model": "model_b", "mae": 0.6, "rmse": 0.9}
        ... ]
        >>> save_result_to_csv("./results/test.csv", data)
        
        >>> # 保存单个字典
        >>> result = {"model": "model_a", "mae": 0.5, "rmse": 0.8}
        >>> save_result_to_csv("./results/test.csv", result)
        
        >>> # 保存DataFrame
        >>> df = pd.DataFrame({"model": ["a", "b"], "mae": [0.5, 0.6]})
        >>> save_result_to_csv("./results/test.csv", df)
    """
    # 验证必须参数
    if not result_csv_path_file:
        raise CSVFileError("result_csv_path_file 参数不能为空, 必须提供包含路径的文件名")
    
    # 转换为Path对象
    file_path = Path(result_csv_path_file)
    
    # 验证文件扩展名
    if file_path.suffix.lower() != ".csv":
        raise CSVFileError(f"文件扩展名必须是.csv, 当前为: {file_path.suffix}")
    
    # 创建父目录(如果不存在)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 数据格式转换
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        # 单个字典转换为单行DataFrame
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if len(data) == 0:
            # 空列表, 创建空DataFrame
            df = pd.DataFrame()
        else:
            # 字典列表转换为DataFrame
            df = pd.DataFrame(data)
    else:
        raise CSVFileError(f"不支持的数据类型: {type(data).__name__}, 支持: List[Dict], Dict, DataFrame")
    
    # 处理追加模式
    if mode == "a" and file_path.exists():
        # 追加模式：不写header
        header = False
    else:
        # 写入模式：写header
        header = True
    
    # 保存CSV
    try:
        df.to_csv(
            file_path,
            index=index,
            mode=mode,
            encoding=encoding,
            header=header,
            **kwargs
        )
    except Exception as e:
        raise CSVFileError(f"保存CSV文件失败: {file_path}\n错误: {e}")
    
    return file_path


def append_to_csv(
    result_csv_path_file: Union[str, Path],
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    encoding: str = "utf-8"
) -> Path:
    """
    追加结果到CSV文件(断点续跑场景)
    
    如果文件不存在, 会创建新文件;如果存在, 则追加数据(不写header)
    
    Args:
        result_csv_path_file: CSV文件路径(包含文件名), 必须参数
        data: 要追加的数据
            - Dict: 单行数据
            - List[Dict]: 多行数据
        encoding: 文件编码, 默认'utf-8'
        
    Returns:
        Path: 追加的文件路径对象
        
    Example:
        >>> result = {"model": "model_a", "mae": 0.5}
        >>> append_result_to_csv("./results/test.csv", result)
    """
    return save_to_csv(
        result_csv_path_file=result_csv_path_file,
        data=data,
        mode="a",
        encoding=encoding
    )


def save_with_json_backup(
    result_csv_path_file: Union[str, Path],
    data: Union[List[Dict[str, Any]], pd.DataFrame],
    save_json: bool = True,
    index: bool = False,
    encoding: str = "utf-8",
    **kwargs
) -> tuple[Path, Optional[Path]]:
    """
    保存结果到CSV, 并可选保存JSON备份
    
    Args:
        result_csv_path_file: CSV文件路径(包含文件名), 必须参数
        data: 要保存的数据
        save_json: 是否同时保存JSON格式, 默认True
        index: 是否保存索引, 默认False
        encoding: 文件编码, 默认'utf-8'
        **kwargs: 额外的参数
        
    Returns:
        tuple[Path, Optional[Path]]: (CSV路径, JSON路径或None)
        
    Example:
        >>> data = [{"model": "a", "mae": 0.5}]
        >>> csv_path, json_path = save_result_with_json_backup("./results/test.csv", data)
    """
    # 保存CSV
    csv_path = save_to_csv(
        result_csv_path_file=result_csv_path_file,
        data=data,
        index=index,
        encoding=encoding,
        **kwargs
    )
    
    json_path = None
    if save_json:
        # 生成JSON文件路径
        csv_path_obj = Path(result_csv_path_file)
        json_path = csv_path_obj.with_suffix(".json")
        
        # 转换数据格式
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient="records")
        else:
            json_data = data
        
        # 保存JSON
        with open(json_path, "w", encoding=encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    return csv_path, json_path


def read_csv_to_dataframe(
    result_csv_path_file: Union[str, Path],
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """
    读取CSV文件为DataFrame
    
    Args:
        result_csv_path_file: CSV文件路径(包含文件名), 必须参数
        encoding: 文件编码, 默认'utf-8'
        **kwargs: 额外的pandas read_csv参数
        
    Returns:
        pd.DataFrame: 读取的数据
        
    Raises:
        CSVFileError: 文件不存在或读取失败时抛出
        
    Example:
        >>> df = read_csv_to_dataframe("./results/test.csv")
    """
    if not result_csv_path_file:
        raise CSVFileError("result_csv_path_file 参数不能为空")
    
    file_path = Path(result_csv_path_file)
    
    if not file_path.exists():
        raise CSVFileError(f"文件不存在: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        return df
    except Exception as e:
        raise CSVFileError(f"读取CSV文件失败: {file_path}\n错误: {e}")


def read_csv_to_list(
    result_csv_path_file: Union[str, Path],
    encoding: str = "utf-8"
) -> List[Dict[str, Any]]:
    """
    读取CSV文件为字典列表
    
    Args:
        result_csv_path_file: CSV文件路径(包含文件名), 必须参数
        encoding: 文件编码, 默认'utf-8'
        
    Returns:
        List[Dict[str, Any]]: 字典列表, 每个字典代表一行数据
        
    Example:
        >>> data = read_csv_to_list("./results/test.csv")
        >>> # 返回: [{"model": "a", "mae": 0.5}, {"model": "b", "mae": 0.6}]
    """
    df = read_csv_to_dataframe(result_csv_path_file, encoding=encoding)
    return df.to_dict(orient="records")


def csv_exists_and_not_empty(result_csv_path_file: Union[str, Path]) -> bool:
    """
    检查CSV文件是否存在且不为空
    
    Args:
        result_csv_path_file: CSV文件路径
        
    Returns:
        bool: True=文件存在且有数据, False=文件不存在或为空
        
    Example:
        >>> if csv_exists_and_not_empty("./results/test.csv"):
        ...     df = read_csv_to_dataframe("./results/test.csv")
    """
    if not result_csv_path_file:
        return False
    
    file_path = Path(result_csv_path_file)
    
    if not file_path.exists():
        return False
    
    # 检查文件是否为空(只有header也算非空)
    try:
        df = pd.read_csv(file_path)
        return len(df) > 0
    except:
        return False

