"""
utils/files.py -- File operation utility module

Provides unified CSV file read/write interface, supporting:
  - Multiple data format saving
  - Resume from breakpoint data appending
  - Unified error handling

Author: Janesong
Create Date: 2026/07/12.
"""

import csv
import json
from pathlib import Path
from typing import Union, List, Dict, Any, Optional
import pandas as pd


class CSVFileError(Exception):
    """CSV file operation exception"""
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
    Unified save results to CSV file

    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        data: Data to save, supports the following formats:
            - List[Dict]: List of dictionaries, each dictionary represents one row of data
            - Dict: Single dictionary, converted to single row of data
            - pd.DataFrame: Directly save DataFrame
        index: Whether to save index, default False
        mode: Write mode, 'w'=overwrite write, 'a'=append write, default 'w'
        encoding: File encoding, default 'utf-8'
        **kwargs: Additional pandas to_csv parameters

    Returns:
        Path: Saved file path object
        
    Raises:
        CSVFileError: Raised when file path or data is invalid
        
    Example:
        >>> # Save list of dictionaries
        >>> data = [
        ...     {"model": "model_a", "mae": 0.5, "rmse": 0.8},
        ...     {"model": "model_b", "mae": 0.6, "rmse": 0.9}
        ... ]
        >>> save_result_to_csv("./results/test.csv", data)
        
        >>> # Save single dictionary
        >>> result = {"model": "model_a", "mae": 0.5, "rmse": 0.8}
        >>> save_result_to_csv("./results/test.csv", result)
        
        >>> # Save DataFrame
        >>> df = pd.DataFrame({"model": ["a", "b"], "mae": [0.5, 0.6]})
        >>> save_result_to_csv("./results/test.csv", df)
    """
    # Validate required parameters
    if not result_csv_path_file:
        raise CSVFileError("result_csv_path_file parameter cannot be empty; must provide a filename including path")

    # Convert to Path object
    file_path = Path(result_csv_path_file)
    
    # Validate file extension
    if file_path.suffix.lower() != ".csv":
        raise CSVFileError(f"File extension must be .csv; current is: {file_path.suffix}")
    
    # Create parent directory (if not exists)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Data format conversion
    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        # Single dictionary converted to single-row DataFrame
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        if len(data) == 0:
            # Empty list, create empty DataFrame
            df = pd.DataFrame()
        else:
            # List of dictionaries converted to DataFrame
            df = pd.DataFrame(data)
    else:
        raise CSVFileError(f"Unsupported data type: {type(data).__name__}, supported: List[Dict], Dict, DataFrame")
    
    # Handle append mode
    if mode == "a" and file_path.exists():
        # Append mode: do not write header
        header = False
    else:
        # Write mode: write header
        header = True
    
    # Save CSV
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
        raise CSVFileError(f"Failed to save CSV file: {file_path}\nError: {e}")
    
    return file_path


def append_to_csv(
    result_csv_path_file: Union[str, Path],
    data: Union[Dict[str, Any], List[Dict[str, Any]]],
    encoding: str = "utf-8"
) -> Path:
    """
    Append results to CSV file (resume from breakpoint scenario)

    If file does not exist, will create new file; if exists, will append data (without writing header)
    
    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        data: Data to append
            - Dict: Single row of data
            - List[Dict]: Multiple rows of data
        encoding: File encoding, default 'utf-8'
        
    Returns:
        Path: Appended file path object
        
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
    Save results to CSV, with optional JSON backup
    
    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        data: Data to save
        save_json: Whether to also save JSON format, default True
        index: Whether to save index, default False
        encoding: File encoding, default 'utf-8'
        **kwargs: Additional parameters
        
    Returns:
        tuple[Path, Optional[Path]]: (CSV path, JSON path or None)
        
    Example:
        >>> data = [{"model": "a", "mae": 0.5}]
        >>> csv_path, json_path = save_result_with_json_backup("./results/test.csv", data)
    """
    # Save CSV
    csv_path = save_to_csv(
        result_csv_path_file=result_csv_path_file,
        data=data,
        index=index,
        encoding=encoding,
        **kwargs
    )
    
    json_path = None
    if save_json:
        # Generate JSON file path
        csv_path_obj = Path(result_csv_path_file)
        json_path = csv_path_obj.with_suffix(".json")
        
        # Convert data format
        if isinstance(data, pd.DataFrame):
            json_data = data.to_dict(orient="records")
        else:
            json_data = data
        
        # Save JSON
        with open(json_path, "w", encoding=encoding) as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
    
    return csv_path, json_path


def read_csv_to_dataframe(
    result_csv_path_file: Union[str, Path],
    encoding: str = "utf-8",
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file as DataFrame
    
    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        encoding: File encoding, default 'utf-8'
        **kwargs: Additional pandas read_csv parameters
        
    Returns:
        pd.DataFrame: Read data
        
    Raises:
        CSVFileError: Raised when file does not exist or read fails
        
    Example:
        >>> df = read_csv_to_dataframe("./results/test.csv")
    """
    if not result_csv_path_file:
        raise CSVFileError("result_csv_path_file parameter cannot be empty")
    
    file_path = Path(result_csv_path_file)
    
    if not file_path.exists():
        raise CSVFileError(f"File does not exist: {file_path}")
    
    try:
        df = pd.read_csv(file_path, encoding=encoding, **kwargs)
        return df
    except Exception as e:
        raise CSVFileError(f"Failed to read CSV file: {file_path}\nError: {e}")


def read_csv_to_list(
    result_csv_path_file: Union[str, Path],
    encoding: str = "utf-8"
) -> List[Dict[str, Any]]:
    """
    Read CSV file as list of dictionaries
    
    Args:
        result_csv_path_file: CSV file path (including filename), required parameter
        encoding: File encoding, default 'utf-8'
        
    Returns:
        List[Dict[str, Any]]: List of dictionaries, each dictionary represents one row of data
        
    Example:
        >>> data = read_csv_to_list("./results/test.csv")
        >>> # Returns: [{"model": "a", "mae": 0.5}, {"model": "b", "mae": 0.6}]
    """
    df = read_csv_to_dataframe(result_csv_path_file, encoding=encoding)
    return df.to_dict(orient="records")


def csv_exists_and_not_empty(result_csv_path_file: Union[str, Path]) -> bool:
    """
    Check if CSV file exists and is not empty
    
    Args:
        result_csv_path_file: CSV file path
        
    Returns:
        bool: True=file exists and has data; False=file does not exist or is empty
        
    Example:
        >>> if csv_exists_and_not_empty("./results/test.csv"):
        ...     df = read_csv_to_dataframe("./results/test.csv")
    """
    if not result_csv_path_file:
        return False
    
    file_path = Path(result_csv_path_file)
    
    if not file_path.exists():
        return False
    
    # Check if file is empty (only header also counts as non-empty)
    try:
        df = pd.read_csv(file_path)
        return len(df) > 0
    except:
        return False

