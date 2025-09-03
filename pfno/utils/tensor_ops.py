"""
张量操作工具模块
提供安全的张量操作和数值稳定性检查
"""

import torch
import numpy as np
import warnings
from typing import Any, Optional, Union, Callable


def safe_tensor_operation(func: Callable, *args, fallback_value=None, 
                         operation_name: str = "tensor operation", **kwargs) -> Any:
    """
    安全的张量操作包装器
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        fallback_value: 失败时的回退值
        operation_name: 操作名称（用于日志）
        **kwargs: 函数关键字参数
        
    Returns:
        操作结果或回退值
    """
    try:
        result = func(*args, **kwargs)
        
        # 检查数值稳定性
        if isinstance(result, torch.Tensor):
            if torch.isnan(result).any():
                warnings.warn(f"{operation_name} 产生了 NaN 值")
                if fallback_value is not None:
                    return fallback_value
            if torch.isinf(result).any():
                warnings.warn(f"{operation_name} 产生了 Inf 值")
                if fallback_value is not None:
                    return fallback_value
        
        return result
    
    except Exception as e:
        warnings.warn(f"{operation_name} 失败: {e}")
        if fallback_value is not None:
            return fallback_value
        raise


def safe_numpy_operation(func: Callable, *args, fallback_value=None,
                        operation_name: str = "numpy operation", **kwargs) -> Any:
    """
    安全的numpy操作包装器
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        fallback_value: 失败时的回退值
        operation_name: 操作名称（用于日志）
        **kwargs: 函数关键字参数
        
    Returns:
        操作结果或回退值
    """
    try:
        result = func(*args, **kwargs)
        
        # 检查数值稳定性
        if isinstance(result, np.ndarray):
            if np.isnan(result).any():
                warnings.warn(f"{operation_name} 产生了 NaN 值")
                if fallback_value is not None:
                    return fallback_value
            if np.isinf(result).any():
                warnings.warn(f"{operation_name} 产生了 Inf 值")
                if fallback_value is not None:
                    return fallback_value
        
        return result
    
    except Exception as e:
        warnings.warn(f"{operation_name} 失败: {e}")
        if fallback_value is not None:
            return fallback_value
        raise


def ensure_tensor(data: Union[torch.Tensor, np.ndarray, list], 
                 device: Optional[torch.device] = None, 
                 dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    确保数据为张量格式
    
    Args:
        data: 输入数据
        device: 目标设备
        dtype: 数据类型
        
    Returns:
        张量格式的数据
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=dtype)
    
    if device is not None:
        data = data.to(device)
    
    return data


def normalize_tensor(tensor: torch.Tensor, method: str = 'minmax', 
                    dim: Optional[int] = None, keepdim: bool = False) -> torch.Tensor:
    """
    张量归一化
    
    Args:
        tensor: 输入张量
        method: 归一化方法 ('minmax', 'zscore', 'l2')
        dim: 归一化维度
        keepdim: 保持维度
        
    Returns:
        归一化后的张量
    """
    if method == 'minmax':
        if dim is not None:
            min_val = tensor.min(dim=dim, keepdim=keepdim)[0]
            max_val = tensor.max(dim=dim, keepdim=keepdim)[0]
        else:
            min_val = tensor.min()
            max_val = tensor.max()
        
        # 避免除零
        range_val = max_val - min_val
        range_val = torch.where(range_val == 0, torch.ones_like(range_val), range_val)
        
        return (tensor - min_val) / range_val
    
    elif method == 'zscore':
        if dim is not None:
            mean = tensor.mean(dim=dim, keepdim=keepdim)
            std = tensor.std(dim=dim, keepdim=keepdim)
        else:
            mean = tensor.mean()
            std = tensor.std()
        
        # 避免除零
        std = torch.where(std == 0, torch.ones_like(std), std)
        
        return (tensor - mean) / std
    
    elif method == 'l2':
        if dim is not None:
            norm = torch.norm(tensor, p=2, dim=dim, keepdim=keepdim)
        else:
            norm = torch.norm(tensor, p=2)
        
        # 避免除零
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        
        return tensor / norm
    
    else:
        raise ValueError(f"未知的归一化方法: {method}")


def check_numerical_stability(tensor: torch.Tensor, 
                            name: str = "tensor") -> dict:
    """
    检查张量的数值稳定性
    
    Args:
        tensor: 输入张量
        name: 张量名称
        
    Returns:
        检查结果字典
    """
    result = {
        'name': name,
        'shape': tensor.shape,
        'dtype': tensor.dtype,
        'device': tensor.device,
        'has_nan': torch.isnan(tensor).any().item(),
        'has_inf': torch.isinf(tensor).any().item(),
        'min': tensor.min().item(),
        'max': tensor.max().item(),
        'mean': tensor.mean().item(),
        'std': tensor.std().item(),
    }
    
    # 检查是否有问题
    if result['has_nan']:
        warnings.warn(f"张量 {name} 包含 NaN 值")
    if result['has_inf']:
        warnings.warn(f"张量 {name} 包含 Inf 值")
    if abs(result['mean']) > 1e6:
        warnings.warn(f"张量 {name} 均值过大: {result['mean']}")
    if result['std'] > 1e6:
        warnings.warn(f"张量 {name} 标准差过大: {result['std']}")
    
    return result


def clamp_tensor(tensor: torch.Tensor, min_val: float = -1e6, 
                max_val: float = 1e6) -> torch.Tensor:
    """
    限制张量值的范围以提高数值稳定性
    
    Args:
        tensor: 输入张量
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        限制后的张量
    """
    return torch.clamp(tensor, min=min_val, max=max_val)


def safe_matrix_operation(operation: str, matrix1: torch.Tensor, 
                         matrix2: Optional[torch.Tensor] = None,
                         **kwargs) -> torch.Tensor:
    """
    安全的矩阵操作
    
    Args:
        operation: 操作类型 ('matmul', 'inverse', 'solve', 'svd')
        matrix1: 第一个矩阵
        matrix2: 第二个矩阵（如果需要）
        **kwargs: 其他参数
        
    Returns:
        操作结果
    """
    if operation == 'matmul':
        if matrix2 is None:
            raise ValueError("矩阵乘法需要两个矩阵")
        return safe_tensor_operation(
            torch.matmul, matrix1, matrix2,
            operation_name="矩阵乘法",
            fallback_value=torch.zeros(matrix1.shape[0], matrix2.shape[1])
        )
    
    elif operation == 'inverse':
        return safe_tensor_operation(
            torch.inverse, matrix1,
            operation_name="矩阵求逆",
            fallback_value=torch.eye(matrix1.shape[0])
        )
    
    elif operation == 'solve':
        if matrix2 is None:
            raise ValueError("矩阵求解需要两个矩阵")
        return safe_tensor_operation(
            torch.solve, matrix2, matrix1,
            operation_name="矩阵求解"
        )
    
    elif operation == 'svd':
        return safe_tensor_operation(
            torch.svd, matrix1,
            operation_name="SVD分解"
        )
    
    else:
        raise ValueError(f"未知的矩阵操作: {operation}")


__all__ = [
    'safe_tensor_operation', 
    'safe_numpy_operation', 
    'ensure_tensor',
    'normalize_tensor', 
    'check_numerical_stability',
    'clamp_tensor',
    'safe_matrix_operation'
] 