"""
模型工具模块
提供模型摘要、参数统计、设备管理等功能
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple


def get_model_summary(model: nn.Module) -> Dict[str, Any]:
    """
    获取模型摘要信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型摘要字典
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = {
        'model_class': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),  # 假设float32
    }
    
    # 添加模型特定信息
    if hasattr(model, 'num_nodes'):
        summary['num_nodes'] = model.num_nodes
    if hasattr(model, 'hidden_channels'):
        summary['hidden_channels'] = model.hidden_channels
    if hasattr(model, 'n_modes'):
        summary['n_modes'] = model.n_modes
    if hasattr(model, 'sequence_length'):
        summary['sequence_length'] = model.sequence_length
    
    # 设备信息
    try:
        device = next(model.parameters()).device
        summary['device'] = str(device)
    except StopIteration:
        summary['device'] = 'no_parameters'
    
    return summary


def count_parameters(model: nn.Module, only_trainable: bool = False) -> int:
    """
    统计模型参数数量
    
    Args:
        model: PyTorch模型
        only_trainable: 是否只统计可训练参数
        
    Returns:
        参数数量
    """
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def check_model_device(model: nn.Module) -> str:
    """
    检查模型所在设备
    
    Args:
        model: PyTorch模型
        
    Returns:
        设备名称
    """
    try:
        device = next(model.parameters()).device
        return str(device)
    except StopIteration:
        return "no_parameters"


def freeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None):
    """
    冻结模型参数
    
    Args:
        model: PyTorch模型
        layer_names: 要冻结的层名称列表，None表示冻结所有层
    """
    if layer_names is None:
        # 冻结所有参数
        for param in model.parameters():
            param.requires_grad = False
    else:
        # 冻结指定层
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False


def unfreeze_parameters(model: nn.Module, layer_names: Optional[List[str]] = None):
    """
    解冻模型参数
    
    Args:
        model: PyTorch模型
        layer_names: 要解冻的层名称列表，None表示解冻所有层
    """
    if layer_names is None:
        # 解冻所有参数
        for param in model.parameters():
            param.requires_grad = True
    else:
        # 解冻指定层
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True


def get_layer_info(model: nn.Module) -> List[Dict[str, Any]]:
    """
    获取模型各层信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        层信息列表
    """
    layer_info = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            # 统计参数
            total_params = sum(p.numel() for p in module.parameters())
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            info = {
                'name': name,
                'type': module.__class__.__name__,
                'total_params': total_params,
                'trainable_params': trainable_params
            }
            
            # 添加层特定信息
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                info['in_features'] = module.in_features
                info['out_features'] = module.out_features
            
            if hasattr(module, 'in_channels') and hasattr(module, 'out_channels'):
                info['in_channels'] = module.in_channels
                info['out_channels'] = module.out_channels
            
            if hasattr(module, 'kernel_size'):
                info['kernel_size'] = module.kernel_size
            
            layer_info.append(info)
    
    return layer_info


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """
    比较两个模型
    
    Args:
        model1: 第一个模型
        model2: 第二个模型
        
    Returns:
        比较结果
    """
    summary1 = get_model_summary(model1)
    summary2 = get_model_summary(model2)
    
    comparison = {
        'model1': summary1,
        'model2': summary2,
        'parameter_difference': summary2['total_parameters'] - summary1['total_parameters'],
        'size_difference_mb': summary2['model_size_mb'] - summary1['model_size_mb']
    }
    
    return comparison


def save_model_checkpoint(model: nn.Module, 
                         optimizer: torch.optim.Optimizer,
                         epoch: int,
                         loss: float,
                         save_path: str,
                         additional_info: Optional[Dict[str, Any]] = None):
    """
    保存模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前轮次
        loss: 当前损失
        save_path: 保存路径
        additional_info: 额外信息
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'model_summary': get_model_summary(model)
    }
    
    if additional_info:
        checkpoint.update(additional_info)
    
    torch.save(checkpoint, save_path)


def load_model_checkpoint(model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         checkpoint_path: str) -> Dict[str, Any]:
    """
    加载模型检查点
    
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
        
    Returns:
        检查点信息
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', 0.0),
        'model_summary': checkpoint.get('model_summary', {})
    }


def initialize_weights(model: nn.Module, method: str = 'xavier_uniform'):
    """
    初始化模型权重
    
    Args:
        model: PyTorch模型
        method: 初始化方法 ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
    """
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
            if method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def get_model_memory_usage(model: nn.Module) -> Dict[str, float]:
    """
    估算模型内存使用
    
    Args:
        model: PyTorch模型
        
    Returns:
        内存使用信息（MB）
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    
    return {
        'parameters_mb': param_size,
        'buffers_mb': buffer_size,
        'total_mb': param_size + buffer_size
    }


__all__ = [
    'get_model_summary',
    'count_parameters',
    'check_model_device',
    'freeze_parameters',
    'unfreeze_parameters',
    'get_layer_info',
    'compare_models',
    'save_model_checkpoint',
    'load_model_checkpoint',
    'initialize_weights',
    'get_model_memory_usage'
] 