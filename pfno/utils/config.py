"""
配置管理工具模块
提供配置验证、加载、保存、合并等功能
"""

import json
import yaml
import warnings
from typing import Dict, Any, Optional, Union
from pathlib import Path
import copy


def validate_config(config: Dict[str, Any], config_type: str = "pfno") -> bool:
    """
    统一的配置验证函数
    
    Args:
        config: 配置字典
        config_type: 配置类型 ("pfno", "enhanced", "basic")
        
    Returns:
        配置是否有效
    """
    if config_type in ["pfno", "enhanced"]:
        required_keys = [
            'architecture', 'probabilistic', 'causal',
            'training', 'data', 'evaluation'
        ]
    else:
        required_keys = ['architecture', 'training', 'data']
    
    # 检查必需键
    for key in required_keys:
        if key not in config:
            warnings.warn(f"缺少必需的配置键: {key}")
            return False
    
    # 验证架构参数
    arch = config.get('architecture', {})
    if arch.get('hidden_channels', 0) <= 0:
        warnings.warn("hidden_channels 必须为正数")
        return False
    
    if 'n_modes' in arch:
        n_modes = arch['n_modes']
        if isinstance(n_modes, list):
            if not all(isinstance(m, int) and m > 0 for m in n_modes):
                warnings.warn("n_modes 列表中的所有元素必须为正整数")
                return False
        elif not (isinstance(n_modes, int) and n_modes > 0):
            warnings.warn("n_modes 必须为正整数或正整数列表")
            return False
    
    # 验证概率参数（如果存在）
    if 'probabilistic' in config:
        prob = config['probabilistic']
        if prob.get('n_samples', 0) <= 0:
            warnings.warn("n_samples 必须为正数")
            return False
        
        confidence_threshold = prob.get('confidence_threshold', 0.5)
        if not (0 < confidence_threshold < 1):
            warnings.warn("confidence_threshold 必须在 (0, 1) 区间内")
            return False
    
    # 验证多尺度参数（如果存在）
    # if 'multi_scale' in config and config['multi_scale'].get('enabled', False):
    #     time_scales = config['multi_scale'].get('time_scales', [])
    #     if not time_scales:
    #         warnings.warn("启用多尺度时必须提供 time_scales")
    #         return False
    #     if not all(0 <= scale < 1 for scale in time_scales):
    #         warnings.warn("时间尺度必须在 [0, 1) 区间内，其中0表示即时响应")
    #         return False
    
    # 验证训练参数
    training = config.get('training', {})
    if training.get('num_epochs', 0) <= 0:
        warnings.warn("num_epochs 必须为正数")
        return False
    
    if training.get('learning_rate', 0) <= 0:
        warnings.warn("learning_rate 必须为正数")
        return False
    
    # 修复：batch_size在training部分而不是data部分
    if training.get('batch_size', 0) <= 0:
        warnings.warn("batch_size 必须为正数")
        return False
    
    # 验证数据参数
    data = config.get('data', {})
    if data.get('sequence_length', 0) <= 0:
        warnings.warn("sequence_length 必须为正数")
        return False
    
    return True


def load_config(config_path: Union[str, Path], 
                config_format: Optional[str] = None) -> Dict[str, Any]:
    """
    从文件加载配置
    
    Args:
        config_path: 配置文件路径
        config_format: 配置格式 ('json', 'yaml', None为自动检测)
        
    Returns:
        配置字典
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    # 自动检测格式
    if config_format is None:
        suffix = config_path.suffix.lower()
        if suffix in ['.json']:
            config_format = 'json'
        elif suffix in ['.yaml', '.yml']:
            config_format = 'yaml'
        else:
            raise ValueError(f"无法自动检测配置格式，文件后缀： {suffix}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_format == 'json':
            config = json.load(f)
        elif config_format == 'yaml':
            config = yaml.safe_load(f)
        else:
            raise ValueError(f"不支持的配置格式: {config_format}")
    
    return config


def save_config(config: Dict[str, Any], 
                config_path: Union[str, Path],
                config_format: Optional[str] = None,
                indent: int = 2) -> None:
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        config_path: 保存路径
        config_format: 配置格式 ('json', 'yaml', None为自动检测)
        indent: 缩进空格数
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 自动检测格式
    if config_format is None:
        suffix = config_path.suffix.lower()
        if suffix in ['.json']:
            config_format = 'json'
        elif suffix in ['.yaml', '.yml']:
            config_format = 'yaml'
        else:
            config_format = 'json'  # 默认使用json
            config_path = config_path.with_suffix('.json')
    
    # 保存配置
    with open(config_path, 'w', encoding='utf-8') as f:
        if config_format == 'json':
            json.dump(config, f, indent=indent, ensure_ascii=False)
        elif config_format == 'yaml':
            yaml.dump(config, f, indent=indent, default_flow_style=False, 
                     allow_unicode=True)
        else:
            raise ValueError(f"不支持的配置格式: {config_format}")


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any],
                 deep_merge: bool = True) -> Dict[str, Any]:
    """
    合并配置
    
    Args:
        base_config: 基础配置
        override_config: 覆盖配置
        deep_merge: 是否深度合并
        
    Returns:
        合并后的配置
    """
    if not deep_merge:
        merged = base_config.copy()
        merged.update(override_config)
        return merged
    
    # 深度合并
    merged = copy.deepcopy(base_config)
    
    def _deep_merge(base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                _deep_merge(base[key], value)
            else:
                base[key] = value
    
    _deep_merge(merged, override_config)
    return merged


def get_default_config(config_type: str = "pfno") -> Dict[str, Any]:
    """
    获取默认配置
    
    Args:
        config_type: 配置类型
        
    Returns:
        默认配置字典
    """
    if config_type == "pfno":
        return {
            'architecture': {
                'n_modes': [16, 16],
                'hidden_channels': 64,
                'n_layers': 3,
                'dropout': 0.1,
                'activation': 'gelu'
            },
            'probabilistic': {
                'n_samples': 100,
                'confidence_threshold': 0.8,
                'uncertainty_method': 'monte_carlo',
                'numerical_stability_check': True
            },
            'causal': {
                'use_dag_constraint': True,
                'causal_reg_weight': 0.01,
                'dag_reg_type': 'spectral_radius',
                'gradient_clipping': True
            },
            'multi_scale': {
                'enabled': False,
                'time_scales': [0.1, 0.3, 0.5, 0.7, 0.9],
                'combine_method': 'attention',
                'analyze_scale_causality': True
            },
            'training': {
                'num_epochs': 100,
                'learning_rate': 0.001,
                'batch_size': 32,
                'weight_decay': 1e-4,
                'patience': 10,
                'scheduler': 'cosine'
            },
            'data': {
                'sequence_length': 25,
                'num_nodes': None,  # 需要根据数据设定
                'normalize': True,
                'augmentation': False
            },
            'evaluation': {
                'metrics': ['f1', 'precision', 'recall', 'accuracy'],
                'thresholds': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                'cross_validation': False
            }
        }
    
    elif config_type == "basic":
        return {
            'architecture': {
                'hidden_channels': 32,
                'n_layers': 2,
                'dropout': 0.0
            },
            'training': {
                'num_epochs': 50,
                'learning_rate': 0.01,
                'batch_size': 16
            },
            'data': {
                'sequence_length': 10,
                'normalize': True
            }
        }
    
    else:
        raise ValueError(f"未知的配置类型: {config_type}")


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    使用点符号路径更新配置
    
    Args:
        config: 原始配置
        updates: 更新字典，键可以使用点符号如 'training.learning_rate'
        
    Returns:
        更新后的配置
    """
    updated_config = copy.deepcopy(config)
    
    for key, value in updates.items():
        # 处理点符号路径
        keys = key.split('.')
        current = updated_config
        
        # 遍历到最后一层
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # 设置最终值
        current[keys[-1]] = value
    
    return updated_config


def validate_and_fix_config(config: Dict[str, Any], 
                           config_type: str = "pfno") -> Dict[str, Any]:
    """
    验证并修复配置
    
    Args:
        config: 输入配置
        config_type: 配置类型
        
    Returns:
        修复后的配置
    """
    # 获取默认配置作为模板
    default_config = get_default_config(config_type)
    
    # 合并配置（用默认值填补缺失项）
    fixed_config = merge_configs(default_config, config)
    
    # 验证修复后的配置
    if not validate_config(fixed_config, config_type):
        warnings.warn("配置修复后仍然无效，请检查配置参数")
    
    return fixed_config


__all__ = [
    'validate_config',
    'load_config', 
    'save_config',
    'merge_configs',
    'get_default_config',
    'update_config',
    'validate_and_fix_config'
] 