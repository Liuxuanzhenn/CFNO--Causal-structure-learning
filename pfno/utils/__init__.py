"""
PFNO工具库 - 统一的工具函数集合
消除代码重复，提供清晰的模块化结构
"""

# 导入各个模块的公共接口
from .tensor_ops import *
from .config import *
from .logging import *
from .evaluation import *
from .experiment import *
from .model_utils import *
from .visualization import *

__version__ = "1.0.0"
__author__ = "PFNO Team"

__all__ = [
    # 张量操作
    'safe_tensor_operation', 'safe_numpy_operation', 'ensure_tensor',
    'normalize_tensor', 'check_numerical_stability',
    
    # 配置管理
    'validate_config', 'load_config', 'save_config', 'merge_configs',
    
    # 日志记录
    'Logger', 'setup_logging', 'log_model_info', 'log_experiment_info',
    
    # 评估工具
    'evaluate_causal_discovery', 'find_optimal_threshold', 'compute_metrics',
    'confusion_matrix_metrics', 'roc_analysis',
    
    # 实验管理
    'save_experiment', 'load_experiment', 'generate_report',
    'compare_experiments', 'create_experiment_summary',
    
    # 模型工具
    'get_model_summary', 'count_parameters', 'check_model_device',
    'freeze_parameters', 'unfreeze_parameters',
    
    # 可视化
    'plot_adjacency_matrix', 'plot_training_curves', 'plot_evaluation_metrics',
    'create_experiment_dashboard', 'save_plots', 'plot_enhanced_time_series_reconstruction',
    'plot_probability_matrix_comparison', 'plot_time_series_reconstruction',
    'plot_comprehensive_results'
] 