"""
PFNO (Physics-informed Fourier Neural Operator) 包
统一的工具库和模型实现
"""

import warnings

# 导入核心工具
try:
    from .utils import (
        # 配置验证
        validate_config, load_config, save_config, merge_configs,
        # 张量操作
        safe_tensor_operation, safe_numpy_operation, ensure_tensor,
        normalize_tensor, check_numerical_stability,
        # 日志记录
        Logger, setup_logging, log_model_info, log_experiment_info,
        # 因果发现评估
        evaluate_causal_discovery, find_optimal_threshold, compute_metrics,
        confusion_matrix_metrics, roc_analysis,
        # 实验管理
        save_experiment, load_experiment, generate_report,
        compare_experiments, create_experiment_summary,
        # 模型工具
        get_model_summary, count_parameters, check_model_device,
        freeze_parameters, unfreeze_parameters,
        # 可视化
        plot_adjacency_matrix, plot_training_curves, plot_evaluation_metrics,
        create_experiment_dashboard, save_plots
    )
    _utils_available = True
except ImportError as e:
    warnings.warn(f"无法导入PFNO工具模块: {e}")
    _utils_available = False

# 配置工具
try:
    from .config_loader import get_pfno_config, print_config_summary, save_config_to_file
    _config_available = True
except ImportError as e:
    warnings.warn(f"无法导入PFNO配置模块: {e}")
    _config_available = False

# 模型相关
try:
    from .probabilistic_causal_pfno import ProbabilisticCausalPFNO
    _basic_model_available = True
except ImportError as e:
    warnings.warn(f"无法导入基础概率PFNO模型: {e}")
    ProbabilisticCausalPFNO = None
    _basic_model_available = False

try:
    from .probabilistic_causal_pfno_enhanced import EnhancedProbabilisticCausalPFNO
    _enhanced_model_available = True
except ImportError as e:
    warnings.warn(f"无法导入增强概率PFNO模型: {e}")
    EnhancedProbabilisticCausalPFNO = None
    _enhanced_model_available = False

# 构建导出列表
__all__ = []

# 核心工具
if _utils_available:
    __all__.extend([
        'safe_tensor_operation', 'safe_numpy_operation', 'ensure_tensor',
        'normalize_tensor', 'check_numerical_stability',
        'validate_config', 'load_config', 'save_config', 'merge_configs',
        'Logger', 'setup_logging', 'log_model_info', 'log_experiment_info',
        'evaluate_causal_discovery', 'find_optimal_threshold', 'compute_metrics',
        'confusion_matrix_metrics', 'roc_analysis',
        'save_experiment', 'load_experiment', 'generate_report',
        'compare_experiments', 'create_experiment_summary',
        'get_model_summary', 'count_parameters', 'check_model_device',
        'freeze_parameters', 'unfreeze_parameters',
        'plot_adjacency_matrix', 'plot_training_curves', 'plot_evaluation_metrics',
        'create_experiment_dashboard', 'save_plots',
    ])

# 配置
if _config_available:
    __all__.extend(['get_pfno_config', 'print_config_summary', 'save_config_to_file'])

# 模型
if _enhanced_model_available:
    __all__.append('EnhancedProbabilisticCausalPFNO')

# 版本信息
__version__ = "1.0.0"
__author__ = "PFNO团队"
__description__ = "统一的PFNO工具库和模型实现"

# 模块状态信息
def get_module_status():
    """获取模块可用性状态"""
    return {
        'utils_available': _utils_available,
        'config_available': _config_available,
        'enhanced_model_available': _enhanced_model_available,
        'available_exports': len(__all__)
    } 