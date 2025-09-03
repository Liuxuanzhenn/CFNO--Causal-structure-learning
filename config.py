"""
集中管理所有配置参数的模块
所有其他文件都应该从这里导入配置
"""
import copy

# 基础配置参数 - 所有任务共享的默认值
BASE_CONFIG = {
    'data': {
        'file_path': "data/generated_time_series2.csv",
        'train_ratio': 0.8,
        'max_time_steps': 1000,
        'normalize': True,
        'augment': False  # 启用数据增强
    },
    'model': {
        'dropout': 0.2,
        'mlp_dropout': 0.2,
        'n_layers': 2,
        'hidden_channels': 64,
        'use_diagonal_mask': True,
        'use_dag_constraint': True,
        'causal_reg_weight': 0.001,

        # 邻接矩阵相关配置
        'use_external_adj': False,  # 是否使用外部邻接矩阵
        'external_adj_source': 'manual',  # 外部邻接矩阵来源: 'granger', 'correlation', 'manual'
        'manual_adj_path': 'waibu_matrix.csv',  # 手动模式下的外部邻接矩阵文件路径
        'internal_adj_init': 'zeros',  # 内部矩阵初始化方法: 'random', 'zeros', 'tiny', 'data_corr'
        'dynamic_adj_weight': True,  # 是否动态调整内外矩阵权重
        'initial_external_weight': 0.6,  # 训练初期外部矩阵权重
        'final_external_weight': 0.25,  # 训练后期外部矩阵权重
        'weight_transition_epochs': 5,  # 权重过渡的epoch数量

        # 格兰杰因果检验相关配置 (整合到一个子字典中)
        'granger': {
            'enabled': False,  # 是否启用格兰杰因果检验
            'max_lag': 3,  # 最大滞后阶数
            'significance': 0.05,  # 显著性水平
            'normalize': False,  # 是否归一化结果
            'as_init_matrix': False  # 是否用作初始邻接矩阵
        }
    },
    'training': {
        'sequence_length': 50,
        'batch_size': 32,
        'num_epochs': 60,
        'learning_rate': 0.001,
            'init_iter': 20,
            'early_stopping_thresh': 1.0,
        'patience': 15
    },
    'random_seed': {
        'use_fixed_seed': True,  # 是否使用固定种子
        'fixed_seed': 21,  # 如果使用固定种子，使用这个值
        'save_seed': True,  # 是否保存随机生成的种子
        'seed_file': 'last_seed.txt'  # 保存种子的文件
    },
    'output': {
        'results_dir': 'results',
        'model_name': 'causal_model'
    },
    'eval': {
        'default_threshold': 0.3,
        'thresholds': [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5],
        'bayesian_iterations': 25  # 移动到这里，使配置结构更清晰
    },
    'augmentation': {
        'factor': 2,  # 增强数据集大小倍数
        'techniques': ['scaling'],  # 使用的增强技术'noise','scaling'
        'noise_level': [0.01, 0.05],  # 噪声水平范围
        'scale_range': [0.8, 1.2],  # 缩放范围
        'time_warp_width': [1, 5],  # 时间扭曲窗口大小，最小值修改为1
        'window_shift': [-2, 2]  # 窗口滑动范围
    },
    'losses': {
        # 模型中现有损失函数的开关
        'mse_loss': True,  # 均方误差损失启用
        'huber_loss': False,  # Huber损失启用
        'l1_reg': True,  # L1正则化启用
        'sparsity_loss': True,  # 稀疏性损失启用
        'nonzero_penalty': True,  # 非零惩罚启用
        'dag_loss': True  # DAG约束损失启用
    },
    'bayesian_optimization': {
        'enabled': False,  # 是否启用贝叶斯优化
        'n_iterations': 25,  # 贝叶斯优化迭代次数
        'initial_points': 10,  # 初始随机采样点数量
        'acquisition_function': 'ei'  # 采集函数类型：'ei'(期望改进), 'ucb'(上置信边界), 'poi'(改进概率)
    }
}

# 创建特定任务的配置，基于基础配置深度复制
CONFIG = copy.deepcopy(BASE_CONFIG)
TRAIN_CONFIG = copy.deepcopy(BASE_CONFIG)
GET_FINAL_DAG_CONFIG = copy.deepcopy(BASE_CONFIG)

# 特定任务的覆盖配置
GET_FINAL_DAG_CONFIG['training']['sequence_length'] = 50

# 向后兼容旧代码，将旧的格兰杰配置转换为新格式
def _convert_old_granger_config():
    """
    如果存在旧的格兰杰因果检验配置项，转换为新格式
    注意：这是临时函数，仅用于向后兼容，未来应移除
    """
    if 'use_granger_causality' in CONFIG.get('model', {}):
        CONFIG['model']['granger']['enabled'] = CONFIG['model'].pop('use_granger_causality')
    
    if 'granger_max_lag' in CONFIG.get('model', {}):
        CONFIG['model']['granger']['max_lag'] = CONFIG['model'].pop('granger_max_lag')
    
    if 'granger_significance' in CONFIG.get('model', {}):
        CONFIG['model']['granger']['significance'] = CONFIG['model'].pop('granger_significance')
    
    if 'granger_normalize' in CONFIG.get('model', {}):
        CONFIG['model']['granger']['normalize'] = CONFIG['model'].pop('granger_normalize')
    
    if 'external_adj_init' in CONFIG.get('model', {}) and CONFIG['model'].get('external_adj_init') == 'granger':
        CONFIG['model']['granger']['as_init_matrix'] = True
        CONFIG['model']['external_adj_source'] = 'granger'

# 执行转换
_convert_old_granger_config()

# 获取配置的函数
def get_config():
    """获取全局配置"""
    return CONFIG

def get_train_config():
    """获取训练特定配置"""
    return TRAIN_CONFIG

def get_final_dag_config():
    """获取DAG提取特定配置"""
    return GET_FINAL_DAG_CONFIG

def merge_config(base_config, override_config):
    """
    将覆盖配置合并到基础配置中
    
    Args:
        base_config: 基础配置字典
        override_config: 覆盖配置字典
    
    Returns:
        合并后的配置字典
    """
    result = copy.deepcopy(base_config)
    
    def _recursive_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _recursive_update(d[k], v)
            else:
                d[k] = v
        return d
    
    return _recursive_update(result, override_config)

def get_merged_config(override_config=None):
    """
    获取合并了覆盖配置的全局配置
    
    Args:
        override_config: 覆盖配置字典，可选
    
    Returns:
        合并后的配置字典
    """
    if override_config is None:
        return copy.deepcopy(CONFIG)
    else:
        return merge_config(CONFIG, override_config) 