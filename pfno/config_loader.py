"""
PFNO配置加载器
使用utils/config.py管理YAML配置文件
"""

import os
import yaml
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

# 使用相对导入替代sys.path.append
from .utils.config import load_config, save_config, validate_config, merge_configs


# 默认配置文件路径
DEFAULT_CONFIG_PATH = Path(__file__).parent / "pfno_config.yaml"


def get_pfno_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    获取PFNO配置
    
    Args:
        config_path: 配置文件路径，默认使用pfno_config.yaml
        
    Returns:
        配置字典
    """
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    # 加载配置
    config = load_config(config_path)
    
    # 🔧 配置键映射 - 处理旧格式到新格式的转换
    config = _map_config_keys(config)
    
    # 验证配置
    if not validate_config(config, config_type="pfno"):
        raise ValueError("配置验证失败")
    
    return config


def _map_config_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    映射配置键，支持不同的配置格式
    
    Args:
        config: 原始配置
        
    Returns:
        映射后的配置
    """
    mapped_config = config.copy()
    
    # 如果有 'model' 键但没有 'architecture' 键，进行映射
    if 'model' in config and 'architecture' not in config:
        mapped_config['architecture'] = config['model'].copy()
        # 移除model键以避免冲突
        del mapped_config['model']
        print("🔧 配置映射: 'model' -> 'architecture'")
    
    # 确保概率配置存在
    if 'probabilistic' not in mapped_config:
        # 如果architecture中有概率参数，提取出来
        if 'architecture' in mapped_config:
            arch = mapped_config['architecture']
            prob_keys = ['n_samples', 'monte_carlo_samples', 'confidence_threshold', 'uncertainty_threshold']
            prob_config = {}
            
            for key in prob_keys:
                if key in arch:
                    prob_config[key] = arch[key]
                    del arch[key]
            
            if prob_config:
                mapped_config['probabilistic'] = prob_config
                print("🔧 配置映射: 概率参数从 'architecture' 提取到 'probabilistic'")
    
    return mapped_config


def update_config_from_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    从命令行参数更新配置
    
    Args:
        config: 基础配置
        args: 命令行参数对象
        
    Returns:
        更新后的配置
    """
    updates = {}
    
    # 训练参数
    if hasattr(args, 'epochs') and args.epochs:
        updates['training'] = {'num_epochs': args.epochs}
    
    if hasattr(args, 'lr') and args.lr:
        if 'training' not in updates:
            updates['training'] = {}
        updates['training']['learning_rate'] = args.lr
    
    if hasattr(args, 'batch_size') and args.batch_size:
        if 'training' not in updates:
            updates['training'] = {}
        updates['training']['batch_size'] = args.batch_size
    
    # 数据参数
    if hasattr(args, 'data_file') and args.data_file:
        updates['data'] = {'file_path': args.data_file}
    
    if hasattr(args, 'ground_truth_file') and args.ground_truth_file:
        if 'data' not in updates:
            updates['data'] = {}
        updates['data']['ground_truth_file'] = args.ground_truth_file
    
    # 预处理方法
    if hasattr(args, 'preprocessing_method') and args.preprocessing_method:
        if 'data' not in updates:
            updates['data'] = {}
        updates['data']['preprocessing'] = {'method': args.preprocessing_method}
    
    # 概率参数
    if hasattr(args, 'n_samples') and args.n_samples:
        updates['probabilistic'] = {'n_samples': args.n_samples}
    
    if hasattr(args, 'confidence_threshold') and args.confidence_threshold:
        if 'probabilistic' not in updates:
            updates['probabilistic'] = {}
        updates['probabilistic']['confidence_threshold'] = args.confidence_threshold
    
    # 调试参数
    if hasattr(args, 'debug') and args.debug:
        updates['debug_visualization'] = {'debug_mode': True, 'verbose_logging': True}
    
    # 合并配置
    if updates:
        config = merge_configs(config, updates)
    
    return config


def print_config_summary(config: Optional[Dict[str, Any]] = None):
    """
    打印配置摘要
    
    Args:
        config: 配置字典，如果为None则加载默认配置
    """
    if config is None:
        config = get_pfno_config()
    
    print("\n" + "="*60)
    print("                PFNO配置摘要")
    print("="*60)
    
    # 架构参数
    arch = config.get('architecture', {})
    print(f"🏗️  架构参数:")
    print(f"   • 傅里叶模式数: {arch.get('n_modes', 'N/A')}")
    print(f"   • 隐藏通道数: {arch.get('hidden_channels', 'N/A')}")
    print(f"   • FNO层数: {arch.get('n_layers', 'N/A')}")
    print(f"   • Dropout率: {arch.get('dropout', 'N/A')}")
    
    # 概率参数
    prob = config.get('probabilistic', {})
    print(f"\n🎲 概率参数:")
    print(f"   • 采样数量: {prob.get('n_samples', 'N/A')}")
    print(f"   • 置信度阈值: {prob.get('confidence_threshold', 'N/A')}")
    print(f"   • 蒙特卡洛采样: {prob.get('monte_carlo_samples', 'N/A')}")
    
    # 训练参数
    training = config.get('training', {})
    loss_weights = training.get('loss_weights', {})
    print(f"\n🚀 训练参数:")
    print(f"   • 训练轮数: {training.get('num_epochs', 'N/A')}")
    print(f"   • 学习率: {training.get('learning_rate', 'N/A')}")
    print(f"   • 批量大小: {training.get('batch_size', 'N/A')}")
    print(f"   • 优化器: {training.get('optimizer', 'N/A')}")
    print(f"   • L1正则化权重: {loss_weights.get('l1_regularization', 'N/A')}")
    print(f"   • L2正则化权重: {loss_weights.get('l2_regularization', 'N/A')}")
    print(f"   • DAG约束权重: {loss_weights.get('dag_constraint', 'N/A')}")
    
    # 数据参数
    data = config.get('data', {})
    print(f"\n📊 数据参数:")
    print(f"   • 数据文件: {data.get('file_path', 'N/A')}")
    print(f"   • 序列长度: {data.get('sequence_length', 'N/A')}")
    print(f"   • 训练比例: {data.get('train_ratio', 'N/A')}")
    preprocessing = data.get('preprocessing', {})
    print(f"   • 预处理方法: {preprocessing.get('method', 'N/A')}")
    
    print("="*60)


def save_config_to_file(config: Dict[str, Any], filepath: str):
    """
    保存配置到文件
    
    Args:
        config: 配置字典
        filepath: 保存路径
    """
    save_config(config, filepath)
    print(f"配置已保存到: {filepath}")


def load_config_from_file(filepath: str) -> Dict[str, Any]:
    """
    从文件加载配置
    
    Args:
        filepath: 配置文件路径
        
    Returns:
        配置字典
    """
    return load_config(filepath)


# 向后兼容的配置常量
def get_legacy_config():
    """获取与原pfno_config.py兼容的配置格式"""
    config = get_pfno_config()
    
    # 转换为原来的格式
    legacy_config = {
        'PFNO_ARCHITECTURE': config.get('architecture', {}),
        'PROBABILISTIC_PARAMS': config.get('probabilistic', {}),
        'CAUSAL_DISCOVERY': config.get('causal', {}),
        'TRAINING_STRATEGY': config.get('training', {}),
        'PFNO_DATA_PARAMS': config.get('data', {}),
        'EVALUATION_PARAMS': config.get('evaluation', {}),
        'PROBABILITY_SCALING': config.get('probability_scaling', {}),
        'RANDOM_SEED': config.get('random_seed', {}),
        'DEBUG_VISUALIZATION': config.get('debug_visualization', {})
    }
    
    return legacy_config


class JointLearningConfigLoader:
    """
    统一的配置加载器，支持从YAML文件和命令行参数加载配置。
    """
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载主配置文件"""
        if self.config_path is None:
            # 如果未提供路径，使用默认配置
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            warnings.warn(f"配置文件 {self.config_path} 未找到，将使用默认配置。")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """提供一个硬编码的默认配置"""
        return get_pfno_config()

    def update_from_args(self, args):
        """用命令行参数更新配置"""
        self.config = update_config_from_args(self.config, args)

    def get_joint_learning_config(self) -> Dict[str, Any]:
        """获取联合学习特定的配置"""
        return self.config.get('joint_learning', {})

    def get_loss_weights(self) -> Dict[str, float]:
        """获取损失权重"""
        return self.config.get('training', {}).get('loss_weights', {})

    def get_model_config(self, num_nodes: int, sequence_length: int) -> Dict[str, Any]:
        """从主配置中提取并构建模型特定的配置字典"""
        arch_config = self.config.get('architecture', {})
        prob_config = self.config.get('probabilistic', {})
        causal_config = self.config.get('causal', {})
        joint_config = self.get_joint_learning_config()
        joint_enabled = joint_config.get('enabled', False)
        
        model_config = {
            # 基础参数
            'n_modes': arch_config.get('n_modes', [8, 8]),
            'hidden_channels': arch_config.get('hidden_channels', 64),
            'lifting_channels': arch_config.get('lifting_channels', 256),
            'projection_channels': arch_config.get('projection_channels', 256),
            'n_layers': arch_config.get('n_layers', 2),
            'dropout': arch_config.get('dropout', 0.1),
            
            # 节点和序列信息
            'num_nodes': num_nodes,
            'sequence_length': sequence_length,
            
            # 概率参数
            'n_samples': prob_config.get('n_samples', 10),
            'numerical_stability_check': prob_config.get('numerical_stability_check', True),
            
            # 因果参数
            'use_dag_constraint': causal_config.get('use_dag_constraint', True),
            'causal_reg_weight': causal_config.get('causal_reg_weight', 0.001),
            'gradient_clipping': causal_config.get('gradient_clipping', 1.0),
            
            # ✨ 联合学习配置
            'joint_learning_config': joint_config if joint_enabled else None
        }
        
        return model_config
    
    def create_model(self, num_nodes: int, sequence_length: int):
        """
        创建配置好的模型实例
        
        Args:
            num_nodes: 节点数量
            sequence_length: 序列长度
            
        Returns:
            配置好的EnhancedProbabilisticCausalPFNO实例
        """
        from .probabilistic_causal_pfno import EnhancedProbabilisticCausalPFNO
        
        model_config = self.get_model_config(num_nodes, sequence_length)
        model = EnhancedProbabilisticCausalPFNO(**model_config)
        
        return model

    def print_config_summary(self):
        """打印最终的配置摘要"""
        print_config_summary(self.config)

def load_joint_learning_config(config_path: str = None) -> JointLearningConfigLoader:
    """
    加载并返回联合学习配置加载器实例
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        JointLearningConfigLoader实例
    """
    return JointLearningConfigLoader(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = get_pfno_config()
    print_config_summary(config)

    # 加载配置
    config_loader = load_joint_learning_config()
    
    # 打印配置摘要
    config_loader.print_config_summary()
    
    # 创建模型
    model = config_loader.create_model(num_nodes=5, sequence_length=20)
    print(f"\n🚀 成功创建模型，参数数量: {sum(p.numel() for p in model.parameters()):,}") 