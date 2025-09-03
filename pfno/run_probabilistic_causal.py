"""
概率因果发现运行脚本
使用概率傅里叶神经算子进行因果关系发现
完全使用pfno_config.py进行配置管理

运行方式:
python run_probabilistic_causal.py --model_type probabilistic --n_samples 20 --confidence_threshold 0.7
"""

import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import datetime
from tqdm import tqdm
from torch.utils.data import DataLoader
import seaborn as sns
import time
import hashlib
import random
import yaml
import json
from sklearn.metrics import auc

# 设置matplotlib支持中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 使用相对导入替代绝对导入
from .probabilistic_causal_pfno_enhanced import EnhancedProbabilisticCausalPFNO

# 导入PFNO专用配置
from .config_loader import (
    get_pfno_config, 
    update_config_from_args, 
    print_config_summary,
    save_config_to_file
)
from .utils.config import validate_config

# 导入现有工具函数 - 这些是项目级别的工具，需要向上导入
import sys
import os
from pathlib import Path

# 添加项目根目录到路径以访问项目级工具
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils import (
        load_data, 
        evaluate_causal_discovery,
        plot_adjacency_matrix,
        set_seed
    )
except ImportError as e:
    print(f"⚠️ 无法导入项目工具函数: {e}")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 导入PFNO统一可视化套件
try:
    # 尝试相对导入
    from .utils.visualization import (
        PFNOVisualizationSuite, 
        plot_training_curves, 
        plot_probability_matrix_comparison,
        plot_enhanced_time_series_reconstruction
    )
    PFNO_VIZ_AVAILABLE = True
    print("✅ PFNO可视化套件已加载 (相对导入)")
except ImportError as e:
    try:
        # 如果相对导入失败，尝试绝对导入
        from pfno.utils.visualization import (
            PFNOVisualizationSuite, 
            plot_training_curves, 
            plot_probability_matrix_comparison,
            plot_enhanced_time_series_reconstruction
        )
        PFNO_VIZ_AVAILABLE = True
        print("✅ PFNO可视化套件已加载 (绝对导入)")
    except ImportError as e2:
        print(f"⚠️ PFNO可视化套件不可用 (相对导入): {e}")
        print(f"⚠️ PFNO可视化套件不可用 (绝对导入): {e2}")
        PFNO_VIZ_AVAILABLE = False
        PFNOVisualizationSuite = None
        plot_training_curves = None
        plot_probability_matrix_comparison = None
        plot_enhanced_time_series_reconstruction = None


# 新增：用于绘制ROC曲线的函数 (高质量出版级)
def plot_roc_curve(fprs, tprs, roc_auc, model_name, results_dir):
    """绘制并保存符合期刊审美的高质量ROC曲线图"""
    try:
        # 使用seaborn进行美化
        sns.set_theme(style="whitegrid")
        plt.figure(figsize=(8, 8))
        
        # 绘制我们的模型的ROC曲线 (模仿示例中的蓝色实线)
        # 使用一个专业且清晰的蓝色
        plt.plot(fprs, tprs, color='#00529B', lw=2.5, 
                 label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # 绘制随机猜测线 (模仿示例中的灰色虚线)
        plt.plot([0, 1], [0, 1], color='grey', lw=2, linestyle='--', label='Random Guess')
        
        # 设置坐标轴范围和标签
        plt.xlim([-0.02, 1.0])
        plt.ylim([0.0, 1.02])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title('ROC Curve Analysis', fontsize=16, pad=12)
        
        # 设置刻度字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # 设置图例
        plt.legend(loc="lower right", fontsize=12)
        
        # 移除顶部和右侧的边框以达到更简洁的学术风格
        sns.despine()

        # 保存图像
        save_path = os.path.join(results_dir, 'roc_curve.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 高质量ROC曲线图已保存: {save_path}")
        return True
    except Exception as e:
        print(f"❌ 绘制ROC曲线失败: {e}")
        return False


# 新增：用于序列化Numpy对象的JSON编码器
class NpEncoder(json.JSONEncoder):
    """ 自定义JSON编码器，用于处理Numpy数据类型 """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def setup_seeds_and_reproducibility(config):
    """
    设置随机种子以确保实验可重现性
    
    Args:
        config: 配置字典
        
    Returns:
        actual_seed: 实际使用的种子值
    """
    seed_config = config.get('random_seed', {})
    use_fixed_seed = seed_config.get('use_fixed_seed', True)
    base_seed = seed_config.get('fixed_seed', 21)
    
    if use_fixed_seed:
        actual_seed = base_seed
        print(f"🔧 使用固定种子: {actual_seed}")
    else:
        # 使用时间戳生成随机种子
        actual_seed = int(time.time() * 1000) % 100000
        print(f"🔧 使用时间戳种子: {actual_seed}")
    
    # 设置所有随机源的种子
    torch.manual_seed(actual_seed)
    np.random.seed(actual_seed)
    random.seed(actual_seed)
    
    # 设置CUDA种子以确保GPU计算的可重现性
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed)
        # 设置CUDA为确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # 保存种子信息
    if seed_config.get('save_seed', True):
        seed_file = seed_config.get('seed_file', 'last_seed.txt')
        with open(seed_file, 'w') as f:
            f.write(f"seed: {actual_seed}\n")
            f.write(f"use_fixed_seed: {use_fixed_seed}\n")
    
    return actual_seed


def setup_experiment():
    """设置实验参数和环境"""
    # 加载pfno配置的默认值
    pfno_cfg = get_pfno_config()
    
    parser = argparse.ArgumentParser(description='概率因果发现实验')
    
    # 新增：用于优化的参数
    parser.add_argument('--config', type=str, default=None, help='从指定的YAML文件加载配置')
    parser.add_argument('--output_json', type=str, default=None, help='将最终评估指标保存为JSON文件的路径')
    parser.add_argument('--no-viz', action='store_true', help='禁用所有可视化输出，加速优化过程')

    # 移除model_type参数，只保留PFNO
    # 移除预设配置功能
    
    # 使用pfno_config中的默认值，只有在命令行明确指定时才覆盖
    parser.add_argument('--n_samples', type=int, 
                       default=pfno_cfg['probabilistic']['n_samples'], 
                       help=f'概率采样数量 (默认: {pfno_cfg["probabilistic"]["n_samples"]})')
    parser.add_argument('--confidence_threshold', type=float, 
                       default=pfno_cfg['probabilistic']['confidence_threshold'], 
                       help=f'置信度阈值 (默认: {pfno_cfg["probabilistic"]["confidence_threshold"]})')
    parser.add_argument('--uncertainty_threshold', type=float, 
                       default=pfno_cfg['probabilistic']['uncertainty_threshold'], 
                       help=f'不确定性阈值 (默认: {pfno_cfg["probabilistic"]["uncertainty_threshold"]})')
    parser.add_argument('--num_epochs', type=int, 
                       default=pfno_cfg['training']['num_epochs'], 
                       help=f'训练轮数 (默认: {pfno_cfg["training"]["num_epochs"]})')
    parser.add_argument('--learning_rate', type=float, 
                       default=pfno_cfg['training']['learning_rate'], 
                       help=f'学习率 (默认: {pfno_cfg["training"]["learning_rate"]})')
    
    args = parser.parse_args()
    
    # 获取默认配置或从文件加载
    if args.config:
        print(f"🧬 从文件加载配置: {args.config}")
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"配置文件未找到: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print("🧬 使用默认配置 (from pfno_config.py)")
        config = get_pfno_config()
    
    # 根据命令行参数更新配置
    config = update_config_from_args(config, args)
    
    # 验证配置
    if not validate_config(config):
        raise ValueError("配置验证失败")
    
    # 设置随机种子
    actual_seed = setup_seeds_and_reproducibility(config)
    config['runtime'] = config.get('runtime', {})
    config['runtime']['actual_seed'] = actual_seed
    
    # 设置结果目录
    # 如果output_json被指定(意味着在优化循环中)，则简化目录结构
    if args.output_json:
        # 在优化模式下，结果可以保存到一个统一的目录，避免创建过多文件夹
        results_dir = os.path.join("results", "hpo_runs", f"trial_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"results/probabilistic_causal_pfno_{timestamp}"

    os.makedirs(results_dir, exist_ok=True)
    
    print(f"=== 概率因果发现实验 ===")
    print(f"数据文件: {config['data']['file_path']}")
    print(f"真实因果矩阵: {config['data']['ground_truth_file']}")
    print(f"模型类型: PFNO")
    print(f"结果目录: {results_dir}")
    print(f"使用随机种子: {actual_seed}")
    print("-" * 50)
    
    # 打印配置摘要
    print_config_summary(config)
    
    # 保存配置到结果目录
    save_config_to_file(config, os.path.join(results_dir, 'experiment_config.json'))
    
    return args, config, results_dir, actual_seed


def load_experimental_data(config):
    """加载实验数据 - 改进的数据加载系统"""
    try:
        # 尝试使用增强预处理系统
        from .preprocessing.data_loader import load_experimental_data_enhanced
        
        print(f"🚀 使用增强预处理系统加载数据...")
        
        # 显示预处理配置信息
        preprocessing_method = config['data']['preprocessing']['method']
        print(f"📋 预处理方法: {preprocessing_method}")
        
        # 使用增强版数据加载器（支持新的预处理系统）
        train_dataloader, val_dataloader, seq_length, num_nodes, ground_truth, raw_data = load_experimental_data_enhanced(config)
        
        print(f"✅ 数据加载完成: 序列长度={seq_length}, 节点数={num_nodes}")
        
        return train_dataloader, val_dataloader, seq_length, num_nodes, ground_truth, raw_data
        
    except ImportError as e:
        print(f"⚠️ 增强预处理系统不可用: {e}")
        print(f"🔄 回退到标准数据加载方法...")
        
        # 回退到标准数据加载
        from utils import load_data
        
        train_dataloader, val_dataloader = load_data(
            file_path=config['data']['file_path'],
            train_ratio=config['data']['train_ratio'],
            batch_size=config['training']['batch_size'],
            sequence_length=config['data']['sequence_length'],
            max_time_steps=config['data']['max_time_steps'],
            normalize=True,
            augment=config['data']['augment']
        )
        
        # 推断数据维度
        raw_data = None
        for batch_x, batch_y in train_dataloader:
            total_features = batch_x.shape[1]
            seq_length = config['data']['sequence_length']
            num_nodes = total_features // seq_length
            
            # 保存原始数据用于可视化
            raw_data = batch_x[0].detach().cpu().numpy() if isinstance(batch_x[0], torch.Tensor) else batch_x[0]
            break
        
        # 尝试直接从文件加载原始数据用于时间序列可视化
        if raw_data is None:
            try:
                import pandas as pd
                original_df = pd.read_csv(config['data']['file_path'])
                if len(original_df.columns) >= num_nodes:
                    # 取前num_nodes列和前seq_length行
                    raw_data = original_df.iloc[:seq_length, :num_nodes].values
                    print(f"   📊 从文件直接加载原始数据用于可视化: {raw_data.shape}")
            except Exception as e:
                print(f"   ⚠️ 无法从文件加载原始数据: {e}")
        
        # 尝试加载真实邻接矩阵
        try:
            ground_truth = pd.read_csv(config['data']['ground_truth_file'], index_col=0).values
        except Exception as e:
            print(f"⚠️ 无法加载真实邻接矩阵: {e}")
            ground_truth = None
        
        print(f"✅ 标准数据加载完成: 序列长度={seq_length}, 节点数={num_nodes}")
        print(f"   📈 原始数据可用于可视化: {'是' if raw_data is not None else '否'}")
        
        return train_dataloader, val_dataloader, seq_length, num_nodes, ground_truth, raw_data


def create_probabilistic_model(num_nodes, seq_length, config):
    """从配置创建概率因果模型"""
    model_config = config['architecture']
    
    model = EnhancedProbabilisticCausalPFNO(
        n_modes=model_config['n_modes'],
        hidden_channels=model_config['hidden_channels'],
        lifting_channels=model_config['lifting_channels'],
        projection_channels=model_config['projection_channels'],
        num_nodes=num_nodes,
        sequence_length=seq_length,
        in_channels=model_config.get('in_channels', 1),
        n_layers=model_config['n_layers'],
        dropout=model_config['dropout'],
        n_attn_heads=model_config.get('n_attn_heads', 4),
        fno_output_layer_indices=model_config.get('fno_output_layer_indices')
    )
    
    print("重构版注意力因果PFNO模型已创建:")
    print(f"   - 节点数: {num_nodes}, 序列长度: {seq_length}")
    print(f"   - Lifting/Hidden/Projection 通道: {model_config['lifting_channels']}/{model_config['hidden_channels']}/{model_config['projection_channels']}")
    print(f"   - FNO模式: {model_config['n_modes']}, FNO层数: {model_config['n_layers']}")
    print(f"   - 注意力头数: {model_config.get('n_attn_heads', 4)}")
    
    return model


def train_probabilistic_model(model, train_dataloader, val_dataloader, config, results_dir):
    """训练概率模型 - 改进的训练过程，解决局部最优问题"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 🔧 改进的优化器配置 - 解决局部最优和类型冲突问题
    real_params = []
    complex_params = []
    
    for name, param in model.named_parameters():
        if param.is_complex():
            complex_params.append(param)
        else:
            real_params.append(param)
    
    # 为邻接矩阵参数设置更高的学习率，帮助跳出局部最优
    base_lr = config['training']['learning_rate']
    param_groups = []
    
    # 实数参数组：可以使用amsgrad
    if real_params:
        param_groups.append({
            'params': real_params,
            'lr': base_lr,
            'weight_decay': config['training']['weight_decay'],
            'amsgrad': False, # 诊断步骤：暂时禁用以避免与复数参数的深层冲突
            'name': 'real_params'
        })
    
    # 复数参数组：必须禁用amsgrad
    if complex_params:
        param_groups.append({
            'params': complex_params,
            'lr': base_lr,
            'weight_decay': 0, # 通常不对复数参数应用权重衰减
            'amsgrad': False, # 禁用amsgrad以避免类型冲突
            'name': 'complex_params'
        })
    
    # 如果没有分组，使用所有参数（这种情况现在不太可能发生）
    if not param_groups:
        param_groups = [{'params': model.parameters(), 'lr': base_lr}]
    
    # 使用更激进的优化器配置
    optimizer = torch.optim.AdamW(
        param_groups,
        betas=(0.9, 0.999),
        eps=1e-8
        # amsgrad 在参数组级别设置
    )
    
    # 🔧 改进的学习率调度器 - 余弦退火重启，帮助跳出局部最优
    scheduler_config = config['training'].get('scheduler', {})
    scheduler_name = scheduler_config.get('name', 'CosineAnnealingWarmRestarts') # 默认为这个

    if scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_config.get('T_0', 20),
            T_mult=scheduler_config.get('T_mult', 2),
            eta_min=scheduler_config.get('eta_min', base_lr * 0.01)
        )
        print("🔧 使用 CosineAnnealingWarmRestarts 学习率调度器")
    else:
        # 可以添加其他调度器或回退到默认
        print("⚠️ 未知的调度器名称，回退到 ReduceLROnPlateau")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    
    print(f"\n{'='*60}")
    print(f"              开始模型训练 (改进版)")
    print(f"{'='*60}")
    print(f"🔧 使用设备: {device}")
    print(f"🔧 优化器: {config['training']['optimizer']}")
    print(f"🔧 学习率: {config['training']['learning_rate']}")
    print(f"🔧 训练轮数: {config['training']['num_epochs']}")
    print(f"🔧 批量大小: {config['training']['batch_size']}")
    print(f"🔧 早停耐心值: {config['training']['patience']}")
    print(f"🔧 DAG约束: {config['causal']['use_dag_constraint']}")
    print(f"🔧 梯度裁剪: {config['training']['gradient_clipping']}")
    print(f"{'='*60}\n")
    
    # 训练循环状态
    train_losses = []
    val_losses = []
    loss_components = {
        'reconstruction': [],
        'sparsity': [],
        'dag_constraint': [],
        'uncertainty': [],
        'scale_regularization': [],
        'l1_regularization': [],        # 🔧 概率学习器L1正则化记录
        'l2_regularization': []         # 🔧 概率学习器L2正则化记录
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # 获取损失权重
    loss_weights = config['training']['loss_weights']
    # 修正：将独立的DAG权重添加到损失权重字典中，确保配置生效
    if 'causal' in config and 'causal_reg_weight' in config['causal']:
        loss_weights['dag_constraint'] = config['causal']['causal_reg_weight']
        print(f"🔧 已从配置加载DAG约束权重: {loss_weights['dag_constraint']:.4f}")
    
    # 获取动态DAG权重配置
    dynamic_dag_config = loss_weights.get('dynamic_dag_weight', {})
    dynamic_dag_enabled = dynamic_dag_config.get('enabled', False)
    if dynamic_dag_enabled:
        print(f"📈 动态DAG权重已启用: 从 epoch {dynamic_dag_config.get('start_epoch', 20)} "
              f"到 epoch {dynamic_dag_config.get('end_epoch', 45)}, "
              f"权重从 {loss_weights.get('dag_constraint', 0.001):.4f} "
              f"增加到 {dynamic_dag_config.get('end_weight', 0.1):.4f}")

    # 训练循环
    for epoch in range(config['training']['num_epochs']):
        # --- 动态DAG权重计算 ---
        current_dag_weight = loss_weights.get('dag_constraint', 0.001)
        if dynamic_dag_enabled:
            start_epoch = dynamic_dag_config.get('start_epoch', 20)
            end_epoch = dynamic_dag_config.get('end_epoch', 45)
            start_weight = loss_weights.get('dag_constraint', 0.001) # 初始权重
            end_weight = dynamic_dag_config.get('end_weight', 0.1)

            if epoch >= start_epoch:
                if epoch >= end_epoch:
                    current_dag_weight = end_weight
                else:
                    # 线性插值
                    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                    current_dag_weight = start_weight + progress * (end_weight - start_weight)
        
        # 将计算出的当前DAG权重更新到loss_weights字典中，以便传递给get_loss
        dynamic_loss_weights = loss_weights.copy()
        dynamic_loss_weights['dag_constraint'] = current_dag_weight
        
        # 训练阶段
        model.train()  # 确保模型处于训练模式
        epoch_train_loss = 0
        epoch_recon_loss = 0 # 新增：单独跟踪重构损失
        epoch_dag_loss = 0  # 单独跟踪DAG损失
        
        with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} - Training", 
                 leave=False) as train_pbar:
            for batch_idx, (batch_x, batch_y) in enumerate(train_pbar):
                try:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    # 前向传播
                    result = model(batch_x)
                    prediction_mean = result['prediction_mean']
                    prediction_log_var = result['prediction_log_var']
                    causal_matrix = result['causal_matrix']
                    causal_logits = result['causal_logits'] # 获取logits
                    
                    # 计算损失 (使用包含动态权重的字典)
                    total_loss, loss_dict = model.get_loss(
                        prediction_mean, 
                        prediction_log_var,
                        batch_y, 
                        causal_matrix, 
                        causal_logits, # 传递logits
                        loss_weights=dynamic_loss_weights
                    )
                    
                    # 累积损失组件用于分析 - 使用detach()避免计算图问题
                    for key, value in loss_dict.items():
                        if key not in loss_components:
                            loss_components[key] = []
                        # 确保将tensor转换为float存储用于记录，并detach切断梯度
                        if isinstance(value, torch.Tensor):
                            loss_components[key].append(value.detach().item())  # 添加detach()
                        else:
                            loss_components[key].append(float(value))
                    
                    # 反向传播 - 🔧 确保只调用一次backward，添加异常处理
                    optimizer.zero_grad()
                    try:
                        total_loss.backward()  # total_loss现在已经包含YAML权重
                    except RuntimeError as e:
                        if "second time" in str(e) or "already been freed" in str(e):
                            print(f"⚠️ 跳过批次 {batch_idx}: 计算图重复使用错误")
                            # 清理梯度并跳过这个批次
                            optimizer.zero_grad()
                            continue
                        else:
                            # 重新抛出其他类型的错误
                            raise e
                    
                    # 梯度裁剪 - 诊断步骤：暂时注释掉，以排查与复数参数的冲突
                    # if 'gradient_clipping' in config['training']:
                    #     torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['gradient_clipping'])
                    
                    optimizer.step()
                    
                    # 🔧 total_loss已经是YAML权重计算的结果，直接使用并detach
                    epoch_train_loss += total_loss.detach().item()  # 添加detach()
                    epoch_recon_loss += loss_dict['recon_loss'].detach().item() if isinstance(loss_dict['recon_loss'], torch.Tensor) else loss_dict['recon_loss']
                    epoch_dag_loss += loss_dict['dag_constraint'].detach().item() if isinstance(loss_dict['dag_constraint'], torch.Tensor) else loss_dict['dag_constraint']
                    
                    # 更新进度条，显示关键损失和logits统计信息
                    logits_mean = causal_logits.detach().mean().item()
                    logits_std = causal_logits.detach().std().item()
                    train_pbar.set_postfix({
                        'Loss': f'{total_loss.detach().item():.4f}',
                        'Recon': f'{loss_dict["recon_loss"].detach().item() if isinstance(loss_dict["recon_loss"], torch.Tensor) else loss_dict["recon_loss"]:.4f}',
                        'DAG': f'{loss_dict["dag_constraint"].detach().item() if isinstance(loss_dict["dag_constraint"], torch.Tensor) else loss_dict["dag_constraint"]:.4f}',
                        'Logits_Mean': f'{logits_mean:.2f}',
                        'Logits_Std': f'{logits_std:.2f}'
                    })
                    
                except Exception as e:
                    print(f"❌ 训练批次 {batch_idx} 失败: {e}")
                    continue
        
        # 平均训练损失
        avg_train_loss = epoch_train_loss / len(train_dataloader)
        avg_recon_loss = epoch_recon_loss / len(train_dataloader) # 新增：计算平均重构损失
        avg_dag_loss = epoch_dag_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        
        with torch.no_grad(), tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']} - Validation", 
                                  leave=False) as val_pbar:
            for batch_x, batch_y in val_pbar:
                try:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    # 前向传播
                    val_result = model(batch_x)
                    val_prediction_mean = val_result['prediction_mean']
                    val_prediction_log_var = val_result['prediction_log_var']
                    val_causal_matrix = val_result['causal_matrix']
                    val_causal_logits = val_result['causal_logits'] # 获取logits
                    
                    # 计算验证损失 (同样使用动态权重)
                    _, val_loss_dict = model.get_loss(
                        val_prediction_mean,
                        val_prediction_log_var,
                        batch_y, 
                        val_causal_matrix, 
                        val_causal_logits, # 传递logits
                        loss_weights=dynamic_loss_weights
                    )
                    
                    # 🔧 val_loss_dict['total']已经是YAML权重计算的结果，直接使用并detach
                    val_loss_value = val_loss_dict['total'].detach().item() if isinstance(val_loss_dict['total'], torch.Tensor) else val_loss_dict['total']
                    epoch_val_loss += val_loss_value
                    
                    val_pbar.set_postfix({'Val_Loss': f'{val_loss_value:.4f}'})
                    
                except Exception as e:
                    print(f"❌ 验证批次失败: {e}")
                    continue
        
        avg_val_loss = epoch_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # 学习率调度
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        else:
            scheduler.step() # CosineAnnealingWarmRestarts 等其他调度器

        current_lr = optimizer.param_groups[0]['lr']
        
        # 早停检查
        if avg_val_loss < best_val_loss - config['training']['min_delta']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # 保存最佳模型
            best_model_path = os.path.join(results_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'config': config
            }, best_model_path)
            
        else:
            patience_counter += 1
        
        # 改进的训练信息显示 (增加当前DAG权重)
        print(f"Epoch {epoch+1:3d}/{config['training']['num_epochs']:3d} | "
              f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
              f"Recon: {avg_recon_loss:.4f} | DAG: {avg_dag_loss:.4f} (W: {current_dag_weight:.4f}) | "
              f"LR: {current_lr:.6f} | Patience: {patience_counter}/{config['training']['patience']}")
        
        # 详细损失组件和logits统计（每5个epoch打印一次）
        if (epoch + 1) % 5 == 0:
            print("   Loss Components & Logits Stats:")
            # 创建一个新的字典，只包含有数据的损失组件
            filtered_components = {k: v for k, v in loss_components.items() if v}
            for key, values in filtered_components.items():
                if values:
                    # 使用更安全的方式获取最后一个值
                    last_value = values[-1] if values else 0.0
                    print(f"      {key}: {last_value:.6f}")
            
            # 打印最新的logits统计信息
            if 'causal_logits' in locals() and causal_logits is not None:
                logits_max = causal_logits.detach().max().item()
                logits_min = causal_logits.detach().min().item()
                print(f"      causal_logits_stats: Mean={logits_mean:.4f}, Std={logits_std:.4f}, Max={logits_max:.4f}, Min={logits_min:.4f}")

            # DAG约束状态检查
            dag_threshold = config.get('causal', {}).get('dag_loss_threshold', 0.1)
            if avg_dag_loss < dag_threshold:
                print(f"   🎯 DAG约束已收敛 (损失: {avg_dag_loss:.6f} < 阈值: {dag_threshold})")
            else:
                print(f"   ⚠️ DAG约束仍在优化 (损失: {avg_dag_loss:.6f} > 阈值: {dag_threshold})")
        
        # 早停
        if patience_counter >= config['training']['patience']:
            print(f"\n⏹️ 早停触发，训练停止 (patience: {config['training']['patience']})")
            break
        
        # 每个epoch的可视化已禁用，仅在训练完成后进行可视化
    
    print(f"\n✅ 训练完成!")
    print(f"   最佳验证损失: {best_val_loss:.6f}")
    print(f"   总训练轮数: {len(train_losses)}")
    
    # 损失记录已完成，返回训练结果
    print(f"\n✅ 训练数据已准备完毕，等待可视化处理...")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'loss_components': loss_components,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses)
    }


def enforce_dag_constraint_postprocess(prob_matrix):
    """
    后处理步骤：强制DAG约束，消除双向边
    
    规则：如果P(i→j) > P(j→i)，则保留i→j，将j→i置为0
    
    Args:
        prob_matrix: 概率矩阵 [num_nodes, num_nodes]
        
    Returns:
        dag_matrix: 强制无环的概率矩阵
        removed_edges: 被移除的边的信息
    """
    n_nodes = prob_matrix.shape[0]
    dag_matrix = prob_matrix.copy()
    removed_edges = []
    
    # 检查所有可能的双向边
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # 只检查上三角，避免重复
            prob_ij = prob_matrix[i, j]  # P(i → j)
            prob_ji = prob_matrix[j, i]  # P(j → i)
            
            # 如果两个方向都有显著概率（都大于某个小阈值）
            if prob_ij > 1e-6 and prob_ji > 1e-6:
                # 比较两个方向的概率，保留较大的，移除较小的
                if prob_ij > prob_ji:
                    # 保留 i → j，移除 j → i
                    dag_matrix[j, i] = 0
                    removed_edges.append(f"Removed edge {j}→{i} (prob={prob_ji:.6f}) due to stronger {i}→{j} (prob={prob_ij:.6f})")
                elif prob_ji > prob_ij:
                    # 保留 j → i，移除 i → j
                    dag_matrix[i, j] = 0
                    removed_edges.append(f"Removed edge {i}→{j} (prob={prob_ij:.6f}) due to stronger {j}→{i} (prob={prob_ji:.6f})")
                # 如果概率相等，保持原状（概率很小的情况下不太可能发生）
    
    return dag_matrix, removed_edges


def analyze_causal_probabilities(model, test_data, config, results_dir):
    """分析因果关系概率 - 使用蒙特卡洛Dropout估计不确定性"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()  # 启用Dropout以进行不确定性估计
    
    print("\n=== 因果关系概率分析 (蒙特卡洛 Dropout) ===")
    
    n_samples = config['evaluation'].get('eval_samples', 100)
    prob_samples = []

    with torch.no_grad():
        for _ in tqdm(range(n_samples), desc="蒙特卡洛采样", leave=False):
            result = model.forward(test_data)
            prob_samples.append(result['causal_matrix'])

    # 将样本堆叠起来并计算均值和方差
    if not prob_samples:
        raise ValueError("蒙特卡洛采样未能生成任何概率样本。")

    prob_stack = torch.stack(prob_samples, dim=0)
    
    # 计算均值作为最终概率
    causal_probs_tensor = torch.mean(prob_stack, dim=0)
    
    # 计算方差作为不确定性
    uncertainty_tensor = torch.var(prob_stack, dim=0)

    # 将模型设置回评估模式
    model.eval()

    # 转换为numpy数组 - 检查维度并正确处理
    if causal_probs_tensor.dim() == 4:
        causal_probs_tensor = causal_probs_tensor.mean(dim=1)
    
    if causal_probs_tensor.dim() == 3:
        causal_probs_raw_numpy = causal_probs_tensor[0].cpu().numpy()
        uncertainty_numpy = uncertainty_tensor[0].cpu().numpy()
    else:
        causal_probs_raw_numpy = causal_probs_tensor.cpu().numpy()
        uncertainty_numpy = uncertainty_tensor.cpu().numpy()

    print(f"\n原始概率矩阵范围: [{causal_probs_raw_numpy.min():.3f}, {causal_probs_raw_numpy.max():.3f}]")
    print(f"原始概率矩阵均值: {causal_probs_raw_numpy.mean():.3f}")
    
    # 应用后处理
    causal_probs_final = causal_probs_raw_numpy.copy()
    removed_edges_info = "未应用DAG约束后处理"
    
    if config.get('probability_scaling', {}).get('enforce_dag_constraint', False):
        print("🔧 应用DAG约束后处理...")
        causal_probs_final, removed_edges = enforce_dag_constraint_postprocess(causal_probs_raw_numpy)
        if removed_edges:
            removed_edges_info = f"移除了 {len(removed_edges)} 条冲突边"
            for edge_info in removed_edges[:5]:
                print(f"  - {edge_info}")
        else:
            removed_edges_info = "DAG约束后处理完成，未发现冲突边"
    
    # 基于最终的概率矩阵（可能经过后处理）重新计算因果图
    confidence_threshold = config['probabilistic']['confidence_threshold']
    uncertainty_threshold = config['probabilistic']['uncertainty_threshold']
    
    causal_graph_numpy = (causal_probs_final > confidence_threshold).astype(int)
    high_confidence_mask = uncertainty_numpy < uncertainty_threshold
    high_conf_graph_numpy = causal_graph_numpy * high_confidence_mask.astype(int)
    
    # 打印因果关系概率
    num_nodes = causal_probs_final.shape[0]
    print("\n最终因果关系概率矩阵:")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and causal_probs_final[i, j] > 0.01: # 只显示有一定可能性的边
                prob = causal_probs_final[i, j]
                unc = uncertainty_numpy[i, j]
                print(f"P(Node{i} → Node{j}) = {prob:.3f} ± {unc:.3f}")
    
    print(f"\n检测到的因果关系: {causal_graph_numpy.sum()} (阈值 > {confidence_threshold})")
    print(f"高置信度因果关系: {high_conf_graph_numpy.sum()} (不确定性 < {uncertainty_threshold})")
    
    # 输出概率邻接矩阵信息
    print(f"\n📊 最终概率邻接矩阵信息:")
    print(f"   - {removed_edges_info}")
    print(f"   - 最终范围: [{causal_probs_final.min():.3f}, {causal_probs_final.max():.3f}]")
    print(f"   - 最终均值: {causal_probs_final.mean():.3f}")
    
    # 🔧 保存结果 - 文件名更清晰
    pd.DataFrame(causal_probs_raw_numpy).to_csv(os.path.join(results_dir, 'causal_probabilities_raw.csv'))
    pd.DataFrame(causal_probs_final).to_csv(os.path.join(results_dir, 'causal_probabilities_final.csv'))
    pd.DataFrame(causal_graph_numpy).to_csv(os.path.join(results_dir, 'causal_graph.csv'))
    
    print("✅ 已保存原始概率矩阵到 causal_probabilities_raw.csv")
    print("✅ 已保存最终概率矩阵到 causal_probabilities_final.csv")
    
    return causal_probs_final, uncertainty_numpy, causal_graph_numpy, high_conf_graph_numpy


def evaluate_against_ground_truth(causal_probs, causal_graph, high_conf_graph, ground_truth, config, results_dir, output_json=None):
    """与真实邻接矩阵对比评估，包含AUROC和详细的逐阈值指标"""
    if ground_truth is None:
        print("警告: 没有真实邻接矩阵，跳过评估")
        if output_json:
            print("❌ 优化模式下无真实矩阵，无法评估。将返回F1=0.0和AUROC=0.5")
            metrics_to_save = {'best_f1_score': 0.0, 'auroc': 0.5, 'error': 'Ground truth not available.'}
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(metrics_to_save, f, indent=4)
        return None
    
    print("\n=== 与真实邻接矩阵对比评估（包含AUROC和详细指标） ===")
    
    # 1. 计算精确的ROC曲线和AUROC
    roc_thresholds = np.unique(causal_probs.ravel())
    if len(roc_thresholds) > 200:
        roc_thresholds = np.linspace(roc_thresholds.min(), roc_thresholds.max(), 200)
    
    all_fprs, all_tprs = [], []
    for threshold in roc_thresholds:
        pred_graph = (causal_probs > threshold).astype(int)
        metrics = evaluate_causal_discovery(pred_adj=pred_graph, true_adj=ground_truth)
        all_fprs.append(metrics['fpr'])
        all_tprs.append(metrics['tpr'])
        
    all_fprs, all_tprs = [0.0] + all_fprs + [1.0], [0.0] + all_tprs + [1.0]
    sorted_indices = np.argsort(all_fprs)
    sorted_fprs, sorted_tprs = np.array(all_fprs)[sorted_indices], np.array(all_tprs)[sorted_indices]
    roc_auc = auc(sorted_fprs, sorted_tprs)
    print(f"📈 AUROC (Area Under ROC Curve): {roc_auc:.4f}")

    # 2. 恢复对一组固定阈值的详细指标报告
    print("\n--- 详细逐阈值性能评估 ---")
    reporting_thresholds = np.arange(0.05, 1.0, 0.05)
    results = {}
    best_f1 = 0
    best_threshold_for_f1 = None
    # 新增：跟踪最佳SHD
    best_shd = float('inf')
    best_threshold_for_shd = None
    
    for threshold in reporting_thresholds:
        pred_graph = (causal_probs > threshold).astype(int)
        metrics = evaluate_causal_discovery(pred_adj=pred_graph, true_adj=ground_truth)
        results[f'threshold_{threshold:.2f}'] = metrics
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold_for_f1 = threshold
        
        # 新增：更新最佳SHD
        if metrics['shd'] < best_shd:
            best_shd = metrics['shd']
            best_threshold_for_shd = threshold
            
        print(f"Threshold {threshold:.2f}: F1={metrics['f1']:.3f}, SHD={metrics['shd']:.0f}, Precision={metrics['precision']:.3f}, "
              f"Recall={metrics['recall']:.3f}, TPR={metrics['tpr']:.3f}, "
              f"FPR={metrics['fpr']:.3f}, Edges={pred_graph.sum()}")

    # 3. 恢复对默认图和高置信度图的评估打印
    print("\n--- 默认阈值与高置信度图性能 ---")
    current_metrics = evaluate_causal_discovery(pred_adj=causal_graph, true_adj=ground_truth)
    results['current_causal_graph'] = current_metrics
    print(f"默认图 (Thr={config['probabilistic']['confidence_threshold']:.2f}): F1={current_metrics['f1']:.3f}, SHD={current_metrics['shd']:.0f}, "
          f"Precision={current_metrics['precision']:.3f}, Recall={current_metrics['recall']:.3f}")

    high_conf_metrics = evaluate_causal_discovery(pred_adj=high_conf_graph, true_adj=ground_truth)
    results['high_confidence_graph'] = high_conf_metrics
    print(f"高置信度图 (Uncertainty<{config['probabilistic']['uncertainty_threshold']:.2f}): F1={high_conf_metrics['f1']:.3f}, SHD={high_conf_metrics['shd']:.0f}, "
          f"Precision={high_conf_metrics['precision']:.3f}, Recall={high_conf_metrics['recall']:.3f}")
    
    print(f"\n🎯 最佳F1分数: {best_f1:.3f} (在阈值 = {best_threshold_for_f1:.2f} 时取得)")
    print(f"🎯 最佳SHD (越小越好): {best_shd:.0f} (在阈值 = {best_threshold_for_shd:.2f} 时取得)")
    
    # 4. 保存详细的评估报告
    report_path = os.path.join(results_dir, 'evaluation_results.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=== Probabilistic Causal Discovery Evaluation Results ===\n")
        f.write(f"AUROC: {roc_auc:.4f}\n")
        f.write(f"Best F1 Score: {best_f1:.4f} at threshold {best_threshold_for_f1:.2f}\n")
        f.write(f"Best SHD: {best_shd:.0f} at threshold {best_threshold_for_shd:.2f}\n\n")
        f.write("--- Detailed Metrics per Threshold ---\n")
        for key, metrics in results.items():
            if 'threshold' in key:
                f.write(f"{key}:\n")
                f.write(f"  F1={metrics['f1']:.4f}, SHD={metrics['shd']:.0f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, "
                        f"TPR={metrics['tpr']:.4f}, FPR={metrics['fpr']:.4f}\n")
    print(f"✅ 详细评估报告已保存: {report_path}")

    # 5. 新增：保存ROC曲线数据以供后续对比绘图
    roc_data_path = os.path.join(results_dir, 'roc_data_lag.json')
    roc_data = {
        'model_name': 'EPC-PFNO', # 或从配置中读取
        'fpr': sorted_fprs.tolist(),
        'tpr': sorted_tprs.tolist(),
        'auroc': roc_auc
    }
    with open(roc_data_path, 'w', encoding='utf-8') as f:
        json.dump(roc_data, f, indent=4)
    print(f"✅ ROC曲线数据已保存: {roc_data_path}")

    # 6. 保存关键指标用于超参数优化
    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        metrics_to_save = {
            'best_f1_score': best_f1,
            'auroc': roc_auc,
            'best_shd': best_shd,
            'best_f1_threshold': best_threshold_for_f1,
            'best_shd_threshold': best_threshold_for_shd,
            'current_graph_metrics': results.get('current_causal_graph', {}),
        }
        
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=4, cls=NpEncoder)
        print(f"✅ 优化指标已保存至: {output_json}")

    return sorted_fprs, sorted_tprs, roc_auc, results


def main():
    """主函数"""
    # 设置实验
    args, config, results_dir, actual_seed = setup_experiment()
    
    # 加载数据
    train_dataloader, val_dataloader, seq_length, num_nodes, ground_truth, raw_data = load_experimental_data(config)
    
    # 创建PFNO模型
    model = create_probabilistic_model(num_nodes, seq_length, config)
    print("创建PFNO模型完成")
    
    # 训练模型
    training_results = train_probabilistic_model(model, train_dataloader, val_dataloader, config, results_dir)
    
    # 准备测试数据
    test_data = next(iter(val_dataloader))[0][:1].to(next(model.parameters()).device)
    
    # 分析因果概率
    causal_probs, uncertainty, causal_graph, high_conf_graph = analyze_causal_probabilities(
        model, test_data, config, results_dir
    )
    
    # 使用统一可视化套件生成所有图表
    if not args.no_viz and PFNO_VIZ_AVAILABLE and PFNOVisualizationSuite is not None:
        print(f"\n📊 启动PFNO可视化套件...")
        viz_suite = PFNOVisualizationSuite(results_dir, config)
        
        # --- 调试代码：检查输入数据 ---
        print("\n" + "="*20 + " 调试信息：检查可视化输入 " + "="*20)
        print(f"   - [model] 类型: {type(model)}")
        print(f"   - [test_data] 类型: {type(test_data)}, 形状: {test_data.shape if hasattr(test_data, 'shape') else 'N/A'}")
        if raw_data is not None:
            print(f"   - [raw_data] 类型: {type(raw_data)}, 形状: {raw_data.shape if hasattr(raw_data, 'shape') else 'N/A'}")
            if hasattr(raw_data, 'dtype'):
                print(f"   - [raw_data] 数据类型: {raw_data.dtype}")
            # 检查原始数据是否有异常值
            if isinstance(raw_data, np.ndarray):
                print(f"   - [raw_data] NaN数量: {np.isnan(raw_data).sum()}, Inf数量: {np.isinf(raw_data).sum()}")
        else:
            print(f"   - [raw_data] 为 None! 这是导致时间序列可视化失败的直接原因。")
        print("="*68 + "\n")
        # --- 调试代码结束 ---

        # 运行所有可视化任务
        success_count, total_tasks = viz_suite.run_all_visualizations(
            train_losses=training_results['train_losses'],
            val_losses=training_results['val_losses'],
            loss_components=training_results['loss_components'],
            predicted_matrix=causal_probs,
            true_matrix=ground_truth,
            model=model,
            test_data=test_data,
            raw_data=raw_data
        )
        
        print(f"\n🎯 可视化任务完成: {success_count}/{total_tasks} 个任务成功")
    else:
        if args.no_viz:
            print("\n🚫 已通过命令行禁用可视化。")
        else:
            print(f"\n⚠️ 可视化套件不可用，跳过图表生成")
            print(f"💡 如需可视化功能，请检查 pfno/utils/visualization.py 模块")
    
    # 评估性能（包含AUROC和TPR/FPR）
    evaluation_outputs = evaluate_against_ground_truth(
        causal_probs, causal_graph, high_conf_graph, ground_truth, config, results_dir,
        output_json=args.output_json  # 传递output_json路径
    )
    
    # 如果评估成功，则进行可视化
    if not args.no_viz and evaluation_outputs:
        sorted_fprs, sorted_tprs, roc_auc, _ = evaluation_outputs
        # 为我们的模型命名，以在图例中显示
        model_name = "EPC-PFNO"
        plot_roc_curve(sorted_fprs, sorted_tprs, roc_auc, model_name, results_dir)
    
    print(f"\n=== 实验完成 ===")
    print(f"结果保存在: {results_dir}")
    print("主要输出文件:")
    print("📊 评估结果:")
    print("- evaluation_results.txt: 评估结果（包含TPR和FPR指标）")
    print("- causal_probabilities_raw.csv: 原始因果关系概率矩阵")
    print("- causal_probabilities_final.csv: 缩放后的因果关系概率矩阵")
    print("- causal_graph.csv: 二值化因果图")
    print("📈 可视化图表（由PFNO可视化套件生成）:")
    if PFNO_VIZ_AVAILABLE and PFNOVisualizationSuite is not None:
        print("- training_curves_final.png: 训练过程曲线（包含平滑处理）")
        print("- adjacency_matrix_comparison.png: 邻接矩阵对比图")
        print("- time_series_reconstruction_detailed.png: 单节点详细重构分析")
        print("- time_series_reconstruction_overview.png: 多节点重构概览")
        print("- time_series_reconstruction_grid.png: 网格化重构分析")
        print("- reconstruction_statistics.txt: 时间序列重构性能统计")
    
    # 添加ROC曲线到输出文件列表
    print("- roc_curve.png: 新增的ROC曲线图")
    
    print("⚙️ 配置文件:")
    print("- experiment_config.json: 实验配置文件")


if __name__ == "__main__":
    main()