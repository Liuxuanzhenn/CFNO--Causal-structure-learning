import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 设置matplotlib支持中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
from .model import PFNO_Causal
from .al_trainer import ALTrainer
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CONFIG  # 从config.py导入配置
import datetime
from sklearn.preprocessing import StandardScaler
import pickle
from torch.utils.data import Dataset
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, confusion_matrix
import networkx as nx
import argparse
import random
import time

# 从utils.py中导入所有需要的函数
from utils import (
    load_data, 
    initialize_model, 
    setup_trainer, 
    train_model, 
    validate_model, 
    plot_losses, 
    save_model, 
    evaluate_causal_discovery,
    granger_causality_matrix,
    initialize_adjacency_matrix,
    correlation_matrix,
    TimeSeriesAugmentation,
    plot_adjacency_matrix,
    plot_adjacency_matrices_comparison,
    optimize_adjacency_matrix_bayesian,
    process_weighted_adjacency_matrix,
    apply_threshold_to_adjacency_matrix,
    set_seed  # 从utils导入种子设置函数
)

# 设置随机种子
def setup_random_seed():
    """设置随机种子，如果配置为使用固定种子则使用指定的种子，否则生成一个随机种子"""
    seed_config = CONFIG.get('random_seed', {})
    use_fixed_seed = seed_config.get('use_fixed_seed', False)
    
    if use_fixed_seed:
        # 使用固定种子
        seed = seed_config.get('fixed_seed', 42)
        print(f"使用固定随机种子: {seed}")
    else:
        # 生成随机种子 (使用当前时间的毫秒部分确保随机性)
        current_time_ms = int(time.time() * 1000) % 100000
        seed = random.randint(0, 100000) + current_time_ms
        print(f"生成随机种子: {seed}")
    
    # 设置种子
    set_seed(seed)
    
    # 如果配置为保存种子，则将种子写入文件
    if seed_config.get('save_seed', False):
        seed_file = seed_config.get('seed_file', 'last_seed.txt')
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(seed_file, 'w') as f:
            f.write(f"Seed: {seed}\n")
            f.write(f"Timestamp: {timestamp}\n")
        print(f"随机种子已保存至 {seed_file}")
    
    return seed

# main.py中特有的函数，这些函数在utils.py中没有定义

def plot_comparison_adjacency_matrix(pred_matrix, ground_truth_matrix, threshold=None, title="Comparison of Adjacency Matrices", save_path=None):
    """
    将预测的邻接矩阵与真实邻接矩阵进行对比可视化
    
    Args:
        pred_matrix: 预测的邻接矩阵
        ground_truth_matrix: 真实的邻接矩阵
        threshold: 应用于预测矩阵的阈值，如果为None则不应用阈值
        title: 图表标题
        save_path: 保存路径，如果为None则显示图形
    """
    if isinstance(pred_matrix, torch.Tensor):
        pred_matrix = pred_matrix.detach().cpu().numpy()
    
    if isinstance(ground_truth_matrix, torch.Tensor):
        ground_truth_matrix = ground_truth_matrix.detach().cpu().numpy()
    
    # 如果提供了阈值，应用到预测矩阵
    if threshold is not None:
        pred_binary = (pred_matrix > threshold).astype(float)
    else:
        pred_binary = pred_matrix
    
    # 确保ground_truth_matrix是二值矩阵
    if np.max(ground_truth_matrix) > 1:
        ground_truth_binary = (ground_truth_matrix > 0).astype(float)
    else:
        ground_truth_binary = ground_truth_matrix
    
    # 计算尺寸和边框间距
    n = pred_matrix.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 绘制预测矩阵
    im0 = axes[0].imshow(pred_matrix, cmap='viridis', vmin=0, vmax=1)
    axes[0].set_title('Weighted Adjacency Matrix')
    plt.colorbar(im0, ax=axes[0])
    
    # 绘制二值化预测矩阵
    im1 = axes[1].imshow(pred_binary, cmap='Blues', vmin=0, vmax=1)
    threshold_info = f" (Threshold = {threshold:.2f})" if threshold is not None else ""
    axes[1].set_title(f'Binary Adjacency Matrix{threshold_info}')
    plt.colorbar(im1, ax=axes[1])
    
    # 绘制真实邻接矩阵
    im2 = axes[2].imshow(ground_truth_binary, cmap='Oranges', vmin=0, vmax=1)
    axes[2].set_title('Ground Truth Adjacency Matrix')
    plt.colorbar(im2, ax=axes[2])
    
    # 添加标签
    for ax in axes:
        ax.set_xlabel('Target Node')
        ax.set_ylabel('Source Node')
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(np.arange(1, n+1))
        ax.set_yticklabels(np.arange(1, n+1))
    
    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def find_optimal_threshold(adj_matrix, ground_truth_file, thresholds=None, save_path=None):
    """
    找到最优阈值，使二值化后的邻接矩阵与真实邻接矩阵的F1分数最高。
    
    Args:
        adj_matrix: 预测的加权邻接矩阵
        ground_truth_file: 真实邻接矩阵的CSV文件路径
        thresholds: 要测试的阈值列表，默认为0.1到0.9的10个值
        save_path: 绘图保存路径，如果为None则显示图形
    
    Returns:
        optimal_threshold: 最优阈值
        max_f1: 最高F1分数
        threshold_results: 所有阈值的评估结果
    """
    # 加载真实邻接矩阵
    try:
        true_adj = pd.read_csv(ground_truth_file, header=0, index_col=0).values
        if true_adj.shape != adj_matrix.shape:
            print(f"警告: 真实邻接矩阵形状 {true_adj.shape} 与预测矩阵形状 {adj_matrix.shape} 不匹配")
            # 尝试调整大小
            if len(true_adj) > len(adj_matrix):
                true_adj = true_adj[:len(adj_matrix), :len(adj_matrix)]
            elif len(true_adj) < len(adj_matrix):
                temp = np.zeros(adj_matrix.shape)
                temp[:len(true_adj), :len(true_adj)] = true_adj
                true_adj = temp
    except Exception as e:
        print(f"警告: 无法加载真实邻接矩阵: {e}")
        print("将使用全零矩阵进行比较")
        true_adj = np.zeros_like(adj_matrix)
    
    # 确保邻接矩阵是numpy数组
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # 创建默认阈值列表如果未提供
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 10)
    
    # 存储结果
    results = []
    
    # 对每个阈值进行测试
    for threshold in thresholds:
        # 应用阈值
        pred_adj_binary = (adj_matrix > threshold).astype(int)
        
        # 计算指标
        # 将邻接矩阵展平为一维数组
        true_flat = true_adj.flatten()
        pred_flat = pred_adj_binary.flatten()
        
        # 如果存在diag掩码，忽略对角线元素
        n = true_adj.shape[0]
        mask = ~np.eye(n, dtype=bool).flatten()
        true_flat = true_flat[mask]
        pred_flat = pred_flat[mask]
        
        # 计算指标
        accuracy = accuracy_score(true_flat, pred_flat)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_flat, pred_flat, average='binary', zero_division=0
        )
        
        # 计算TPR (True Positive Rate) 和 FPR (False Positive Rate)
        tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 计算有向图的结构汉明距离(SHD)
        shd = np.sum(np.abs(true_adj - pred_adj_binary))
        
        results.append({
            'threshold': threshold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'tpr': tpr,  # 添加TPR
            'fpr': fpr,  # 添加FPR
            'f1': f1,
            'shd': shd
        })
    
    # 找到最优F1分数的阈值
    results_df = pd.DataFrame(results)
    max_f1_idx = results_df['f1'].idxmax()
    optimal_result = results_df.iloc[max_f1_idx]
    optimal_threshold = optimal_result['threshold']
    max_f1 = optimal_result['f1']
    
    print(f"最优阈值: {optimal_threshold:.3f}, F1分数: {max_f1:.3f}")
    print(f"- 在此阈值下: 准确率={optimal_result['accuracy']:.3f}, 精确率={optimal_result['precision']:.3f}, 召回率={optimal_result['recall']:.3f}")
    print(f"- TPR={optimal_result['tpr']:.3f}, FPR={optimal_result['fpr']:.3f}, SHD={optimal_result['shd']}")
    
    # 可视化结果
    plt.figure(figsize=(10, 6))
    
    plt.subplot(2, 1, 1)
    plt.plot(results_df['threshold'], results_df['accuracy'], 'o-', label='Accuracy')
    plt.plot(results_df['threshold'], results_df['precision'], 's-', label='Precision')
    plt.plot(results_df['threshold'], results_df['recall'], '^-', label='Recall')
    plt.plot(results_df['threshold'], results_df['f1'], 'D-', label='F1', linewidth=2)
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', alpha=0.5, label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.axhline(y=max_f1, color='r', linestyle='--', alpha=0.5, label=f'Max F1: {max_f1:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('Metric Value')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(results_df['threshold'], results_df['shd'], 'o-', label='SHD', color='purple')
    plt.plot(results_df['threshold'], results_df['tpr'], 's-', label='TPR', color='green')
    plt.plot(results_df['threshold'], results_df['fpr'], '^-', label='FPR', color='red')
    plt.axvline(x=optimal_threshold, color='k', linestyle='--', alpha=0.5, label=f'Optimal Threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Value')
    plt.title('SHD, TPR & FPR vs. Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return optimal_threshold, max_f1, results

def evaluate_at_threshold(adj_matrix, ground_truth_matrix, threshold=0.3):
    """
    在指定阈值下评估因果发现性能
    
    Args:
        adj_matrix: 预测的加权邻接矩阵
        ground_truth_matrix: 真实的邻接矩阵
        threshold: 应用于预测矩阵的阈值
    
    Returns:
        metrics: 包含性能指标的字典
    """
    # 确保输入是numpy数组
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    if isinstance(ground_truth_matrix, torch.Tensor):
        ground_truth_matrix = ground_truth_matrix.detach().cpu().numpy()
    
    # 应用阈值
    pred_adj_binary = (adj_matrix > threshold).astype(int)
    
    # 处理真实矩阵如果不是二值的话
    if np.max(ground_truth_matrix) > 1:
        true_adj_binary = (ground_truth_matrix > 0).astype(int)
    else:
        true_adj_binary = ground_truth_matrix
    
    # 将邻接矩阵展平为一维数组，排除对角线元素
    n = true_adj_binary.shape[0]
    mask = ~np.eye(n, dtype=bool).flatten()
    true_flat = true_adj_binary.flatten()[mask]
    pred_flat = pred_adj_binary.flatten()[mask]
    
    # 计算指标
    accuracy = accuracy_score(true_flat, pred_flat)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_flat, pred_flat, average='binary', zero_division=0
    )
    
    # 计算TPR (True Positive Rate) 和 FPR (False Positive Rate)
    tn, fp, fn, tp = confusion_matrix(true_flat, pred_flat).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # 计算AUROC，如果可能的话
    auroc = -1  # 默认值，表示无法计算
    if len(np.unique(true_flat)) > 1:
        # 只有当真实标签包含正例和负例时才能计算AUROC
        try:
            # 对展平后的权重矩阵使用ROC
            pred_scores = adj_matrix.flatten()[mask]
            auroc = roc_auc_score(true_flat, pred_scores)
        except Exception as e:
            print(f"无法计算AUROC: {e}")
    
    # 计算有向图的结构汉明距离(SHD)
    shd = np.sum(np.abs(true_adj_binary - pred_adj_binary))
    
    # 返回指标
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'tpr': tpr,  # 添加TPR
        'fpr': fpr,  # 添加FPR
        'f1': f1,
        'auroc': auroc,
        'shd': shd
    }
    
    return metrics

def main():
    """
    主函数，实现整个流程的执行
    """
    # 设置随机种子
    current_seed = setup_random_seed()
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='训练因果发现模型')
    parser.add_argument('--seed', type=int, default=None, help='要使用的数据的随机种子')
    args = parser.parse_args()

    # 使用提供的种子确定输入文件名，如果未提供则使用默认文件
    seed_suffix = f"_{args.seed}" if args.seed is not None else "2"
    input_file = f"data/generated_time_series{seed_suffix}.csv"
    ground_truth_file = f"data/causal_adjacency_matrix{seed_suffix}.csv"
    
    # 更新配置中的文件路径
    CONFIG['data']['file_path'] = input_file
    
    # 设置结果目录
    base_results_dir = CONFIG['output']['results_dir']
    os.makedirs(base_results_dir, exist_ok=True)
    
    # 记录实验时间和种子信息
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = CONFIG['output']['model_name']
    seed_info = f"_seed{args.seed}" if args.seed is not None else ""
    experiment_name = f"{model_name}{seed_info}_{timestamp}_rseed{current_seed}"  # 添加随机种子信息
    
    # 为本次实验创建独立子目录
    results_dir = os.path.join(base_results_dir, f"{model_name}{seed_info}_{timestamp}_rseed{current_seed}")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"实验名称: {experiment_name}")
    print(f"使用数据文件: {input_file}")
    print(f"结果将保存至: {results_dir}")
    print("-" * 50)
    
    # 创建实验信息文件
    with open(os.path.join(results_dir, "info.txt"), "w", encoding="utf-8") as f:
        f.write(f"实验名称: {experiment_name}\n")
        f.write(f"实验时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"数据文件: {input_file}\n")
        f.write(f"使用的数据种子: {args.seed if args.seed is not None else '默认(2)'}\n")
        f.write(f"训练随机种子: {current_seed}\n")  # 添加训练随机种子信息
        f.write(f"模型配置:\n")
        f.write(f"  - 隐藏层通道数: {CONFIG['model']['hidden_channels']}\n")
        f.write(f"  - 层数: {CONFIG['model']['n_layers']}\n")
        f.write(f"  - Dropout率: {CONFIG['model']['dropout']}\n")
        f.write(f"  - 因果正则化权重: {CONFIG['model']['causal_reg_weight']}\n")
        f.write(f"训练配置:\n")
        f.write(f"  - 批量大小: {CONFIG['training']['batch_size']}\n")
        f.write(f"  - 序列长度: {CONFIG['training']['sequence_length']}\n")
        f.write(f"  - 学习率: {CONFIG['training']['learning_rate']}\n")
        f.write(f"  - 训练轮数: {CONFIG['training']['num_epochs']}\n")
        f.write(f"  - 早停耐心值: {CONFIG['training']['patience']}\n")
        f.write(f"随机种子配置:\n")  # 添加随机种子配置信息
        f.write(f"  - 固定种子模式: {CONFIG['random_seed']['use_fixed_seed']}\n")
        if CONFIG['random_seed']['use_fixed_seed']:
            f.write(f"  - 固定种子值: {CONFIG['random_seed']['fixed_seed']}\n")
        f.write(f"  - 种子保存文件: {CONFIG['random_seed']['seed_file']}\n")
    
    # 加载数据
    file_path = CONFIG['data']['file_path']
    train_ratio = CONFIG['data']['train_ratio']
    max_time_steps = CONFIG['data']['max_time_steps']
    normalize = CONFIG['data']['normalize']
    augment = CONFIG['data']['augment']
    
    sequence_length = CONFIG['training']['sequence_length']
    batch_size = CONFIG['training']['batch_size']
    
    train_dataloader, val_dataloader, seq_length, num_nodes, scalers, raw_data = load_data(
        file_path=file_path,
        train_ratio=train_ratio,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_time_steps=max_time_steps,
        normalize=normalize,
        augment=augment
    )
    
    print(f"数据加载完成: 序列长度 = {seq_length}, 节点数 = {num_nodes}")
    
    # 如果启用外部邻接矩阵，则生成外部矩阵
    external_adj = None
    use_external_adj = CONFIG['model']['use_external_adj']
    
    if use_external_adj:
        print("生成外部邻接矩阵...")
        # 使用格兰杰因果检验生成外部矩阵
        if CONFIG['model']['external_adj_source'] == 'granger':
            # 使用新的配置结构
            granger_config = CONFIG['model']['granger']
            max_lag = granger_config['max_lag']
            significance = granger_config['significance']
            
            print(f"使用格兰杰因果检验 (max_lag={max_lag}, significance={significance})...")
            external_adj, pvalues = granger_causality_matrix(
                raw_data, 
                max_lag=max_lag,
                significance=significance,
                verbose=True
            )
            print(f"格兰杰邻接矩阵形状: {external_adj.shape}")
            
            # 是否归一化格兰杰检验结果
            if granger_config['normalize'] and np.sum(external_adj) > 0:
                # 归一化邻接矩阵
                print("归一化格兰杰因果检验结果...")
                row_sums = external_adj.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0] = 1  # 避免除零错误
                external_adj = external_adj / row_sums
                print(f"归一化后的最大值: {external_adj.max()}, 最小值: {external_adj.min()}")
            
            # 保存格兰杰因果检验结果
            np.savetxt(os.path.join(results_dir, "granger_adj_matrix.csv"), external_adj, delimiter=",")
            np.savetxt(os.path.join(results_dir, "granger_pvalues.csv"), pvalues, delimiter=",")
            print(f"格兰杰因果检验结果已保存到 {results_dir}")
            
        elif CONFIG['model']['external_adj_source'] == 'manual':
            # 从指定文件加载手动定义的邻接矩阵
            manual_path = CONFIG['model']['manual_adj_path']
            if os.path.exists(manual_path):
                try:
                    external_adj = pd.read_csv(manual_path, header=0, index_col=0).values
                    print(f"已从{manual_path}加载手动定义的外部邻接矩阵，形状: {external_adj.shape}")
                    
                    # 确保矩阵尺寸与节点数匹配
                    if external_adj.shape[0] != num_nodes or external_adj.shape[1] != num_nodes:
                        print(f"警告：加载的矩阵形状{external_adj.shape}与节点数{num_nodes}不匹配，将进行调整")
                        # 如果矩阵尺寸小于节点数，用零填充
                        if external_adj.shape[0] < num_nodes or external_adj.shape[1] < num_nodes:
                            temp = np.zeros((num_nodes, num_nodes))
                            temp[:external_adj.shape[0], :external_adj.shape[1]] = external_adj
                            external_adj = temp
                            print(f"已将矩阵填充至{external_adj.shape}")
                        # 如果矩阵尺寸大于节点数，裁剪
                        else:
                            external_adj = external_adj[:num_nodes, :num_nodes]
                            print(f"已将矩阵裁剪至{external_adj.shape}")
                except Exception as e:
                    print(f"加载手动邻接矩阵失败: {str(e)}，将使用零矩阵代替")
                    external_adj = np.zeros((num_nodes, num_nodes))
            else:
                print(f"未找到指定的外部邻接矩阵文件: {manual_path}，将使用零矩阵代替")
                external_adj = np.zeros((num_nodes, num_nodes))
        else:
            # 使用其他方式初始化外部邻接矩阵（如correlation等）
            external_adj = initialize_adjacency_matrix(raw_data, CONFIG, internal=False)
        
        # 保存外部邻接矩阵供参考
        external_adj_path = os.path.join(results_dir, "external_adjacency_matrix.csv")
        pd.DataFrame(external_adj).to_csv(external_adj_path, index=False)
        print(f"外部邻接矩阵已保存到 {external_adj_path}")
        
        # 创建外部邻接矩阵的可视化热图
        external_adj_plot_path = os.path.join(results_dir, "external_adjacency_matrix.png")
        plot_adjacency_matrix(
            adjacency_matrix=external_adj, 
            title="External Adjacency Matrix from Granger Causality", 
            save_path=external_adj_plot_path
        )
        print(f"外部邻接矩阵热图已保存到 {external_adj_plot_path}")
        
        # 保存最终使用的外部邻接矩阵（可能经过处理）
        final_ext_adj_path = os.path.join(results_dir, "final_ext_adj.csv")
        pd.DataFrame(external_adj).to_csv(final_ext_adj_path, index=False)
    else:
        external_adj = None
        print("不使用外部邻接矩阵")
    
    # 初始化内部邻接矩阵
    internal_adj = initialize_adjacency_matrix(raw_data, CONFIG, internal=True)
    
    # 初始化模型
    model = initialize_model(
        num_nodes=num_nodes,
        sequence_length=seq_length,
        n_modes=[16, 16],
        hidden_channels=CONFIG['model']['hidden_channels'],
        n_layers=CONFIG['model']['n_layers'],
        dropout=CONFIG['model']['dropout'],
        mlp_dropout=CONFIG['model']['mlp_dropout'],
        use_diagonal_mask=CONFIG['model']['use_diagonal_mask'],
        use_dag_constraint=CONFIG['model']['use_dag_constraint'],
        causal_reg_weight=CONFIG['model']['causal_reg_weight']
    )
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 计算样本数和特征维度
    n = len(train_dataloader.dataset)
    d = num_nodes
    
    # 设置ALTrainer
    trainer = setup_trainer(
        n=n,
        d=d,
        model=model,
        lr=CONFIG['training']['learning_rate'],
        init_iter=CONFIG['training']['init_iter'],
        early_stopping=True,
        early_stopping_thresh=CONFIG['training']['early_stopping_thresh'],
        patience=CONFIG['training']['patience'],
        seed=current_seed,
        device=device,
        sequence_length=seq_length,
        dag_constraint=CONFIG['model']['use_dag_constraint'],
        use_external_adj=use_external_adj,
        external_adj=external_adj,
        initial_external_weight=CONFIG['model']['initial_external_weight'],
        final_external_weight=CONFIG['model']['final_external_weight'],
        weight_transition_epochs=CONFIG['model']['weight_transition_epochs']
    )
    
    # 打印使用外部邻接矩阵的信息
    if use_external_adj and external_adj is not None:
        print(f"已设置外部邻接矩阵，初始权重: {trainer.initial_external_weight}, 最终权重: {trainer.final_external_weight}")
    else:
        print("仅使用内部邻接矩阵")
    
    # 训练模型
    num_epochs = CONFIG['training']['num_epochs']
    print(f"开始训练，轮数: {num_epochs}")
    
    train_losses, val_losses = train_model(
        trainer=trainer,
        train_dataloader=train_dataloader,
        num_epochs=num_epochs,
        validation_dataloader=val_dataloader
    )
    
    # 绘制损失曲线
    loss_plot_path = os.path.join(results_dir, "loss_curves.png")
    plot_losses(train_losses, val_losses, save_path=loss_plot_path)
    print(f"损失曲线已保存到 {loss_plot_path}")
    
    # 获取最终邻接矩阵
    final_adj = trainer.get_current_adj()
    if isinstance(final_adj, torch.Tensor):
        final_adj = final_adj.cpu().detach().numpy()
    
    # 确保邻接矩阵是2D的
    if len(final_adj.shape) > 2:
        # 如果是3D张量(batch, nodes, nodes)，取第一个元素
        final_adj = final_adj[0]
    
    # 1. 保存原始加权邻接矩阵（ADAMW优化结果）
    raw_adj_path = os.path.join(results_dir, "raw_adjacency_matrix.csv")
    pd.DataFrame(final_adj).to_csv(raw_adj_path, index=False)
    print(f"原始加权邻接矩阵已保存到 {raw_adj_path}")
    
    # 2. 应用贝叶斯优化，只在配置中启用时执行
    if CONFIG['bayesian_optimization'].get('enabled', True):  # 默认为True以兼容旧配置
        print("正在应用贝叶斯优化处理邻接矩阵...")
        try:
            bayesian_adj = optimize_adjacency_matrix_bayesian(
                model=trainer.model,
                dataloader=train_dataloader,
                initial_adj=torch.tensor(final_adj).to(device),
                device=device,
                n_iterations=CONFIG['eval'].get('bayesian_iterations', 25),
                init_points=CONFIG['bayesian_optimization'].get('init_points', 5)
            )
            
            # 转换为NumPy数组
            if isinstance(bayesian_adj, torch.Tensor):
                bayesian_adj = bayesian_adj.cpu().detach().numpy()
                
            # 保存贝叶斯优化后的邻接矩阵
            bayesian_adj_path = os.path.join(results_dir, "bayesian_adjacency_matrix.csv")
            pd.DataFrame(bayesian_adj).to_csv(bayesian_adj_path, index=False)
            print(f"贝叶斯优化后的邻接矩阵已保存到 {bayesian_adj_path}")
        except Exception as e:
            print(f"贝叶斯优化过程中出错: {e}")
            print("使用原始加权邻接矩阵继续...")
            bayesian_adj = final_adj
    else:
        print("贝叶斯优化已在配置中禁用，使用原始邻接矩阵继续...")
        bayesian_adj = final_adj
    
    # 3. 应用Wij和Wji的值比较处理
    print("正在处理Wij和Wji的值比较...")
    try:
        processed_adj = process_weighted_adjacency_matrix(bayesian_adj)
        
        # 保存处理后的邻接矩阵
        processed_adj_path = os.path.join(results_dir, "processed_adjacency_matrix.csv")
        pd.DataFrame(processed_adj).to_csv(processed_adj_path, index=False)
        print(f"处理Wij和Wji值比较后的邻接矩阵已保存到 {processed_adj_path}")
    except Exception as e:
        print(f"处理Wij和Wji值比较过程中出错: {e}")
        print("使用贝叶斯优化后的邻接矩阵继续...")
        processed_adj = bayesian_adj
    
    # 最终结果使用处理后的邻接矩阵
    final_adj = processed_adj
    
    # 保存最终邻接矩阵
    adj_path = os.path.join(results_dir, "adjacency_matrix.csv")
    pd.DataFrame(final_adj).to_csv(adj_path, index=False)
    print(f"最终加权邻接矩阵已保存到 {adj_path}")
    
    # 保存模型
    model_save_path = os.path.join(results_dir, "model.pth")
    metadata = {
        'num_nodes': num_nodes,
        'sequence_length': seq_length,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': CONFIG,
        'adjacency_matrix': final_adj,
        'experiment_name': experiment_name,
        'timestamp': timestamp
    }
    save_model(trainer.model, optimizer=trainer.optimizer, path=model_save_path, metadata=metadata)
    print(f"模型已保存到 {model_save_path}")
    
    # 在项目根目录保存一个副本，方便其他脚本使用
    root_model_path = "model.pth"
    save_model(trainer.model, optimizer=trainer.optimizer, path=root_model_path, metadata=metadata)
    print(f"模型副本已保存到项目根目录 {root_model_path}")
    
    # 可视化邻接矩阵
    adj_plot_path = os.path.join(results_dir, "adjacency_matrix.png")
    plot_adjacency_matrix(final_adj, title="Learned Adjacency Matrix", save_path=adj_plot_path)
    
    # 阈值搜索和评估
    if os.path.exists(ground_truth_file):
        # 加载真实邻接矩阵，处理第一行和第一列的索引
        true_adj = pd.read_csv(ground_truth_file, index_col=0, header=0).values.astype(float)
        
        # 比较可视化
        comparison_path = os.path.join(results_dir, "adjacency_comparison.png")
        plot_comparison_adjacency_matrix(
            pred_matrix=final_adj,
            ground_truth_matrix=true_adj,
            threshold=CONFIG['eval']['default_threshold'],
            title="Comparison of Predicted vs True Adjacency",
            save_path=comparison_path
        )
        
        # 固定阈值评估
        default_threshold = CONFIG['eval']['default_threshold']
        default_metrics = evaluate_at_threshold(final_adj, true_adj, threshold=default_threshold)
        
        print(f"\n固定阈值评估结果 (阈值 = {default_threshold}):")
        print(f"  - F1分数: {default_metrics['f1']:.4f}")
        print(f"  - 准确率: {default_metrics['accuracy']:.4f}")
        print(f"  - 精确率: {default_metrics['precision']:.4f}")
        print(f"  - 召回率: {default_metrics['recall']:.4f} (TPR: {default_metrics['tpr']:.4f})")
        print(f"  - FPR: {default_metrics['fpr']:.4f}")
        print(f"  - SHD: {default_metrics['shd']}")
        
        # 4. 最优阈值搜索和二值化处理
        print("\n进行阈值搜索和二值化处理...")
        thresholds = CONFIG['eval']['thresholds']
        threshold_path = os.path.join(results_dir, "threshold_search.png")
        optimal_threshold, max_f1, threshold_results = find_optimal_threshold(
            adj_matrix=final_adj,
            ground_truth_file=ground_truth_file,
            thresholds=thresholds,
            save_path=threshold_path
        )
        
        # 使用最优阈值生成二值邻接矩阵
        binary_adj = apply_threshold_to_adjacency_matrix(final_adj, threshold=optimal_threshold)
        binary_adj_path = os.path.join(results_dir, "binary_adjacency_matrix.csv")
        pd.DataFrame(binary_adj).to_csv(binary_adj_path, index=False)
        print(f"二值化邻接矩阵已保存到 {binary_adj_path}")
        
        # 创建二值化邻接矩阵可视化
        binary_plot_path = os.path.join(results_dir, "binary_adjacency_matrix.png")
        plot_adjacency_matrix(
            adjacency_matrix=binary_adj,
            title=f"Binary Adjacency Matrix (threshold={optimal_threshold:.3f})",
            save_path=binary_plot_path
        )
        
        # 最优阈值下的评估结果
        optimal_metrics = evaluate_at_threshold(final_adj, true_adj, threshold=optimal_threshold)
        
        # 输出结果摘要
        print("\n=== 最终评估结果 ===")
        print(f"最优阈值: {optimal_threshold:.4f}")
        print(f"在最优阈值下:")
        print(f"  - F1分数: {max_f1:.4f}")
        print(f"  - 准确率: {optimal_metrics['accuracy']:.4f}")
        print(f"  - 精确率: {optimal_metrics['precision']:.4f}")
        print(f"  - 召回率: {optimal_metrics['recall']:.4f} (TPR: {optimal_metrics['tpr']:.4f})")
        print(f"  - FPR: {optimal_metrics['fpr']:.4f}")
        # 修复AUROC格式化
        auroc_str = f"{optimal_metrics['auroc']:.4f}" if optimal_metrics['auroc'] >= 0 else "N/A"
        print(f"  - AUROC: {auroc_str}")
        print(f"  - SHD: {optimal_metrics['shd']}")
        
        # 保存结果摘要到文本文件
        summary_path = os.path.join(results_dir, "evaluation_summary.txt")
        with open(summary_path, "w") as f:
            f.write(f"实验名称: {experiment_name}\n")
            f.write(f"时间戳: {timestamp}\n\n")
            
            f.write(f"固定阈值评估结果 (阈值 = {default_threshold}):\n")
            f.write(f"  - F1分数: {default_metrics['f1']:.4f}\n")
            f.write(f"  - 准确率: {default_metrics['accuracy']:.4f}\n")
            f.write(f"  - 精确率: {default_metrics['precision']:.4f}\n")
            f.write(f"  - 召回率: {default_metrics['recall']:.4f} (TPR: {default_metrics['tpr']:.4f})\n")
            f.write(f"  - FPR: {default_metrics['fpr']:.4f}\n")
            f.write(f"  - AUROC: {auroc_str}\n")
            f.write(f"  - SHD: {default_metrics['shd']}\n\n")
            
            f.write(f"最优阈值: {optimal_threshold:.4f}\n")
            f.write(f"最高F1分数: {max_f1:.4f}\n")
            f.write(f"在最优阈值下的指标:\n")
            f.write(f"  - 准确率: {optimal_metrics['accuracy']:.4f}\n")
            f.write(f"  - 精确率: {optimal_metrics['precision']:.4f}\n")
            f.write(f"  - 召回率: {optimal_metrics['recall']:.4f} (TPR: {optimal_metrics['tpr']:.4f})\n")
            f.write(f"  - FPR: {optimal_metrics['fpr']:.4f}\n")
            f.write(f"  - AUROC: {auroc_str}\n")
            f.write(f"  - SHD: {optimal_metrics['shd']}\n")
        
        print(f"评估摘要已保存到 {summary_path}")
    else:
        print(f"警告: 未找到真实邻接矩阵文件 {ground_truth_file}，跳过评估步骤")
    
    print("\n训练和评估完成！")


# 执行主函数
if __name__ == "__main__":
    main() 