"""
可视化工具模块
提供PFNO实验结果的可视化功能
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import warnings
import torch
import os
import time

# 设置matplotlib中文支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")


def plot_adjacency_matrix(adj_matrix: np.ndarray, 
                         title: str = "邻接矩阵",
                         save_path: Optional[str] = None,
                         **kwargs) -> None:
    """
    绘制邻接矩阵热力图
    
    Args:
        adj_matrix: 邻接矩阵
        title: 图表标题
        save_path: 保存路径
        **kwargs: 其他绘图参数
    """
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
    
    # 绘制热力图
    im = ax.imshow(adj_matrix, cmap=kwargs.get('cmap', 'RdBu_r'), 
                   vmin=kwargs.get('vmin', 0), vmax=kwargs.get('vmax', 1))
    
    # 设置标题和标签
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('目标节点', fontsize=12)
    ax.set_ylabel('源节点', fontsize=12)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('连接强度', fontsize=10)
    
    # 添加数值标注（如果矩阵不太大）
    if adj_matrix.shape[0] <= 10 and kwargs.get('annotate', True):
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                text = ax.text(j, i, f'{adj_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.show()


def plot_training_curves(training_history: Dict[str, List], 
                        save_path: Optional[str] = None,
                        **kwargs) -> None:
    """
    绘制训练曲线
    
    Args:
        training_history: 训练历史字典
        save_path: 保存路径
        **kwargs: 其他绘图参数
    """
    fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (15, 10)))
    
    # 主损失曲线
    if 'train_losses' in training_history and 'val_losses' in training_history:
        axes[0, 0].plot(training_history['train_losses'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(training_history['val_losses'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 损失组件
    if 'loss_components' in training_history:
        for key, values in training_history['loss_components'].items():
            if values and any(v > 0 for v in values):
                axes[0, 1].plot(values, label=key, alpha=0.8)
        axes[0, 1].set_title('Loss Components')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss Value')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 学习率曲线
    if 'learning_rates' in training_history:
        axes[1, 0].plot(training_history['learning_rates'], alpha=0.8)
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 性能指标
    if 'metrics' in training_history:
        metrics = training_history['metrics']
        for metric_name, values in metrics.items():
            if isinstance(values, list) and values:
                axes[1, 1].plot(values, label=metric_name, alpha=0.8)
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to: {save_path}")
    
    plt.show()


def plot_evaluation_metrics(metrics: Dict[str, float],
                           save_path: Optional[str] = None,
                           **kwargs) -> None:
    """
    绘制评估指标条形图
    
    Args:
        metrics: 评估指标字典
        save_path: 保存路径
        **kwargs: 其他绘图参数
    """
    # 过滤掉非数值指标
    numeric_metrics = {k: v for k, v in metrics.items() 
                      if isinstance(v, (int, float)) and not k.startswith('t')}
    
    if not numeric_metrics:
        print("没有可绘制的数值指标")
        return
    
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (10, 6)))
    
    metric_names = list(numeric_metrics.keys())
    metric_values = list(numeric_metrics.values())
    
    bars = ax.bar(range(len(metric_names)), metric_values, 
                  alpha=0.8, color=sns.color_palette("husl", len(metric_names)))
    
    # 设置标签和标题
    ax.set_xlabel('评估指标', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.set_title('模型评估结果', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, rotation=45, ha='right')
    
    # 添加数值标注
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"评估指标图已保存到: {save_path}")
    
    plt.show()


def plot_comparison_results(results_list: List[Dict[str, Any]],
                           experiment_names: List[str],
                           metrics: List[str] = None,
                           save_path: Optional[str] = None,
                           **kwargs) -> None:
    """
    绘制多个实验结果对比图
    
    Args:
        results_list: 实验结果列表
        experiment_names: 实验名称列表
        metrics: 要比较的指标列表
        save_path: 保存路径
        **kwargs: 其他绘图参数
    """
    if metrics is None:
        metrics = ['f1', 'precision', 'recall', 'accuracy']
    
    # 提取数据
    data = []
    for i, (results, exp_name) in enumerate(zip(results_list, experiment_names)):
        if 'evaluation' in results:
            eval_results = results['evaluation']
            for metric in metrics:
                if metric in eval_results:
                    data.append({
                        'experiment': exp_name,
                        'metric': metric,
                        'value': eval_results[metric]
                    })
    
    if not data:
        print("没有可比较的数据")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(data)
    
    # 绘制分组条形图
    fig, ax = plt.subplots(figsize=kwargs.get('figsize', (12, 6)))
    
    sns.barplot(data=df, x='metric', y='value', hue='experiment', ax=ax)
    
    ax.set_title('实验结果对比', fontsize=14, fontweight='bold')
    ax.set_xlabel('评估指标', fontsize=12)
    ax.set_ylabel('指标值', fontsize=12)
    ax.legend(title='实验', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()


def plot_threshold_analysis(threshold_results: Dict[str, List],
                           save_path: Optional[str] = None,
                           **kwargs) -> None:
    """
    绘制阈值分析图
    
    Args:
        threshold_results: 阈值分析结果
        save_path: 保存路径
        **kwargs: 其他绘图参数
    """
    fig, axes = plt.subplots(2, 2, figsize=kwargs.get('figsize', (12, 8)))
    
    thresholds = threshold_results.get('thresholds', [])
    
    # F1分数 vs 阈值
    if 'f1' in threshold_results:
        axes[0, 0].plot(thresholds, threshold_results['f1'], 'o-', alpha=0.8)
        axes[0, 0].set_title('F1分数 vs 阈值')
        axes[0, 0].set_xlabel('阈值')
        axes[0, 0].set_ylabel('F1分数')
        axes[0, 0].grid(True, alpha=0.3)
    
    # 精确率和召回率 vs 阈值
    if 'precision' in threshold_results and 'recall' in threshold_results:
        axes[0, 1].plot(thresholds, threshold_results['precision'], 'o-', 
                       label='精确率', alpha=0.8)
        axes[0, 1].plot(thresholds, threshold_results['recall'], 's-', 
                       label='召回率', alpha=0.8)
        axes[0, 1].set_title('精确率和召回率 vs 阈值')
        axes[0, 1].set_xlabel('阈值')
        axes[0, 1].set_ylabel('指标值')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 准确率 vs 阈值
    if 'accuracy' in threshold_results:
        axes[1, 0].plot(thresholds, threshold_results['accuracy'], '^-', alpha=0.8)
        axes[1, 0].set_title('准确率 vs 阈值')
        axes[1, 0].set_xlabel('阈值')
        axes[1, 0].set_ylabel('准确率')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 结构汉明距离 vs 阈值
    if 'shd' in threshold_results:
        axes[1, 1].plot(thresholds, threshold_results['shd'], 'd-', alpha=0.8)
        axes[1, 1].set_title('结构汉明距离 vs 阈值')
        axes[1, 1].set_xlabel('阈值')
        axes[1, 1].set_ylabel('SHD')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"阈值分析图已保存到: {save_path}")
    
    plt.show()


def create_experiment_dashboard(results: Dict[str, Any],
                               config: Dict[str, Any],
                               save_dir: str,
                               experiment_name: str = "experiment") -> str:
    """
    创建实验结果仪表板
    
    Args:
        results: 实验结果
        config: 配置信息
        save_dir: 保存目录
        experiment_name: 实验名称
        
    Returns:
        保存目录路径
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📊 正在创建实验仪表板...")
    
    # 1. 因果矩阵可视化
    if 'causal_matrix' in results:
        causal_matrix = results['causal_matrix']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 概率矩阵
        im1 = axes[0].imshow(causal_matrix, cmap='RdBu_r', vmin=0, vmax=1)
        axes[0].set_title('因果概率矩阵')
        axes[0].set_xlabel('目标节点')
        axes[0].set_ylabel('源节点')
        plt.colorbar(im1, ax=axes[0])
        
        # 二值化矩阵
        threshold = results.get('evaluation', {}).get('best_threshold', 0.5)
        binary_matrix = (causal_matrix > threshold).astype(int)
        im2 = axes[1].imshow(binary_matrix, cmap='RdBu_r', vmin=0, vmax=1)
        axes[1].set_title(f'二值化因果图 (阈值={threshold:.2f})')
        axes[1].set_xlabel('目标节点')
        axes[1].set_ylabel('源节点')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        causal_path = save_path / f"{experiment_name}_causal_matrices.png"
        plt.savefig(causal_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. 训练过程可视化
    if 'training' in results:
        plot_training_curves(results['training'], 
                            save_path=save_path / f"{experiment_name}_training.png")
        plt.close()
    
    # 3. 评估指标可视化
    if 'evaluation' in results:
        plot_evaluation_metrics(results['evaluation'],
                               save_path=save_path / f"{experiment_name}_metrics.png")
        plt.close()
    
    print(f"✅ 实验仪表板已创建: {save_path}")
    return str(save_path)


def save_plots(save_dir: str, **plots) -> None:
    """
    批量保存图表
    
    Args:
        save_dir: 保存目录
        **plots: 图表字典，键为文件名，值为图表对象
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    for filename, fig in plots.items():
        if hasattr(fig, 'savefig'):
            file_path = save_path / f"{filename}.png"
            fig.savefig(file_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存: {file_path}")
        else:
            print(f"警告: {filename} 不是有效的图表对象")


def plot_probability_matrix_comparison(predicted_prob_matrix, true_causal_matrix, 
                                     save_path=None, title_prefix="", 
                                     threshold=0.5, cmap='Blues'):
    """
    绘制概率矩阵（缩放后）与真实因果矩阵的对比图
    
    Args:
        predicted_prob_matrix: 预测的概率矩阵 (缩放后)
        true_causal_matrix: 真实因果矩阵
        save_path: 保存路径
        title_prefix: 标题前缀
        threshold: 二值化阈值（用于生成预测的二值图）
        cmap: 颜色映射
    """
    # 确保输入是numpy数组
    if isinstance(predicted_prob_matrix, torch.Tensor):
        predicted_prob_matrix = predicted_prob_matrix.detach().cpu().numpy()
    if isinstance(true_causal_matrix, torch.Tensor):
        true_causal_matrix = true_causal_matrix.detach().cpu().numpy()
    
    # 处理维度
    if predicted_prob_matrix.ndim > 2:
        predicted_prob_matrix = predicted_prob_matrix.squeeze()
    if true_causal_matrix.ndim > 2:
        true_causal_matrix = true_causal_matrix.squeeze()
    
    # 生成二值化的预测矩阵
    predicted_binary = (predicted_prob_matrix > threshold).astype(float)
    
    # 计算差异矩阵（用于错误分析）
    diff_matrix = np.abs(predicted_binary - true_causal_matrix)
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{title_prefix}Probability Matrix vs Ground Truth Comparison', fontsize=16, y=0.98)
    
    # 子图1: 真实因果矩阵
    im1 = axes[0, 0].imshow(true_causal_matrix, cmap='Reds', vmin=0, vmax=1)
    axes[0, 0].set_title('Ground Truth Causal Matrix')
    axes[0, 0].set_xlabel('Target Node')
    axes[0, 0].set_ylabel('Source Node')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # 添加数值标注（适用于小矩阵）
    n_nodes = true_causal_matrix.shape[0]
    if n_nodes <= 10:
        for i in range(n_nodes):
            for j in range(n_nodes):
                text = axes[0, 0].text(j, i, f'{true_causal_matrix[i, j]:.0f}',
                                     ha="center", va="center", color="white" if true_causal_matrix[i, j] > 0.5 else "black")
    
    # 子图2: 预测概率矩阵（缩放后）
    im2 = axes[0, 1].imshow(predicted_prob_matrix, cmap=cmap, vmin=0, vmax=1)
    axes[0, 1].set_title(f'Predicted Probability Matrix (Scaled)')
    axes[0, 1].set_xlabel('Target Node')
    axes[0, 1].set_ylabel('Source Node')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # 添加数值标注（适用于小矩阵）
    if n_nodes <= 10:
        for i in range(n_nodes):
            for j in range(n_nodes):
                text = axes[0, 1].text(j, i, f'{predicted_prob_matrix[i, j]:.2f}',
                                     ha="center", va="center", 
                                     color="white" if predicted_prob_matrix[i, j] > 0.5 else "black",
                                     fontsize=8)
    
    # 子图3: 预测二值矩阵（阈值化后）
    im3 = axes[1, 0].imshow(predicted_binary, cmap='Greens', vmin=0, vmax=1)
    axes[1, 0].set_title(f'Predicted Binary Matrix (Threshold={threshold})')
    axes[1, 0].set_xlabel('Target Node')
    axes[1, 0].set_ylabel('Source Node')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 添加数值标注
    if n_nodes <= 10:
        for i in range(n_nodes):
            for j in range(n_nodes):
                text = axes[1, 0].text(j, i, f'{predicted_binary[i, j]:.0f}',
                                     ha="center", va="center", 
                                     color="white" if predicted_binary[i, j] > 0.5 else "black")
    
    # 子图4: 错误矩阵（差异分析）
    im4 = axes[1, 1].imshow(diff_matrix, cmap='Oranges', vmin=0, vmax=1)
    axes[1, 1].set_title('Error Matrix (|Predicted - Truth|)')
    axes[1, 1].set_xlabel('Target Node')
    axes[1, 1].set_ylabel('Source Node')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    # 标记错误位置
    if n_nodes <= 10:
        for i in range(n_nodes):
            for j in range(n_nodes):
                if diff_matrix[i, j] > 0:
                    axes[1, 1].text(j, i, '✗', ha="center", va="center", 
                                   color="red", fontsize=12, fontweight='bold')
                else:
                    axes[1, 1].text(j, i, '✓', ha="center", va="center", 
                                   color="green", fontsize=10)
    
    # 计算和显示统计信息
    try:
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        
        true_flat = true_causal_matrix.flatten()
        pred_flat = predicted_binary.flatten()
        
        accuracy = accuracy_score(true_flat, pred_flat)
        precision = precision_score(true_flat, pred_flat, zero_division=0)
        recall = recall_score(true_flat, pred_flat, zero_division=0)
        f1 = f1_score(true_flat, pred_flat, zero_division=0)
        
        # 添加统计信息文本
        stats_text = f"""Evaluation Metrics:
Accuracy: {accuracy:.3f}
Precision: {precision:.3f}
Recall: {recall:.3f}
F1 Score: {f1:.3f}
Error Edges: {int(np.sum(diff_matrix))}
Threshold: {threshold}"""
        
        plt.figtext(0.02, 0.02, stats_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    except ImportError:
        print("Warning: sklearn not installed, skipping evaluation metrics calculation")
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability matrix comparison saved to: {save_path}")
    
    plt.show()


def plot_time_series_reconstruction(original_data, reconstructed_data,
                                  selected_nodes=None, max_timesteps=200,
                                  save_path=None, title_prefix="Time Series Reconstruction - "):
    """
    Plots a comparison of original and reconstructed time series data.
    Optimized for handling long sequences and providing clear visual feedback.
    
    Args:
        original_data (np.ndarray): Original time series data [timesteps, nodes].
        reconstructed_data (np.ndarray): Reconstructed time series data [timesteps, nodes].
        selected_nodes (list, optional): List of node indices to visualize. Defaults to auto-selection.
        max_timesteps (int, optional): Maximum number of timesteps to display. Defaults to 200.
        save_path (str, optional): Path to save the plot. Defaults to None (shows plot).
        title_prefix (str, optional): Prefix for the plot title.
    """
    # 确保输入是numpy数组
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.detach().cpu().numpy()
    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.detach().cpu().numpy()
    
    # 处理批次维度（如果存在）
    if original_data.ndim == 3:
        original_data = original_data[0]
    if reconstructed_data.ndim == 3:
        reconstructed_data = reconstructed_data[0]
    
    # 确保数据形状正确 [timesteps, nodes]
    if original_data.ndim != 2 or reconstructed_data.ndim != 2:
        print(f"⚠️ 数据维度异常，跳过可视化")
        return
    
    # 限制时间步数为用户要求的200
    timesteps = min(original_data.shape[0], reconstructed_data.shape[0], max_timesteps)
    original_data = original_data[:timesteps]
    reconstructed_data = reconstructed_data[:timesteps]
    
    n_nodes = min(original_data.shape[1], reconstructed_data.shape[1])
    original_data = original_data[:, :n_nodes]
    reconstructed_data = reconstructed_data[:, :n_nodes]
    
    print(f"📊 可视化数据: {timesteps}时间步 × {n_nodes}节点")
    
    # 自动选择要显示的节点（针对CSV数据优化）
    if selected_nodes is None:
        # 选择方差最大的前6个节点（CSV数据有10个节点）
        node_variances = np.var(original_data, axis=0)
        top_variance_nodes = np.argsort(node_variances)[-6:]  # 选择方差最大的6个节点
        selected_nodes = sorted(top_variance_nodes)  # 按索引排序
    
    # 限制选择的节点数量
    selected_nodes = [node for node in selected_nodes if 0 <= node < n_nodes]
    n_selected = min(len(selected_nodes), 6)  # 最多显示6个节点
    selected_nodes = selected_nodes[:n_selected]
    
    # 创建子图
    fig, axes = plt.subplots(n_selected, 1, figsize=(16, 4 * n_selected), constrained_layout=True)
    if n_selected == 1:
        axes = [axes]
    
    fig.suptitle(f'{title_prefix}Comparison (First {timesteps} Timesteps)', fontsize=18, fontweight='bold')
    
    # 为每个选中的节点绘制对比图
    time_axis = np.arange(timesteps)
    
    for i, node_idx in enumerate(selected_nodes):
        ax = axes[i]
        
        # 原始数据
        ax.plot(time_axis, original_data[:, node_idx], 
               label=f'Original Data', color='#2E86C1', linewidth=2.5, alpha=0.9)
        
        # 重构数据
        ax.plot(time_axis, reconstructed_data[:, node_idx], 
               label=f'Reconstructed Data', color='#E74C3C', linewidth=2, alpha=0.9, linestyle='--')
        
        # 计算重构误差指标
        mse = np.mean((original_data[:, node_idx] - reconstructed_data[:, node_idx]) ** 2)
        mae = np.mean(np.abs(original_data[:, node_idx] - reconstructed_data[:, node_idx]))
        
        # 计算相关系数
        if np.std(original_data[:, node_idx]) > 0 and np.std(reconstructed_data[:, node_idx]) > 0:
            correlation = np.corrcoef(original_data[:, node_idx], reconstructed_data[:, node_idx])[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        ax.set_title(f'Node {node_idx + 1} | MSE: {mse:.4f}, MAE: {mae:.4f}, Correlation: {correlation:.3f}', 
                    fontsize=14)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.4, linestyle=':')
        
        # 添加误差填充区域（浅色）
        ax.fill_between(time_axis, 
                       original_data[:, node_idx], 
                       reconstructed_data[:, node_idx],
                       alpha=0.15, color='orange')
        
        # 设置坐标轴样式
        ax.tick_params(axis='both', which='major', labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # 调整布局
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # 计算整体统计
    overall_mse = np.mean((original_data[:, selected_nodes] - reconstructed_data[:, selected_nodes]) ** 2)
    overall_mae = np.mean(np.abs(original_data[:, selected_nodes] - reconstructed_data[:, selected_nodes]))
    
    # 添加整体统计信息
    stats_text = f"""Overall Performance (for {n_selected} selected nodes):
    MSE: {overall_mse:.6f} | MAE: {overall_mae:.6f}
    Displayed Nodes: {[f'Node {i+1}' for i in selected_nodes]}
    Time Window: 1-{timesteps} (out of {original_data.shape[0]} total steps)"""
    
    fig.text(0.5, -0.01, stats_text, fontsize=11, ha='center',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F0F0F0", alpha=0.8))
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✅ Time series reconstruction plot saved to: {save_path}")
    
    plt.close(fig)


def plot_comprehensive_results(model_result, original_data, true_causal_matrix,
                             selected_nodes=None, max_timesteps=100,
                             threshold=0.5, save_dir=None, title_prefix=""):
    """
    绘制综合结果可视化（包括概率矩阵对比和时间序列重构）
    
    Args:
        model_result: 模型前向传播结果字典，包含 'prediction_samples', 'adj_probs' 等
        original_data: 原始时间序列数据
        true_causal_matrix: 真实因果矩阵
        selected_nodes: 要显示的节点列表
        max_timesteps: 最大时间步数
        threshold: 概率矩阵二值化阈值
        save_dir: 保存目录
        title_prefix: 标题前缀
    """
    import os
    
    # 准备数据
    predicted_prob_matrix = model_result.get('adj_probs')
    prediction_samples = model_result.get('prediction_samples')
    
    # 处理重构数据
    if prediction_samples is not None:
        # 取预测样本的均值作为重构数据
        if isinstance(prediction_samples, torch.Tensor):
            reconstructed_data = prediction_samples.mean(dim=-1).detach().cpu().numpy()
        else:
            reconstructed_data = np.mean(prediction_samples, axis=-1)
    else:
        print("警告: 未找到预测样本数据，跳过时间序列重构图")
        reconstructed_data = None
    
    # 1. 绘制概率矩阵对比图
    if predicted_prob_matrix is not None and true_causal_matrix is not None:
        prob_save_path = None
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            prob_save_path = os.path.join(save_dir, f"{title_prefix}probability_matrix_comparison.png")
        
        plot_probability_matrix_comparison(
            predicted_prob_matrix, true_causal_matrix,
            save_path=prob_save_path, title_prefix=title_prefix,
            threshold=threshold
        )
    
    # 2. 绘制时间序列重构图
    if reconstructed_data is not None and original_data is not None:
        # 确保数据形状兼容
        try:
            # 重构数据需要reshape为时间序列格式
            if reconstructed_data.ndim == 2:
                # [batch_size, features] -> [timesteps, nodes]
                batch_size, features = reconstructed_data.shape
                # 假设数据是 [batch_size, sequence_length * num_nodes] 格式
                if hasattr(model_result.get('model', None), 'num_nodes'):
                    num_nodes = model_result['model'].num_nodes
                    sequence_length = features // num_nodes
                elif true_causal_matrix is not None:
                    num_nodes = true_causal_matrix.shape[0]
                    sequence_length = features // num_nodes
                else:
                    # 估算节点数（假设是方阵的边数）
                    num_nodes = int(np.sqrt(features))
                    sequence_length = features // num_nodes
                
                reconstructed_data = reconstructed_data[0].reshape(sequence_length, num_nodes)
            
            recon_save_path = None
            if save_dir:
                recon_save_path = os.path.join(save_dir, f"{title_prefix}time_series_reconstruction.png")
            
            plot_time_series_reconstruction(
                original_data, reconstructed_data,
                selected_nodes=selected_nodes, max_timesteps=max_timesteps,
                save_path=recon_save_path, title_prefix=title_prefix
            )
            
        except Exception as e:
            print(f"警告: 时间序列重构图绘制失败: {e}")
    
    # 3. 绘制额外的分析图（可选）
    if save_dir:
        # 保存模型结果的详细信息
        info_path = os.path.join(save_dir, f"{title_prefix}model_info.txt")
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"模型结果分析报告\n")
            f.write(f"=" * 50 + "\n\n")
            
            if predicted_prob_matrix is not None:
                f.write(f"概率矩阵统计:\n")
                f.write(f"  - 形状: {predicted_prob_matrix.shape}\n")
                f.write(f"  - 最小值: {np.min(predicted_prob_matrix):.4f}\n")
                f.write(f"  - 最大值: {np.max(predicted_prob_matrix):.4f}\n")
                f.write(f"  - 均值: {np.mean(predicted_prob_matrix):.4f}\n")
                f.write(f"  - 标准差: {np.std(predicted_prob_matrix):.4f}\n")
                f.write(f"  - 稀疏度: {np.mean(predicted_prob_matrix < 0.1):.3f}\n\n")
            
            if reconstructed_data is not None and original_data is not None:
                mse = np.mean((original_data - reconstructed_data) ** 2)
                mae = np.mean(np.abs(original_data - reconstructed_data))
                f.write(f"重构性能:\n")
                f.write(f"  - MSE: {mse:.6f}\n")
                f.write(f"  - MAE: {mae:.6f}\n")
                f.write(f"  - 显示的时间步数: {min(max_timesteps, original_data.shape[0])}\n")
                if selected_nodes:
                    f.write(f"  - 显示的节点: {selected_nodes}\n")
        
        print(f"结果分析报告已保存到: {info_path}")


def plot_enhanced_time_series_reconstruction(original_data, reconstructed_data, 
                                           selected_node=None, max_timesteps=200,
                                           show_error_band=True, show_statistics=True,
                                           save_path=None, title_prefix="",
                                           comparison_mode="single", **kwargs):
    """
    Orchestrator for generating various types of time series reconstruction visualizations.
    All text output is in English.
    """
    # 设置matplotlib后端（在导入pyplot之前）
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    plt.style.use('default')
    
    # 参数设置
    figsize = kwargs.get('figsize', (15, 8))
    line_width = kwargs.get('line_width', 2)
    alpha = kwargs.get('alpha', 0.8)
    error_alpha = kwargs.get('error_alpha', 0.3)
    
    # 数据预处理
    original_data, reconstructed_data, num_nodes, timesteps = _preprocess_time_series_data(
        original_data, reconstructed_data, max_timesteps
    )
    
    print(f"📊 数据预处理完成:")
    print(f"   • 时间步数: {timesteps}")
    print(f"   • 节点数量: {num_nodes}")
    print(f"   • 数据形状: 原始{original_data.shape}, 重构{reconstructed_data.shape}")
    
    # 节点选择逻辑
    if selected_node is None:
        selected_node = _auto_select_representative_node(original_data)
        print(f"   • 自动选择节点: {selected_node + 1} (最具代表性)")
    else:
        if selected_node >= num_nodes:
            selected_node = num_nodes - 1
            print(f"   ⚠️  节点索引超出范围，自动调整为: {selected_node + 1}")
    
    # 根据对比模式绘制
    if comparison_mode == "single":
        _plot_single_node_detailed(original_data, reconstructed_data, selected_node,
                                 timesteps, show_error_band, show_statistics,
                                 figsize, line_width, alpha, error_alpha, title_prefix)
    elif comparison_mode == "multi":
        _plot_multi_node_overview(original_data, reconstructed_data, 
                                max_nodes=min(6, num_nodes), timesteps=timesteps,
                                figsize=(16, 12), title_prefix=title_prefix)
    elif comparison_mode == "grid":
        _plot_grid_comparison(original_data, reconstructed_data,
                            selected_node, timesteps, title_prefix)
    
    # 保存图像
    if save_path:
        try:
            # 确保目录存在
            import os
            save_dir = os.path.dirname(save_path)
            if save_dir:  # 如果有目录部分
                os.makedirs(save_dir, exist_ok=True)
                print(f"   📂 确保保存目录存在: {save_dir}")
            
            # 设置matplotlib后端
            import matplotlib
            matplotlib.use('Agg')  # 使用非交互式后端
            
            # 保存图像，使用更多参数确保成功
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none', 
                       format='png', transparent=False,
                       pad_inches=0.1)  # 添加一些边距
            
            print(f"   💾 尝试保存到: {save_path}")
            
            # 强制刷新并等待
            plt.draw()
            import time
            time.sleep(0.1)  # 短暂等待确保保存完成
            
            # 验证文件是否确实保存
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                if file_size > 0:
                    print(f"   ✅ 图像已成功保存到: {save_path}")
                    print(f"   📁 文件大小: {file_size} 字节")
                else:
                    print(f"   ❌ 文件已创建但大小为0: {save_path}")
            else:
                print(f"   ❌ 文件保存失败，文件不存在: {save_path}")
                # 尝试使用绝对路径保存
                abs_path = os.path.abspath(save_path)
                print(f"   🔄 尝试使用绝对路径: {abs_path}")
                plt.savefig(abs_path, dpi=300, bbox_inches='tight', 
                           facecolor='white', edgecolor='none', 
                           format='png', transparent=False)
                
        except Exception as save_error:
            print(f"   ❌ 保存图像时发生错误: {save_error}")
            import traceback
            traceback.print_exc()
            
            # 尝试备用保存方法
            try:
                backup_path = save_path.replace('.png', '_backup.png')
                plt.savefig(backup_path, dpi=200)  # 降低DPI再试
                print(f"   🔄 已尝试备用保存: {backup_path}")
            except Exception as backup_error:
                print(f"   ❌ 备用保存也失败: {backup_error}")
    
    plt.close('all')  # 释放内存而不是显示
    
    # 返回成功状态
    if save_path:
        import os
        return os.path.exists(save_path) and os.path.getsize(save_path) > 0
    else:
        return True  # 如果没有指定保存路径，认为成功


def _preprocess_time_series_data(original_data, reconstructed_data, max_timesteps):
    """预处理时间序列数据，统一格式"""
    
    # 转换为numpy数组
    if isinstance(original_data, torch.Tensor):
        original_data = original_data.detach().cpu().numpy()
    if isinstance(reconstructed_data, torch.Tensor):
        reconstructed_data = reconstructed_data.detach().cpu().numpy()
    
    # 处理不同的输入格式
    original_data = _reshape_to_time_series(original_data)
    reconstructed_data = _reshape_to_time_series(reconstructed_data)
    
    # 确保两个数据形状一致
    if original_data.shape != reconstructed_data.shape:
        print(f"⚠️  数据形状不匹配: 原始{original_data.shape} vs 重构{reconstructed_data.shape}")
        # 尝试对齐
        min_timesteps = min(original_data.shape[0], reconstructed_data.shape[0])
        min_nodes = min(original_data.shape[1], reconstructed_data.shape[1])
        original_data = original_data[:min_timesteps, :min_nodes]
        reconstructed_data = reconstructed_data[:min_timesteps, :min_nodes]
    
    # 限制时间步数
    timesteps = min(original_data.shape[0], max_timesteps)
    original_data = original_data[:timesteps]
    reconstructed_data = reconstructed_data[:timesteps]
    
    num_nodes = original_data.shape[1]
    
    return original_data, reconstructed_data, num_nodes, timesteps


def _reshape_to_time_series(data):
    """将数据重塑为 [timesteps, nodes] 格式"""
    
    if data.ndim == 3:
        # [batch, timesteps, nodes] 或 [batch, nodes, timesteps]
        if data.shape[0] == 1:
            data = data[0]  # 移除batch维度
        else:
            data = data[0]  # 取第一个batch
            print(f"   ℹ️  检测到多个batch，使用第一个batch")
    
    if data.ndim == 2:
        # 检查是否需要转置
        if data.shape[0] < data.shape[1]:
            # 可能是 [nodes, timesteps]，转置为 [timesteps, nodes]
            if data.shape[0] < 20:  # 启发式判断：节点数通常小于20
                data = data.T
                print(f"   🔄 数据已转置为 [timesteps, nodes] 格式")
    
    return data


def _auto_select_representative_node(data):
    """自动选择最具代表性的节点"""
    
    if data.ndim == 1 or data.shape[1] == 1:
        return 0
    
    try:
        # 计算每个节点的代表性指标
        variances = np.var(data, axis=0)      # 方差（信号强度）
        ranges = np.ptp(data, axis=0)         # 极差（动态范围）
        
        # 避免选择常数节点（方差为0）
        valid_nodes = np.where(variances > 1e-10)[0]
        if len(valid_nodes) == 0:
            print(f"   ⚠️ 所有节点方差过小，使用节点0")
            return 0
        
        # 避免NaN和无穷大值
        variances = np.nan_to_num(variances, nan=0.0, posinf=0.0, neginf=0.0)
        ranges = np.nan_to_num(ranges, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 综合评分：优先选择方差大且动态范围大的节点
        scores = variances + 0.5 * ranges
        valid_scores = scores[valid_nodes]
        
        if np.all(valid_scores == 0):
            print(f"   ⚠️ 所有节点评分为0，使用节点0")
            return 0
            
        best_idx = np.argmax(valid_scores)
        best_node = valid_nodes[best_idx]
        
        print(f"   🎯 自动选择节点 {best_node}，方差={variances[best_node]:.6f}")
        return best_node
        
    except Exception as e:
        print(f"   ❌ 节点选择失败: {e}")
        return 0


def _plot_single_node_detailed(original_data, reconstructed_data, selected_node,
                              timesteps, show_error_band, show_statistics,
                              figsize, line_width, alpha, error_alpha, title_prefix):
    """绘制单节点详细对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    time_axis = np.arange(timesteps)
    
    # 提取选中节点的数据
    orig_series = original_data[:, selected_node]
    recon_series = reconstructed_data[:, selected_node]
    
    # 计算统计指标 - 添加安全检查
    try:
        mse = np.mean((orig_series - recon_series) ** 2)
        mae = np.mean(np.abs(orig_series - recon_series))
        rmse = np.sqrt(mse)
        
        # 安全的相关性计算
        if len(orig_series) > 1 and np.std(orig_series) > 1e-10 and np.std(recon_series) > 1e-10:
            correlation = np.corrcoef(orig_series, recon_series)[0, 1]
            # 处理NaN值
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 1.0 if np.allclose(orig_series, recon_series, atol=1e-6) else 0.0
    except Exception as e:
        print(f"   ⚠️ 统计计算出错: {e}")
        mse = mae = rmse = correlation = 0.0
    
    # 1. 主要对比图 (左上)
    axes[0, 0].plot(time_axis, orig_series, label='Original Data', 
                   color='#1f77b4', linewidth=line_width, alpha=alpha)
    axes[0, 0].plot(time_axis, recon_series, label='Reconstructed Data', 
                   color='#ff7f0e', linewidth=line_width, alpha=alpha, linestyle='--')
    
    if show_error_band:
        axes[0, 0].fill_between(time_axis, orig_series, recon_series,
                               alpha=error_alpha, color='red', label='Reconstruction Error')
    
    axes[0, 0].set_title(f'Node {selected_node + 1} - Time Series Reconstruction Comparison')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 误差序列图 (右上)
    error_series = orig_series - recon_series
    axes[0, 1].plot(time_axis, error_series, color='red', linewidth=1.5, alpha=0.8)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0, 1].fill_between(time_axis, error_series, 0, alpha=0.3, color='red')
    axes[0, 1].set_title('Reconstruction Error Series')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Error Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 散点图对比 (左下)
    axes[1, 0].scatter(orig_series, recon_series, alpha=0.6, s=20, color='purple')
    
    # 添加理想对角线
    min_val = min(orig_series.min(), recon_series.min())
    max_val = max(orig_series.max(), recon_series.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Ideal Reconstruction')
    
    axes[1, 0].set_xlabel('Original Data')
    axes[1, 0].set_ylabel('Reconstructed Data')
    axes[1, 0].set_title(f'Scatter Plot Comparison (Correlation: {correlation:.3f})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 统计信息展示 (右下)
    if show_statistics:
        axes[1, 1].axis('off')
        
        # 更多统计指标
        percentile_errors = np.percentile(np.abs(error_series), [50, 90, 95, 99])
        signal_power = np.mean(orig_series ** 2)
        noise_power = np.mean(error_series ** 2)
        snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        stats_text = f"""📊 Reconstruction Performance Statistics
        
🎯 Basic Metrics:
   • MSE: {mse:.6f}
   • RMSE: {rmse:.6f}  
   • MAE: {mae:.6f}
   • Correlation: {correlation:.4f}

📈 Error Distribution:
   • Median Error: {percentile_errors[0]:.6f}
   • 90th Percentile: {percentile_errors[1]:.6f}
   • 95th Percentile: {percentile_errors[2]:.6f}
   • 99th Percentile: {percentile_errors[3]:.6f}

🔊 Signal-to-Noise Ratio:
   • SNR: {snr:.2f} dB
   
📏 Data Range:
   • Original Range: [{orig_series.min():.3f}, {orig_series.max():.3f}]
   • Reconstructed Range: [{recon_series.min():.3f}, {recon_series.max():.3f}]
   
⏱️ Display Info:
   • Node Index: {selected_node + 1}
   • Time Steps: {timesteps}
        """
        
        axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    plt.suptitle(f'{title_prefix}Node {selected_node + 1} Time Series Reconstruction Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def _plot_multi_node_overview(original_data, reconstructed_data, max_nodes, timesteps, figsize, title_prefix):
    """绘制多节点概览对比图"""
    
    num_nodes = original_data.shape[1]
    nodes_to_show = min(max_nodes, num_nodes)
    
    # 选择最具代表性的节点
    variances = np.var(original_data, axis=0)
    top_nodes = np.argsort(variances)[-nodes_to_show:]
    
    fig, axes = plt.subplots(nodes_to_show, 1, figsize=figsize, sharex=True)
    if nodes_to_show == 1:
        axes = [axes]
    
    time_axis = np.arange(timesteps)
    
    for i, node_idx in enumerate(top_nodes):
        orig_series = original_data[:, node_idx]
        recon_series = reconstructed_data[:, node_idx]
        
        # 计算关键指标 - 安全计算
        try:
            mse = np.mean((orig_series - recon_series) ** 2)
            if len(orig_series) > 1 and np.std(orig_series) > 1e-10 and np.std(recon_series) > 1e-10:
                correlation = np.corrcoef(orig_series, recon_series)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
            else:
                correlation = 1.0 if np.allclose(orig_series, recon_series, atol=1e-6) else 0.0
        except Exception as e:
            print(f"   ⚠️ 节点{node_idx}统计计算出错: {e}")
            mse = correlation = 0.0
        
        axes[i].plot(time_axis, orig_series, label='Original', color='blue', linewidth=1.5, alpha=0.8)
        axes[i].plot(time_axis, recon_series, label='Reconstructed', color='red', linewidth=1.5, alpha=0.8, linestyle='--')
        
        axes[i].set_title(f'Node {node_idx + 1} - MSE: {mse:.4f}, Correlation: {correlation:.3f}')
        axes[i].set_ylabel('Values')
        axes[i].legend(loc='upper right')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Steps')
    plt.suptitle(f'{title_prefix}Multi-Node Time Series Reconstruction Overview (First {timesteps} Steps)', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


def _plot_grid_comparison(original_data, reconstructed_data, selected_node, timesteps, title_prefix):
    """绘制网格布局对比图"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    time_axis = np.arange(timesteps)
    orig_series = original_data[:, selected_node]
    recon_series = reconstructed_data[:, selected_node]
    
    # 主图 - 占据左侧两列
    ax_main = fig.add_subplot(gs[:, :2])
    ax_main.plot(time_axis, orig_series, label='Original Data', color='blue', linewidth=2, alpha=0.8)
    ax_main.plot(time_axis, recon_series, label='Reconstructed Data', color='red', linewidth=2, alpha=0.8, linestyle='--')
    ax_main.fill_between(time_axis, orig_series, recon_series, alpha=0.2, color='orange')
    ax_main.set_title(f'Node {selected_node + 1} - Complete Time Series Comparison ({timesteps} Steps)')
    ax_main.set_xlabel('Time Steps')
    ax_main.set_ylabel('Values')
    ax_main.legend()
    ax_main.grid(True, alpha=0.3)
    
    # 局部放大图1 - 前1/3
    ax_zoom1 = fig.add_subplot(gs[0, 2])
    start_idx, end_idx = 0, timesteps // 3
    _plot_zoomed_section(ax_zoom1, time_axis[start_idx:end_idx], 
                        orig_series[start_idx:end_idx], recon_series[start_idx:end_idx],
                        f'First 1/3 Section (Steps {start_idx+1}-{end_idx})')
    
    # 局部放大图2 - 中间1/3  
    ax_zoom2 = fig.add_subplot(gs[1, 2])
    start_idx, end_idx = timesteps // 3, 2 * timesteps // 3
    _plot_zoomed_section(ax_zoom2, time_axis[start_idx:end_idx],
                        orig_series[start_idx:end_idx], recon_series[start_idx:end_idx],
                        f'Middle 1/3 Section (Steps {start_idx+1}-{end_idx})')
    
    # 局部放大图3 - 后1/3
    ax_zoom3 = fig.add_subplot(gs[2, 2])
    start_idx, end_idx = 2 * timesteps // 3, timesteps
    _plot_zoomed_section(ax_zoom3, time_axis[start_idx:end_idx],
                        orig_series[start_idx:end_idx], recon_series[start_idx:end_idx],
                        f'Last 1/3 Section (Steps {start_idx+1}-{end_idx})')
    
    plt.suptitle(f'{title_prefix}Node {selected_node + 1} Grid-Based Detailed Comparison Analysis', fontsize=16, fontweight='bold')


def _plot_zoomed_section(ax, time_axis, orig_data, recon_data, title):
    """绘制局部放大区域"""
    ax.plot(time_axis, orig_data, color='blue', linewidth=1.5, alpha=0.8, label='Original')
    ax.plot(time_axis, recon_data, color='red', linewidth=1.5, alpha=0.8, linestyle='--', label='Reconstructed')
    ax.fill_between(time_axis, orig_data, recon_data, alpha=0.3, color='orange')
    
    # 计算该区间的MSE
    mse = np.mean((orig_data - recon_data) ** 2)
    ax.set_title(f'{title}\nMSE: {mse:.6f}', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)


class PFNOVisualizationSuite:
    """PFNO可视化套件 - 统一管理所有可视化功能"""
    
    def __init__(self, results_dir, config=None):
        """
        初始化可视化套件
        
        Args:
            results_dir: 结果保存目录
            config: 可视化配置字典（可选）
        """
        self.results_dir = results_dir
        self.config = config or {}
        
        # 默认可视化配置
        self.viz_config = {
            'training_curves': {
                'figsize': (16, 12),
                'smooth_window': 'auto',  # auto, int, or None
                'show_components': True,
                'save_format': 'png'
            },
            'matrix_comparison': {
                'figsize': (14, 10),
                'cmap': 'Blues',
                'threshold': 0.5,
                'show_colorbar': True,
                'title_prefix': "PFNO Causal Discovery Results - "
            },
            'time_series': {
                'figsize_single': (16, 10),
                'figsize_multi': (16, 12),
                'figsize_grid': (20, 14),
                'max_timesteps': 200,
                'selected_nodes': 'auto',  # 'auto', list of indices, or int
                'show_error_band': True,
                'show_statistics': True,
                'title_prefix': "PFNO重构分析 - "
            }
        }
        
        # 更新用户配置
        if 'visualization' in self.config:
            self._update_config(self.config['visualization'])
        
        print(f"🎨 PFNO可视化套件已初始化")
        print(f"   - 结果目录: {self.results_dir}")
    
    def _update_config(self, user_config):
        """更新可视化配置"""
        for category, settings in user_config.items():
            if category in self.viz_config:
                self.viz_config[category].update(settings)
    
    def plot_training_curves_enhanced(self, train_losses, val_losses, loss_components):
        """绘制增强的训练损失曲线"""
        try:
            print(f"   📈 正在生成训练曲线...")
            print(f"   📊 训练轮数: {len(train_losses)}")
            
            # 准备训练历史数据
            training_history = {
                'train_losses': train_losses,
                'val_losses': val_losses,
                'loss_components': loss_components
            }
            
            # 添加平滑处理
            smooth_config = self.viz_config['training_curves']
            if len(train_losses) >= 5 and smooth_config['smooth_window'] != 'none':
                try:
                    import pandas as pd
                    
                    # 自适应窗口大小
                    if smooth_config['smooth_window'] == 'auto':
                        window = max(3, min(9, len(train_losses) // 4))
                    else:
                        window = smooth_config['smooth_window']
                    
                    print(f"   🔧 使用移动平均窗口: {window}")
                    
                    # 计算移动平均
                    train_smooth = pd.Series(train_losses).rolling(window=window, center=True).mean()
                    val_smooth = pd.Series(val_losses).rolling(window=window, center=True).mean()
                    
                    # 处理边界NaN值
                    training_history['train_losses_smooth'] = train_smooth.bfill().ffill().tolist()
                    training_history['val_losses_smooth'] = val_smooth.bfill().ffill().tolist()
                    
                    print(f"   ✨ 已应用平滑处理")
                except ImportError:
                    print(f"   ⚠️ pandas不可用，跳过平滑处理")
            
            # 绘制训练曲线
            save_path = f"{self.results_dir}/training_curves_final.png"
            plot_training_curves(
                training_history, 
                save_path=save_path, 
                figsize=smooth_config['figsize']
            )
            
            print(f"   ✅ 训练曲线已保存: {save_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ 绘制训练曲线失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def plot_matrix_comparison_enhanced(self, predicted_matrix, true_matrix):
        """绘制增强的矩阵对比图"""
        if true_matrix is None:
            print("   ⚠️ 没有真实邻接矩阵，跳过对比图绘制")
            return False
        
        try:
            print(f"   📊 正在生成邻接矩阵对比图...")
            
            matrix_config = self.viz_config['matrix_comparison']
            print(f"   🎯 使用置信度阈值: {matrix_config['threshold']:.3f}")
            print(f"   📐 矩阵尺寸: {predicted_matrix.shape}")
            
            # 使用现有的矩阵对比可视化函数
            save_path = f"{self.results_dir}/adjacency_matrix_comparison.png"
            plot_probability_matrix_comparison(
                predicted_prob_matrix=predicted_matrix,
                true_causal_matrix=true_matrix,
                save_path=save_path,
                title_prefix=matrix_config['title_prefix'],
                threshold=matrix_config['threshold'],
                cmap=matrix_config['cmap']
            )
            
            print(f"   ✅ 邻接矩阵对比图已保存: {save_path}")
            return True
            
        except Exception as e:
            print(f"   ❌ 绘制矩阵对比图失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_time_series_visualizations(self, model, test_data, raw_data):
        """
        Generates a comprehensive set of long-sequence time series reconstruction visualizations.
        This function creates detailed, overview, and grid plots for up to 200 timesteps.
        """
        try:
            print(f"   📈 Generating long-sequence time series reconstruction visualizations...")
            import os
            os.makedirs(self.results_dir, exist_ok=True)
            print(f"   📂 Results will be saved in: {self.results_dir}")

            # --- 1. Infer model's processing structure (chunk size and number of nodes) ---
            structure_info = self._infer_model_structure(model, test_data)
            if not structure_info:
                print("   ❌ Could not infer model structure. Visualization terminated.")
                return False
            timesteps_per_chunk, num_nodes = structure_info
            
            # --- 2. Prepare original data for 200 timesteps ---
            print("   📖 Step 2: Preparing original data for 200 timesteps...")
            full_original_data = self._prepare_original_data(raw_data, test_data)
            if full_original_data is None:
                print(f"   ❌ Could not load original data. Visualization terminated.")
                return False
            
            max_timesteps_to_show = 200
            if full_original_data.shape[0] < max_timesteps_to_show:
                print(f"   ⚠️ Original data has fewer than 200 timesteps ({full_original_data.shape[0]} available). Using all available data.")
                max_timesteps_to_show = full_original_data.shape[0]

            original_ts_long = full_original_data[:max_timesteps_to_show, :num_nodes]
            print(f"   ✅ Prepared original data for comparison, shape: {original_ts_long.shape}")

            # --- 3. Iteratively reconstruct the long sequence ---
            print(f"   🚀 Step 3: Reconstructing sequence of length {max_timesteps_to_show}...")
            reconstructed_ts_long = self._reconstruct_long_sequence(
                model, original_ts_long, timesteps_per_chunk, num_nodes
            )
            if reconstructed_ts_long is None:
                print("   ❌ Long sequence reconstruction failed. Visualization terminated.")
                return False
            print(f"   ✅ Long sequence reconstruction complete, shape: {reconstructed_ts_long.shape}")

            # --- 4. Generate and save all visualizations ---
            print("   🎨 Step 4: Plotting and saving all visualization types...")
            ts_config = self.viz_config['time_series']
            selected_nodes = self._select_nodes_for_visualization(original_ts_long, ts_config['selected_nodes'])
            success_count = 0
            
            # Plot 1: Detailed single-node analysis
            try:
                save_path_detailed = os.path.join(self.results_dir, "time_series_reconstruction_detailed.png")
                plot_enhanced_time_series_reconstruction(
                    original_data=original_ts_long,
                    reconstructed_data=reconstructed_ts_long,
                    selected_node=selected_nodes[0] if selected_nodes else None,
                    max_timesteps=max_timesteps_to_show,
                    comparison_mode="single",
                    save_path=save_path_detailed,
                    title_prefix="Detailed Reconstruction Analysis - "
                )
                if os.path.exists(save_path_detailed): success_count += 1
            except Exception as e:
                print(f"   ❌ Failed to generate detailed plot: {e}")

            # Plot 2: Multi-node overview
            try:
                save_path_overview = os.path.join(self.results_dir, "time_series_reconstruction_overview.png")
                plot_enhanced_time_series_reconstruction(
                    original_data=original_ts_long,
                    reconstructed_data=reconstructed_ts_long,
                    max_timesteps=max_timesteps_to_show,
                    comparison_mode="multi",
                    save_path=save_path_overview,
                    title_prefix="Multi-Node Reconstruction Overview - "
                )
                if os.path.exists(save_path_overview): success_count += 1
            except Exception as e:
                print(f"   ❌ Failed to generate overview plot: {e}")

            # Plot 3: Grid comparison
            try:
                save_path_grid = os.path.join(self.results_dir, "time_series_reconstruction_grid.png")
                plot_enhanced_time_series_reconstruction(
                    original_data=original_ts_long,
                    reconstructed_data=reconstructed_ts_long,
                    selected_node=selected_nodes[0] if selected_nodes else 0,
                    max_timesteps=max_timesteps_to_show,
                    comparison_mode="grid",
                    save_path=save_path_grid,
                    title_prefix="Grid-Based Reconstruction Analysis - "
                )
                if os.path.exists(save_path_grid): success_count += 1
            except Exception as e:
                print(f"   ❌ Failed to generate grid plot: {e}")

            print(f"   ✅ Visualization generation finished. {success_count}/3 plots saved successfully.")

            # --- 5. Save statistical data ---
            print("   💾 Step 5: Saving detailed statistics...")
            try:
                self._save_reconstruction_statistics(original_ts_long, reconstructed_ts_long)
            except Exception as e:
                print(f"   ❌ Failed to save reconstruction statistics: {e}")

            return success_count > 0

        except Exception as e:
            print(f"   ❌ Failed to generate time series visualizations: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _infer_model_structure(self, model, test_data):
        """通过单次运行推断模型的输入结构（时间步长和节点数）。"""
        try:
            print("   🧩 正在推断模型结构...")
            if not hasattr(test_data, 'shape') or len(test_data.shape) < 2:
                 raise ValueError(f"测试数据形状无效: {test_data.shape if hasattr(test_data, 'shape') else 'N/A'}")

            num_features = test_data.shape[1]
            
            # 优先从模型属性获取节点数
            num_nodes = getattr(model, 'num_nodes', None)
            if num_nodes is None:
                # 再次尝试从配置中获取
                num_nodes = self.config.get('num_nodes', None)
                if num_nodes is None:
                    # 最后，作为备选，假定为10
                    num_nodes = 10
                    print(f"   ⚠️ 无法从模型或配置中获取节点数，假设为: {num_nodes}")

            if num_features % num_nodes != 0:
                raise ValueError(f"特征数({num_features})不能被节点数({num_nodes})整除")
            
            timesteps_per_chunk = num_features // num_nodes
            
            # 将结构保存在实例中，方便复用
            self.ts_structure = {'timesteps': timesteps_per_chunk, 'nodes': num_nodes}
            
            print(f"   ✅ 模型结构推断成功: 每个切片包含 {timesteps_per_chunk} 时间步 × {num_nodes} 节点。")
            return timesteps_per_chunk, num_nodes
        except Exception as e:
            print(f"   ❌ 推断模型结构失败: {e}")
            return None

    def _reconstruct_long_sequence(self, model, original_sequence, timesteps_per_chunk, num_nodes):
        """用滑动窗口的方式重构一个长序列"""
        model.eval()
        device = next(model.parameters()).device
        print(f"\n🔄 长序列将被切分为 {len(original_sequence) // timesteps_per_chunk} 个块进行处理...")
        reconstructed_chunks = []
        num_total_timesteps = original_sequence.shape[0]
        
        # 确保总时间步是切片长度的整数倍
        if num_total_timesteps % timesteps_per_chunk != 0:
            new_len = (num_total_timesteps // timesteps_per_chunk) * timesteps_per_chunk
            print(f"   ⚠️ 总时间步长({num_total_timesteps})不是切片长度({timesteps_per_chunk})的整数倍。将截断为 {new_len}。")
            original_sequence = original_sequence[:new_len, :]
            num_total_timesteps = new_len
        
        num_chunks = num_total_timesteps // timesteps_per_chunk
        print(f"   🔄 长序列将被切分为 {num_chunks} 个块进行处理...")
        
        with torch.no_grad():
            model.eval()
            device = next(model.parameters()).device
            
            for i in range(num_chunks):
                start = i * timesteps_per_chunk
                end = start + timesteps_per_chunk
                chunk_original_np = original_sequence[start:end, :]
                
                # Reshape to model input format [1, features]
                model_input_flat = torch.tensor(chunk_original_np.flatten(), dtype=torch.float32).unsqueeze(0)
                model_input_flat = model_input_flat.to(device)
                
                # Run model - 显式传递 only_prediction=True
                result = model(model_input_flat, only_prediction=True)

                # Extract prediction from dict or direct tensor
                prediction_output = None
                if isinstance(result, dict):
                    key_found = False
                    # 修正：只寻找 'prediction' 或 'prediction_mean'
                    if 'prediction' in result:
                        prediction_output = result['prediction']
                        key_found = True
                    elif 'prediction_mean' in result:
                        prediction_output = result['prediction_mean']
                        key_found = True

                    if not key_found:
                        print("   ❌ 在模型输出字典中找不到可用的预测键。")
                        return None
                else:
                    prediction_output = result
                    
                # Process prediction samples: e.g., [1, 50, 400] -> [1, 50]
                if prediction_output.ndim == 3:
                    reconstructed_flat = prediction_output.mean(dim=-1)
                else:
                    reconstructed_flat = prediction_output
                
                # Reshape back to [timesteps, nodes]
                reconstructed_chunk_np = reconstructed_flat.detach().cpu().numpy().reshape(timesteps_per_chunk, num_nodes)
                reconstructed_chunks.append(reconstructed_chunk_np)
        
        print("   ✅ 所有块处理完毕。")
        return np.vstack(reconstructed_chunks)
    
    def _prepare_reconstruction_data(self, model, test_data, raw_data):
        """准备重构数据，并提取时间序列的结构信息"""
        try:
            print(f"   🔄 准备重构数据...")
            
            import torch
            import numpy as np
            
            with torch.no_grad():
                model.eval()
                device = next(model.parameters()).device
                test_data_tensor = test_data.to(device) if isinstance(test_data, torch.Tensor) else torch.tensor(test_data, device=device).float()
                
                print(f"   📊 输入数据形状: {test_data_tensor.shape}")
                
                result = model(test_data_tensor)
                
                prediction_key = None
                if isinstance(result, dict):
                    for key in ['prediction_samples', 'predictions', 'reconstructed', 'output', 'x_pred']:
                        if key in result:
                            prediction_key = key
                            break
                    if not prediction_key:
                        raise ValueError("模型输出字典中未找到预测数据")
                    prediction_output = result[prediction_key]
                else:
                    prediction_output = result

                print(f"   📊 预测输出形状: {prediction_output.shape}")

                # 提取重构数据
                if prediction_output.ndim == 3:  # [batch, features, samples]
                    reconstructed_data = prediction_output.mean(dim=-1)
                elif prediction_output.ndim <= 2: # [batch, features] or [features]
                    reconstructed_data = prediction_output
                else:
                    raise ValueError(f"不支持的预测输出维度: {prediction_output.ndim}D")
                
                # 确保是2D [batch, features]
                if reconstructed_data.ndim == 1:
                    reconstructed_data = reconstructed_data.unsqueeze(0)
                
                print(f"   🔄 重构数据形状 (处理后): {reconstructed_data.shape}")

                # 提取时间序列结构
                # 尝试从模型属性获取
                num_nodes = getattr(model, 'num_nodes', None)
                if num_nodes is None:
                    # 如果模型没有节点信息，从原始数据推断
                    if raw_data is not None and hasattr(raw_data, 'shape') and raw_data.ndim == 2:
                        num_nodes = raw_data.shape[1]
                    else: # 最后尝试从配置中获取
                        num_nodes = self.config.get('num_nodes', 10) # 默认为10
                
                features = reconstructed_data.shape[1]
                if features % num_nodes != 0:
                     print(f"   ⚠️ 特征数({features})不能被节点数({num_nodes})整除，将使用原始数据形状")
                     if raw_data is not None and hasattr(raw_data, 'shape') and raw_data.ndim == 2:
                         num_nodes = raw_data.shape[1]
                         timesteps = raw_data.shape[0]
                     else:
                        raise ValueError("无法确定时间序列结构")
                else:
                    timesteps = features // num_nodes

                print(f"   🧩 推断出的时间序列结构: {timesteps} 时间步 × {num_nodes} 节点")

                # 保存结构信息以供后续步骤使用
                self.ts_structure = {'timesteps': timesteps, 'nodes': num_nodes}
                
                reconstructed_data_np = reconstructed_data.detach().cpu().numpy()
                return reconstructed_data_np

        except Exception as e:
            print(f"   ❌ 准备重构数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _prepare_original_data(self, raw_data, test_data):
        """准备原始数据 - 专门适配CSV格式数据（行为样本数，列为节点种类）"""
        # 方案1: 直接从CSV文件加载数据
        csv_path = "data/generated_time_series2.csv"
        if Path(csv_path).exists():
            try:
                import pandas as pd
                print(f"   📁 从CSV文件加载数据: {csv_path}")
                df = pd.read_csv(csv_path, header=None)
                original_data = df.values
                print(f"   ✅ 成功加载CSV数据: {original_data.shape}")
                return original_data
            except Exception as e:
                print(f"   ❌ 从CSV文件加载数据失败: {e}")

        # 方案2: 使用传入的raw_data
        if raw_data is not None and hasattr(raw_data, 'shape'):
            try:
                original_data = raw_data.detach().cpu().numpy() if isinstance(raw_data, torch.Tensor) else np.array(raw_data)
                print(f"   📊 使用传入的原始数据: {original_data.shape}")
                return original_data
            except Exception as e:
                print(f"   ⚠️ 处理传入的原始数据失败: {e}")

        # 方案3: 从test_data恢复
        if test_data is not None and hasattr(test_data, 'shape'):
            try:
                original_data = test_data.detach().cpu().numpy() if isinstance(test_data, torch.Tensor) else np.array(test_data)
                print(f"   📊 使用测试数据作为原始数据: {original_data.shape}")
                return original_data
            except Exception as e:
                print(f"   ⚠️ 处理测试数据失败: {e}")

        print(f"   ❌ 无法获取原始数据")
        return None
    
    def _align_data_shapes(self, original_data, reconstructed_data):
        """利用已推断的结构信息，对齐原始和重构数据形状"""
        try:
            print(f"   🔄 开始数据形状对齐...")
            print(f"   📐 输入形状: 原始{original_data.shape}, 重构{reconstructed_data.shape}")

            # 检查是否有可用的结构信息
            if not hasattr(self, 'ts_structure') or not self.ts_structure:
                raise ValueError("时间序列结构信息 (ts_structure) 不可用")
                
            timesteps = self.ts_structure['timesteps']
            nodes = self.ts_structure['nodes']
            print(f"   🧩 使用结构信息: {timesteps} 时间步 × {nodes} 节点")

            # 重塑函数
            def reshape_data(data, name):
                if data.ndim == 2 and data.shape[0] == 1: # [1, features]
                    data = data.flatten()
                
                if data.ndim == 1:
                    expected_size = timesteps * nodes
                    if len(data) == expected_size:
                        reshaped = data.reshape(timesteps, nodes)
                        print(f"   ✅ 已将 {name} 从1D重塑为 {reshaped.shape}")
                        return reshaped
                    else:
                        raise ValueError(f"{name} 数据长度({len(data)})与期望大小({expected_size})不匹配")
                elif data.ndim == 2:
                    return data # 已经是2D，直接返回
                else:
                    raise ValueError(f"{name} 数据维度({data.ndim}D)异常")

            original_data = reshape_data(original_data, "原始数据")
            reconstructed_data = reshape_data(reconstructed_data, "重构数据")

            # 截取到相同的最小长度
            min_len = min(original_data.shape[0], reconstructed_data.shape[0])
            original_data = original_data[:min_len, :]
            reconstructed_data = reconstructed_data[:min_len, :]
            
            print(f"   ✅ 数据形状对齐完成: {original_data.shape}")
            return original_data, reconstructed_data

        except Exception as e:
            print(f"   ❌ 数据对齐失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _select_nodes_for_visualization(self, data, selection_config):
        """智能选择要可视化的节点"""
        num_nodes = data.shape[1]
        
        if selection_config == 'auto':
            # 自动选择最有代表性的节点
            variances = np.var(data, axis=0)
            selected_nodes = [np.argmax(variances)]  # 选择方差最大的节点
            if num_nodes > 1:
                sorted_indices = np.argsort(variances)[::-1]
                selected_nodes = sorted_indices[:min(3, num_nodes)].tolist()
        elif isinstance(selection_config, list):
            selected_nodes = [i for i in selection_config if 0 <= i < num_nodes]
        elif isinstance(selection_config, int):
            selected_nodes = [selection_config] if 0 <= selection_config < num_nodes else [0]
        else:
            selected_nodes = [0]  # 默认选择第一个节点
        
        return selected_nodes
    
    def _save_reconstruction_statistics(self, original_data, reconstructed_data):
        """保存重构性能统计 - 增强版本，包含每个节点的详细指标"""
        try:
            print(f"   📈 保存重构性能统计...")
            
            # 修正：在计算前确保原始数据和重构数据的长度一致
            # 这是为了防止因滑动窗口处理而导致的末端数据不匹配问题
            min_len = min(len(original_data), len(reconstructed_data))
            original_data = original_data[:min_len]
            reconstructed_data = reconstructed_data[:min_len]
            
            # 计算整体统计指标
            mse = np.mean((original_data - reconstructed_data) ** 2)
            mae = np.mean(np.abs(original_data - reconstructed_data))
            rmse = np.sqrt(mse)
            
            # 计算相关系数
            correlation_matrix = np.corrcoef(original_data.flatten(), reconstructed_data.flatten())
            overall_correlation = correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
            
            # 计算R²分数（决定系数）
            ss_res = np.sum((original_data - reconstructed_data) ** 2)
            ss_tot = np.sum((original_data - np.mean(original_data)) ** 2)
            r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            # 计算每个节点的统计指标
            num_nodes = original_data.shape[1]
            node_statistics = []
            
            for node_idx in range(num_nodes):
                orig_node = original_data[:, node_idx]
                recon_node = reconstructed_data[:, node_idx]
                
                node_mse = np.mean((orig_node - recon_node) ** 2)
                node_mae = np.mean(np.abs(orig_node - recon_node))
                node_rmse = np.sqrt(node_mse)
                
                # 节点相关系数
                if np.std(orig_node) > 0 and np.std(recon_node) > 0:
                    node_corr = np.corrcoef(orig_node, recon_node)[0, 1]
                    if np.isnan(node_corr):
                        node_corr = 0.0
                else:
                    node_corr = 0.0
                
                # 节点R²分数
                node_ss_res = np.sum((orig_node - recon_node) ** 2)
                node_ss_tot = np.sum((orig_node - np.mean(orig_node)) ** 2)
                node_r2 = 1 - (node_ss_res / node_ss_tot) if node_ss_tot != 0 else 0.0
                
                # 节点数据范围
                orig_range = [float(np.min(orig_node)), float(np.max(orig_node))]
                recon_range = [float(np.min(recon_node)), float(np.max(recon_node))]
                
                node_statistics.append({
                    'node_id': node_idx + 1,
                    'mse': float(node_mse),
                    'mae': float(node_mae),
                    'rmse': float(node_rmse),
                    'correlation': float(node_corr),
                    'r2_score': float(node_r2),
                    'original_range': orig_range,
                    'reconstructed_range': recon_range,
                    'original_mean': float(np.mean(orig_node)),
                    'reconstructed_mean': float(np.mean(recon_node)),
                    'original_std': float(np.std(orig_node)),
                    'reconstructed_std': float(np.std(recon_node))
                })
            
            # 准备完整的统计报告
            statistics = {
                'dataset_info': {
                    'original_data_shape': original_data.shape,
                    'reconstructed_data_shape': reconstructed_data.shape,
                    'num_timesteps': int(original_data.shape[0]),
                    'num_nodes': int(original_data.shape[1]),
                    'data_type': 'CSV时间序列数据（行为样本数，列为节点种类）'
                },
                'overall_metrics': {
                    'mse': float(mse),
                    'mae': float(mae),
                    'rmse': float(rmse),
                    'correlation': float(overall_correlation),
                    'r2_score': float(r2_score)
                },
                'data_statistics': {
                    'original_data_range': [float(np.min(original_data)), float(np.max(original_data))],
                    'reconstructed_data_range': [float(np.min(reconstructed_data)), float(np.max(reconstructed_data))],
                    'original_data_mean': float(np.mean(original_data)),
                    'reconstructed_data_mean': float(np.mean(reconstructed_data)),
                    'original_data_std': float(np.std(original_data)),
                    'reconstructed_data_std': float(np.std(reconstructed_data))
                },
                'node_wise_metrics': node_statistics,
                'best_reconstructed_nodes': {
                    'highest_correlation': sorted(node_statistics, key=lambda x: x['correlation'], reverse=True)[:3],
                    'lowest_mse': sorted(node_statistics, key=lambda x: x['mse'])[:3],
                    'highest_r2': sorted(node_statistics, key=lambda x: x['r2_score'], reverse=True)[:3]
                },
                'visualization_info': {
                    'max_timesteps_displayed': int(min(200, original_data.shape[0])),
                    'total_available_timesteps': int(original_data.shape[0]),
                    'selected_for_visualization': f"前{min(200, original_data.shape[0])}个时间步"
                }
            }
            
            # 保存到JSON文件
            import json
            stats_path = os.path.join(self.results_dir, "reconstruction_statistics.json")
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, ensure_ascii=False)
            
            print(f"   ✅ 重构统计已保存到: {stats_path}")
            
            # 保存原始和重构数据
            np.save(os.path.join(self.results_dir, "original_data.npy"), original_data)
            np.save(os.path.join(self.results_dir, "reconstructed_data.npy"), reconstructed_data)
            print(f"   💾 原始和重构数据已保存为.npy文件")
            
            # 打印简要统计
            print(f"   📊 重构性能概览:")
            print(f"      整体MSE: {mse:.6f}")
            print(f"      整体相关系数: {overall_correlation:.3f}")
            print(f"      整体R²分数: {r2_score:.3f}")
            print(f"      最佳重构节点(相关系数): 节点{statistics['best_reconstructed_nodes']['highest_correlation'][0]['node_id']}")
            
        except Exception as e:
            print(f"   ❌ 保存重构统计失败: {e}")
            import traceback
            traceback.print_exc()
    
    def run_all_visualizations(self, train_losses=None, val_losses=None, loss_components=None,
                              predicted_matrix=None, true_matrix=None,
                              model=None, test_data=None, raw_data=None):
        """运行所有可视化 - 统一入口"""
        print(f"\n{'='*60}")
        print(f"              运行PFNO可视化套件")
        print(f"{'='*60}")
        
        success_count = 0
        total_tasks = 0
        
        # 1. 训练曲线可视化
        if train_losses is not None and val_losses is not None:
            total_tasks += 1
            print(f"\n🎯 任务 1/3: 训练曲线可视化")
            if self.plot_training_curves_enhanced(train_losses, val_losses, loss_components or {}):
                success_count += 1
        
        # 2. 矩阵对比可视化
        if predicted_matrix is not None:
            total_tasks += 1
            print(f"\n🎯 任务 2/3: 邻接矩阵对比可视化")
            if self.plot_matrix_comparison_enhanced(predicted_matrix, true_matrix):
                success_count += 1
        
        # 3. 时间序列重构可视化
        if model is not None and test_data is not None:
            total_tasks += 1
            print(f"\n🎯 任务 3/3: 时间序列重构可视化")
            if self.generate_time_series_visualizations(model, test_data, raw_data):
                success_count += 1
        
        print(f"\n{'='*60}")
        print(f"    可视化套件完成: {success_count}/{total_tasks} 个任务成功")
        print(f"{'='*60}")
        
        return success_count, total_tasks


# 更新 __all__ 列表
__all__ = [
    'plot_adjacency_matrix',
    'plot_training_curves', 
    'plot_evaluation_metrics',
    'plot_comparison_results',
    'plot_threshold_analysis',
    'create_experiment_dashboard',
    'save_plots',
    'plot_probability_matrix_comparison',
    'plot_time_series_reconstruction',
    'plot_comprehensive_results',
    'plot_enhanced_time_series_reconstruction',
    'PFNOVisualizationSuite'  # 新增统一可视化套件
] 