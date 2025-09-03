"""
评估工具模块
提供因果发现评估和各种性能指标计算
"""

import numpy as np
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.metrics import (
    precision_recall_fscore_support, accuracy_score, confusion_matrix, 
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)


def evaluate_causal_discovery(pred_matrix: np.ndarray, 
                            true_matrix: np.ndarray,
                            threshold: float = 0.5,
                            ignore_self_loops: bool = True,
                            return_detailed: bool = False) -> Dict[str, Any]:
    """
    统一的因果发现评估函数
    
    Args:
        pred_matrix: 预测的邻接矩阵
        true_matrix: 真实的邻接矩阵
        threshold: 二值化阈值
        ignore_self_loops: 是否忽略自环
        return_detailed: 是否返回详细结果
        
    Returns:
        评估结果字典
    """
    # 输入验证
    if pred_matrix.shape != true_matrix.shape:
        raise ValueError(f"矩阵形状不匹配: {pred_matrix.shape} vs {true_matrix.shape}")
    
    # 处理自环
    if ignore_self_loops:
        np.fill_diagonal(pred_matrix, 0)
        np.fill_diagonal(true_matrix, 0)
    
    # 二值化预测矩阵
    pred_binary = (pred_matrix > threshold).astype(int)
    
    # 展平矩阵（移除对角线元素）
    if ignore_self_loops:
        mask = ~np.eye(true_matrix.shape[0], dtype=bool)
        y_true = true_matrix[mask].flatten()
        y_pred = pred_binary[mask].flatten()
        y_pred_prob = pred_matrix[mask].flatten()
    else:
        y_true = true_matrix.flatten()
        y_pred = pred_binary.flatten()
        y_pred_prob = pred_matrix.flatten()
    
    # 计算基本指标
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算混淆矩阵指标
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    
    # 计算AUROC（如果有概率预测）
    try:
        auroc = roc_auc_score(y_true, y_pred_prob)
    except:
        auroc = -1
    
    # 计算AUPRC
    try:
        auprc = average_precision_score(y_true, y_pred_prob)
    except:
        auprc = -1
    
    # 计算结构指标
    shd = np.sum(pred_binary != true_matrix)  # Structural Hamming Distance
    
    # 基本结果
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tpr': tpr,
        'fpr': fpr,
        'tnr': tnr,
        'auroc': auroc,
        'auprc': auprc,
        'shd': shd,
        'threshold': threshold
    }
    
    # 详细结果
    if return_detailed:
        results.update({
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'total_edges_true': int(np.sum(true_matrix)),
            'total_edges_pred': int(np.sum(pred_binary)),
            'matrix_shape': pred_matrix.shape,
            'pred_matrix_stats': {
                'min': float(pred_matrix.min()),
                'max': float(pred_matrix.max()),
                'mean': float(pred_matrix.mean()),
                'std': float(pred_matrix.std())
            },
            'edge_ratio_true': float(np.sum(true_matrix)) / (true_matrix.size - (true_matrix.shape[0] if ignore_self_loops else 0)),
            'edge_ratio_pred': float(np.sum(pred_binary)) / (pred_binary.size - (pred_binary.shape[0] if ignore_self_loops else 0))
        })
    
    return results


def find_optimal_threshold(pred_matrix: np.ndarray, 
                          true_matrix: np.ndarray,
                          thresholds: Optional[List[float]] = None,
                          metric: str = 'f1',
                          return_all_results: bool = False) -> Union[Tuple[float, float], Tuple[float, float, List[Dict]]]:
    """
    寻找最优阈值
    
    Args:
        pred_matrix: 预测矩阵
        true_matrix: 真实矩阵
        thresholds: 要测试的阈值列表
        metric: 优化的指标 ('f1', 'accuracy', 'precision', 'recall', 'auroc')
        return_all_results: 是否返回所有结果
        
    Returns:
        (最优阈值, 最优分数) 或 (最优阈值, 最优分数, 所有结果)
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50).tolist()
    
    results = []
    best_score = -np.inf if metric == 'auroc' else 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        try:
            eval_result = evaluate_causal_discovery(
                pred_matrix, true_matrix, threshold=threshold
            )
            
            score = eval_result.get(metric, 0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
            
            results.append({
                'threshold': threshold,
                'score': score,
                **eval_result
            })
            
        except Exception as e:
            warnings.warn(f"阈值 {threshold} 评估失败: {e}")
            continue
    
    if return_all_results:
        return best_threshold, best_score, results
    else:
        return best_threshold, best_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                   y_pred_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    计算各种分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_pred_prob: 预测概率
        
    Returns:
        指标字典
    """
    metrics = {}
    
    # 基本分类指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1'] = f1
    
    # 混淆矩阵相关指标
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # 如果有概率预测，计算ROC和PR相关指标
    if y_pred_prob is not None:
        try:
            metrics['auroc'] = roc_auc_score(y_true, y_pred_prob)
            metrics['auprc'] = average_precision_score(y_true, y_pred_prob)
        except:
            metrics['auroc'] = -1
            metrics['auprc'] = -1
    
    return metrics


def confusion_matrix_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    计算混淆矩阵及相关指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        
    Returns:
        混淆矩阵指标字典
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # 基本混淆矩阵信息
    result = {
        'confusion_matrix': cm,
        'n_classes': cm.shape[0]
    }
    
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        result.update({
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        })
    
    return result


def roc_analysis(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, Any]:
    """
    ROC分析
    
    Args:
        y_true: 真实标签
        y_pred_prob: 预测概率
        
    Returns:
        ROC分析结果
    """
    try:
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        auroc = roc_auc_score(y_true, y_pred_prob)
        
        # 找到最优阈值（Youden index）
        youden_index = tpr - fpr
        optimal_idx = np.argmax(youden_index)
        optimal_threshold = thresholds[optimal_idx]
        
        return {
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds,
            'auroc': auroc,
            'optimal_threshold': optimal_threshold,
            'optimal_youden_index': youden_index[optimal_idx]
        }
    except Exception as e:
        warnings.warn(f"ROC分析失败: {e}")
        return {'auroc': -1}


def precision_recall_analysis(y_true: np.ndarray, y_pred_prob: np.ndarray) -> Dict[str, Any]:
    """
    精确率-召回率分析
    
    Args:
        y_true: 真实标签
        y_pred_prob: 预测概率
        
    Returns:
        PR分析结果
    """
    try:
        # 计算PR曲线
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)
        auprc = average_precision_score(y_true, y_pred_prob)
        
        # 找到最优阈值（F1最大）
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        
        if optimal_idx < len(thresholds):
            optimal_threshold = thresholds[optimal_idx]
        else:
            optimal_threshold = 0.5
        
        return {
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'auprc': auprc,
            'optimal_threshold': optimal_threshold,
            'optimal_f1': f1_scores[optimal_idx]
        }
    except Exception as e:
        warnings.warn(f"PR分析失败: {e}")
        return {'auprc': -1}


def edge_orientation_accuracy(pred_matrix: np.ndarray, true_matrix: np.ndarray,
                             threshold: float = 0.5) -> Dict[str, float]:
    """
    计算边方向准确性（用于有向图）
    
    Args:
        pred_matrix: 预测的邻接矩阵
        true_matrix: 真实的邻接矩阵
        threshold: 二值化阈值
        
    Returns:
        方向准确性指标
    """
    pred_binary = (pred_matrix > threshold).astype(int)
    
    # 移除自环
    np.fill_diagonal(pred_binary, 0)
    np.fill_diagonal(true_matrix, 0)
    
    # 计算边的存在准确性
    edge_existence = np.logical_or(pred_binary, pred_binary.T)
    true_edge_existence = np.logical_or(true_matrix, true_matrix.T)
    
    # 计算方向准确性（只考虑存在的边）
    existing_edges = np.logical_and(edge_existence, true_edge_existence)
    if np.sum(existing_edges) > 0:
        correct_directions = np.sum(np.logical_and(pred_binary, true_matrix))
        total_directions = np.sum(existing_edges) // 2  # 每条边计算一次
        direction_accuracy = correct_directions / total_directions if total_directions > 0 else 0
    else:
        direction_accuracy = 0
    
    return {
        'direction_accuracy': direction_accuracy,
        'total_edges': int(np.sum(existing_edges) // 2),
        'correct_directions': int(np.sum(np.logical_and(pred_binary, true_matrix)))
    }


def structural_hamming_distance(pred_matrix: np.ndarray, true_matrix: np.ndarray,
                               threshold: float = 0.5) -> int:
    """
    计算结构汉明距离
    
    Args:
        pred_matrix: 预测的邻接矩阵
        true_matrix: 真实的邻接矩阵
        threshold: 二值化阈值
        
    Returns:
        结构汉明距离
    """
    pred_binary = (pred_matrix > threshold).astype(int)
    
    # 移除自环
    np.fill_diagonal(pred_binary, 0)
    np.fill_diagonal(true_matrix, 0)
    
    return int(np.sum(pred_binary != true_matrix))


def evaluate_multiple_thresholds(pred_matrix: np.ndarray, true_matrix: np.ndarray,
                                thresholds: Optional[List[float]] = None) -> Dict[str, List]:
    """
    在多个阈值下评估性能
    
    Args:
        pred_matrix: 预测矩阵
        true_matrix: 真实矩阵
        thresholds: 阈值列表
        
    Returns:
        每个阈值的评估结果
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 9).tolist()
    
    results = {
        'thresholds': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'auroc': [],
        'shd': []
    }
    
    for threshold in thresholds:
        try:
            eval_result = evaluate_causal_discovery(pred_matrix, true_matrix, threshold)
            
            results['thresholds'].append(threshold)
            results['accuracy'].append(eval_result['accuracy'])
            results['precision'].append(eval_result['precision'])
            results['recall'].append(eval_result['recall'])
            results['f1'].append(eval_result['f1'])
            results['auroc'].append(eval_result['auroc'])
            results['shd'].append(eval_result['shd'])
            
        except Exception as e:
            warnings.warn(f"阈值 {threshold} 评估失败: {e}")
            continue
    
    return results


__all__ = [
    'evaluate_causal_discovery',
    'find_optimal_threshold',
    'compute_metrics',
    'confusion_matrix_metrics',
    'roc_analysis',
    'precision_recall_analysis',
    'edge_orientation_accuracy',
    'structural_hamming_distance',
    'evaluate_multiple_thresholds'
] 