"""
实验管理工具模块
提供实验保存、加载、报告生成等功能
"""

import json
import pickle
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def save_experiment(results: Dict[str, Any], 
                   config: Dict[str, Any], 
                   save_dir: str,
                   experiment_name: str = "pfno_experiment") -> str:
    """
    保存实验结果和配置
    
    Args:
        results: 实验结果字典
        config: 配置字典
        save_dir: 保存目录
        experiment_name: 实验名称
        
    Returns:
        保存路径
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # 添加时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{experiment_name}_{timestamp}"
    
    # 保存配置
    config_path = save_path / f"{experiment_name}_config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    # 保存结果（处理numpy数组）
    def serialize_results(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    results_json_path = save_path / f"{experiment_name}_results.json"
    with open(results_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=serialize_results)
    
    # 保存pickle版本（保留原始数据类型）
    results_pkl_path = save_path / f"{experiment_name}_results.pkl"
    with open(results_pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    # 生成报告
    report_path = save_path / f"{experiment_name}_report.txt"
    generate_report(results, config, str(report_path))
    
    print(f"✅ 实验结果已保存到: {save_path}")
    return str(save_path)


def load_experiment(experiment_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    加载实验结果和配置
    
    Args:
        experiment_path: 实验路径
        
    Returns:
        (results, config) 元组
    """
    experiment_path = Path(experiment_path)
    
    if experiment_path.is_file():
        # 如果是文件路径，从文件名推断实验名
        experiment_dir = experiment_path.parent
        experiment_name = experiment_path.stem
        if experiment_name.endswith('_config'):
            experiment_name = experiment_name[:-7]
        elif experiment_name.endswith('_results'):
            experiment_name = experiment_name[:-8]
    else:
        # 如果是目录路径，查找最新的实验
        experiment_dir = experiment_path
        experiment_files = list(experiment_dir.glob("*_config.json"))
        if not experiment_files:
            raise FileNotFoundError(f"在 {experiment_dir} 中找不到实验配置文件")
        
        # 按修改时间排序，获取最新的
        latest_config = max(experiment_files, key=lambda p: p.stat().st_mtime)
        experiment_name = latest_config.stem[:-7]
    
    # 加载配置
    config_path = experiment_dir / f"{experiment_name}_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 加载结果
    results_path = experiment_dir / f"{experiment_name}_results.pkl"
    with open(results_path, 'rb') as f:
        results = pickle.load(f)
    
    return results, config


def generate_report(results: Dict[str, Any], 
                   config: Dict[str, Any], 
                   report_path: str):
    """
    生成实验报告
    
    Args:
        results: 实验结果
        config: 配置信息
        report_path: 报告保存路径
    """
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("                    PFNO 实验报告\n")
        f.write("="*80 + "\n\n")
        
        # 实验基本信息
        f.write("📋 实验信息:\n")
        f.write("-"*40 + "\n")
        f.write(f"   生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 配置信息
        f.write("\n📐 配置参数:\n")
        f.write("-"*40 + "\n")
        
        for section, settings in config.items():
            f.write(f"   {section.upper()}:\n")
            if isinstance(settings, dict):
                for key, value in settings.items():
                    f.write(f"     {key}: {value}\n")
            else:
                f.write(f"     {settings}\n")
        
        # 评估结果
        if 'evaluation' in results:
            eval_results = results['evaluation']
            f.write("\n🎯 评估结果:\n")
            f.write("-"*40 + "\n")
            
            for metric, value in eval_results.items():
                if isinstance(value, (int, float)):
                    f.write(f"   {metric}: {value:.4f}\n")
                else:
                    f.write(f"   {metric}: {value}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")


def compare_experiments(experiment_paths: List[str]) -> Dict[str, Any]:
    """
    比较多个实验的结果
    
    Args:
        experiment_paths: 实验路径列表
        
    Returns:
        比较结果字典
    """
    comparison_data = []
    
    for exp_path in experiment_paths:
        try:
            results, config = load_experiment(exp_path)
            
            # 提取实验名称
            exp_name = Path(exp_path).name if Path(exp_path).is_dir() else Path(exp_path).stem
            
            # 提取指标
            row_data = {'experiment': exp_name}
            
            # 从评估结果中提取指标
            if 'evaluation' in results:
                eval_results = results['evaluation']
                for metric in ['f1', 'precision', 'recall', 'accuracy']:
                    if metric in eval_results:
                        row_data[metric] = eval_results[metric]
            
            comparison_data.append(row_data)
            
        except Exception as e:
            print(f"⚠️ 加载实验 {exp_path} 时出错: {e}")
            continue
    
    return {'experiments': comparison_data}


def create_experiment_summary(experiment_dir: str) -> Dict[str, Any]:
    """
    创建实验目录的摘要
    
    Args:
        experiment_dir: 实验目录路径
        
    Returns:
        实验摘要字典
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        raise FileNotFoundError(f"实验目录不存在: {experiment_dir}")
    
    # 查找所有实验文件
    config_files = list(experiment_dir.glob("*_config.json"))
    
    summary = {
        'directory': str(experiment_dir),
        'total_experiments': len(config_files),
        'experiment_list': [],
        'created_time': datetime.datetime.now().isoformat()
    }
    
    # 分析每个实验
    for config_file in config_files:
        experiment_name = config_file.stem[:-7]  # 移除_config后缀
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            exp_info = {
                'name': experiment_name,
                'config_file': str(config_file),
                'created_time': datetime.datetime.fromtimestamp(
                    config_file.stat().st_ctime
                ).isoformat()
            }
            
            summary['experiment_list'].append(exp_info)
            
        except Exception as e:
            print(f"⚠️ 分析实验 {experiment_name} 时出错: {e}")
            continue
    
    return summary


def cleanup_old_experiments(experiment_dir: str, 
                           keep_last: int = 10,
                           dry_run: bool = True) -> List[str]:
    """
    清理旧的实验文件
    
    Args:
        experiment_dir: 实验目录
        keep_last: 保留最近的实验数量
        dry_run: 是否为试运行（不实际删除）
        
    Returns:
        被删除（或将被删除）的文件列表
    """
    experiment_dir = Path(experiment_dir)
    
    if not experiment_dir.exists():
        return []
    
    # 获取所有配置文件并按时间排序
    config_files = list(experiment_dir.glob("*_config.json"))
    config_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # 确定要删除的文件
    to_delete = []
    
    if len(config_files) > keep_last:
        old_configs = config_files[keep_last:]
        
        for config_file in old_configs:
            experiment_name = config_file.stem[:-7]  # 移除_config后缀
            
            # 找到相关的所有文件
            related_files = list(experiment_dir.glob(f"{experiment_name}_*"))
            to_delete.extend(related_files)
    
    # 执行删除操作
    deleted_files = []
    
    for file_path in to_delete:
        if not dry_run:
            try:
                file_path.unlink()
                deleted_files.append(str(file_path))
            except Exception as e:
                print(f"⚠️ 删除文件 {file_path} 时出错: {e}")
        else:
            deleted_files.append(str(file_path))
    
    if dry_run:
        print(f"🧪 试运行模式：将删除 {len(deleted_files)} 个文件")
    else:
        print(f"🗑️ 已删除 {len(deleted_files)} 个旧实验文件")
    
    return deleted_files


__all__ = [
    'save_experiment',
    'load_experiment',
    'generate_report',
    'compare_experiments',
    'create_experiment_summary',
    'cleanup_old_experiments'
] 