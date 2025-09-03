import os
import time
import numpy as np
import pandas as pd
import torch
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import traceback

# 导入自定义模块
from models.model import PFNO_Causal
from train import train_model, setup_trainer, load_data
# 从main.py导入评估函数，避免函数实现不一致问题
from main import evaluate_causal_discovery, find_optimal_threshold, plot_adjacency_matrix, plot_comparison_adjacency_matrix

class GridSearch:
    def __init__(self, param_grid, results_dir='results/grid_search', base_config=None):
        """
        初始化网格搜索类
        
        Args:
            param_grid: 包含需要搜索的参数及其取值范围的字典
            results_dir: 结果保存目录
            base_config: 基础配置，如果为None则使用config.py中的CONFIG
        """
        # 设置matplotlib为非交互模式
        import matplotlib
        matplotlib.use('Agg')
        
        self.param_grid = param_grid
        self.results_dir = results_dir
        
        # 导入配置，放在这里避免循环导入问题
        from config import CONFIG
        self.base_config = base_config if base_config is not None else CONFIG.copy()
        
        # 确保结果目录存在
        os.makedirs(results_dir, exist_ok=True)
        
        # 初始化结果列表
        self.results = []
        
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def _update_config(self, params):
        """
        根据当前参数更新配置
        
        Args:
            params: 当前参数组合字典
            
        Returns:
            更新后的配置字典
        """
        config = self.base_config.copy()
        
        # 更新配置中的参数
        for param_name, param_value in params.items():
            # 解析参数名称和配置部分
            parts = param_name.split('.')
            if len(parts) == 2:
                section, key = parts
                if section in config and key in config[section]:
                    # 确保正确的类型转换
                    if key in ['learning_rate', 'causal_reg_weight', 'dropout', 'mlp_dropout', 'early_stopping_thresh']:
                        param_value = float(param_value)
                    elif key in ['n_layers', 'hidden_channels', 'init_iter', 'patience', 'batch_size', 'num_epochs', 'sequence_length']:
                        param_value = int(param_value)
                    elif key in ['use_diagonal_mask', 'use_dag_constraint', 'normalize']:
                        param_value = bool(param_value)
                    
                    config[section][key] = param_value
                    print(f"更新参数: {section}.{key} = {param_value} (类型: {type(param_value).__name__})")
                else:
                    print(f"警告: 配置中不存在 {section}.{key}")
            else:
                print(f"警告: 参数名称格式错误 {param_name}")
        
        return config
    
    def run(self, num_epochs=20, evaluation_metric='f1'):
        """
        执行网格搜索
        
        Args:
            num_epochs: 每个模型训练的最大轮次
            evaluation_metric: 用于比较模型的评估指标 ('f1', 'accuracy', 'precision', 'recall')
            
        Returns:
            最佳参数组合和结果
        """
        # 确保matplotlib不显示图表
        import matplotlib
        matplotlib.use('Agg')  # 重新设置非交互式后端
        
        # 确保num_epochs是整数
        num_epochs = int(num_epochs)
        
        # 生成所有参数组合
        param_keys = list(self.param_grid.keys())
        param_values = list(self.param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        print(f"开始网格搜索，共 {len(param_combinations)} 种参数组合")
        print(f"搜索参数: {param_keys}")
        print(f"评估指标: {evaluation_metric}")
        start_time = time.time()
        
        # 添加参数类型检查
        for key, values in self.param_grid.items():
            param_type = type(values[0]).__name__
            print(f"参数 {key}: 类型={param_type}, 范围={values}")
        
        try:
            # 加载一次数据，所有模型共用
            file_path = self.base_config['data']['file_path']
            
            # 检查load_data函数所需的参数和返回值
            print(f"正在加载数据...")
            try:
                data_results = load_data(
                    file_path=file_path,
                    sequence_length=self.base_config['training']['sequence_length'],
                    train_ratio=self.base_config['data']['train_ratio'],
                    batch_size=self.base_config['training']['batch_size'],
                    max_time_steps=self.base_config['data']['max_time_steps'],
                    normalize=self.base_config['data']['normalize'],
                    augment=self.base_config['data'].get('augment', True)
                )
                print(f"load_data返回类型: {type(data_results)}")
                print(f"load_data返回元组长度: {len(data_results) if isinstance(data_results, tuple) else 'Not a tuple'}")
                
                # 安全地解包数据结果
                if isinstance(data_results, tuple) and len(data_results) >= 4:
                    if len(data_results) == 5:
                        train_dataloader, val_dataloader, sequence_length, num_nodes, scalers = data_results
                    elif len(data_results) == 4:
                        train_dataloader, val_dataloader, sequence_length, num_nodes = data_results
                        scalers = None
                    
                    print(f"  元素 0: 类型={type(train_dataloader)}")
                    print(f"  元素 1: 类型={type(val_dataloader)}")
                    print(f"  元素 2: 类型={type(sequence_length)}")
                    print(f"  元素 3: 类型={type(num_nodes)}")
                    if len(data_results) > 4:
                        print(f"  元素 4: 类型={type(data_results[4])}")
                else:
                    # 尝试其他可能的解包方式
                    print(f"警告: load_data返回结果数量不匹配，尝试备选解包方式")
                    try:
                        # 尝试直接获取数据加载器
                        if isinstance(data_results, tuple) and len(data_results) >= 2:
                            train_dataloader = data_results[0]
                            val_dataloader = data_results[1]
                            
                            # 尝试从数据推断其他参数
                            try:
                                sample_batch = next(iter(train_dataloader))
                                inputs = sample_batch[0]
                                if isinstance(inputs, torch.Tensor):
                                    # 假设是[batch_size, num_nodes*sequence_length]格式
                                    batch_size, feat_dim = inputs.shape
                                    
                                    # 尝试根据配置推断节点数和序列长度
                                    num_nodes = int(self.base_config['data'].get('num_nodes', 10))
                                    sequence_length = int(feat_dim / num_nodes)
                                    
                                    print(f"从数据推断 - 序列长度: {sequence_length}, 节点数: {num_nodes}")
                                else:
                                    raise ValueError(f"输入数据格式异常: {type(inputs)}")
                            except Exception as e:
                                print(f"从数据推断参数失败: {str(e)}")
                                # 使用配置中的默认值
                                sequence_length = int(self.base_config['training']['sequence_length'])
                                num_nodes = 10  # 默认值
                                print(f"使用配置默认值 - 序列长度: {sequence_length}, 节点数: {num_nodes}")
                        else:
                            raise ValueError(f"无法解析load_data的返回结果，类型: {type(data_results)}")
                    except Exception as e:
                        print(f"无法处理load_data返回结果: {str(e)}")
                        traceback.print_exc()
                        return None, None
            except Exception as e:
                print(f"加载数据时出错: {str(e)}")
                traceback.print_exc()
                return None, None
            
            ground_truth_file = "causal_adjacency_matrix2.csv"
            ground_truth_exists = os.path.exists(ground_truth_file)
            
            if ground_truth_exists:
                print(f"找到真实邻接矩阵: {ground_truth_file}")
                gt_df = pd.read_csv(ground_truth_file, index_col=0)
                ground_truth = gt_df.values
                # 确保是二值矩阵
                if not np.all(np.isin(ground_truth, [0, 1])):
                    ground_truth = (ground_truth > 0).astype(float)
                print(f"真实邻接矩阵形状: {ground_truth.shape}")
            else:
                print("未找到真实邻接矩阵，无法计算评估指标")
                return None, None
            
            # 遍历所有参数组合
            for i, param_values in enumerate(tqdm(param_combinations, desc="参数组合")):
                # 构建当前参数字典
                current_params = {param_keys[j]: param_values[j] for j in range(len(param_keys))}
                
                # 更新配置
                current_config = self._update_config(current_params)
                
                # 构建模型ID和保存路径
                model_id = f"model_{i:03d}"
                model_dir = os.path.join(self.results_dir, model_id)
                os.makedirs(model_dir, exist_ok=True)
                
                print(f"\n\n===== 正在训练模型 {i+1}/{len(param_combinations)} =====")
                print(f"参数组合: {current_params}")
                
                try:
                    # 确保所有参数类型正确
                    hidden_channels = int(current_config['model']['hidden_channels'])
                    n_layers = int(current_config['model']['n_layers'])
                    dropout = float(current_config['model']['dropout'])
                    mlp_dropout = float(current_config['model']['mlp_dropout'])
                    use_diagonal_mask = bool(current_config['model']['use_diagonal_mask'])
                    use_dag_constraint = bool(current_config['model']['use_dag_constraint'])
                    causal_reg_weight = float(current_config['model'].get('causal_reg_weight', 0.01))
                    
                    # 打印参数类型用于调试
                    print(f"模型参数类型: hidden_channels={type(hidden_channels)}, n_layers={type(n_layers)}, causal_reg_weight={type(causal_reg_weight)}")
                    
                    # 先打印变量类型，帮助诊断
                    print(f"sequence_length类型: {type(sequence_length)}")
                    print(f"num_nodes类型: {type(num_nodes)}")

                    # 尝试从数据中获取维度信息
                    if not isinstance(sequence_length, int):
                        # 如果sequence_length不是整数，尝试从配置中获取
                        seq_len = int(self.base_config['training']['sequence_length'])
                        print(f"从配置获取序列长度: {seq_len}")
                    else:
                        seq_len = sequence_length
                        print(f"序列长度: {seq_len}")
                    
                    # 确保num_nodes是整数
                    if not isinstance(num_nodes, int):
                        # 尝试从第一个批次中推断节点数
                        try:
                            sample_batch = next(iter(train_dataloader))
                            inputs = sample_batch[0]
                            nodes = inputs.shape[-1] if len(inputs.shape) > 1 else int(np.sqrt(inputs.shape[1]))
                            print(f"从数据批次推断节点数: {nodes}")
                        except Exception as e:
                            print(f"无法从数据批次推断节点数: {e}")
                            nodes = 10  # 默认值
                            print(f"使用默认节点数: {nodes}")
                    else:
                        nodes = num_nodes
                        print(f"节点数: {nodes}")
                    
                    # 计算in_channels - 如果可能，检查序列长度是否已经包含在输入维度中
                    try:
                        sample_batch = next(iter(train_dataloader))
                        inputs = sample_batch[0]
                        if inputs.shape[1] == seq_len * nodes:
                            # 输入已经被展平为[batch_size, seq_len*nodes]
                            in_channels = inputs.shape[1]
                        else:
                            # 否则遵循标准计算方式
                            in_channels = seq_len * nodes
                        print(f"计算得到的in_channels: {in_channels}")
                    except Exception as e:
                        print(f"计算in_channels时出错: {e}")
                        in_channels = seq_len * nodes
                        print(f"使用默认计算的in_channels: {in_channels}")
                    
                    # 初始化模型使用处理后的in_channels和nodes
                    model = PFNO_Causal(
                        n_modes=[16, 16],
                        hidden_channels=hidden_channels,
                        num_nodes=nodes,
                        in_channels=in_channels,
                        out_channels=nodes,
                        n_layers=n_layers,
                        dropout=dropout,
                        use_mlp=True,
                        mlp_dropout=mlp_dropout,
                        non_linearity=torch.nn.GELU(),
                        use_diagonal_mask=use_diagonal_mask,
                        use_dag_constraint=use_dag_constraint,
                        causal_reg_weight=causal_reg_weight
                    )
                    
                    # 设置trainer
                    trainer = setup_trainer(
                        n=len(train_dataloader.dataset),
                        d=nodes,
                        model=model,
                        lr=float(current_config['training']['learning_rate']),
                        init_iter=int(current_config['training']['init_iter']),
                        early_stopping_thresh=float(current_config['training']['early_stopping_thresh']),
                        patience=int(current_config['training']['patience']),
                        device=self.device,
                        sequence_length=seq_len,
                        dag_constraint=True
                    )
                    
                    # 训练模型
                    train_losses, val_losses, optimized_weighted_adj, final_adj = train_model(
                        trainer, 
                        train_dataloader, 
                        num_epochs=num_epochs, 
                        validation_dataloader=val_dataloader
                    )
                    
                    # 评估不同阈值下的性能
                    thresholds = np.arange(0.1, 0.55, 0.05)
                    best_threshold = None
                    best_metric_value = -np.inf
                    threshold_results = []
                    
                    for threshold in thresholds:
                        # 二值化预测矩阵
                        binarized_adj = (optimized_weighted_adj.cpu().numpy() > threshold).astype(float)
                        
                        # 使用main.py中的evaluate_causal_discovery函数
                        if ground_truth_exists:
                            metrics = evaluate_causal_discovery(binarized_adj, ground_truth_file)
                        else:
                            continue
                        
                        # 确保所有plt图表都被关闭
                        plt.close('all')
                        
                        if metrics:
                            threshold_results.append({
                                'threshold': threshold,
                                'accuracy': metrics['accuracy'],
                                'precision': metrics['precision'],
                                'recall': metrics['recall'],
                                'f1': metrics['f1'],
                                'jaccard': metrics['jaccard'],
                                'auroc': metrics.get('auroc', -1),  # 添加AUROC指标
                                'shd': metrics.get('shd', float('inf'))  # 添加SHD指标
                            })
                            
                            # 更新最佳阈值
                            current_metric = metrics[evaluation_metric]
                            if current_metric > best_metric_value:
                                best_metric_value = current_metric
                                best_threshold = threshold
                    
                    # 记录结果
                    result = {
                        'model_id': model_id,
                        'params': current_params,
                        'best_threshold': best_threshold,
                        'best_val_loss': min(val_losses) if val_losses else None
                    }
                    
                    # 添加最佳指标
                    if threshold_results:
                        best_result = next((r for r in threshold_results if r['threshold'] == best_threshold), None)
                        if best_result:
                            for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'jaccard', 'auroc', 'shd']:
                                result[f'best_{metric_name}'] = best_result[metric_name]
                    
                    self.results.append(result)
                    
                    # 保存结果到CSV
                    results_df = pd.DataFrame(self.results)
                    results_df.to_csv(os.path.join(self.results_dir, 'grid_search_results.csv'), index=False)
                    
                    # 保存当前模型的阈值性能结果
                    if threshold_results:
                        threshold_df = pd.DataFrame(threshold_results)
                        threshold_df.to_csv(os.path.join(model_dir, 'threshold_performance.csv'), index=False)
                    
                    # 保存配置
                    pd.DataFrame([{'.'.join([section, key]): value 
                                   for section, section_dict in current_config.items() 
                                   for key, value in section_dict.items()}]).to_csv(
                        os.path.join(model_dir, 'config.csv'), index=False)
                    
                    # 保存邻接矩阵
                    np.save(os.path.join(model_dir, 'optimized_weighted_adj.npy'), 
                            optimized_weighted_adj.cpu().numpy())
                    np.save(os.path.join(model_dir, 'final_adj.npy'), 
                            final_adj.cpu().numpy())
                    
                    # 绘制损失曲线
                    plt.figure(figsize=(10, 6))
                    plt.plot(train_losses, label='Training Loss')
                    if val_losses:
                        plt.plot(val_losses, label='Validation Loss')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title(f'Loss Curves for {model_id}')
                    plt.legend()
                    save_and_close_fig(plt, os.path.join(model_dir, 'loss_curves.png'))
                    
                except Exception as e:
                    print(f"模型 {model_id} 训练失败: {str(e)}")
                    traceback.print_exc()  # 打印完整堆栈信息
                    # 记录失败结果
                    self.results.append({
                        'model_id': model_id,
                        'params': current_params,
                        'error': str(e)
                    })
                    # 保存中间结果
                    results_df = pd.DataFrame(self.results)
                    results_df.to_csv(os.path.join(self.results_dir, 'grid_search_results.csv'), index=False)
            
            # 计算总耗时
            total_time = time.time() - start_time
            print(f"\n网格搜索完成，总耗时: {total_time:.2f} 秒")
            
            # 找出最佳参数组合
            if self.results:
                valid_results = [r for r in self.results if f'best_{evaluation_metric}' in r]
                if valid_results:
                    best_result = max(valid_results, key=lambda x: x[f'best_{evaluation_metric}'])
                    print(f"\n最佳参数组合 (基于 {evaluation_metric}):")
                    for param, value in best_result['params'].items():
                        print(f"  {param}: {value}")
                    print(f"最佳阈值: {best_result['best_threshold']}")
                    print(f"最佳 {evaluation_metric}: {best_result[f'best_{evaluation_metric}']}")
                    
                    # 打印全部指标
                    print("\n全部评估指标:")
                    print(f"  准确率 (Accuracy): {best_result['best_accuracy']:.4f}")
                    print(f"  精确率 (Precision): {best_result['best_precision']:.4f}")
                    print(f"  召回率 (Recall): {best_result['best_recall']:.4f}")
                    print(f"  F1 分数: {best_result['best_f1']:.4f}")
                    print(f"  Jaccard 相似度: {best_result['best_jaccard']:.4f}")
                    
                    # 打印新增指标
                    if 'best_auroc' in best_result:
                        auroc = best_result['best_auroc']
                        print(f"  AUROC: {auroc:.4f}" if auroc >= 0 else "  AUROC: N/A")
                    
                    if 'best_shd' in best_result:
                        shd = best_result['best_shd']
                        print(f"  结构汉明距离 (SHD): {shd}")
                    
                    # 创建可视化结果
                    self._visualize_results(evaluation_metric)
                    
                    # 结束时确保所有图表都被关闭
                    plt.close('all')
                    
                    return best_result['params'], best_result
                
            # 结束时确保所有图表都被关闭
            plt.close('all')
            
            return None, None
            
        except Exception as e:
            print(f"执行网格搜索时发生错误: {str(e)}")
            traceback.print_exc()
            # 结束时确保所有图表都被关闭
            plt.close('all')
            return None, None
    
    def _visualize_results(self, evaluation_metric='f1'):
        """
        可视化网格搜索结果
        
        Args:
            evaluation_metric: 用于比较模型的评估指标
        """
        # 转换结果为DataFrame
        results_df = pd.DataFrame(self.results)
        
        # 过滤掉出错的结果
        results_df = results_df[~results_df.get('error', pd.Series([False] * len(results_df))).notna()]
        
        if len(results_df) == 0 or f'best_{evaluation_metric}' not in results_df.columns:
            print("没有足够的结果进行可视化")
            return
        
        # 为每个参数创建单独的图表
        for param in self.param_grid.keys():
            plt.figure(figsize=(10, 6))
            
            if param in results_df['params'].iloc[0]:
                # 提取参数值
                param_values = results_df['params'].apply(lambda x: x.get(param))
                
                # 绘制参数值与指标的关系
                plt.scatter(param_values, results_df[f'best_{evaluation_metric}'])
                plt.xlabel(param)
                plt.ylabel(f'Best {evaluation_metric}')
                plt.title(f'Effect of {param} on {evaluation_metric}')
                plt.grid(True)
                
                # 如果是数值参数，尝试拟合曲线
                if all(isinstance(x, (int, float)) for x in param_values if x is not None):
                    try:
                        from scipy import stats
                        
                        # 过滤掉None值
                        valid_mask = param_values.notna()
                        x = param_values[valid_mask].astype(float)
                        y = results_df[f'best_{evaluation_metric}'][valid_mask]
                        
                        # 计算相关系数
                        corr, _ = stats.pearsonr(x, y)
                        plt.annotate(f'Correlation: {corr:.2f}', 
                                    xy=(0.05, 0.95), 
                                    xycoords='axes fraction')
                        
                        # 拟合线性回归
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                        plt.plot(x, intercept + slope * x, 'r', 
                                label=f'y={intercept:.2f}+{slope:.2f}x')
                        plt.legend()
                    except:
                        pass
                
                save_and_close_fig(plt, os.path.join(self.results_dir, f'{param}_effect.png'))
        
        # 绘制F1与学习率和正则化权重的关系
        if 'model.causal_reg_weight' in self.param_grid.keys() and 'training.learning_rate' in self.param_grid.keys():
            try:
                # 创建热图
                pivot_params = []
                for param in self.results:
                    if 'error' not in param:
                        pivot_params.append({
                            'lr': param['params'].get('training.learning_rate'),
                            'reg_weight': param['params'].get('model.causal_reg_weight'),
                            'f1': param.get(f'best_{evaluation_metric}')
                        })
                
                if pivot_params:
                    pivot_df = pd.DataFrame(pivot_params)
                    pivot_table = pivot_df.pivot('lr', 'reg_weight', 'f1')
                    
                    plt.figure(figsize=(10, 8))
                    heatmap = plt.pcolor(pivot_table)
                    plt.colorbar(heatmap)
                    plt.xlabel('Causal Regularization Weight')
                    plt.ylabel('Learning Rate')
                    plt.title(f'{evaluation_metric} Score for Different Learning Rates and Regularization Weights')
                    plt.xticks(np.arange(0.5, len(pivot_table.columns)), pivot_table.columns)
                    plt.yticks(np.arange(0.5, len(pivot_table.index)), pivot_table.index)
                    
                    # 在每个单元格中添加值
                    for i in range(len(pivot_table.index)):
                        for j in range(len(pivot_table.columns)):
                            if not np.isnan(pivot_table.iloc[i, j]):
                                plt.text(j + 0.5, i + 0.5, f'{pivot_table.iloc[i, j]:.3f}',
                                        horizontalalignment='center',
                                        verticalalignment='center')
                    
                    save_and_close_fig(plt, os.path.join(self.results_dir, 'lr_reg_weight_heatmap.png'))
            except Exception as e:
                print(f"创建热图失败: {str(e)}")

# 添加关闭图表的函数
def save_and_close_fig(plt, save_path):
    """保存图表到文件并立即关闭，避免阻塞"""
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """
    执行网格搜索的主函数
    """
    # 设置matplotlib为非交互模式，避免显示图形
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    
    # 定义要搜索的参数及其范围
    param_grid = {
        'model.hidden_channels': [64, 128, 256],
        'model.n_layers': [2, 4, 6],
        'model.dropout': [0.1, 0.2, 0.3],
        'model.causal_reg_weight': [0.01, 0.05, 0.1, 0.2],
        'training.learning_rate': [0.0001, 0.0005, 0.001]
    }
    
    # 验证参数类型
    print("参数类型检查:")
    for key, values in param_grid.items():
        for value in values:
            print(f"{key}: {value} (类型: {type(value).__name__})")
            # 尝试进行必要的类型转换，确保没有问题
            if key in ["model.hidden_channels", "model.n_layers"]:
                try:
                    converted = int(value)
                    print(f"  转换为整数: {converted}")
                except Exception as e:
                    print(f"  ！！转换为整数失败: {str(e)}")
            elif key in ["model.dropout", "model.causal_reg_weight", "training.learning_rate"]:
                try:
                    converted = float(value)
                    print(f"  转换为浮点数: {converted}")
                except Exception as e:
                    print(f"  ！！转换为浮点数失败: {str(e)}")
    
    # 创建结果目录
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"grid_search_{timestamp}")
    
    # 检查load_data函数的返回格式，确保可以正确处理，但避免生成图形
    try:
        print("\n测试数据加载函数:")
        from config import CONFIG
        file_path = CONFIG['data']['file_path']
        
        # 禁用可能的图形生成
        import matplotlib
        matplotlib.use('Agg')
        plt.ioff()  # 关闭交互模式
        
        # 临时修改配置，避免测试时进行数据增强或其他可能生成图形的操作
        test_config = CONFIG.copy()
        test_config['data']['augment'] = False  # 禁用数据增强
        
        # 只加载一个小样本进行测试
        mini_config = test_config.copy()
        mini_config['data']['max_time_steps'] = min(100, test_config['data']['max_time_steps'])
        mini_config['training']['batch_size'] = 2
        
        # 加载少量数据
        data_results = load_data(
            file_path=file_path,
            train_ratio=mini_config['data']['train_ratio'],
            batch_size=mini_config['training']['batch_size'],
            sequence_length=mini_config['training']['sequence_length'],
            max_time_steps=mini_config['data']['max_time_steps'],
            normalize=mini_config['data']['normalize'],
            augment=False  # 明确禁用数据增强
        )
        
        print(f"load_data返回类型: {type(data_results)}")
        if isinstance(data_results, tuple):
            print(f"load_data返回元组长度: {len(data_results)}")
            for i, item in enumerate(data_results):
                print(f"  元素 {i}: 类型={type(item)}")
        else:
            print("警告: load_data未返回元组")
        
        # 确保关闭所有可能打开的图形
        plt.close('all')
        
    except Exception as e:
        print(f"测试数据加载函数时出错: {str(e)}")
        traceback.print_exc()
        plt.close('all')  # 确保关闭所有图形
    
    # 初始化网格搜索
    grid_search = GridSearch(param_grid, results_dir=results_dir)
    
    # 运行网格搜索
    try:
        best_params, best_result = grid_search.run(num_epochs=30, evaluation_metric='f1')
        
        if best_params:
            print("\n网格搜索找到的最佳超参数:")
            for param, value in best_params.items():
                print(f"  {param}: {value}")
            
            # 保存最佳参数
            with open(os.path.join(results_dir, 'best_params.txt'), 'w') as f:
                f.write("最佳超参数组合:\n")
                for param, value in best_params.items():
                    f.write(f"{param}: {value}\n")
                
                if best_result:
                    f.write("\n性能指标:\n")
                    for key, value in best_result.items():
                        if key not in ['params', 'model_id']:
                            f.write(f"{key}: {value}\n")
        else:
            print("网格搜索未找到有效的超参数组合")
    except Exception as e:
        print(f"执行网格搜索时出错: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 