import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from config import TRAIN_CONFIG as CONFIG  # 从config.py导入特定的训练配置
from tqdm import tqdm
import argparse

# 从utils模块导入所有共享函数
from utils import (
    load_data, 
    initialize_model, 
    setup_trainer, 
    train_model, 
    validate_model, 
    plot_losses, 
    save_model, 
    evaluate_causal_discovery,
    process_weighted_adjacency_matrix,
    apply_threshold_to_adjacency_matrix,
    plot_adjacency_matrix,
    plot_adjacency_matrices_comparison,
    optimize_adjacency_matrix_bayesian,
    initialize_adjacency_matrix,
    granger_causality_matrix
)

# 设置随机种子以确保可重复性
torch.manual_seed(21)
np.random.seed(21)


def main():
    """
    训练主函数
    """
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='单独训练模型')
    parser.add_argument('--seed', type=int, default=None, help='要使用的数据的随机种子')
    parser.add_argument('--use_external', action='store_true', help='是否使用外部邻接矩阵')
    args = parser.parse_args()

    # 根据命令行参数确定文件名
    seed_suffix = f"_{args.seed}" if args.seed is not None else "2"
    input_file = f"generated_time_series{seed_suffix}.csv"
    ground_truth_file = f"causal_adjacency_matrix{seed_suffix}.csv"
    
    # 更新配置
    CONFIG['data']['file_path'] = input_file
    # 增加训练轮数以确保充分训练
    CONFIG['training']['num_epochs'] = 80  # 增加到80轮
    # 降低DAG约束权重
    CONFIG['model']['causal_reg_weight'] = 0.0001  # 大幅降低DAG约束权重
    # 增加学习率，帮助邻接矩阵更快收敛
    CONFIG['training']['learning_rate'] = 0.002
    # 设置是否使用外部矩阵
    CONFIG['model']['use_external_adj'] = args.use_external
    
    print(f"读取数据文件: {input_file}")
    print(f"读取真实邻接矩阵: {ground_truth_file}")
    print(f"训练轮数设置为: {CONFIG['training']['num_epochs']}")
    print(f"DAG约束权重设置为: {CONFIG['model']['causal_reg_weight']}")
    print(f"是否使用外部邻接矩阵: {CONFIG['model']['use_external_adj']}")
    
    # 创建结果目录
    results_dir = os.path.join(CONFIG['output']['results_dir'])
    os.makedirs(results_dir, exist_ok=True)
    print(f"结果将保存在: {results_dir}")
    
    # 加载数据
    file_path = CONFIG['data']['file_path']
    train_ratio = CONFIG['data']['train_ratio']
    max_time_steps = CONFIG['data']['max_time_steps']
    normalize = CONFIG['data']['normalize']
    
    sequence_length = CONFIG['training']['sequence_length']
    batch_size = CONFIG['training']['batch_size']
    
    train_dataloader, val_dataloader, seq_length, num_nodes, scalers, raw_data = load_data(
        file_path=file_path,
        train_ratio=train_ratio,
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_time_steps=max_time_steps,
        normalize=normalize,
        augment=CONFIG['data']['augment']
    )
    
    print(f"数据加载完成: 序列长度 = {seq_length}, 节点数 = {num_nodes}")
    
    # 如果启用外部邻接矩阵，则生成外部矩阵
    external_adj = None
    if CONFIG['model']['use_external_adj']:
        print("生成外部邻接矩阵...")
        # 使用格兰杰因果检验生成外部矩阵
        if CONFIG['model']['external_adj_source'] == 'granger':
            max_lag = CONFIG['model']['granger_max_lag']
            significance = CONFIG['model']['granger_significance']
            print(f"使用格兰杰因果检验 (max_lag={max_lag}, significance={significance})...")
            external_adj, pvalues = granger_causality_matrix(
                raw_data, 
                max_lag=max_lag,
                significance=significance,
                verbose=True
            )
            print(f"格兰杰邻接矩阵形状: {external_adj.shape}")
        else:
            # 使用通用函数初始化外部矩阵
            external_adj = initialize_adjacency_matrix(raw_data, CONFIG, internal=False)
            print(f"生成的外部邻接矩阵形状: {external_adj.shape}")
    
    # 初始化模型
    model = initialize_model(
        num_nodes=num_nodes,
        sequence_length=seq_length,
        hidden_channels=CONFIG['model']['hidden_channels'],
        n_layers=CONFIG['model']['n_layers'],
        use_diagonal_mask=CONFIG['model']['use_diagonal_mask'],
        dropout=CONFIG['model']['dropout'],
        mlp_dropout=CONFIG['model']['mlp_dropout'],
        use_dag_constraint=CONFIG['model']['use_dag_constraint'],
        causal_reg_weight=CONFIG['model']['causal_reg_weight']
    )
    
    print(f"模型初始化完成")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"使用设备: {device}")
    
    # 创建专门的优化器，给adj_generator参数设置更高的学习率
    adj_params = []
    other_params = []
    
    # 将参数分组，给adj_generator设置更高的学习率
    for name, param in model.named_parameters():
        if 'adj_generator' in name:
            adj_params.append(param)
        else:
            other_params.append(param)
    
    # 创建优化器
    optimizer = torch.optim.AdamW([
        {'params': adj_params, 'lr': CONFIG['training']['learning_rate'] * 5.0},  # 5倍学习率
        {'params': other_params, 'lr': CONFIG['training']['learning_rate']}
    ])
    
    print(f"优化器配置完成: adj_generator学习率 = {CONFIG['training']['learning_rate'] * 5.0}")
    
    # 训练模型
    num_epochs = CONFIG['training']['num_epochs']
    
    train_losses = []
    val_losses = []
    
    # 早停设置
    best_val_loss = float('inf')
    best_model_state = None
    patience = CONFIG['training']['patience']
    counter = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        # 训练
        model.train()
        train_loss = 0.0
        train_batch_count = 0
        
        for batch_idx, (inputs, targets) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 计算当前外部权重
            if CONFIG['model']['use_external_adj'] and external_adj is not None:
                # 动态调整外部矩阵权重
                weight_transition_epochs = CONFIG['model']['weight_transition_epochs']
                initial_external_weight = CONFIG['model']['initial_external_weight']
                final_external_weight = CONFIG['model']['final_external_weight']
                
                # 线性插值计算当前权重
                progress = min(epoch / weight_transition_epochs, 1.0)
                current_external_weight = initial_external_weight + (final_external_weight - initial_external_weight) * progress
                
                # 确保外部矩阵在正确的设备上
                if not isinstance(external_adj, torch.Tensor):
                    external_adj_tensor = torch.tensor(external_adj, device=device, dtype=torch.float32)
                else:
                    external_adj_tensor = external_adj.to(device)
                
                # 处理外部矩阵批次维度
                if external_adj_tensor.dim() == 2:
                    external_adj_batch = external_adj_tensor.unsqueeze(0).expand(inputs.size(0), -1, -1)
                else:
                    external_adj_batch = external_adj_tensor
                
                # 前向传播
                outputs, adj_matrix, dag_loss, l1_reg = model(inputs, external_adj=external_adj_batch, external_weight=current_external_weight)
            else:
                # 前向传播
                outputs, adj_matrix, dag_loss, l1_reg = model(inputs)
            
            # 计算损失
            recon_loss = torch.nn.functional.mse_loss(outputs, targets)
            
            # 总损失
            loss = recon_loss
            
            # 添加非零惩罚
            if CONFIG['losses']['nonzero_penalty']:
                nonzero_penalty = torch.mean((torch.abs(adj_matrix) < 1e-5).float() * 0.01)
                loss += nonzero_penalty
            
            # 添加DAG约束损失
            if CONFIG['losses']['dag_loss'] and dag_loss is not None:
                loss += CONFIG['model']['causal_reg_weight'] * dag_loss
            
            # 添加L1正则化
            if CONFIG['losses']['l1_reg'] and l1_reg is not None:
                loss += 0.01 * l1_reg
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_batch_count += 1
        
        avg_train_loss = train_loss / train_batch_count
        train_losses.append(avg_train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_batch_count = 0
        
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 前向传播，验证时不使用外部矩阵
                outputs, adj_matrix, dag_loss, l1_reg = model(inputs)
                
                # 计算损失
                recon_loss = torch.nn.functional.mse_loss(outputs, targets)
                
                # 总损失
                loss = recon_loss
                
                # 添加DAG约束损失
                if CONFIG['losses']['dag_loss'] and dag_loss is not None:
                    loss += CONFIG['model']['causal_reg_weight'] * dag_loss
                
                # 添加L1正则化
                if CONFIG['losses']['l1_reg'] and l1_reg is not None:
                    loss += 0.01 * l1_reg
                
                val_loss += loss.item()
                val_batch_count += 1
        
        avg_val_loss = val_loss / val_batch_count
        val_losses.append(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        # 周期性地保存邻接矩阵，用于监控训练过程
        if (epoch + 1) % 10 == 0 or epoch == 0:
            # 提取邻接矩阵
            with torch.no_grad():
                # 使用一批数据进行前向传播
                batch = next(iter(train_dataloader))
                inputs = batch[0].to(device)
                
                _, current_adj_matrix, _, _ = model(inputs)
                current_adj_matrix = current_adj_matrix[0].cpu().numpy()  # 使用第一个样本的邻接矩阵
                
                # 保存邻接矩阵
                epoch_adj_path = os.path.join(results_dir, f"adjacency_matrix_epoch_{epoch+1}.csv")
                pd.DataFrame(current_adj_matrix).to_csv(epoch_adj_path, index=False)
                print(f"Epoch {epoch+1} 邻接矩阵已保存")
        
        # 早停检查
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"早停: 验证损失 {patience} 轮未改善")
                break
    
    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # 保存模型
    model_path = os.path.join(results_dir, f"train_model_seed{seed_suffix}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'num_nodes': num_nodes,
        'sequence_length': seq_length,
        'config': CONFIG
    }, model_path)
    print(f"模型已保存到 {model_path}")
    
    # 绘制损失曲线
    plot_losses(train_losses, val_losses, save_path=os.path.join(results_dir, f"losses_seed{seed_suffix}.png"))
    
    # 提取最终邻接矩阵
    model.eval()
    with torch.no_grad():
        # 使用一批数据进行前向传播
        batch = next(iter(train_dataloader))
        inputs = batch[0].to(device)
        
        _, adj_matrix, _, _ = model(inputs)
        adj_matrix = adj_matrix[0].cpu().numpy()  # 使用第一个样本的邻接矩阵
    
    # 显示邻接矩阵
    plt.figure(figsize=(10, 8))
    plt.imshow(adj_matrix, cmap='Blues')
    plt.colorbar()
    plt.title(f"训练后的邻接矩阵 (Seed: {args.seed if args.seed is not None else 'default'})")
    plt.savefig(os.path.join(results_dir, f"adjacency_matrix_seed{seed_suffix}.png"))
    
    # 保存邻接矩阵为CSV
    adj_matrix_path = os.path.join(results_dir, f"adjacency_matrix_seed{seed_suffix}.csv")
    pd.DataFrame(adj_matrix).to_csv(adj_matrix_path, index=False)
    print(f"邻接矩阵已保存到 {adj_matrix_path}")
    
    # 添加额外的可视化 - 查看邻接矩阵的值分布
    plt.figure(figsize=(10, 6))
    plt.hist(adj_matrix.flatten(), bins=50)
    plt.title("邻接矩阵值分布")
    plt.xlabel("权重值")
    plt.ylabel("频次")
    plt.savefig(os.path.join(results_dir, f"adjacency_weights_distribution_seed{seed_suffix}.png"))
    
    # 评估因果发现性能
    if os.path.exists(ground_truth_file):
        # 使用默认阈值评估
        default_threshold = 0.3
        metrics = evaluate_causal_discovery(adj_matrix, ground_truth_file, threshold=default_threshold)
        
        print(f"\n固定阈值评估结果 (阈值 = {default_threshold}):")
        print(f"精确率: {metrics['precision']:.4f}")
        print(f"召回率: {metrics['recall']:.4f} (TPR: {metrics['tpr']:.4f})")
        print(f"FPR: {metrics['fpr']:.4f}")
        print(f"F1分数: {metrics['f1']:.4f}")
        print(f"准确率: {metrics['accuracy']:.4f}")
        print(f"结构汉明距离: {metrics['shd']}")
        
        # 添加对多个阈值的评估
        print("\n不同阈值下的评估结果:")
        thresholds = CONFIG['eval']['thresholds']
        all_metrics = []
        
        for threshold in thresholds:
            metrics = evaluate_causal_discovery(adj_matrix, ground_truth_file, threshold=threshold)
            all_metrics.append(metrics)
            print(f"阈值 {threshold:.2f}: F1={metrics['f1']:.4f}, 精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}, TPR={metrics['tpr']:.4f}, FPR={metrics['fpr']:.4f}")
        
        # 找到最优F1分数的阈值
        best_metric = max(all_metrics, key=lambda x: x['f1'])
        print(f"\n最优阈值: {best_metric['threshold']:.2f}, F1分数: {best_metric['f1']:.4f}")
        print(f"- 精确率: {best_metric['precision']:.4f}")
        print(f"- 召回率: {best_metric['recall']:.4f} (TPR: {best_metric['tpr']:.4f})")
        print(f"- FPR: {best_metric['fpr']:.4f}")
        print(f"- 准确率: {best_metric['accuracy']:.4f}")
        print(f"- 结构汉明距离: {best_metric['shd']}")
        
        # 保存评估结果
        with open(os.path.join(results_dir, f"evaluation_seed{seed_suffix}.txt"), "w") as f:
            f.write(f"数据文件: {input_file}\n")
            f.write(f"真实邻接矩阵: {ground_truth_file}\n\n")
            
            f.write(f"固定阈值评估结果 (阈值 = {default_threshold}):\n")
            f.write(f"精确率: {metrics['precision']:.4f}\n")
            f.write(f"召回率: {metrics['recall']:.4f} (TPR: {metrics['tpr']:.4f})\n")
            f.write(f"FPR: {metrics['fpr']:.4f}\n")
            f.write(f"F1分数: {metrics['f1']:.4f}\n")
            f.write(f"准确率: {metrics['accuracy']:.4f}\n")
            f.write(f"结构汉明距离: {metrics['shd']}\n\n")
            
            f.write("不同阈值下的评估结果:\n")
            for metric in all_metrics:
                threshold = metric['threshold']
                f.write(f"阈值 {threshold:.2f}: F1={metric['f1']:.4f}, 精确率={metric['precision']:.4f}, 召回率={metric['recall']:.4f}, TPR={metric['tpr']:.4f}, FPR={metric['fpr']:.4f}\n")
            
            f.write(f"\n最优阈值: {best_metric['threshold']:.2f}, F1分数: {best_metric['f1']:.4f}\n")
            f.write(f"- 精确率: {best_metric['precision']:.4f}\n")
            f.write(f"- 召回率: {best_metric['recall']:.4f} (TPR: {best_metric['tpr']:.4f})\n")
            f.write(f"- FPR: {best_metric['fpr']:.4f}\n")
            f.write(f"- 准确率: {best_metric['accuracy']:.4f}\n")
            f.write(f"- 结构汉明距离: {best_metric['shd']}\n")
    else:
        print(f"未找到真实邻接矩阵文件 {ground_truth_file}，跳过评估")
    
    print("\n训练完成!")


if __name__ == "__main__":
    main()