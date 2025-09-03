import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from utils import (
    initialize_model, 
    setup_trainer, 
    load_data, 
    evaluate_causal_discovery,
    process_weighted_adjacency_matrix, 
    apply_threshold_to_adjacency_matrix, 
    plot_adjacency_matrix,
    plot_adjacency_matrices_comparison,
    optimize_adjacency_matrix_bayesian
)
from models.model import PFNO_Causal
from config import GET_FINAL_DAG_CONFIG as CONFIG  # 从config.py导入特定的配置
from skimage.filters import threshold_otsu
from sklearn.metrics import roc_curve, auc

def adaptive_threshold_selection(adj_matrix, true_A=None):
    """
    根据邻接矩阵分布自适应选择阈值
    
    如果有ground truth数据，使用ROC曲线找最佳点
    如果没有ground truth，使用基于分布的方法
    """
    # 展平邻接矩阵获取所有边的权重
    edge_weights = adj_matrix.flatten()
    
    if true_A is not None:
        # 如果有真实值，使用ROC曲线找最佳阈值
        true_edges = (true_A > 0).flatten()
        fpr, tpr, thresholds = roc_curve(true_edges, edge_weights)
        
        # 找到最大化TPR-FPR的阈值 (Youden's J statistic)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        best_threshold = thresholds[best_idx]
    else:
        # 没有真实值时，使用统计方法
        # 方法1: Otsu阈值法 (常用于图像分割)
        best_threshold = threshold_otsu(edge_weights)
        
        # 方法2: 使用高斯混合模型区分噪声和信号
        # 假设边权重是两个高斯分布的混合 (噪声边和实际边)
        try:
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=2, max_iter=1000)
            gmm.fit(edge_weights.reshape(-1, 1))
            
            # 两个分布的均值
            means = gmm.means_.flatten()
            # 找到均值较大的分布(假设为实际边)的权重
            signal_mean = max(means)
            noise_mean = min(means)
            
            # 在两个分布之间取阈值
            best_threshold = (signal_mean + noise_mean) / 2
        except:
            # 如果GMM失败，使用Otsu方法
            pass
    
    print(f"自适应选择的阈值为: {best_threshold:.4f}")
    return best_threshold

def main():
    """
    主函数 - 最终DAG生成流程
    """
    # 添加命令行参数解析
    import argparse
    parser = argparse.ArgumentParser(description='生成最终因果DAG')
    parser.add_argument('--seed', type=int, default=None, help='要使用的数据的随机种子')
    parser.add_argument('--model_path', type=str, default=None, help='预训练模型路径')
    parser.add_argument('--force_train', action='store_true', help='如果没有找到模型，是否强制重新训练一个模型')
    args = parser.parse_args()

    # 基于种子确定文件名
    seed_suffix = f"_{args.seed}" if args.seed is not None else "2"
    input_file = f"generated_time_series{seed_suffix}.csv"
    ground_truth_file = f"causal_adjacency_matrix{seed_suffix}.csv"
    
    # 更新配置
    CONFIG['data']['file_path'] = input_file
    
    # 确定模型路径 - 修改模型加载逻辑，确保使用正确的模型
    if args.model_path:
        # 如果直接指定了模型路径，使用指定的路径
        model_path = args.model_path
        if not os.path.exists(model_path):
            print(f"错误: 指定的模型文件 {model_path} 不存在")
            return
    else:
        # 尝试寻找与当前数据种子匹配的模型
        seed_specific_model = f"results/causal_model_seed{seed_suffix}_*/model.pth"
        seed_specific_train_model = f"results/train_model_seed{seed_suffix}.pth"
        
        import glob
        matching_models = glob.glob(seed_specific_model)
        
        if os.path.exists(seed_specific_train_model):
            # 优先使用特定种子的训练模型
            model_path = seed_specific_train_model
            print(f"使用种子特定的训练模型: {model_path}")
        elif matching_models:
            # 其次使用匹配的模型目录中的模型
            model_path = matching_models[0]
            print(f"使用匹配的模型文件: {model_path}")
        else:
            # 如果没有找到匹配的模型
            if args.force_train:
                # 如果指定了强制训练，调用train.py重新训练
                print(f"未找到种子 {args.seed} 对应的模型，开始重新训练...")
                import subprocess
                train_cmd = f"python train.py --seed {args.seed}" if args.seed is not None else "python train.py"
                result = subprocess.run(train_cmd, shell=True)
                
                if result.returncode == 0:
                    # 训练成功，使用新训练的模型
                    model_path = f"results/train_model_seed{seed_suffix}.pth"
                    if not os.path.exists(model_path):
                        print(f"训练完成，但未找到生成的模型文件 {model_path}")
                        return
                    print(f"训练完成，使用新模型: {model_path}")
                else:
                    print(f"训练失败，退出程序")
                    return
            else:
                # 未指定强制训练且找不到模型，给出错误提示并退出
                print(f"错误: 未找到种子 {args.seed} 对应的模型文件")
                print(f"请先运行 'python train.py --seed {args.seed}' 训练模型")
                print(f"或者使用 --force_train 参数自动训练模型")
                print(f"或者使用 --model_path 参数指定现有模型路径")
                return
            
    print(f"使用数据文件: {input_file}")
    print(f"使用真实邻接矩阵: {ground_truth_file}")
    print(f"使用模型文件: {model_path}")
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建结果目录
    results_dir = os.path.join(CONFIG['output']['results_dir'], 'final_dag')
    os.makedirs(results_dir, exist_ok=True)
    
    # 时间戳
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"final_dag_{timestamp}"
    
    # 加载数据以获取模型参数
    train_dataloader, val_dataloader, sequence_length, num_nodes, scalers = load_data(
        input_file, 
        train_ratio=CONFIG['data']['train_ratio'], 
        batch_size=CONFIG['training']['batch_size'], 
        sequence_length=CONFIG['training']['sequence_length'],
        max_time_steps=CONFIG['data']['max_time_steps'],
        normalize=CONFIG['data']['normalize'],
        augment=False
    )
    
    print(f"数据加载完成: {num_nodes}个节点, {sequence_length}长度的序列, 使用前{CONFIG['data']['max_time_steps']}个时间步")
    
    # 初始化模型
    model = initialize_model(
        num_nodes=num_nodes,
        sequence_length=sequence_length,
        dropout=CONFIG['model']['dropout'],
        mlp_dropout=CONFIG['model']['mlp_dropout'],
        n_layers=CONFIG['model']['n_layers'],
        hidden_channels=CONFIG['model']['hidden_channels'],
        use_diagonal_mask=True,
        use_dag_constraint=True,
        causal_reg_weight=0.01
    )
    
    # 加载训练好的模型
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型加载成功: {model_path}")
    else:
        print(f"错误: 找不到模型文件 {model_path}")
        return
    
    # 设置模型为评估模式
    model.eval()
    model = model.to(device)
    
    # 1. ADAMW优化后得到的加权邻接矩阵
    print("\n===== 步骤1: 从ADAMW优化后的模型提取初始邻接矩阵 =====")
    with torch.no_grad():
        # 使用一个批次的数据提取邻接矩阵
        inputs = torch.randn(1, sequence_length * num_nodes).to(device)  # 创建一个随机输入
        _, initial_adj, _, _, _ = model(inputs)
        initial_adj = initial_adj[0].cpu().detach()  # 获取批次中第一个样本的邻接矩阵
    
    # 保存初始邻接矩阵
    initial_adj_path = os.path.join(results_dir, f"{run_name}_initial_adjacency.csv")
    pd.DataFrame(initial_adj.numpy()).to_csv(initial_adj_path, index=False)
    print(f"初始邻接矩阵已保存到 {initial_adj_path}")
    
    # 2. 贝叶斯优化
    print("\n===== 步骤2: 使用贝叶斯优化优化邻接矩阵 =====")
    if CONFIG['bayesian_optimization'].get('enabled', True):  # 默认为True以兼容旧配置
        optimized_adj = optimize_adjacency_matrix_bayesian(
            model=model,
            dataloader=val_dataloader, 
            initial_adj=initial_adj, 
            device=device,
            n_iterations=20,
            init_points=CONFIG['bayesian_optimization'].get('init_points', 5)
        )
        
        # 保存贝叶斯优化后的邻接矩阵
        bayesian_adj_path = os.path.join(results_dir, f"{run_name}_bayesian_adjacency.csv")
        pd.DataFrame(optimized_adj.numpy()).to_csv(bayesian_adj_path, index=False)
        print(f"贝叶斯优化后的邻接矩阵已保存到 {bayesian_adj_path}")
    else:
        print("贝叶斯优化已在配置中禁用，使用初始邻接矩阵继续...")
        optimized_adj = initial_adj
    
    # 3. 处理Wij和Wji的关系
    print("\n===== 步骤3: 处理Wij和Wji的关系 =====")
    final_weighted_adj = process_weighted_adjacency_matrix(optimized_adj)
    
    # 保存最终的加权邻接矩阵
    final_weighted_adj_path = os.path.join(results_dir, f"{run_name}_final_weighted_adjacency.csv")
    pd.DataFrame(final_weighted_adj).to_csv(final_weighted_adj_path, index=False)
    print(f"最终加权邻接矩阵已保存到 {final_weighted_adj_path}")
    
    # 4. 应用阈值处理，生成二值化邻接矩阵
    print("\n===== 步骤4: 应用阈值处理生成二值化邻接矩阵 =====")
    # 使用配置中的默认阈值
    default_threshold = CONFIG['eval']['default_threshold']
    binary_adj = apply_threshold_to_adjacency_matrix(final_weighted_adj, threshold=default_threshold)
    
    # 同时尝试自适应阈值
    print("\n===== 步骤4(备选): 使用自适应阈值选择 =====")
    # 读取真实邻接矩阵用于比较
    true_A = None
    if os.path.exists(ground_truth_file):
        true_A = pd.read_csv(ground_truth_file, index_col=0).values
        print(f"已加载真实邻接矩阵，形状: {true_A.shape}")
    else:
        print(f"警告: 找不到真实邻接矩阵文件 {ground_truth_file}")
        true_A = np.zeros_like(final_weighted_adj)
        
    adaptive_threshold = adaptive_threshold_selection(final_weighted_adj, true_A)
    adaptive_binary_adj = (final_weighted_adj > adaptive_threshold).astype(float)
    
    # 保存自适应阈值的二值化邻接矩阵
    adaptive_binary_adj_path = os.path.join(results_dir, f"{run_name}_adaptive_binary_adjacency.csv")
    pd.DataFrame(adaptive_binary_adj).to_csv(adaptive_binary_adj_path, index=False)
    print(f"自适应阈值的二值化邻接矩阵已保存到 {adaptive_binary_adj_path}")
    
    # 保存最终的二值化邻接矩阵（默认阈值）
    binary_adj_path = os.path.join(results_dir, f"{run_name}_binary_adjacency.csv")
    pd.DataFrame(binary_adj).to_csv(binary_adj_path, index=False)
    print(f"最终二值化邻接矩阵已保存到 {binary_adj_path}")
    
    # 可视化最终结果
    binary_plot_path = os.path.join(results_dir, f"{run_name}_binary_adjacency_matrix.png")
    plot_adjacency_matrix(binary_adj, title="最终因果关系图", save_path=binary_plot_path)
    
    # 同时保存最终的邻接矩阵到项目根目录，方便其他脚本使用
    pd.DataFrame(final_weighted_adj).to_csv("final_weighted_dag.csv", index=False)
    np.save("final_weighted_dag.npy", final_weighted_adj)
    pd.DataFrame(binary_adj).to_csv("final_dag_2d.csv", index=False)
    np.save("final_dag.npy", binary_adj)
    print("最终邻接矩阵已保存到项目根目录")
    
    # 评估模型
    if os.path.exists(ground_truth_file):
        print("\n===== 评估因果发现性能 =====")
        # 使用默认阈值
        print("使用默认阈值评估:")
        accuracy, precision, recall, f1, auroc, shd = evaluate_causal_discovery(
            true_adj=true_A, 
            pred_adj=binary_adj, 
            threshold=0.5,  # 已经二值化，所以使用0.5只是形式参数
            ignore_self_loops=True
        )
        print(f"默认阈值 {default_threshold:.2f} - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}, AUROC: {auroc:.4f if auroc >= 0 else 'N/A'}, SHD: {shd}")
        
        # 使用自适应阈值
        print("\n使用自适应阈值评估:")
        accuracy, precision, recall, f1, auroc, shd = evaluate_causal_discovery(
            true_adj=true_A, 
            pred_adj=adaptive_binary_adj, 
            threshold=0.5,  # 已经二值化，所以使用0.5只是形式参数
            ignore_self_loops=True
        )
        print(f"自适应阈值 {adaptive_threshold:.4f} - 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}, AUROC: {auroc:.4f if auroc >= 0 else 'N/A'}, SHD: {shd}")
    
    print("\n因果发现流程完成！")


if __name__ == "__main__":
    main() 