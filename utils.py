import torch
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
# 移除循环导入
# from models.model import PFNO_Causal
# from al_trainer import ALTrainer  # 已移动到fno模块
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import datetime
import warnings
import random
# 用于格兰杰因果检验
try:
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels库未安装，格兰杰因果检验功能不可用。请安装: pip install statsmodels")

# 设置matplotlib支持中文字体
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 导出所有工具函数，方便其他模块导入
__all__ = [
    'TimeSeriesAugmentation', 'load_data', 'initialize_model', 'setup_trainer',
    'train_model', 'validate_model', 'plot_losses', 'save_model',
    'evaluate_causal_discovery', 'granger_causality_matrix', 
    'correlation_matrix', 'initialize_adjacency_matrix',
    'process_weighted_adjacency_matrix', 'apply_threshold_to_adjacency_matrix',
    'plot_adjacency_matrix', 'plot_adjacency_matrices_comparison',
    'optimize_adjacency_matrix_bayesian', 'set_seed', 'calculate_dag_constraint',
    'enhance_adjacency_probabilities', 'find_optimal_threshold_auto', 'adaptive_evaluate_causal_discovery'
]

def set_seed(seed):
    """
    设置所有随机数生成器的种子以确保可复现性
    
    Args:
        seed: 整数种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # 打印设置信息
    print(f"已设置随机种子: {seed}")
    return seed

class TimeSeriesAugmentation(torch.utils.data.Dataset):
    """时间序列数据增强类，仅用于训练集增强"""
    
    def __init__(self, dataset, num_nodes, sequence_length, config=None):
        """
        初始化时间序列数据增强类
        
        Args:
            dataset: 原始数据集
            num_nodes: 节点数量
            sequence_length: 序列长度
            config: 配置参数，如果为None则使用默认值
        """
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        
        # 使用配置或默认值
        if config is None:
            # 提供默认的数据增强配置，避免依赖config.py
            config = {
                'augmentation': {
                    'factor': 1.5,
                    'techniques': ['noise', 'scaling', 'time_warp', 'jitter'],
                    'noise_level': [0.001, 0.01],
                    'scale_range': [0.9, 1.1],
                    'time_warp_width': [1, 3],
                    'window_shift': 2
                }
            }
        
        self.augmentation_factor = config['augmentation']['factor']
        self.techniques = config['augmentation']['techniques']
        self.noise_level = config['augmentation']['noise_level']
        self.scale_range = config['augmentation']['scale_range']
        self.time_warp_width = config['augmentation']['time_warp_width']
        self.window_shift = config['augmentation']['window_shift']
        
        self.orig_length = len(dataset)
        self.augmented_length = int(self.orig_length * self.augmentation_factor)
        self.augmentation_indices = np.random.choice(self.orig_length, self.augmented_length - self.orig_length)
        
        print(f"应用时间序列数据增强，增强因子: {self.augmentation_factor}")
        print(f"原始样本数: {self.orig_length}, 增强后样本数: {self.augmented_length}")
        print(f"使用的增强技术: {self.techniques}")
    
    def __len__(self):
        return self.augmented_length
    
    def __getitem__(self, idx):
        if idx < self.orig_length:
            # 返回原始数据
            return self.dataset[idx]
        else:
            # 应用数据增强
            aug_idx = self.augmentation_indices[idx - self.orig_length]
            x, y = self.dataset[aug_idx]
            
            # 重塑为原始形状以便进行时间序列特定的增强
            x_3d = x.reshape(-1, self.num_nodes, self.sequence_length)
            y_3d = y.reshape(-1, self.num_nodes, self.sequence_length)
            
            # 随机选择增强技术
            aug_technique = np.random.choice(self.techniques)
            
            if aug_technique == 'noise':
                # 添加高斯噪声
                noise_level = np.random.uniform(self.noise_level[0], self.noise_level[1])
                x_aug = x_3d + torch.randn_like(x_3d) * noise_level
                y_aug = y_3d  # 目标保持不变
                
            elif aug_technique == 'scaling':
                # 随机缩放
                scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
                x_aug = x_3d * scale_factor
                y_aug = y_3d  # 目标保持不变
                
            elif aug_technique == 'time_warp':
                # 时间扭曲（拉伸/压缩某些时间段）
                x_aug = x_3d.clone()
                for i in range(self.num_nodes):
                    # 随机选择一个扭曲点和扭曲窗口大小
                    warp_point = np.random.randint(1, self.sequence_length-1) if self.sequence_length > 2 else 1
                    
                    # 计算最大可能的扭曲宽度
                    max_width = min(self.time_warp_width[1], warp_point, self.sequence_length-warp_point)
                    min_width = self.time_warp_width[0]
                    
                    # 确保最小宽度不超过最大宽度
                    if min_width >= max_width:
                        min_width = 1
                        max_width = max(2, max_width)  # 确保max_width至少为2
                    
                    # 安全地生成随机宽度
                    warp_width = np.random.randint(min_width, max_width)
                    
                    # 创建扭曲索引（确保长度合法）
                    if warp_width > 0:
                        try:
                            warp_indices = torch.linspace(
                                warp_point - warp_width, 
                                warp_point + warp_width,
                                2 * warp_width
                            ).long()
                            
                            # 应用线性插值扭曲
                            if len(warp_indices) > 0:
                                # 确保索引在有效范围内
                                valid_indices = warp_indices[(warp_indices >= 0) & (warp_indices < self.sequence_length)]
                                if len(valid_indices) > 0:
                                    try:
                                        x_aug[0, i, valid_indices] = torch.nn.functional.interpolate(
                                            x_3d[0, i, valid_indices].unsqueeze(0).unsqueeze(0), 
                                            size=len(valid_indices),
                                            mode='linear'
                                        ).squeeze(0).squeeze(0)
                                    except RuntimeError:
                                        # 如果插值失败，保持原样
                                        pass
                        except RuntimeError:
                            # 如果生成索引失败，保持原样
                            pass
                
                y_aug = y_3d  # 目标保持不变
                
            elif aug_technique == 'jitter':
                # 随机抖动 - 每个时间点添加不同噪声
                jitter_factor = np.random.uniform(self.noise_level[0], self.noise_level[1])
                noise = torch.randn_like(x_3d) * jitter_factor
                x_aug = x_3d + noise
                y_aug = y_3d  # 目标保持不变
            
            else:
                # 默认行为：保持原样
                x_aug = x_3d
                y_aug = y_3d
            
            # 重塑回原始维度
            x_aug_flat = x_aug.reshape(-1, self.num_nodes * self.sequence_length)
            y_aug_flat = y_aug.reshape(-1, self.num_nodes * self.sequence_length)  # 确保y也被正确重塑
            
            # 确保返回与原始数据集相同的形状格式
            if x_aug_flat.shape[0] == 1:
                x_aug_flat = x_aug_flat.squeeze(0)
            if y_aug_flat.shape[0] == 1:
                y_aug_flat = y_aug_flat.squeeze(0)
            
            return x_aug_flat, y_aug_flat  # 修复：返回重塑后的y而不是原始y


def load_data(file_path, train_ratio=0.8, batch_size=32, sequence_length=25, max_time_steps=None, normalize=True, augment=True):
    """
    加载数据并生成训练集和验证集 DataLoader。
    
    Args:
        file_path: CSV文件路径
        train_ratio: 训练集所占比例
        batch_size: 批量大小
        sequence_length: 序列长度
        max_time_steps: 要使用的最大时间步长数，如果为None则使用全部数据
        normalize: 是否启用数据标准化
        augment: 是否对训练集进行数据增强
        
    Returns:
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        sequence_length: 序列长度
        num_nodes: 节点数量
        scalers: 标准化器字典
        raw_data: 原始时间序列数据
    """
    # 从 CSV 文件中读取数据
    data = pd.read_csv(file_path)
    print(f"原始CSV文件形状: {data.shape}")
    
    # 保存原始数据用于格兰杰因果检验
    raw_data = data.copy()
    
    # 如果指定了最大时间步长，则只使用前max_time_steps行
    if max_time_steps is not None and max_time_steps < data.shape[0]:
        data = data.iloc[:max_time_steps, :]
        print(f"限制为前{max_time_steps}个时间步，实际使用数据形状: {data.shape}")
    
    # 参数定义
    num_nodes = data.shape[1]  # 节点数从数据推断
    sequence_length = sequence_length  # 可配置的时间序列长度
    print(f"推断节点数: {num_nodes}, 序列长度: {sequence_length}")

    # 将数据转换为 NumPy 数组
    data_values = data.values  # 形状 (num_samples, num_nodes)
    
    # 生成时间序列样本
    samples = []
    targets = []
    for i in range(0, len(data_values) - sequence_length - 1):  # 确保有下一个序列
        sample = data_values[i:i + sequence_length]  # 输入序列 [t0, t1, ..., t(sequence_length-1)]
        target = data_values[i + 1:i + sequence_length + 1]  # 目标序列 [t1, t2, ..., t(sequence_length)]
        samples.append(sample)
        targets.append(target)

    # 转换为 NumPy 数组
    x = np.array(samples)  # [num_samples, sequence_length, num_nodes]
    y = np.array(targets)  # [num_samples, sequence_length, num_nodes]

    # 调整维度为 (num_samples, num_nodes, sequence_length)
    x = x.transpose(0, 2, 1)  # [num_samples, num_nodes, sequence_length]
    y = y.transpose(0, 2, 1)  # [num_samples, num_nodes, sequence_length]

    # 先划分训练集和验证集（在归一化前）
    train_size = int(x.shape[0] * train_ratio)
    val_size = x.shape[0] - train_size
    
    # 划分数据
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # 创建存储标准化器的字典
    scalers = {}

    # 对每个节点的数据进行标准化，仅使用训练集数据计算统计量
    if normalize:
        print("对数据进行标准化处理（仅基于训练集统计量）...")
        x_train_norm = np.zeros_like(x_train)
        x_val_norm = np.zeros_like(x_val)
        
        for i in range(num_nodes):
            # 为每个节点创建一个标准化器，仅使用训练数据拟合
            scaler = StandardScaler()
            # 拟合训练数据（注意：这里只使用训练集计算均值和标准差）
            scaler.fit(x_train[:, i, :].reshape(-1, 1))
            
            # 对训练集数据进行变换
            x_train_norm[:, i, :] = scaler.transform(x_train[:, i, :].reshape(-1, 1)).reshape(x_train.shape[0], sequence_length)
            # 对验证集数据应用相同的变换
            x_val_norm[:, i, :] = scaler.transform(x_val[:, i, :].reshape(-1, 1)).reshape(x_val.shape[0], sequence_length)
            
            # 保存标准化器
            scalers[i] = scaler
        
        # 更新数据为标准化后的数据
        x_train, x_val = x_train_norm, x_val_norm
        
        # 保存标准化器以便后续使用
        scaler_dir = "scalers"
        os.makedirs(scaler_dir, exist_ok=True)
        with open(os.path.join(scaler_dir, "node_scalers.pkl"), "wb") as f:
            pickle.dump(scalers, f)
        print(f"标准化器已保存到 {os.path.join(scaler_dir, 'node_scalers.pkl')}")
    else:
        print("跳过数据标准化...")

    # 展平输入和目标（保持时间步连续性）
    x_train = x_train.reshape(-1, num_nodes * sequence_length)  # [num_samples, num_nodes * sequence_length]
    y_train = y_train.reshape(-1, num_nodes * sequence_length)  # [num_samples, num_nodes * sequence_length]
    x_val = x_val.reshape(-1, num_nodes * sequence_length)
    y_val = y_val.reshape(-1, num_nodes * sequence_length)

    # 将数据转换为 PyTorch 张量
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    # 创建数据集
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    # 仅对训练集应用数据增强
    if augment:
        # 使用简单的数据增强，避免依赖config.py
        augmented_train_dataset = TimeSeriesAugmentation(train_dataset, num_nodes, sequence_length, None)
        print(f"数据增强后训练集大小: {len(augmented_train_dataset)}")
        train_dataloader = DataLoader(augmented_train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # 验证集保持不变
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, sequence_length, num_nodes, scalers, raw_data


def initialize_model(num_nodes, sequence_length, n_modes=[16, 16], hidden_channels=64,
                     n_layers=2, dropout=0.12, use_mlp=True, mlp_dropout=0.12,
                     use_diagonal_mask=True, use_dag_constraint=False, causal_reg_weight=0.01):
    """
    初始化 PFNO_Causal 模型。
    
    Args:
        num_nodes: 节点数量
        sequence_length: 序列长度
        n_modes: 模式数量
        hidden_channels: 隐藏层通道数
        n_layers: 层数
        dropout: dropout率
        use_mlp: 是否使用MLP
        mlp_dropout: MLP dropout率
        use_diagonal_mask: 是否使用对角线掩码
        use_dag_constraint: 是否强制DAG约束
        causal_reg_weight: 因果正则化权重
    """
    # 动态导入PFNO_Causal，避免循环导入
    from fno.model import PFNO_Causal
    
    model = PFNO_Causal(
        n_modes=n_modes,
        hidden_channels=hidden_channels,
        num_nodes=num_nodes,
        in_channels=sequence_length * num_nodes,
        out_channels=num_nodes,
        n_layers=n_layers,
        dropout=dropout,
        use_mlp=use_mlp,
        mlp_dropout=mlp_dropout,
        non_linearity=torch.nn.GELU(),
        use_diagonal_mask=use_diagonal_mask,
        use_dag_constraint=use_dag_constraint,
        causal_reg_weight=causal_reg_weight
    )
    return model


def setup_trainer(n, d, model, lr=1e-3, init_iter=10, 
                early_stopping=True, early_stopping_thresh=1.0, 
                patience=20, seed=21, device=None, sequence_length=None, dag_constraint=True,
                use_external_adj=False, external_adj=None, 
                initial_external_weight=0.5, final_external_weight=0.1,
                weight_transition_epochs=5):
    """
    设置ALTrainer用于模型训练
    
    Args:
        n: 样本数量
        d: 特征维度
        model: 训练模型
        lr: 学习率
        init_iter: 初始迭代次数
        early_stopping: 是否使用早停
        early_stopping_thresh: 早停阈值
        patience: 早停耐心值，连续多少轮验证损失没有改善后停止训练
        device: 计算设备
        sequence_length: 序列长度
        dag_constraint: 是否强制DAG约束
        use_external_adj: 是否使用外部邻接矩阵
        external_adj: 外部邻接矩阵
        initial_external_weight: 外部邻接矩阵的初始权重
        final_external_weight: 外部邻接矩阵的最终权重
        weight_transition_epochs: 权重过渡的轮数
    Returns:
        trainer: ALTrainer实例
    """
    
    try:
        # 动态导入ALTrainer，避免循环依赖
        from fno.al_trainer import ALTrainer
    except ImportError:
        print("警告: 无法导入ALTrainer，setup_trainer功能不可用")
        return None
    
    trainer = ALTrainer(
        n=n,
        d=d,
        model=model,
        lr=lr,
        init_iter=init_iter,
        early_stopping=early_stopping,
        early_stopping_thresh=early_stopping_thresh,
        patience=patience, 
        device=device,
        sequence_length=sequence_length,
        dag_constraint=dag_constraint
    )
    
    # 设置外部邻接矩阵相关参数
    if use_external_adj and external_adj is not None:
        trainer.use_external_adj = use_external_adj
        trainer.external_adj = torch.tensor(external_adj, dtype=torch.float32, device=device if device else trainer.device)
        trainer.initial_external_weight = initial_external_weight
        trainer.final_external_weight = final_external_weight
        trainer.weight_transition_epochs = weight_transition_epochs
        trainer.current_epoch = 0
    
    # 添加处理epoch参数的逻辑，以兼容train_step的新参数
    # 将原来的train_step方法替换为新的接受epoch参数的train_step方法
    original_train_step = trainer.train_step
    
    def train_step_with_epoch(batch, epoch=0):
        # 新的train_step方法，接受epoch参数
        inputs, targets = batch  # 解包batch元组
        if hasattr(original_train_step, '__code__') and len(original_train_step.__code__.co_varnames) > 2:
            # 如果原方法已经接受epoch参数
            return original_train_step(inputs, targets, epoch)
        else:
            # 保持向后兼容性
            return original_train_step(inputs, targets)
    
    # 替换train_step方法
    trainer.train_step = train_step_with_epoch
    
    return trainer


def train_model(trainer, train_dataloader, num_epochs=20, validation_dataloader=None):
    """
    训练模型并记录损失。
    
    Args:
        trainer: ALTrainer实例
        train_dataloader: 训练数据加载器
        num_epochs: 训练轮数
        validation_dataloader: 验证数据加载器
    
    Returns:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer.model.to(device)
    trainer.device = device

    # 使用 AdamW 优化器
    trainer.optimizer = torch.optim.AdamW([
        {'params': trainer.model.parameters(), 'weight_decay': 0.05}
    ], lr=trainer.lr, weight_decay=0.005)

    # 使用 ReduceLROnPlateau 调度器替换 OneCycleLR
    trainer.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer.optimizer,
        mode='min',
        factor=0.5,  # 设置更大的衰减比率，每次降低为原来的50%
        patience=5,  # 5个epoch没有改善就降低学习率
        verbose=True,  # 打印学习率变化信息
        threshold=1e-4,  # 改善阈值
        min_lr=1e-7  # 最小学习率
    )

    # 记录损失
    train_losses = []
    val_losses = []
    dag_losses = []  # 新增：记录DAG约束损失
    
    # 使用tqdm显示进度条
    for epoch in tqdm(range(num_epochs), desc="Training"):
        # 更新当前epoch，用于外部矩阵权重计算
        trainer.current_epoch = epoch
        
        # 训练模式
        trainer.model.train()
        epoch_loss = 0.0
        epoch_dag_loss = 0.0  # 新增：记录每个epoch的DAG约束损失
        batch_count = 0
        
        # 迭代训练数据
        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
            # 将数据移动到适当的设备
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 训练步骤
            batch_loss = trainer.train_step((inputs, targets), epoch)
            
            # 新增：获取当前批次的DAG约束损失
            if hasattr(trainer.model, 'forward'):
                with torch.no_grad():
                    _, adj_matrix, dag_loss, _ = trainer.model(inputs)
                    epoch_dag_loss += dag_loss.item() if isinstance(dag_loss, torch.Tensor) else dag_loss
            
            if batch_loss is not None:
                epoch_loss += batch_loss
                batch_count += 1
                
        # 计算平均训练损失
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else float('inf')
        train_losses.append(avg_train_loss)
        
        # 计算平均DAG约束损失
        avg_dag_loss = epoch_dag_loss / batch_count if batch_count > 0 else float('inf')
        dag_losses.append(avg_dag_loss)
        
        # 验证
        if validation_dataloader is not None:
            # 验证模式
            trainer.model.eval()
            val_loss = validate_model(trainer.model, validation_dataloader, device)
            val_losses.append(val_loss)
            
            # 更新学习率调度器
            trainer.scheduler.step(val_loss)
            
            # 保存最佳模型
            if val_loss < trainer.best_loss:
                trainer.best_loss = val_loss
                trainer.best_model_state = trainer.model.state_dict().copy()
                trainer.no_improvement_count = 0
                # 记录最佳邻接矩阵
                with torch.no_grad():
                    if hasattr(trainer.model, 'get_adj_matrix'):
                        trainer.best_adj = trainer.model.get_adj_matrix()[0].cpu().numpy()
            else:
                trainer.no_improvement_count += 1
            
            # 早停
            if trainer.early_stopping and trainer.no_improvement_count >= trainer.patience:
                print(f"\n早停在第 {epoch+1} 轮，连续 {trainer.patience} 轮没有改善")
                break
            
            tqdm.write(f"轮次 {epoch+1}/{num_epochs} - 训练损失: {avg_train_loss:.6f}, 验证损失: {val_loss:.6f}, DAG约束损失: {avg_dag_loss:.6f}")
        else:
            tqdm.write(f"轮次 {epoch+1}/{num_epochs} - 训练损失: {avg_train_loss:.6f}, DAG约束损失: {avg_dag_loss:.6f}")
    
    # 加载最佳模型权重
    if trainer.best_model_state is not None:
        trainer.model.load_state_dict(trainer.best_model_state)
    
    # 绘制DAG约束损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(dag_losses, label='DAG Constraint Loss')
    plt.title('DAG Constraint Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('dag_constraint_loss.png')
    print(f"DAG约束损失曲线已保存到 dag_constraint_loss.png")
    
    return train_losses, val_losses


def validate_model(model, dataloader, device, causal_adjacency_file=None):
    """
    在验证集上评估模型
    
    Args:
        model: 要评估的模型
        dataloader: 验证数据加载器
        device: 计算设备
        causal_adjacency_file: 可选的因果邻接矩阵文件
    
    Returns:
        avg_loss: 平均验证损失
    """
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # 加载真实邻接矩阵（如果提供）
    true_adj = None
    if causal_adjacency_file is not None and os.path.exists(causal_adjacency_file):
        try:
            true_adj = pd.read_csv(causal_adjacency_file).values
            print(f"已加载真实邻接矩阵，形状: {true_adj.shape}")
        except Exception as e:
            print(f"无法加载邻接矩阵文件: {e}")
    
    with torch.no_grad():
        # 处理验证数据批次
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 模型前向传播
            outputs, adj_matrix, dag_loss, _ = model(inputs)
            
            # 计算损失
            loss, mse_loss, reg_loss = model.get_loss(outputs, targets, adj_matrix, dag_loss)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    
    return avg_loss


def plot_losses(train_losses, val_losses, save_path=None):
    """
    绘制训练和验证损失曲线
    
    Args:
        train_losses: 训练损失列表
        val_losses: 验证损失列表
        save_path: 保存图表的路径
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    
    if val_losses:
        # 如果验证损失在不同的频率下记录，调整x轴以匹配
        if len(val_losses) != len(train_losses):
            val_epochs = np.linspace(0, len(train_losses) - 1, len(val_losses))
            plt.plot(val_epochs, val_losses, label='Validation Loss', linestyle='--')
        else:
            plt.plot(val_losses, label='Validation Loss', linestyle='--')
    
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        print(f"损失曲线已保存到 {save_path}")
    
    plt.show()


def save_model(model, optimizer=None, path="model.pth", metadata=None):
    """
    保存模型及其元数据
    
    Args:
        model: 要保存的模型
        optimizer: 优化器（可选）
        path: 保存路径
        metadata: 额外的元数据字典（可选）
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'num_nodes': model.num_nodes if hasattr(model, 'num_nodes') else None,
            'n_modes': model.n_modes if hasattr(model, 'n_modes') else None,
            'hidden_channels': model.hidden_channels if hasattr(model, 'hidden_channels') else None,
        },
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if optimizer is not None:
        save_dict['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata is not None:
        save_dict['metadata'] = metadata
    
    torch.save(save_dict, path)
    print(f"模型已保存到 {path}")


def evaluate_causal_discovery(adjacency_matrix=None, ground_truth_file=None, true_adj=None, pred_adj=None, threshold=0.1, ignore_self_loops=True):
    """
    评估因果发现的性能
    
    Args:
        adjacency_matrix: 预测的邻接矩阵（兼容旧接口）
        ground_truth_file: 真实邻接矩阵的文件路径（兼容旧接口）
        true_adj: 真实邻接矩阵（新接口）
        pred_adj: 预测的邻接矩阵（新接口）
        threshold: 二值化阈值
        ignore_self_loops: 是否忽略自环
        
    Returns:
        metrics: 包含精确率、召回率和F1分数的字典或元组(accuracy, precision, recall, f1, auroc, shd)
    """
    # 处理输入，确定预测矩阵和真实矩阵
    if pred_adj is not None:
        predicted_matrix = pred_adj
    else:
        predicted_matrix = adjacency_matrix
    
    if true_adj is not None:
        ground_truth = true_adj
    elif ground_truth_file is not None:
        try:
            # 加载真实邻接矩阵
            ground_truth = pd.read_csv(ground_truth_file).values
        except Exception as e:
            print(f"无法加载真实邻接矩阵文件: {ground_truth_file} - {e}")
            return None
    else:
        print("错误: 必须提供ground_truth_file或true_adj")
        return None
    
    # 确保两个矩阵有相同的形状
    if ground_truth.shape != predicted_matrix.shape:
        print(f"警告: 真实矩阵形状 {ground_truth.shape} 与预测矩阵形状 {predicted_matrix.shape} 不一致")
        # 尝试调整形状
        if len(ground_truth.shape) < len(predicted_matrix.shape):
            # 真实矩阵维度较低，扩展维度
            ground_truth = np.expand_dims(ground_truth, axis=tuple(range(len(predicted_matrix.shape) - len(ground_truth.shape))))
        elif len(predicted_matrix.shape) < len(ground_truth.shape):
            # 预测矩阵维度较低，扩展维度
            predicted_matrix = np.expand_dims(predicted_matrix, axis=tuple(range(len(ground_truth.shape) - len(predicted_matrix.shape))))
        
        # 重新检查形状
        if ground_truth.shape != predicted_matrix.shape:
            print(f"错误: 无法使形状匹配: {ground_truth.shape} vs {predicted_matrix.shape}")
            return None
    
    if ignore_self_loops:
        # 创建对角线掩码
        diag_mask = ~np.eye(ground_truth.shape[-2], dtype=bool)
        
        # 应用掩码去除自环
        ground_truth = ground_truth * diag_mask
        predicted_matrix = predicted_matrix * diag_mask
    
    # 二值化预测的邻接矩阵（如果还未二值化）
    if not np.array_equal(predicted_matrix, predicted_matrix.astype(bool).astype(predicted_matrix.dtype)):
        predicted_binary = (predicted_matrix > threshold).astype(int)
    else:
        predicted_binary = predicted_matrix
    
    # 展平评估（确保适用于任何形状）
    ground_truth_flat = ground_truth.flatten()
    predicted_binary_flat = predicted_binary.flatten()
    predicted_raw_flat = predicted_matrix.flatten() if not np.array_equal(predicted_matrix, predicted_binary) else None
    
    # 评估指标
    try:
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score, confusion_matrix
        
        # 计算标准评估指标
        precision, recall, f1, _ = precision_recall_fscore_support(
            ground_truth_flat, 
            predicted_binary_flat,
            average='binary',
            zero_division=0
        )
        
        accuracy = accuracy_score(ground_truth_flat, predicted_binary_flat)
        
        # 计算TPR (True Positive Rate) 和 FPR (False Positive Rate)
        tn, fp, fn, tp = confusion_matrix(ground_truth_flat, predicted_binary_flat).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR = Recall
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 如果预测矩阵不是二值的，可以计算AUROC
        try:
            if predicted_raw_flat is not None:
                auroc = roc_auc_score(ground_truth_flat, predicted_raw_flat)
            else:
                auroc = -1  # 已经二值化，无法计算AUROC
        except Exception:
            auroc = -1
        
        # 计算结构Hamming距离 (SHD)
        shd = np.sum(np.abs(predicted_binary - ground_truth))
        
        # 计算正确识别的边数
        true_positives = np.sum((predicted_binary == 1) & (ground_truth == 1))
        total_edges = np.sum(ground_truth == 1)
        
        # 计算Jaccard相似度
        intersection = np.sum((predicted_binary == 1) & (ground_truth == 1))
        union = np.sum((predicted_binary == 1) | (ground_truth == 1))
        jaccard = intersection / union if union > 0 else 0
        
        # 根据函数调用来源决定返回格式
        # 检查调用栈判断调用源
        import inspect
        calling_frame = inspect.currentframe().f_back
        calling_module = inspect.getmodule(calling_frame)
        caller_name = calling_module.__name__ if calling_module else ""
        
        # 创建指标字典
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'tpr': tpr,  # 添加了TPR
            'fpr': fpr,  # 添加了FPR
            'true_positives': true_positives,
            'total_edges': total_edges,
            'edge_detection_rate': true_positives / total_edges if total_edges > 0 else 0,
            'auroc': auroc,
            'shd': shd,
            'jaccard': jaccard,
            'threshold': threshold  # 记录使用的阈值
        }
        
        # 对于get_final_dag.py的调用，返回元组
        if 'get_final_dag' in caller_name:
            return accuracy, precision, recall, f1, auroc, shd
        else:
            return metrics
            
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        return None


def granger_causality_matrix(time_series, max_lag=5, significance=0.05, verbose=False):
    """
    使用格兰杰因果检验计算时间序列之间的因果关系矩阵
    
    Args:
        time_series: 形状为(num_samples, num_variables)的时间序列数据
        max_lag: 最大滞后阶数，默认为5
        significance: 显著性水平，默认为0.05
        verbose: 是否打印详细信息，默认为False
    
    Returns:
        adjacency_matrix: 格兰杰因果邻接矩阵 (1表示存在因果关系，0表示不存在)
        p_values: p值矩阵，每个元素代表对应关系的p值
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        print("缺少statsmodels库，无法执行格兰杰因果检验")
        print("请使用 'pip install statsmodels' 安装")
        # 返回空矩阵
        return np.zeros((time_series.shape[1], time_series.shape[1])), np.ones((time_series.shape[1], time_series.shape[1]))
    
    # 处理非2D数据
    if len(time_series.shape) != 2:
        print(f"警告：输入数据维度为{len(time_series.shape)}，需要2D数据")
        if len(time_series.shape) > 2:
            # 如果是多维数据，展平为2D
            time_series = time_series.reshape(time_series.shape[0], -1)
        else:
            # 如果是1D数据，扩展为2D
            time_series = time_series.reshape(-1, 1)
    
    n_variables = time_series.shape[1]
    
    # 初始化结果矩阵
    adjacency_matrix = np.zeros((n_variables, n_variables))
    p_values = np.ones((n_variables, n_variables))
    
    # 设置进度条
    if verbose:
        import tqdm
        iterator = tqdm.tqdm(range(n_variables), desc="格兰杰因果检验")
    else:
        iterator = range(n_variables)
    
    # 执行格兰杰因果检验
    for i in iterator:
        for j in range(n_variables):
            # 跳过自身
            if i == j:
                continue
            
            # 提取两个变量的时间序列
            y = time_series[:, i]
            x = time_series[:, j]
            data = np.column_stack([y, x])
            
            try:
                # 执行格兰杰因果检验
                test_results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # 提取p值 (使用F检验的p值)
                # 对所有滞后取最小p值
                min_p_value = min([test_results[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag+1)])
                p_values[j, i] = min_p_value  # 注意：方向是 j -> i
                
                # 判断是否存在因果关系
                if min_p_value < significance:
                    adjacency_matrix[j, i] = 1  # 因果关系方向是 j -> i
                    if verbose:
                        print(f"检测到因果关系: X{j+1} -> X{i+1}, p值 = {min_p_value:.4f}")
            
            except Exception as e:
                if verbose:
                    print(f"计算X{j+1} -> X{i+1}的格兰杰因果关系时出错: {e}")
                continue
    
    # 根据配置管理变更，确保引用的函数使用新的接口
    # 由于此函数是通用函数，不直接依赖特定配置
    
    # 记录结果
    if verbose:
        print(f"\n检测到{np.sum(adjacency_matrix)}个因果关系，总计{n_variables * (n_variables-1)}个可能的关系")
        print(f"检测率: {np.sum(adjacency_matrix) / (n_variables * (n_variables-1)) * 100:.2f}%")
    
    return adjacency_matrix, p_values


def correlation_matrix(time_series, threshold=0.3, absolute=True):
    """
    计算相关性矩阵，并根据阈值生成邻接矩阵
    
    Args:
        time_series: 时间序列数据
        threshold: 相关系数阈值
        absolute: 是否使用相关系数的绝对值
        
    Returns:
        adj_matrix: 相关性邻接矩阵
        corr_matrix: 相关性系数矩阵
    """
    # 计算相关系数矩阵
    if isinstance(time_series, pd.DataFrame):
        corr_matrix = time_series.corr().values
    else:
        corr_matrix = np.corrcoef(time_series, rowvar=False)
    
    # 使用绝对值
    if absolute:
        corr_abs = np.abs(corr_matrix)
    else:
        corr_abs = corr_matrix
    
    # 根据阈值生成邻接矩阵
    adj_matrix = (corr_abs > threshold).astype(float)
    
    # 移除自环 (对角线置零)
    np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix, corr_matrix


def initialize_adjacency_matrix(time_series, config=None, internal=False):
    """
    根据配置初始化邻接矩阵
    
    Args:
        time_series: 时间序列数据
        config: 配置字典，如果为None则使用默认配置
        internal: 是否初始化内部矩阵
        
    Returns:
        adj_matrix: 初始化的邻接矩阵
    """
    # 加载配置或使用默认值
    if config is None:
        # 提供默认配置，避免依赖config.py
        config = {
            'model': {
                'internal_adj_init': 'zeros',
                'external_adj_source': 'correlation',
                'granger_max_lag': 5,
                'granger_significance': 0.05,
                'granger_normalize': True,
                'manual_adj_path': None
            }
        }
    
    n_nodes = time_series.shape[1] if len(time_series.shape) > 1 else int(np.sqrt(time_series.shape[0]))
    
    # 根据类型选择初始化方法
    if internal:
        init_type = config['model']['internal_adj_init']
        
        if init_type == 'zeros':
            # 零矩阵初始化
            adj_matrix = np.zeros((n_nodes, n_nodes))
            print(f"内部邻接矩阵使用零矩阵初始化，形状为{adj_matrix.shape}")
            
        elif init_type == 'tiny':
            # 极小值初始化
            adj_matrix = np.random.uniform(0, 0.01, size=(n_nodes, n_nodes))
            # 确保无自环
            np.fill_diagonal(adj_matrix, 0)
            print(f"内部邻接矩阵使用极小值初始化，形状为{adj_matrix.shape}")
            
        elif init_type == 'random':
            # 随机初始化
            adj_matrix = np.random.uniform(0.12, 0.2, size=(n_nodes, n_nodes))
            # 确保无自环
            np.fill_diagonal(adj_matrix, 0)
            print(f"内部邻接矩阵使用随机值初始化，形状为{adj_matrix.shape}")
            
        elif init_type == 'data_corr':
            # 使用数据相关性初始化
            adj_matrix, _ = correlation_matrix(time_series, threshold=0.1)
            print(f"内部邻接矩阵使用数据相关性初始化，形状为{adj_matrix.shape}")
            
        else:
            raise ValueError(f"未知的内部邻接矩阵初始化方法: {init_type}")
    else:
        # 外部邻接矩阵初始化
        adj_source = config['model']['external_adj_source']
        
        if adj_source == 'granger':
            # 使用格兰杰因果检验初始化
            max_lag = config['model']['granger_max_lag']
            significance = config['model']['granger_significance']
            
            print(f"使用格兰杰因果检验初始化外部邻接矩阵 (max_lag={max_lag}, significance={significance})...")
            adj_matrix, pvalues = granger_causality_matrix(
                time_series, 
                max_lag=max_lag, 
                significance=significance,
                verbose=True
            )
            
            # 是否归一化格兰杰检验结果
            if config['model']['granger_normalize'] and np.sum(adj_matrix) > 0:
                # 使用1-p值作为权重，p值越小，权重越大
                weights = 1.0 - pvalues
                weights[weights < 0] = 0  # 确保权重非负
                # 只保留显著的因果关系
                weights = weights * adj_matrix
                # 归一化权重
                weights_max = np.max(weights)
                if weights_max > 0:
                    weights = weights / weights_max
                adj_matrix = weights
                print("已将格兰杰检验结果转换为归一化权重")
            
        elif adj_source == 'correlation':
            # 使用相关性初始化
            threshold = 0.3  # 默认阈值
            adj_matrix, _ = correlation_matrix(time_series, threshold=threshold)
            print(f"使用相关性矩阵初始化外部邻接矩阵(threshold={threshold})，形状为{adj_matrix.shape}")
            
        elif adj_source == 'manual':
            # 手动指定的邻接矩阵
            manual_path = config['model'].get('manual_adj_path', None)
            if manual_path and os.path.exists(manual_path):
                try:
                    adj_matrix = pd.read_csv(manual_path, header=0, index_col=0).values
                    print(f"从 {manual_path} 加载手动指定的邻接矩阵，形状为 {adj_matrix.shape}")
                    
                    # 确保尺寸匹配
                    if adj_matrix.shape[0] != n_nodes or adj_matrix.shape[1] != n_nodes:
                        if adj_matrix.shape[0] < n_nodes or adj_matrix.shape[1] < n_nodes:
                            # 尺寸小于需要的，用零填充
                            temp = np.zeros((n_nodes, n_nodes))
                            temp[:adj_matrix.shape[0], :adj_matrix.shape[1]] = adj_matrix
                            adj_matrix = temp
                        else:
                            # 尺寸大于需要的，裁剪
                            adj_matrix = adj_matrix[:n_nodes, :n_nodes]
                except Exception as e:
                    print(f"加载手动邻接矩阵失败: {str(e)}，使用零矩阵")
                    adj_matrix = np.zeros((n_nodes, n_nodes))
            else:
                print(f"未找到手动指定的邻接矩阵文件，使用零矩阵")
                adj_matrix = np.zeros((n_nodes, n_nodes))
        
        else:
            raise ValueError(f"未知的外部邻接矩阵来源: {adj_source}")
    
    return adj_matrix 

# 添加从train.py中提取的函数
def process_weighted_adjacency_matrix(adj_matrix):
    """
    处理加权邻接矩阵，处理Wij和Wji的关系
    
    对于每对边(i,j)和(j,i)，只保留权重较大的那个
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # 创建副本，避免修改原始数据
    processed_adj = adj_matrix.copy()
    
    n = processed_adj.shape[0]
    
    # 对于每对节点(i,j)，比较Wij和Wji的权重，保留较大的一个
    for i in range(n):
        for j in range(i+1, n):  # 只需要遍历上三角矩阵
            if processed_adj[i, j] >= processed_adj[j, i]:
                # 如果i->j的权重大于等于j->i，保留i->j
                processed_adj[j, i] = 0
            else:
                # 否则保留j->i
                processed_adj[i, j] = 0
    
    # 确保对角线元素为0（无自环）
    np.fill_diagonal(processed_adj, 0)
    
    # 归一化权重到[0,1]范围
    max_weight = np.max(processed_adj)
    if max_weight > 0:
        processed_adj = processed_adj / max_weight
    
    return processed_adj

def apply_threshold_to_adjacency_matrix(weighted_adj, threshold=0.3):
    """
    对加权邻接矩阵应用阈值，生成二值化的邻接矩阵
    
    Args:
        weighted_adj: 加权邻接矩阵
        threshold: 阈值，大于该值的边将被保留
        
    Returns:
        binary_adj: 二值化的邻接矩阵
    """
    if isinstance(weighted_adj, torch.Tensor):
        weighted_adj = weighted_adj.detach().cpu().numpy()
    
    # 应用阈值
    binary_adj = (weighted_adj > threshold).astype(float)
    
    # 确保对角线元素为0（无自环）
    np.fill_diagonal(binary_adj, 0)
    
    return binary_adj

def plot_adjacency_matrix(adjacency_matrix, title="Adjacency Matrix", save_path=None):
    """
    绘制邻接矩阵的热图。
    
    Args:
        adjacency_matrix: 邻接矩阵
        title: 图表标题
        save_path: 保存路径，如果为None则显示图形
    """
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.detach().cpu().numpy()
    
    n = adjacency_matrix.shape[0]
    
    plt.figure(figsize=(7, 6))
    
    # 根据矩阵类型选择colormap
    if np.allclose(adjacency_matrix, adjacency_matrix.astype(bool).astype(float)):
        cmap = 'Blues'  # 二值矩阵
    else:
        cmap = 'viridis'  # 加权矩阵
    
    im = plt.imshow(adjacency_matrix, cmap=cmap, vmin=0, vmax=np.max(adjacency_matrix))
    plt.colorbar(im)
    
    plt.title(title)
    plt.xlabel('目标节点')
    plt.ylabel('源节点')
    plt.xticks(np.arange(n), np.arange(1, n+1))
    plt.yticks(np.arange(n), np.arange(1, n+1))
    
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

# 添加从train.py中提取的函数
def plot_adjacency_matrices_comparison(matrices_dict, save_path=None):
    """
    绘制多个邻接矩阵之间的对比图
    
    Args:
        matrices_dict: 字典，键为矩阵名称，值为邻接矩阵
        save_path: 保存图像的路径
    """
    n_matrices = len(matrices_dict)
    if n_matrices == 0:
        return
    
    # 创建图表
    fig, axes = plt.subplots(1, n_matrices, figsize=(5 * n_matrices, 5))
    
    # 如果只有一个矩阵，确保axes是列表
    if n_matrices == 1:
        axes = [axes]
    
    # 确定颜色映射范围
    vmin = 0
    vmax = max([np.max(mat) for mat in matrices_dict.values() if not np.all(mat == 0)])
    if vmax == 0:
        vmax = 1  # 防止所有矩阵都为零
    
    # 遍历绘制每个矩阵
    for i, (name, matrix) in enumerate(matrices_dict.items()):
        im = axes[i].imshow(matrix, cmap='Blues', vmin=vmin, vmax=vmax)
        axes[i].set_title(name)
        axes[i].set_xlabel('Target Node')
        axes[i].set_ylabel('Source Node')
        
        # 添加网格
        axes[i].set_xticks(np.arange(matrix.shape[1] - 0.5), minor=True)
        axes[i].set_yticks(np.arange(matrix.shape[0] - 0.5), minor=True)
        axes[i].grid(which="minor", color="gray", linestyle='-', linewidth=0.5, alpha=0.3)
        
        # 隐藏网格线
        axes[i].tick_params(which="minor", bottom=False, left=False)
    
    # 添加颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"邻接矩阵比较图已保存到 {save_path}")
    
    plt.show()

def optimize_adjacency_matrix_bayesian(model, dataloader, initial_adj, device, 
                                      n_iterations=25, init_points=5):
    """
    使用贝叶斯优化方法优化邻接矩阵和模型超参数
    
    Args:
        model: 训练好的模型
        dataloader: 训练数据加载器
        initial_adj: 初始邻接矩阵
        device: 计算设备
        n_iterations: 迭代次数
        init_points: 初始点数
    
    Returns:
        optimized_adj: 优化后的邻接矩阵
    """
    # 确保model是PFNO_Causal类型
    if not hasattr(model, '_calculate_dag_constraint'):
        # 动态导入DAG约束计算函数
        from models.model import PFNO_Causal
    print("\n开始贝叶斯邻接矩阵优化...")
    
    # 确保模型在评估模式
    model.eval()
    
    # 确保初始邻接矩阵是2D的
    if initial_adj.dim() > 2:
        initial_adj = initial_adj.squeeze()
    if initial_adj.dim() == 1:
        n_nodes = int(np.sqrt(initial_adj.numel()))
        initial_adj = initial_adj.view(n_nodes, n_nodes)
    
    # 初始化贝叶斯变量
    num_nodes = initial_adj.shape[-1]
    A = initial_adj.clone().to(device)  # 主变量
    Z = torch.zeros_like(A).to(device)  # 辅助变量
    U = torch.zeros_like(A).to(device)  # 拉格朗日乘子
    
    # 创建对角线掩码
    diag_mask = (1.0 - torch.eye(num_nodes)).to(device)
    
    # 获取一批数据用于评估
    batch = next(iter(dataloader))
    inputs, targets = batch[0].to(device), batch[1].to(device)
    
    # 定义可优化的超参数范围
    param_bounds = {
        'temperature': (0.1, 2.0),
        'sparsity_weight': (0.05, 2.0),  # 增加稀疏性权重的范围
        'dag_weight': (0.001, 0.1),
        'attention_dropout': (0.0, 0.5),
        'feature_dropout': (0.0, 0.5),
        'internal_weight': (0.0, 1.0)
    }
    
    # 贝叶斯迭代优化
    with tqdm(range(n_iterations), desc="贝叶斯优化进度") as pbar:
        for iteration in pbar:
            # 步骤1: 更新超参数
            with torch.no_grad():
                # 使用贝叶斯优化更新超参数
                for param_name, (min_val, max_val) in param_bounds.items():
                    if hasattr(model, param_name):
                        param = getattr(model, param_name)
                        # 使用高斯分布采样新的参数值，增加采样方差
                        new_value = torch.clamp(
                            param + torch.randn_like(param) * 0.2,  # 增加采样方差
                            min=min_val,
                            max=max_val
                        )
                        param.data.copy_(new_value)
            
            # 步骤2: 更新A - 模型拟合步骤
            A = A.detach().clone().requires_grad_(True)
            optimizer_A = torch.optim.LBFGS([A], lr=0.01, max_iter=5)
            
            def closure():
                optimizer_A.zero_grad()
                
                # 应用对角线掩码确保无自环
                A_masked = A * diag_mask
                
                # 使用当前邻接矩阵进行前向传播
                outputs, _, _, _ = model(inputs, external_adj=A_masked)
                
                # 计算重建损失
                recon_loss = torch.nn.functional.mse_loss(outputs, targets)
                
                # 计算DAG约束损失
                if A_masked.dim() > 2:
                    A_2d = A_masked.squeeze()
                else:
                    A_2d = A_masked
                
                # 使用新的通用函数计算DAG约束
                dag_violation = calculate_dag_constraint(A_2d, method='polynomial', degree=4)
                
                # 计算稀疏性损失，增加L1正则化强度
                sparsity_loss = torch.mean(torch.abs(A_masked)) * 2.0  # 增加稀疏性权重
                
                # 总损失，调整权重
                total_loss = (
                    recon_loss + 
                    0.01 * dag_violation + 
                    0.08 * sparsity_loss  # 增加稀疏性损失权重
                )
                
                total_loss.backward()
                
                # 确保梯度正确更新，增加梯度裁剪范围
                if hasattr(A, 'grad') and A.grad is not None:
                    A.grad.data.clamp_(-2, 2)  # 增加梯度裁剪范围
                
                return total_loss
            
            optimizer_A.step(closure)
            
            # 应用对角线掩码确保无自环
            A = A.detach() * diag_mask
            
            # 步骤3: 更新Z - 结构约束步骤
            Z_prev = Z.clone()
            Z = A + U
            
            # 应用软阈值操作实现L1正则化（稀疏性），增加阈值
            Z = torch.sign(Z) * torch.maximum(torch.abs(Z) - 0.05, torch.zeros_like(Z))  # 增加阈值
            
            # 应用DAG约束
            if Z.dim() > 2:
                Z = Z.squeeze()
            
            # 限制矩阵值范围
            Z = torch.clamp(Z, -0.9, 0.9)
            
            # 使用新的通用函数计算Z的DAG测度
            dag_measure = calculate_dag_constraint(Z, method='polynomial', degree=4)
            
            # 确保对角线为零（无自环）
            Z = Z * diag_mask
            
            # 步骤4: 更新拉格朗日乘子U
            U = U + (A - Z)
            
            # 计算当前邻接矩阵质量
            with torch.no_grad():
                outputs, _, _, _ = model(inputs, external_adj=A)
                recon_loss = torch.nn.functional.mse_loss(outputs, targets)
                
                # 监控DAG违反度，使用新的函数
                if A.dim() > 2:
                    A_2d = A.squeeze()
                else:
                    A_2d = A
                dag_violation = calculate_dag_constraint(A_2d, method='polynomial', degree=4)
                
                # 计算邻接矩阵变化量
                A_change = torch.norm(A - Z)
                
                # 更新进度条信息
                pbar.set_postfix({
                    "recon_loss": f"{recon_loss.item():.4f}",
                    "dag_violation": f"{dag_violation.item():.4f}",
                    "A_change": f"{A_change.item():.4f}"
                })
    
    # 返回最终优化的邻接矩阵
    return A.detach().cpu()

def calculate_dag_constraint(adjacency_matrix, method='polynomial', degree=4):
    """
    计算DAG约束的通用函数，使用数值稳定的方法
    
    Args:
        adjacency_matrix: 邻接矩阵，形状为[num_nodes, num_nodes]或batch形式
        method: 计算方法，'polynomial'使用多项式近似，'matrix_exp'使用矩阵指数
        degree: 多项式近似的阶数，默认为4
    
    Returns:
        dag_constraint: DAG约束值
    """
    # 确保邻接矩阵是2D的
    if adjacency_matrix.dim() > 2:
        adj = adjacency_matrix.squeeze()
    else:
        adj = adjacency_matrix
    
    # 限制邻接矩阵的值范围，防止数值不稳定
    adj = torch.clamp(adj, -0.9, 0.9)
    
    # 计算A^2
    A_squared = torch.matmul(adj, adj)
    
    num_nodes = adj.shape[0]
    
    if method == 'polynomial':
        # 使用多项式展开近似 exp(A) ≈ I + A + A²/2! + A³/3! + ...
        identity = torch.eye(num_nodes, device=adj.device)
        
        if degree == 2:
            poly_exp = identity + A_squared
        elif degree == 3:
            A_cubed = torch.matmul(A_squared, adj)
            poly_exp = identity + A_squared + A_cubed / 6.0
        elif degree == 4:
            A_cubed = torch.matmul(A_squared, adj)
            A_fourth = torch.matmul(A_squared, A_squared)
            poly_exp = identity + A_squared + A_cubed / 6.0 + A_fourth / 24.0
        else:
            # 默认使用2阶近似
            poly_exp = identity + A_squared
        
        dag_constraint = torch.trace(poly_exp) - num_nodes
    
    elif method == 'matrix_exp':
        # 如果仍需要使用矩阵指数，添加额外的安全措施
        try:
            dag_constraint = torch.trace(torch.matrix_exp(A_squared)) - num_nodes
        except Exception as e:
            print(f"矩阵指数计算错误: {e}，使用多项式近似替代")
            # 在出错时回退到多项式近似
            identity = torch.eye(num_nodes, device=adj.device)
            poly_exp = identity + A_squared + torch.matmul(A_squared, A_squared) / 2.0
            dag_constraint = torch.trace(poly_exp) - num_nodes
    
    # 检查是否为NaN并进行处理
    if torch.isnan(dag_constraint):
        print("警告：DAG约束为NaN，将其设置为大值")
        dag_constraint = torch.tensor(1000.0, device=adj.device)
    
    return dag_constraint 

def enhance_adjacency_probabilities(adj_matrix, method='sigmoid_amplify', temperature=2.0, power=0.5):
    """
    增强邻接矩阵的概率值，解决概率过低问题
    
    Args:
        adj_matrix: 原始邻接矩阵
        method: 增强方法 ('sigmoid_amplify', 'power_transform', 'percentile_rescale', 'temperature_scaling')
        temperature: 温度参数（用于temperature_scaling）
        power: 幂次参数（用于power_transform）
    
    Returns:
        enhanced_matrix: 增强后的邻接矩阵
    """
    if isinstance(adj_matrix, torch.Tensor):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    
    # 确保非负值
    adj_matrix = np.clip(adj_matrix, 0, 1)
    
    if method == 'sigmoid_amplify':
        # 使用sigmoid函数放大差异
        # 将[0,1]映射到更大的范围，然后重新sigmoid
        amplified = (adj_matrix - 0.5) * temperature + 0.5
        enhanced = 1 / (1 + np.exp(-amplified * 6))  # 6是放大系数
        
    elif method == 'power_transform':
        # 幂次变换，提升小概率值
        enhanced = np.power(adj_matrix, power)
        
    elif method == 'percentile_rescale':
        # 基于百分位数重新缩放
        p75 = np.percentile(adj_matrix, 75)
        p25 = np.percentile(adj_matrix, 25)
        if p75 > p25:
            enhanced = (adj_matrix - p25) / (p75 - p25)
            enhanced = np.clip(enhanced, 0, 1)
        else:
            enhanced = adj_matrix
            
    elif method == 'temperature_scaling':
        # 温度缩放
        enhanced = np.exp(adj_matrix / temperature)
        enhanced = enhanced / np.sum(enhanced, axis=1, keepdims=True)
        
    else:
        enhanced = adj_matrix
    
    return enhanced


def find_optimal_threshold_auto(pred_matrix, true_matrix, min_threshold=0.01, max_threshold=0.9, n_points=50):
    """
    自动寻找最优阈值
    
    Args:
        pred_matrix: 预测的邻接矩阵
        true_matrix: 真实邻接矩阵
        min_threshold: 最小阈值
        max_threshold: 最大阈值
        n_points: 搜索点数
    
    Returns:
        best_threshold: 最优阈值
        best_f1: 最优F1分数
        results: 所有阈值的结果
    """
    thresholds = np.linspace(min_threshold, max_threshold, n_points)
    results = []
    best_f1 = 0
    best_threshold = min_threshold
    
    print(f"搜索最优阈值范围: [{min_threshold:.3f}, {max_threshold:.3f}]")
    print(f"预测矩阵值范围: [{pred_matrix.min():.6f}, {pred_matrix.max():.6f}]")
    print(f"预测矩阵平均值: {pred_matrix.mean():.6f}")
    
    for threshold in thresholds:
        pred_binary = (pred_matrix > threshold).astype(int)
        
        # 统计边数
        pred_edges = pred_binary.sum()
        true_edges = true_matrix.sum()
        
        if pred_edges == 0:
            # 如果预测边数为0，跳过
            continue
            
        metrics = evaluate_causal_discovery(
            pred_adj=pred_binary,
            true_adj=true_matrix,
            threshold=0.5
        )
        
        if metrics is not None and metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold
            
        results.append({
            'threshold': threshold,
            'pred_edges': pred_edges,
            'true_edges': true_edges,
            'f1': metrics['f1'] if metrics else 0,
            'precision': metrics['precision'] if metrics else 0,
            'recall': metrics['recall'] if metrics else 0
        })
    
    print(f"最优阈值: {best_threshold:.6f}, 最优F1: {best_f1:.3f}")
    return best_threshold, best_f1, results


def adaptive_evaluate_causal_discovery(pred_matrix, true_matrix, enhancement_method='sigmoid_amplify'):
    """
    自适应评估因果发现结果，包含概率增强和动态阈值
    
    Args:
        pred_matrix: 预测的邻接矩阵
        true_matrix: 真实邻接矩阵
        enhancement_method: 概率增强方法
    
    Returns:
        results: 评估结果字典
    """
    print("=== 自适应因果发现评估 ===")
    
    # 原始评估
    print("\n1. 原始矩阵评估:")
    original_threshold, original_f1, original_results = find_optimal_threshold_auto(
        pred_matrix, true_matrix, min_threshold=0.001, max_threshold=0.5
    )
    
    # 增强后评估
    print(f"\n2. 使用 {enhancement_method} 方法增强概率值:")
    enhanced_matrix = enhance_adjacency_probabilities(pred_matrix, method=enhancement_method)
    print(f"增强后矩阵值范围: [{enhanced_matrix.min():.6f}, {enhanced_matrix.max():.6f}]")
    print(f"增强后矩阵平均值: {enhanced_matrix.mean():.6f}")
    
    enhanced_threshold, enhanced_f1, enhanced_results = find_optimal_threshold_auto(
        enhanced_matrix, true_matrix, min_threshold=0.01, max_threshold=0.9
    )
    
    # 比较结果
    print("\n3. 结果比较:")
    print(f"原始最优: 阈值={original_threshold:.6f}, F1={original_f1:.3f}")
    print(f"增强最优: 阈值={enhanced_threshold:.6f}, F1={enhanced_f1:.3f}")
    print(f"F1提升: {enhanced_f1 - original_f1:.3f}")
    
    # 返回最佳结果
    if enhanced_f1 > original_f1:
        print("推荐使用增强后的结果")
        return {
            'method': 'enhanced',
            'matrix': enhanced_matrix,
            'threshold': enhanced_threshold,
            'f1': enhanced_f1,
            'enhancement_method': enhancement_method,
            'all_results': enhanced_results
        }
    else:
        print("推荐使用原始结果")
        return {
            'method': 'original',
            'matrix': pred_matrix,
            'threshold': original_threshold,
            'f1': original_f1,
            'enhancement_method': None,
            'all_results': original_results
        } 