"""
PFNO增强数据加载器

集成了新的预处理系统，支持四种不同级别的数据预处理
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle
from sklearn.preprocessing import StandardScaler

# 导入预处理模块
from .core import preprocess_pfno_data, TimeSeriesPreprocessor


class TimeSeriesAugmentation(torch.utils.data.Dataset):
    """时间序列数据增强类"""
    
    def __init__(self, dataset, num_nodes, sequence_length, config=None):
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.original_size = len(dataset)
        
        # 从配置中获取增强参数，或使用默认值
        if config and 'augmentation' in config:
            self.augment_factor = config['augmentation'].get('factor', 1.5)
            self.techniques = config['augmentation'].get('techniques', ['scaling'])
            self.noise_levels = config['augmentation'].get('noise_level', [0.01, 0.05])
            self.scale_range = config['augmentation'].get('scale_range', [0.8, 1.2])
            self.time_warp_width = config['augmentation'].get('time_warp_width', [1, 5])
            self.window_shift = config['augmentation'].get('window_shift', [-2, 2])
        else:
            # 默认配置
            self.augment_factor = 1.5
            self.techniques = ['scaling']
            self.noise_levels = [0.01, 0.05]
            self.scale_range = [0.8, 1.2]
            self.time_warp_width = [1, 5]
            self.window_shift = [-2, 2]
        
        # 计算总大小
        self.total_size = int(self.original_size * self.augment_factor)
        
    def __len__(self):
        return self.total_size
    
    def __getitem__(self, idx):
        # 如果索引在原始范围内，返回原始数据
        if idx < self.original_size:
            return self.dataset[idx]
        
        # 否则返回增强数据
        original_idx = idx % self.original_size
        inputs, targets = self.dataset[original_idx]
        
        # 随机选择增强技术
        technique = np.random.choice(self.techniques)
        
        if technique == 'noise':
            noise_level = np.random.uniform(self.noise_levels[0], self.noise_levels[1])
            noise = torch.randn_like(inputs) * noise_level
            inputs = inputs + noise
            targets = targets + noise
            
        elif technique == 'scaling':
            scale_factor = np.random.uniform(self.scale_range[0], self.scale_range[1])
            inputs = inputs * scale_factor
            targets = targets * scale_factor
            
        elif technique == 'time_warp':
            # 简单的时间扭曲
            warp_width = np.random.randint(self.time_warp_width[0], self.time_warp_width[1] + 1)
            if warp_width < inputs.shape[-1]:
                start_idx = np.random.randint(0, inputs.shape[-1] - warp_width)
                warp_factor = np.random.uniform(0.8, 1.2)
                inputs[:, start_idx:start_idx + warp_width] *= warp_factor
                targets[:, start_idx:start_idx + warp_width] *= warp_factor
                
        elif technique == 'window_shift':
            shift = np.random.randint(self.window_shift[0], self.window_shift[1] + 1)
            if shift != 0:
                inputs = torch.roll(inputs, shifts=shift, dims=-1)
                targets = torch.roll(targets, shifts=shift, dims=-1)
        
        return inputs, targets


def load_pfno_data(config):
    """
    为PFNO加载和预处理数据
    
    Args:
        config: PFNO配置字典
        
    Returns:
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器
        seq_length: 序列长度
        num_nodes: 节点数量
        preprocessor: 预处理器实例
        raw_data: 原始数据
        preprocessing_info: 预处理信息
    """
    data_config = config['data']
    preprocessing_config = data_config['preprocessing']
    
    print(f"\n{'='*60}")
    print(f"              PFNO数据加载与预处理")
    print(f"{'='*60}")
    
    # 1. 读取原始数据
    file_path = data_config['file_path']
    print(f"📂 读取数据文件: {file_path}")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    
    raw_data = pd.read_csv(file_path)
    print(f"📊 原始数据形状: {raw_data.shape}")
    
    # 保存原始数据副本
    original_data = raw_data.copy()
    
    # 2. 限制时间步数
    max_time_steps = data_config.get('max_time_steps')
    if max_time_steps and max_time_steps < raw_data.shape[0]:
        raw_data = raw_data.iloc[:max_time_steps, :]
        print(f"🔢 限制为前{max_time_steps}个时间步，实际使用数据形状: {raw_data.shape}")
    
    # 3. 执行预处理
    print(f"\n🔧 预处理方法: {preprocessing_config['method']}")
    processed_data, preprocessor, preprocessing_info = preprocess_pfno_data(
        raw_data, preprocessing_config
    )
    
    print(f"✅ 预处理完成")
    for step in preprocessing_info['steps']:
        print(f"   • {step}")
    
    # 4. 生成时间序列样本
    sequence_length = data_config['sequence_length']
    num_nodes = processed_data.shape[1]
    
    print(f"\n📊 数据维度信息:")
    print(f"   • 节点数: {num_nodes}")
    print(f"   • 序列长度: {sequence_length}")
    print(f"   • 时间步数: {len(processed_data)}")
    
    # 转换为numpy数组
    data_values = processed_data.values
    
    # 生成序列样本
    samples = []
    targets = []
    
    for i in range(len(data_values) - sequence_length):
        sample = data_values[i:i + sequence_length]  # 输入序列
        target = data_values[i + 1:i + sequence_length + 1]  # 目标序列（下一时刻）
        samples.append(sample)
        targets.append(target)
    
    # 转换为numpy数组并调整维度
    x = np.array(samples)  # [num_samples, sequence_length, num_nodes]
    y = np.array(targets)  # [num_samples, sequence_length, num_nodes]
    
    # 调整维度为 (num_samples, num_nodes, sequence_length)
    x = x.transpose(0, 2, 1)
    y = y.transpose(0, 2, 1)
    
    print(f"   • 生成样本数: {x.shape[0]}")
    print(f"   • 样本形状: {x.shape}")
    
    # 5. 划分训练集和验证集
    train_ratio = data_config['train_ratio']
    train_size = int(x.shape[0] * train_ratio)
    
    x_train, x_val = x[:train_size], x[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    print(f"   • 训练集大小: {len(x_train)}")
    print(f"   • 验证集大小: {len(x_val)}")
    
    # 6. 展平数据（PFNO需要的格式）
    x_train = x_train.reshape(-1, num_nodes * sequence_length)
    y_train = y_train.reshape(-1, num_nodes * sequence_length)
    x_val = x_val.reshape(-1, num_nodes * sequence_length)
    y_val = y_val.reshape(-1, num_nodes * sequence_length)
    
    # 7. 转换为PyTorch张量
    x_train = torch.tensor(x_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    x_val = torch.tensor(x_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    
    # 8. 创建数据集
    train_dataset = TensorDataset(x_train, y_train)
    val_dataset = TensorDataset(x_val, y_val)
    
    # 9. 数据增强（仅对训练集）
    if data_config.get('augment', False):
        print(f"\n🔄 应用数据增强...")
        augmented_train_dataset = TimeSeriesAugmentation(
            train_dataset, num_nodes, sequence_length, data_config
        )
        print(f"   • 原始训练集大小: {len(train_dataset)}")
        print(f"   • 增强后训练集大小: {len(augmented_train_dataset)}")
        train_dataset = augmented_train_dataset
    
    # 10. 创建数据加载器
    batch_size = config['training']['batch_size']
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # 在某些环境中避免多进程问题
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    print(f"\n✅ 数据加载完成!")
    print(f"   • 训练批次数: {len(train_dataloader)}")
    print(f"   • 验证批次数: {len(val_dataloader)}")
    print(f"{'='*60}\n")
    
    return (train_dataloader, val_dataloader, sequence_length, num_nodes, 
            preprocessor, original_data, preprocessing_info)


def load_experimental_data_enhanced(config):
    """
    增强版的实验数据加载函数，使用新的预处理系统
    
    替换原来的load_experimental_data函数
    """
    # 使用新的数据加载器
    (train_dataloader, val_dataloader, seq_length, num_nodes, 
     preprocessor, raw_data, preprocessing_info) = load_pfno_data(config)
    
    # 加载真实邻接矩阵
    ground_truth_file = config['data']['ground_truth_file']
    ground_truth = None
    
    if os.path.exists(ground_truth_file):
        try:
            ground_truth_df = pd.read_csv(ground_truth_file, index_col=0, header=0)
            ground_truth = ground_truth_df.values
            print(f"📊 真实邻接矩阵形状: {ground_truth.shape}")
            print(f"📊 真实邻接矩阵节点标签: {ground_truth_df.columns.tolist()}")
            
            # 验证矩阵维度匹配
            if ground_truth.shape[0] != num_nodes or ground_truth.shape[1] != num_nodes:
                print(f"⚠️  警告: 真实邻接矩阵维度 {ground_truth.shape} 与数据节点数 {num_nodes} 不匹配")
                if ground_truth.shape[0] == ground_truth.shape[1]:
                    print(f"将使用 {ground_truth.shape[0]}x{ground_truth.shape[0]} 的真实邻接矩阵")
                    num_nodes = ground_truth.shape[0]
                else:
                    print("真实邻接矩阵不是方阵，将设置为None")
                    ground_truth = None
            
            if ground_truth is not None:
                print(f"📊 真实邻接矩阵中的边数: {ground_truth.sum()}")
                
        except Exception as e:
            print(f"❌ 加载真实邻接矩阵时出错: {e}")
            ground_truth = None
    else:
        print(f"⚠️  警告: 未找到真实邻接矩阵文件 {ground_truth_file}")
    
    # 保存预处理器和预处理信息
    results_dir = config.get('output', {}).get('results_base_dir', 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存预处理器
    preprocessor_path = os.path.join(results_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump(preprocessor, f)
    print(f"💾 预处理器已保存到: {preprocessor_path}")
    
    # 保存预处理信息
    preprocessing_info_path = os.path.join(results_dir, 'preprocessing_info.pkl')
    with open(preprocessing_info_path, 'wb') as f:
        pickle.dump(preprocessing_info, f)
    print(f"💾 预处理信息已保存到: {preprocessing_info_path}")
    
    return train_dataloader, val_dataloader, seq_length, num_nodes, ground_truth, raw_data 