import numpy as np
import torch
import logging
import torch.nn.functional as F
from consts import LOG_FREQUENCY, LOG_FORMAT
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import networkx as nx
import random

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


class ALTrainer:
    def __init__(self, n, d, model, lr=1e-3, init_iter=10, early_stopping=True,
                 early_stopping_thresh=1e-5, patience=30, min_lr=1e-7,
                 seed=21, sequence_length=None, device=None, dag_constraint=True,
                 use_external_adj=False, external_adj=None,
                 initial_external_weight=0.5, final_external_weight=0.05, 
                 weight_transition_epochs=50):
        """
        初始化ALTrainer
        
        Args:
            n: 样本数量
            d: 特征维度
            model: 训练模型
            lr: 学习率
            init_iter: 初始迭代次数
            early_stopping: 是否使用早停
            early_stopping_thresh: 早停阈值
            patience: 早停耐心值
            min_lr: 最小学习率
            seed: 随机种子
            sequence_length: 序列长度
            device: 计算设备
            dag_constraint: 是否强制DAG约束
            use_external_adj: 是否使用外部邻接矩阵
            external_adj: 外部邻接矩阵
            initial_external_weight: 外部邻接矩阵的初始权重
            final_external_weight: 外部邻接矩阵的最终权重
            weight_transition_epochs: 权重过渡的轮数
        """
        self.n = n
        self.d = d
        self.model = model
        self.lr = lr
        self.init_iter = init_iter
        self.early_stopping = early_stopping
        self.early_stopping_thresh = early_stopping_thresh
        self.patience = patience
        self.min_lr = min_lr
        self.seed = seed
        self.device = device
        self.sequence_length = sequence_length
        self.no_improvement_count = 0
        self.dag_constraint = dag_constraint
        self.training = True  # 添加training属性
        
        # 新增与外部邻接矩阵相关的属性
        self.use_external_adj = use_external_adj
        self.external_adj = None if external_adj is None else torch.tensor(
            external_adj, device=device, dtype=torch.float32)
        self.initial_external_weight = initial_external_weight
        self.final_external_weight = final_external_weight
        self.weight_transition_epochs = weight_transition_epochs
        self.current_external_weight = initial_external_weight
        
        # 设置随机种子
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
        
        # 更新优化器配置，只包含模型参数
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.parameters(), 'weight_decay': 0.1}
        ], lr=lr * 0.5)
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True,
            threshold=1e-4,
            min_lr=1e-7
        )

        self.best_adj = None
        self.best_loss = float('inf')
        self.best_model_state = None
        self.current_epoch = 0
        
        self.train_losses = []
        self.val_losses = []
        
        self.adj_history = []
        
        self.best_epoch = 0
        self.plateau_count = 0
        self.min_delta = 1e-4

    def train(self, train_loader, val_loader, epochs=100, validate_every=5):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            validate_every: 每隔多少轮验证一次
            
        Returns:
            train_losses: 训练损失列表
            val_losses: 验证损失列表
            best_adj: 最佳邻接矩阵
        """
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            self.training = True  # 设置训练模式
            
            # 训练一个epoch
            epoch_loss = 0.0
            num_batches = 0
            
            for inputs, targets in train_loader:
                loss = self.train_step(inputs, targets)
                if loss is not None:
                    epoch_loss += loss
                    num_batches += 1
            
            if num_batches > 0:
                epoch_loss /= num_batches
                train_losses.append(epoch_loss)
            
            # 验证
            if (epoch + 1) % validate_every == 0:
                self.training = False  # 设置评估模式
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                
                # 更新学习率 - 修改为使用验证损失来调整学习率
                self.scheduler.step(val_loss)
                
                # 早停检查
                if self.early_stopping:
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        self.no_improvement_count = 0
                        # 保存最佳邻接矩阵
                        with torch.no_grad():
                            dummy_input = next(iter(train_loader))[0][:1].to(self.device)
                            _, best_adj, _, _ = self.model(dummy_input)
                            self.best_adj = best_adj[0]  # 取第一个样本的邻接矩阵
                    else:
                        self.no_improvement_count += 1
                        if self.no_improvement_count >= self.patience:
                            print(f"Early stopping triggered after {epoch + 1} epochs")
                            break
            
            print(f"Epoch [{epoch + 1}/{epochs}], "
                  f"Train Loss: {epoch_loss:.4f}, "
                  f"Val Loss: {val_losses[-1] if val_losses else 'N/A'}")
        
        return train_losses, val_losses, self.best_adj

    def train_step(self, inputs, targets, epoch=0):
        """
        执行一个训练步骤
        
        Args:
            inputs: 输入数据
            targets: 目标数据
            epoch: 当前epoch
            
        Returns:
            loss: 训练损失
        """
        self.training = True  # 确保在训练步骤中设置training为True
        self.model.train()  # 设置模型为训练模式
        
        # 计算当前的外部矩阵权重（如果使用外部矩阵）
        if self.use_external_adj and self.external_adj is not None:
            progress = min(self.current_epoch / self.weight_transition_epochs, 1.0)
            self.current_external_weight = self.initial_external_weight + \
                (self.final_external_weight - self.initial_external_weight) * progress
        
        # 使用混合精度训练
        with torch.amp.autocast('cuda', enabled=True):
            # 前向传播（传递外部矩阵和当前权重）
            if self.use_external_adj and self.external_adj is not None:
                # 确保外部矩阵形状与批次大小匹配
                if self.external_adj.dim() == 2:
                    external_adj_batch = self.external_adj.unsqueeze(0).expand(inputs.size(0), -1, -1)
                else:
                    external_adj_batch = self.external_adj
                
                outputs, adj_matrix, dag_loss, _ = self.model(
                    inputs, 
                    external_adj=external_adj_batch, 
                    external_weight=self.current_external_weight
                )
            else:
                # 不再传递生成的 A_dag，直接使用模型内部生成的邻接矩阵
                outputs, adj_matrix, dag_loss, _ = self.model(inputs)
            
            # 计算损失
            loss, mse_loss, dag_loss = self.model.get_loss(outputs, targets, adj_matrix, dag_loss)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 优化器步进
            self.optimizer.step()
            
            return loss.item()

    def validate(self, val_loader):
        """验证函数"""
        self.model.eval()
        val_loss = 0.0
        total_samples = 0
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
            for inputs, targets in val_loader:
                batch_size = inputs.size(0)
                inputs = inputs.to(self.device) if self.device else inputs
                
                outputs, adj_matrix, dag_loss, _ = self.model(inputs)
                
                total_loss, _, _ = self.model.get_loss(
                    outputs, inputs, adj_matrix, dag_loss
                )
                
                val_loss += total_loss.item() * batch_size
                total_samples += batch_size
        
        return val_loss / total_samples if total_samples > 0 else float('inf')

    def plot_losses(self, save_path=None):
        """绘制损失曲线"""
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        if len(self.val_losses) > 0:
            validate_every = len(self.train_losses) // len(self.val_losses)
            val_epochs = range(validate_every, len(self.train_losses) + 1, validate_every)
            plt.plot(val_epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.draw()
            plt.pause(0.1)

    def get_current_adj(self):
        """获取当前的邻接矩阵"""
        with torch.no_grad():
            # 使用模型的 get_adj_matrix 方法代替旧的逻辑
            return self.model.get_adj_matrix()

    def get_final_adjacency_matrix(self, threshold=0.5, force_dag=True):
        """
        获取最终的邻接矩阵，使用改进的稀疏性约束和边检测策略
        """
        with torch.no_grad():
            # 获取当前最佳的邻接矩阵权重
            if hasattr(self, 'best_adj'):
                adj_matrix = self.best_adj
            else:
                adj_matrix = self.model.get_adj_matrix()
            
            # 将adj_matrix转换为numpy数组
            if isinstance(adj_matrix, torch.Tensor):
                adj_matrix = adj_matrix.cpu().numpy()
            
            if adj_matrix.ndim == 4:  # 如果是4D张量 (batch, seq_len, n, n)
                adj_matrix = adj_matrix[0, 0]  # 取第一个batch和时间步
            
            n_nodes = adj_matrix.shape[0]
            print(f"Debug: Initial adj_matrix shape: {adj_matrix.shape}")
            print(f"Debug: Value range: [{np.min(adj_matrix):.4f}, {np.max(adj_matrix):.4f}]")
            
            # 1. 每个节点的出度控制
            target_edges = min(n_nodes * 3, n_nodes * (n_nodes-1) // 2)  # 基于average_degree=3设置目标边数
            edges_per_node = max(1, target_edges // n_nodes)  # 每个节点的平均出度
            
            # 2. 使用自适应阈值策略
            binary_adj = np.zeros_like(adj_matrix)
            for i in range(n_nodes):
                # 对每个节点的出边进行排序
                out_edges = adj_matrix[i, :]
                # 忽略自环
                out_edges[i] = -np.inf
                
                # 选择每个节点的top-k个最强出边
                if np.max(out_edges) > 0:  # 只有当存在正值时才选择边
                    k = min(edges_per_node, np.sum(out_edges > 0))
                    if k > 0:
                        top_indices = np.argpartition(out_edges, -k)[-k:]
                        # 只保留权重大于0.2的边
                        for idx in top_indices:
                            if out_edges[idx] > 0.2:
                                binary_adj[i, idx] = 1
            
            print(f"初始边数: {np.sum(binary_adj)}")
            
            # 3. 应用DAG约束
            if force_dag:
                original_weights = adj_matrix.copy()
                while not self.is_dag(binary_adj):
                    # 找到最弱的导致环的边
                    min_edge_weight = float('inf')
                    min_edge = None
                    
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            if binary_adj[i, j] > 0:
                                # 暂时移除这条边
                                binary_adj[i, j] = 0
                                
                                # 如果移除后仍然不是DAG，说明这条边不是导致环的关键边
                                if not self.is_dag(binary_adj):
                                    binary_adj[i, j] = 1
                                    continue
                                
                                # 评估边的重要性：使用原始权重
                                edge_importance = original_weights[i, j]
                                
                                if edge_importance < min_edge_weight:
                                    min_edge_weight = edge_importance
                                    min_edge = (i, j)
                                
                                # 恢复边
                                binary_adj[i, j] = 1
                    
                    if min_edge:
                        i, j = min_edge
                        binary_adj[i, j] = 0
                    else:
                        break  # 如果找不到边可以移除，退出循环
            
            # 4. 后处理：移除弱连接
            final_adj = binary_adj.copy()
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if final_adj[i, j] > 0 and original_weights[i, j] < 0.15:  # 移除权重过小的边
                        final_adj[i, j] = 0
            
            print(f"最终边数: {np.sum(final_adj)}")
            print(f"每个节点的平均出度: {np.sum(final_adj) / n_nodes:.2f}")
            
            return final_adj

    def is_dag(self, adj_matrix):
        """
        检查邻接矩阵是否代表有向无环图(DAG)
        使用图理论的方法检测环
        
        Args:
            adj_matrix: 邻接矩阵，可以是2D或3D numpy数组或tensor
        
        Returns:
            bool: 如果是DAG返回True，否则返回False
        """
        if isinstance(adj_matrix, torch.Tensor):
            adj_matrix = adj_matrix.detach().cpu().numpy()
        
        if adj_matrix.ndim == 3:
            adj_matrix = adj_matrix[0]
        
        if adj_matrix.ndim != 2:
            print(f"警告: is_dag收到了形状为{adj_matrix.shape}的非二维矩阵")
            if adj_matrix.ndim == 1:
                n_nodes = int(np.sqrt(len(adj_matrix) + 0.25))
                if n_nodes * n_nodes == len(adj_matrix):
                    adj_matrix = adj_matrix.reshape(n_nodes, n_nodes)
                else:
                    return False
            else:
                return False
        
        n = adj_matrix.shape[0]
        
        visited = [0] * n
        rec_stack = [0] * n
        
        def is_cyclic_util(v, visited, rec_stack):
            visited[v] = 1
            rec_stack[v] = 1
            
            for i in range(n):
                if adj_matrix[v, i] > 0:
                    if visited[i] == 0:
                        if is_cyclic_util(i, visited, rec_stack):
                            return True
                    elif rec_stack[i] == 1:
                        return True
            
            rec_stack[v] = 0
            return False
        
        for i in range(n):
            if visited[i] == 0:
                if is_cyclic_util(i, visited, rec_stack):
                    return False
        
        return True

    def train_epoch(self, train_loader):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            
        Returns:
            epoch_loss: 当前epoch的平均损失
        """
        self.model.train()  # 设置模型为训练模式
        epoch_loss = 0.0
        num_batches = 0
        
        for inputs, targets in train_loader:
            # 将数据移动到正确的设备
            inputs = inputs.to(self.device) if self.device else inputs
            targets = targets.to(self.device) if self.device else targets
            
            # 执行一个训练步骤
            loss = self.train_step(inputs, targets, self.current_epoch)
            if loss is not None:
                epoch_loss += loss
                num_batches += 1
        
        # 计算平均损失
        return epoch_loss / num_batches if num_batches > 0 else float('inf')