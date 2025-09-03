import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.pfno import PFNO
from models.layers import MLP


class ProbabilisticCausalPFNO(nn.Module):
    """
    概率因果PFNO模型
    使用重参数化技巧生成邻接矩阵的概率分布
    邻接矩阵元素A(i,j)表示节点i是节点j原因的概率
    """
    
    def __init__(self, 
                 n_modes, 
                 hidden_channels,
                 num_nodes,
                 sequence_length,
                 n_samples=10,
                 in_channels=None,
                 out_channels=None,
                 lifting_channels=256,
                 projection_channels=256,
                 n_layers=4,
                 dropout=0.1,
                 use_dag_constraint=True,
                 causal_reg_weight=0.001,
                 **kwargs):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.n_samples = n_samples
        self.use_dag_constraint = use_dag_constraint
        self.causal_reg_weight = causal_reg_weight
        
        # 计算输入输出通道
        if in_channels is None:
            in_channels = sequence_length * num_nodes
        if out_channels is None:
            out_channels = sequence_length * num_nodes
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 主PFNO网络用于时间序列预测
        self.pfno_backbone = PFNO(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_samples=n_samples,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            dropout=dropout,
            **kwargs
        )
        
        # 概率邻接矩阵生成器
        self.adj_mu_generator = MLP(
            in_channels=hidden_channels,
            out_channels=num_nodes * num_nodes,
            hidden_channels=projection_channels,
            n_layers=3,
            dropout_rate=dropout,
            non_linearity=nn.GELU()
        )
        
        self.adj_sigma_generator = MLP(
            in_channels=hidden_channels,
            out_channels=num_nodes * num_nodes,
            hidden_channels=projection_channels,
            n_layers=3,
            dropout_rate=dropout,
            non_linearity=nn.GELU()
        )
        
        # 用于特征提取的中间层（从PFNO中提取）
        self.feature_extractor = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels)
        )
        
        # 对角线掩码（防止自环）
        self.register_buffer('diag_mask', 
                           1.0 - torch.eye(num_nodes).unsqueeze(0))
    
    def forward(self, x, n_samples=None, return_adj_samples=False):
        """
        前向传播
        
        Args:
            x: 输入时间序列 [batch_size, sequence_length * num_nodes]
            n_samples: 采样数量
            return_adj_samples: 是否返回邻接矩阵的多个样本
            
        Returns:
            prediction_samples: 预测的多个样本 [batch_size, out_channels, n_samples]
            adj_probs: 邻接矩阵概率 [batch_size, num_nodes, num_nodes]
            adj_uncertainty: 邻接矩阵不确定性 [batch_size, num_nodes, num_nodes]
            dag_loss: DAG约束损失
        """
        if n_samples is None:
            n_samples = self.n_samples
            
        batch_size = x.shape[0]
        
        # 1. 使用PFNO生成预测样本
        prediction_samples = self.pfno_backbone(x, n_samples=n_samples)
        
        # 2. 提取特征用于邻接矩阵生成
        features = self.feature_extractor(x)  # [batch_size, hidden_channels]
        
        # 3. 生成邻接矩阵的均值和方差参数
        adj_mu_logits = self.adj_mu_generator(features)  # [batch_size, num_nodes^2]
        adj_sigma_logits = self.adj_sigma_generator(features)  # [batch_size, num_nodes^2]
        
        # 重塑为邻接矩阵形状
        adj_mu = adj_mu_logits.view(batch_size, self.num_nodes, self.num_nodes)
        adj_log_sigma = adj_sigma_logits.view(batch_size, self.num_nodes, self.num_nodes)
        
        # 4. 使用重参数化技巧生成概率邻接矩阵
        adj_sigma = F.softplus(adj_log_sigma) + 1e-6  # 确保方差为正
        
        if return_adj_samples:
            # 生成多个邻接矩阵样本
            epsilon = torch.randn(batch_size, self.num_nodes, self.num_nodes, n_samples, device=x.device)
            adj_samples = adj_mu.unsqueeze(-1) + adj_sigma.unsqueeze(-1) * epsilon
            # 转换为概率
            adj_prob_samples = torch.sigmoid(adj_samples)
            # 应用对角线掩码
            adj_prob_samples = adj_prob_samples * self.diag_mask.unsqueeze(-1)
            
            # 计算均值作为最终概率
            adj_probs = torch.mean(adj_prob_samples, dim=-1)
            # 计算方差作为不确定性
            adj_uncertainty = torch.var(adj_prob_samples, dim=-1)
        else:
            # 单次采样
            epsilon = torch.randn_like(adj_mu)
            adj_sample = adj_mu + adj_sigma * epsilon
            # 转换为概率
            adj_probs = torch.sigmoid(adj_sample)
            # 应用对角线掩码
            adj_probs = adj_probs * self.diag_mask
            # 不确定性估计
            adj_uncertainty = adj_sigma
        
        # 5. 计算DAG约束损失
        dag_loss = 0
        if self.use_dag_constraint:
            for i in range(batch_size):
                dag_loss += self._calculate_dag_constraint(adj_probs[i])
            dag_loss = dag_loss / batch_size
        
        return prediction_samples, adj_probs, adj_uncertainty, dag_loss
    
    def _calculate_dag_constraint(self, adj_prob_matrix, degree=3):
        """
        计算概率邻接矩阵的DAG约束
        使用多项式近似 tr(exp(A)) - n
        """
        n = adj_prob_matrix.shape[0]
        A = adj_prob_matrix
        
        # 使用多项式近似
        I = torch.eye(n, device=A.device)
        A_squared = torch.matmul(A, A)
        
        if degree >= 3:
            A_cubed = torch.matmul(A_squared, A)
            exp_A_approx = I + A + A_squared/2 + A_cubed/6
        else:
            exp_A_approx = I + A + A_squared/2
            
        dag_constraint = torch.trace(exp_A_approx) - n
        return dag_constraint
    
    def get_causal_probabilities(self, x, n_samples=50):
        """
        获取因果关系概率矩阵
        
        Args:
            x: 输入数据
            n_samples: 用于估计的样本数量
            
        Returns:
            causal_probs: 因果概率矩阵 [batch_size, num_nodes, num_nodes]
            uncertainty: 不确定性矩阵 [batch_size, num_nodes, num_nodes]
        """
        self.eval()
        with torch.no_grad():
            _, adj_probs, adj_uncertainty, _ = self.forward(
                x, n_samples=n_samples, return_adj_samples=True
            )
        return adj_probs, adj_uncertainty
    
    def get_causal_graph(self, x, confidence_threshold=0.5, uncertainty_threshold=0.1):
        """
        获取因果图，基于概率和不确定性阈值
        
        Args:
            x: 输入数据
            confidence_threshold: 概率阈值
            uncertainty_threshold: 不确定性阈值
            
        Returns:
            causal_graph: 二值因果图
            high_confidence_graph: 高置信度因果图
        """
        causal_probs, uncertainty = self.get_causal_probabilities(x)
        
        # 基于概率的因果图
        causal_graph = (causal_probs > confidence_threshold).float()
        
        # 基于低不确定性的高置信度因果图
        high_confidence_mask = uncertainty < uncertainty_threshold
        high_confidence_graph = causal_graph * high_confidence_mask.float()
        
        return causal_graph, high_confidence_graph
    
    def get_loss(self, pred_samples, target, adj_probs=None, dag_loss=None):
        """
        计算总损失
        
        Args:
            pred_samples: 预测样本 [batch_size, out_channels, n_samples]
            target: 目标值 [batch_size, out_channels]
            adj_probs: 邻接矩阵概率
            dag_loss: DAG约束损失
            
        Returns:
            total_loss: 总损失
            recon_loss: 重构损失
            uncertainty_loss: 不确定性损失
        """
        # 1. 预测损失（负对数似然）
        pred_mean = torch.mean(pred_samples, dim=-1)
        pred_var = torch.var(pred_samples, dim=-1) + 1e-6
        
        # 高斯负对数似然
        nll_loss = 0.5 * torch.log(2 * np.pi * pred_var) + \
                   0.5 * (target - pred_mean)**2 / pred_var
        recon_loss = torch.mean(nll_loss)
        
        # 2. 邻接矩阵稀疏性损失
        sparsity_loss = 0
        if adj_probs is not None:
            # 鼓励稀疏性
            sparsity_loss = torch.mean(adj_probs)
        
        # 3. DAG约束损失
        dag_penalty = 0
        if dag_loss is not None:
            dag_penalty = self.causal_reg_weight * dag_loss
        
        # 4. 不确定性正则化
        uncertainty_loss = 0
        if adj_probs is not None:
            # 鼓励低不确定性（但不过度惩罚）
            uncertainty_loss = 0.01 * torch.mean(pred_var)
        
        total_loss = recon_loss + 0.2 * sparsity_loss + dag_penalty + uncertainty_loss
        
        return total_loss, recon_loss, uncertainty_loss 