import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import math
import numpy as np

sys.path.append(os.getcwd())
from .layers import FNOBlocks, SpectralConv, MLP
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.spherical_layers import SphericalConv
from neuralop.models.base_model import BaseModel
from neuralop.layers.padding import DomainPadding
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.resample import resample
from neuralop.models.fno import partialclass
# 移除循环导入
# from utils import calculate_dag_constraint


# 定义Mish激活函数
class Mish(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        # Mish = x * tanh(softplus(x))
        return x * torch.tanh(F.softplus(x))


class PNO_Wrapper(nn.Module):
    """
    Takes a model and wraps it to generate n_samples from the predictive distribution.
    The model should be inherently stochastic (e.g. via dropout), otherwise all created samples will be identical.
    """

    def __init__(self, model: nn.Module, n_samples: int = 3):
        """Initialize the PNO_Wrapper.

        Args:
            model (nn.Module): Neural network.
            n_samples (int, optional): Number of output samples. Defaults to 3.
        """
        super(PNO_Wrapper, self).__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, input: torch.Tensor, n_samples: int = None) -> torch.Tensor:
        """Generate samples from the predictive distribution.

        Args:
            input (torch.Tensor): Input to the network.
            n_samples (int, optional): Number of samples to generate. Defaults to None.

        Returns:
            torch.Tensor: Samples from the predictive distribution.
        """
        if n_samples is None:
            n_samples = self.n_samples

        outputs = [self.model(input) for _ in range(n_samples)]

        # Stack along the second dimension and add a last dimension if missing.
        return torch.atleast_3d(torch.stack(outputs, dim=-1))

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if list(state_dict.keys())[0].startswith('model'):
            return super().load_state_dict(state_dict, strict, assign)
        else:
            self.model.load_state_dict(state_dict, strict, assign)
            return self


class PFNO_Causal(BaseModel, name="PFNO_Causal"):
    def __init__(
        self,
        n_modes,
        hidden_channels,
        num_nodes,
        n_samples=3,
        in_channels=1,  # 输入通道为 sequence_length * num_nodes
        out_channels=1,  # 输出通道为 num_nodes
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        dropout=0.0,
        fourier_dropout=None,
        output_scaling_factor=None,
        max_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        mlp_dropout=0.0,
        mlp_expansion=0.5,
        non_linearity=None,
        stabilizer=None,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="linear",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        conv_module=SpectralConv,
        use_diagonal_mask=False,  # 参数保留，但简化实现
        use_dag_constraint=True,  # 参数保留，但简化实现
        causal_reg_weight=0.001,  # 参数保留，但简化实现
        **kwargs
    ):
        super().__init__()
        self.num_nodes = num_nodes  # 节点数量
        self.d = num_nodes  # 添加 d 属性，与节点数量相同
        self.sequence_length = in_channels // num_nodes  # 从输入维度推断序列长度
        self.n_dim = len(n_modes)
        self.use_diagonal_mask = use_diagonal_mask  # 是否使用对角线掩码
        self.use_dag_constraint = use_dag_constraint

        # 简化隐藏通道计算，不再依赖注意力头数
        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mlp_skip = (mlp_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.fno_block_precision = fno_block_precision
        self.n_samples = n_samples
        self.causal_reg_weight = causal_reg_weight

        # Define dropout
        self.dropout = dropout
        if fourier_dropout is None:
            self.fourier_dropout = self.dropout
        else:
            self.fourier_dropout = fourier_dropout

        # Domain padding
        if domain_padding is not None and (
                (isinstance(domain_padding, list) and sum(domain_padding) > 0)
                or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        # Define FNO Blocks
        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            dropout=self.fourier_dropout,
            output_scaling_factor=output_scaling_factor,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs
        )

        # Lifting layer
        self.lifting = MLP(
            in_channels=in_channels,  # 输入通道为 sequence_length * num_nodes
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            dropout_rate=self.dropout,
        )

        # Reshape layer to convert 2D to 4D
        self.reshape = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels * n_modes[0] * n_modes[1]),
            nn.Unflatten(1, (hidden_channels, n_modes[0], n_modes[1]))  # 转换为 4D 张量
        )

        # 邻接矩阵生成器 - 简化版本
        self.adj_generator = MLP(
            in_channels=self.hidden_channels,
            out_channels=self.num_nodes * self.num_nodes,  # 简化：直接输出节点间的邻接矩阵
            hidden_channels=self.projection_channels,
            n_layers=2,  # 简化为2层
            dropout_rate=self.dropout,
            non_linearity=non_linearity,
        )

        # 初始化掩码
        if self.use_diagonal_mask:
            # 只屏蔽对角线元素，允许双向因果关系
            self.mask = torch.ones(num_nodes, num_nodes)
            self.mask.fill_diagonal_(0)
        else:
            # 不使用掩码，允许任意因果关系
            self.mask = torch.ones(num_nodes, num_nodes)

        # 设置默认的非线性激活函数为 GELU 模块
        if non_linearity is None:
            non_linearity = nn.GELU()

        # 直接预测输出的简化模型 - 移除了注意力机制和路径步骤
        self.final_projection = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            non_linearity,
            nn.Dropout(dropout),
            nn.Linear(hidden_channels*2, self.sequence_length * num_nodes)
        )

        # 添加非线性函数拟合增强组件 - 保留这部分以增强表达能力
        self.nonlinear_estimator = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels*2),
            nn.LayerNorm(hidden_channels*2),
            nn.GELU(),  # 使用GELU替代Mish激活函数
            nn.Dropout(dropout),
            nn.Linear(hidden_channels*2, hidden_channels)
        )

    def forward(self, x, y=None, output_shape=None, n_samples=None, external_adj=None, external_weight=None, **kwargs):
        if n_samples is None:
            n_samples = self.n_samples

        if output_shape is None:
            output_shape = [None] * self.n_layers

        batch_size = x.shape[0]
        
        # 记录输入形状，用于确保输出维度匹配
        original_shape = x.shape  # [batch_size, sequence_length * num_nodes]

        # Lifting layer
        x_lifted = self.lifting(x)  # [batch_size, hidden_channels]
        
        # Reshape to 4D
        x_reshaped = self.reshape(x_lifted)  # [batch_size, hidden_channels, n_modes[0], n_modes[1]]
        
        if self.domain_padding is not None:
            x_reshaped = self.domain_padding.pad(x_reshaped)
            
        # 应用 FNO blocks
        for layer_idx in range(self.n_layers):
            x_reshaped = self.fno_blocks(x_reshaped, layer_idx, output_shape=output_shape[layer_idx])
            
        if self.domain_padding is not None:
            x_reshaped = self.domain_padding.unpad(x_reshaped)
            
        # 全局平均池化
        x_pooled = torch.mean(x_reshaped, dim=(2, 3))  # [batch_size, hidden_channels]
        
        # 应用非线性函数拟合增强组件
        x_nonlinear = self.nonlinear_estimator(x_pooled)
        # 结合线性和非线性特征，使用残差连接
        x_pooled = x_pooled + 0.3 * x_nonlinear  # 使用权重因子控制非线性特征的影响
        
        # 生成邻接矩阵 - 改进版本
        adj_logits = self.adj_generator(x_pooled)
        # 重塑为 [batch_size, num_nodes, num_nodes]
        internal_adj_matrix = adj_logits.view(batch_size, self.num_nodes, self.num_nodes)
        
        # 应用掩码 - 避免就地操作
        if self.use_diagonal_mask:
            mask = self.mask.to(internal_adj_matrix.device)
            internal_adj_matrix = internal_adj_matrix * mask.unsqueeze(0)  # 扩展维度以匹配adj_matrix
        
        # 邻接矩阵生成改进
        # 1. 使用较大的温度系数使sigmoid函数不那么陡峭
        temperature = 1.5  # 提高温度使分布更平缓(值大于1使分布更平缓)
        
        # 2. 添加偏置鼓励非零边 - 创建新张量而非就地修改
        small_bias = 0.1  # 减小偏置值，避免过度影响
        internal_adj_matrix = internal_adj_matrix + small_bias  # 非就地操作
        
        # 应用sigmoid函数
        internal_adj_matrix = torch.sigmoid(internal_adj_matrix / temperature)
        
        # 确保对角线为0(无自环) - 避免就地操作
        diag_mask = 1.0 - torch.eye(self.num_nodes, device=internal_adj_matrix.device).unsqueeze(0)
        internal_adj_matrix = internal_adj_matrix * diag_mask
        
        # 4. 【新增】融合外部矩阵和内部矩阵
        if external_adj is not None:
            # 确保外部矩阵在正确的设备上
            if not isinstance(external_adj, torch.Tensor):
                external_adj = torch.tensor(external_adj, device=internal_adj_matrix.device, dtype=torch.float32)
            
            # 确保形状匹配
            if external_adj.dim() == 4:  # [batch_size, sequence_length, num_nodes, num_nodes]
                # 取时间维度的第一个时间步
                external_adj = external_adj[:, 0, :, :]

            # 融合外部和内部矩阵，使用可配置的权重
            if external_weight is None:
                internal_weight = 0.7  # 增加内部矩阵权重
                external_weight = 0.3  # 降低外部矩阵权重
            else:
                # 使用传入的外部权重参数
                external_weight = min(max(external_weight, 0.0), 1.0)  # 确保权重在[0,1]范围内
                internal_weight = 1.0 - external_weight
            
            # 避免就地操作，创建新张量
            adj_matrix = internal_weight * internal_adj_matrix + external_weight * external_adj
            
            # 重新确保对角线为0
            adj_matrix = adj_matrix * diag_mask
        else:
            # 如果没有外部矩阵，直接使用内部矩阵
            adj_matrix = internal_adj_matrix
        
        # 计算DAG损失 - 简化版本
        batch_size = adj_matrix.size(0)
        dag_loss = 0
        for i in range(batch_size):
            adj_batch_i = adj_matrix[i]
            # 使用内部方法计算DAG约束，避免循环导入
            dag_loss += self._calculate_dag_constraint(adj_batch_i)
        
        dag_loss = dag_loss / batch_size
        
        # 检查是否为NaN并进行处理
        if torch.isnan(dag_loss):
            print("警告：DAG损失为NaN，将其设置为大值")
            dag_loss = torch.tensor(1000.0, device=adj_matrix.device)
        
        # 简化：直接从池化特征生成输出，不使用注意力
        output = self.final_projection(x_pooled)
        
        # 确保输出维度与输入维度匹配
        output = output.view(original_shape)  # [batch_size, sequence_length * num_nodes]
        
        # 返回输出、邻接矩阵、DAG损失和None（替代原来的注意力权重）
        return output, adj_matrix, dag_loss, None

    def get_adj_matrix(self, input_samples=None):
        """
        获取当前模型的邻接矩阵
        
        Args:
            input_samples: 可选的输入样本，如果为None，则生成随机样本
        
        Returns:
            avg_adj_matrix: 平均邻接矩阵
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            
            # 如果没有提供输入样本，则生成随机样本
            if input_samples is None:
                # 使用多个随机样本以提高稳定性
                num_samples = 15
                adj_matrices = []
                
                for i in range(num_samples):
                    # 生成随机输入，使用较小的方差以避免极端值
                    dummy_input = torch.randn(1, self.in_channels, device=device) * 0.1
                    
                    # 前向传播获取邻接矩阵
                    _, adj_matrix, _, _ = self.forward(dummy_input)
                    adj_matrices.append(adj_matrix.squeeze(0))  # 移除批次维度
                
                # 求平均
                avg_adj_matrix = torch.mean(torch.stack(adj_matrices), dim=0)
            else:
                # 使用提供的输入样本
                if not isinstance(input_samples, torch.Tensor):
                    input_samples = torch.tensor(input_samples, device=device, dtype=torch.float32)
                
                # 确保输入格式正确
                if input_samples.dim() == 2:
                    input_samples = input_samples.unsqueeze(0)  # 添加批次维度
                
                # 前向传播
                _, adj_matrix, _, _ = self.forward(input_samples)
                avg_adj_matrix = adj_matrix.mean(dim=0)  # 对批次维度取平均
            
            # 确保对角线为零（无自环）
            mask = torch.ones_like(avg_adj_matrix) - torch.eye(self.num_nodes, device=device)
            avg_adj_matrix = avg_adj_matrix * mask
            
            return avg_adj_matrix
    
    def get_loss(self, pred, target, adj_matrix=None, dag_loss=None):
        """
        计算总损失，根据配置中的开关决定启用哪些损失函数
        Args:
            pred: shape [batch_size, sequence_length * num_nodes]
            target: shape [batch_size, sequence_length * num_nodes]
            adj_matrix: shape [batch_size, num_nodes, num_nodes]
            dag_loss: DAG约束损失
        """
        # 导入配置
        from config import CONFIG
        losses_config = CONFIG.get('losses', {})
        
        # 确保预测值和目标值具有相同的形状
        if pred.shape != target.shape:
            raise ValueError(f"预测值形状 {pred.shape} 与目标值形状 {target.shape} 不匹配")
        
        # 基础 MSE 重建损失
        mse_loss = F.mse_loss(pred, target)
        
        # 添加非线性重构损失：使用Huber损失改进对异常值的处理
        huber_loss = F.smooth_l1_loss(pred, target, beta=0.1)
        
        # 根据配置决定是否启用MSE和Huber损失
        if losses_config.get('mse_loss', True) and losses_config.get('huber_loss', True):
            recon_loss = 0.95 * mse_loss + 0.05 * huber_loss
        elif losses_config.get('mse_loss', True):
            recon_loss = mse_loss
        elif losses_config.get('huber_loss', True):
            recon_loss = huber_loss
        else:
            # 至少启用一种损失
            recon_loss = mse_loss

        # 初始化总损失为改进的重构损失
        total_loss = recon_loss
        
        # 如果提供了邻接矩阵，计算额外的正则化项
        if adj_matrix is not None:
            # 计算平均邻接矩阵（跨时间步）
            if len(adj_matrix.shape) == 4:  # [batch_size, sequence_length, num_nodes, num_nodes]
                avg_adj = adj_matrix.mean(dim=1)  # [batch_size, num_nodes, num_nodes]
            else:
                avg_adj = adj_matrix

            # 初始化损失组件
            l1_reg = torch.tensor(0.0, device=pred.device)
            sparsity_loss = torch.tensor(0.0, device=pred.device)
            nonzero_penalty = torch.tensor(0.0, device=pred.device)
            dag_penalty = torch.tensor(0.0, device=pred.device)

            # 1. 使用L1正则化，如果启用
            if losses_config.get('l1_reg', True):
                small_weights = avg_adj * (avg_adj <= 0.3).float()
                medium_weights = avg_adj * ((avg_adj > 0.3) & (avg_adj <= 0.7)).float()
                large_weights = avg_adj * (avg_adj > 0.7).float()
                l1_reg = (
                        torch.mean(torch.abs(small_weights)) * 0.15 +  # 大幅降低小权重惩罚
                        torch.mean(torch.abs(medium_weights)) * 0.10 +  # 降低中等权重惩罚
                        torch.mean(torch.abs(large_weights)) * 0.12  # 降低大权重惩罚
                )

            # 2. 稀疏性损失，如果启用
            if losses_config.get('sparsity_loss', True):
                sparsity_loss = torch.mean(torch.abs(avg_adj)) * 0.10

            # 3. 非零惩罚，如果启用
            if losses_config.get('nonzero_penalty', True):
                nonzero_penalty = -0.12 * torch.log(torch.mean(avg_adj) + 1e-8)

            # 4. DAG约束损失，如果启用
            if losses_config.get('dag_loss', True) and self.use_dag_constraint and dag_loss is not None:
                # 使用模型的causal_reg_weight替代硬编码值，以支持动态调整
                dag_penalty = self.causal_reg_weight * dag_loss

            # 组合所有损失 - 根据配置启用各部分
            total_loss = 1.50 * recon_loss  # 重构损失始终启用

            if losses_config.get('sparsity_loss', True):
                total_loss += 1.1 * sparsity_loss

            if losses_config.get('dag_loss', True) and self.use_dag_constraint and dag_loss is not None:
                total_loss += dag_penalty

            if losses_config.get('l1_reg', True):
                total_loss += l1_reg

            if losses_config.get('nonzero_penalty', True):
                total_loss += nonzero_penalty

        return total_loss, mse_loss, dag_loss

    def _calculate_dag_constraint(self, adjacency_matrix, degree=4):
        """
        计算DAG约束的内部方法，避免循环导入
        
        Args:
            adjacency_matrix: 邻接矩阵，形状为[num_nodes, num_nodes]
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
        
        # 检查是否为NaN并进行处理
        if torch.isnan(dag_constraint):
            print("警告：DAG约束为NaN，将其设置为大值")
            dag_constraint = torch.tensor(1000.0, device=adj.device)
        
        return dag_constraint

    # 其余方法保持不变
    def get_dag(self, threshold=0.1):
        """
        获取DAG邻接矩阵（简化版本）
        
        Args:
            threshold: 阈值，默认为0.1
        Returns:
            adj_binary: 二值化的邻接矩阵
        """
        # 创建一个零输入进行前向传播
        with torch.no_grad():
            # 使用较少的随机输入取平均，提高效率
            num_samples = 10
            adj_matrices = []
            
            device = next(self.parameters()).device
            
            for i in range(num_samples):
                # 生成随机输入
                dummy_input = torch.randn(1, self.in_channels, device=device) * 0.2
                
                # 前向传播获取邻接矩阵
                _, adj_matrix, _, _ = self.forward(dummy_input)
                adj_matrices.append(adj_matrix)
            
            # 求平均
            adj_matrix = torch.mean(torch.stack(adj_matrices), dim=0)
            
            # 使用阈值获取二值化邻接矩阵
            adj_binary = (adj_matrix > threshold).float()
            
            # 确保对角线为零（无自环）
            adj_binary = adj_binary * (1 - torch.eye(self.num_nodes, device=device))
            
            return adj_binary

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


# SFNO
PSFNO = partialclass("PSFNO", PFNO_Causal, conv_module=SphericalConv)