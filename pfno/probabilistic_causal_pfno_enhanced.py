"""
CFNO-P/PFNO模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from .layers import FNOBlocks
from torch.fft import rfft2, irfft2



def safe_reshape_for_pfno(x: torch.Tensor, num_nodes: int, sequence_length: int) -> torch.Tensor:
    """统一的维度变换函数，确保PFNO输入维度正确"""
    batch_size = x.shape[0]
    x_reshaped = x.view(batch_size, 1, num_nodes, sequence_length)
    return x_reshaped


def check_numerical_stability(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """检查并修复数值稳定性问题"""
    if torch.isnan(tensor).any():
        warnings.warn(f"NaN detected in {name}, replacing with 0.")
        tensor = torch.where(torch.isnan(tensor), torch.zeros_like(tensor), tensor)
    if torch.isinf(tensor).any():
        warnings.warn(f"Inf detected in {name}, clamping to +/-1e6.")
        tensor = torch.clamp(tensor, -1e6, 1e6)
    return tensor


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x_out = irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x_out


class SlidingWindowEncoder(nn.Module):
    """使用一维卷积（滑动窗口）对时间序列进行编码，保留时间维度。"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class EnhancedProbabilisticCausalPFNO(nn.Module):
    """
    重构为基于注意力机制的模型，实现"重构优先，因果涌现"。
    完整保留了多级特征融合和维度/数值检查等鲁棒性设计。
    """
    def __init__(self, n_modes, hidden_channels, num_nodes, sequence_length,
                 lifting_channels, projection_channels,
                 in_channels=1, n_layers=4, dropout=0.1, n_attn_heads=4, 
                 fno_output_layer_indices: list = None, **kwargs):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.in_channels = in_channels
        self.fno_output_layer_indices = fno_output_layer_indices if fno_output_layer_indices is not None else [n_layers - 1]

        # 1. Lifting层 & FNO频谱卷积 (2D FNO)
        self.lifting = nn.Conv2d(self.in_channels, lifting_channels, 1)
        self.pre_fno_conv = nn.Conv2d(lifting_channels, hidden_channels, 1)
        self.spectral_layers = FNOBlocks(
            in_channels=hidden_channels, out_channels=hidden_channels,
            n_modes=n_modes, n_layers=n_layers, use_mlp=True,
            mlp_dropout=dropout, non_linearity=F.gelu
        )
        
        # 2. 节点特征提取器升级: 输出特征序列以保留时序信息
        # 动态计算窗口大小，以防止其超过输入序列长度
        if sequence_length < 4:
            safe_window_size = 2
        else:
            safe_window_size = min(10, sequence_length // 2)
        safe_window_size = max(2, safe_window_size)
        
        self.local_pattern_extractor = SlidingWindowEncoder(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=safe_window_size
        )

        # 3. 动态因果头升级: 手动实现交叉注意力以获取pre-softmax分数
        self.n_attn_heads = n_attn_heads
        self.causal_q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.causal_k_proj = nn.Linear(hidden_channels, hidden_channels)
        
        # 新增: LayerNorm 和可学习的缩放参数
        self.causal_logit_norm = nn.LayerNorm(num_nodes)
        self.logit_scale = nn.Parameter(torch.ones(1))

        # 改进初始化，确保更好的初始分布
        nn.init.xavier_uniform_(self.causal_q_proj.weight)
        nn.init.xavier_uniform_(self.causal_k_proj.weight)
        nn.init.zeros_(self.causal_q_proj.bias)
        nn.init.zeros_(self.causal_k_proj.bias)

        # 4. 核心：多头注意力层 (现在用于时空重构)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_channels, num_heads=n_attn_heads, 
            dropout=dropout, batch_first=True
        )

        # 5. 全新的融合预测头 (Convolutional) - 使用 projection_channels
        # 接收拼接后的特征 (原始时序特征 + 注意力上下文)，通道数加倍
        self.prediction_head = nn.Sequential(
            nn.Conv2d(hidden_channels * 2, projection_channels, 1),
            nn.BatchNorm2d(projection_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(projection_channels, projection_channels // 2, 1),
            nn.BatchNorm2d(projection_channels // 2),
            nn.GELU(),
            nn.Conv2d(projection_channels // 2, 2, 1) # 输出2个通道: 均值和对数方差
        )

        # 缓冲区
        self.register_buffer('diag_mask', 1.0 - torch.eye(num_nodes, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')).unsqueeze(0))
        
        # 可学习的因果偏置参数，用于打破对称性
        self.causal_bias = nn.Parameter(torch.randn(1, num_nodes, num_nodes) * 0.01)

    def forward(self, x, tau=0.5, only_prediction=False, temperature=1.0, **kwargs):
        batch_size = x.shape[0]
        x_grid = safe_reshape_for_pfno(x, self.num_nodes, self.sequence_length)
        x_grid = check_numerical_stability(x_grid, "input_grid")
        
        lifted_features = self.lifting(x_grid)
        features_for_fno = self.pre_fno_conv(lifted_features)
        
        # 直接获取最后一层的输出，而不是中间特征字典
        fused_features_grid, _ = self.spectral_layers(
            features_for_fno, output_layer_indices=self.fno_output_layer_indices
        )
        
        # 如果只进行预测，则跳过复杂的因果和注意力计算
        if only_prediction:
            # 使用一个简化的头进行快速预测
            reconstruction_output = self.prediction_head(
                torch.cat([fused_features_grid, fused_features_grid], dim=1) # 提供一个形状正确的输入
            )
            mean_grid, _ = reconstruction_output.chunk(2, dim=1)
            prediction_mean = mean_grid.squeeze(1).view(batch_size, -1)
            return {'prediction': prediction_mean}

        B, C, N, T = fused_features_grid.shape
        
        # --- 1. 因果发现模块：升级为"历史事件驱动的动态因果评估" ---
        # a. 提取每个节点的局部动态特征序列 (每个节点的历史发展脉络)
        node_features_temporal = fused_features_grid.permute(0, 2, 1, 3).reshape(B * N, C, T)
        local_feature_sequences = self.local_pattern_extractor(node_features_temporal)
        
        # b. 重塑为 (B, N, T_out, C) 格式，其中T_out代表历史时期的数量
        T_out = local_feature_sequences.shape[-1]
        node_sequences = local_feature_sequences.view(B, N, C, T_out).permute(0, 1, 3, 2)

        # c. 将每个历史时期的特征投影为 Query 和 Key
        queries = self.causal_q_proj(node_sequences) # (B, N, T_out, C)
        keys = self.causal_k_proj(node_sequences)    # (B, N, T_out, C)

        # d. 使用爱因斯坦求和(einsum)高效地完成所有历史时期的因果评估与分数累加
        # 'binc,bjnc->bij' 的含义:
        #   b: batch, i/j: 节点, n: 历史时期(T_out), c: 特征
        #   在每个batch(b)和历史时期(n)内，计算节点i的query和节点j的key的点积(c)，
        #   然后将所有历史时期(n)的结果累加起来，得到最终的 i 对 j 的总影响力。
        #   注意：这里计算的是 j -> i 的影响力
        attn_logits = torch.einsum('binc,bjnc->bij', queries, keys)

        # e. 转置以获得我们期望的格式: causal_logits[b, i, j] 代表 i -> j
        causal_logits = attn_logits.transpose(1, 2)
        
        # f. 新增: 应用LayerNorm进行稳定
        causal_logits = self.causal_logit_norm(causal_logits)
        
        # g. 使用可学习的参数进行缩放
        causal_logits = causal_logits * self.logit_scale

        # h. 应用掩码，确保对角线为0
        causal_logits = causal_logits * self.diag_mask
        
        # i. 添加可学习的偏置以打破对称性
        causal_logits = causal_logits + self.causal_bias * self.diag_mask
        
        # --- 2. 因果引导的时空重构 ---
        # a. 准备时空特征
        spatio_temporal_features = fused_features_grid.view(B, C, N * T).permute(0, 2, 1)

        # b. 手动分解多头注意力以获取 pre-softmax logits
        qkv = F.linear(spatio_temporal_features, self.attention.in_proj_weight, self.attention.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        H = self.attention.num_heads
        D_head = self.attention.embed_dim // H
        q = q.contiguous().view(B, N * T, H, D_head).transpose(1, 2)
        k = k.contiguous().view(B, N * T, H, D_head).transpose(1, 2)
        v = v.contiguous().view(B, N * T, H, D_head).transpose(1, 2)
        
        # c. 计算原始注意力分 (logits)
        scale = D_head ** -0.5
        attn_logits_flat = torch.matmul(q, k.transpose(-1, -2)) * scale
        
        # --- 修正：使用加性引导代替乘性门控 ---
        # d. 将causal_logits扩展到与注意力logits匹配的维度
        # (B, N, N) -> (B, 1, N, 1, N, 1) -> (B, H, N, T, N, T) -> (B, H, N*T, N*T)
        causal_guidance = causal_logits.view(B, 1, N, 1, N, 1).expand(B, H, N, T, N, T)
        causal_guidance_flat = causal_guidance.reshape(B, H, N*T, N*T)

        # e. 将因果引导（logits）加到原始注意力logits上
        guided_attn_logits = attn_logits_flat + causal_guidance_flat

        # f. 通过Softmax获得最终的注意力权重，自然归一化
        attn_weights_flat = F.softmax(guided_attn_logits, dim=-1)
        
        # g. 应用 Dropout
        attn_weights_flat = F.dropout(attn_weights_flat, p=self.attention.dropout, training=self.training)
        
        # h. 计算注意力输出
        attn_output_flat = torch.matmul(attn_weights_flat, v)
        attn_output_flat = attn_output_flat.transpose(1, 2).contiguous().view(B, N * T, C)
        attn_output_projected = self.attention.out_proj(attn_output_flat)

        # g. 将注意力输出重塑为网格状的上下文特征
        context_features = attn_output_projected.view(B, N, T, C).permute(0, 3, 1, 2)

        # --- 3. 融合与预测 ---
        combined_features = torch.cat([fused_features_grid, context_features], dim=1)
        reconstruction_output = self.prediction_head(combined_features)
        
        # 将输出分解为均值和对数方差
        mean_grid, log_var_grid = reconstruction_output.chunk(2, dim=1)
        
        prediction_mean = mean_grid.squeeze(1).view(batch_size, -1)
        prediction_log_var = log_var_grid.squeeze(1).view(batch_size, -1)
        
        # --- 4. 准备输出 ---
        # 从logits计算最终的概率矩阵用于评估和损失计算
        # 应用温度缩放 (T > 1 会使概率分布更平滑)
        # 关键修正：使用Sigmoid函数，为每个潜在的因果关系(i->j)独立地计算其存在的概率。
        # 这种方式下，一个节点可以有多个父节点，概率之和不为1。
        
        # 确保数值稳定性
        causal_logits = check_numerical_stability(causal_logits, "causal_logits")
        
        # 应用温度缩放和sigmoid
        scaled_logits = causal_logits / temperature
        causal_matrix = torch.sigmoid(scaled_logits)
        
        # 最后应用掩码，确保对角线为0 (解决sigmoid(0)=0.5的问题)
        causal_matrix = causal_matrix * self.diag_mask

        return {
            'prediction': prediction_mean, # 保持'prediction'以兼容
            'prediction_mean': prediction_mean,
            'prediction_log_var': prediction_log_var,
            'causal_matrix': causal_matrix,
            'causal_logits': causal_logits,
        }

    def _calculate_dag_constraint(self, adj_prob_matrix, degree=3):
        """
        更稳定的DAG约束损失计算，基于 'DAGs with NO TEARS' 方法.
        h(A) = tr(e^(A◦A)) - n
        其中 ◦ 是逐元素乘积.
        """
        A = adj_prob_matrix
        n = A.shape[1]
        # Element-wise product
        A_squared = A * A
        # Differentiable matrix exponential
        matrix_exp = torch.matrix_exp(A_squared)
        # The trace of the matrix exponential
        trace = torch.diagonal(matrix_exp, dim1=-2, dim2=-1).sum(-1)
        # The DAG constraint loss
        dag_loss = trace - n
        return torch.mean(dag_loss)

    def get_loss(self, prediction_mean, prediction_log_var, target, causal_matrix, causal_logits, 
                 loss_weights=None, epoch=None, total_epochs=None, **kwargs):
        """
        完整的损失函数，包含重构损失和所有正则化项。
        使用静态权重配置。
        """
        lw = loss_weights if loss_weights is not None else {}
        rw = lw.get('reconstruction_weights', {})

        # 1. 重构损失 - 分为直接损失和分布损失
        # 1a. 直接损失 (MSE)
        direct_loss = F.mse_loss(prediction_mean, target)

        # 修正：使用更稳定的NLL公式
        variance = torch.exp(prediction_log_var)
        # 增加epsilon以防止除以零，增强数值稳定性
        nll_loss = 0.5 * (prediction_log_var + (prediction_mean - target) ** 2 / (variance + 1e-8))
        # 确保损失非负
        distributional_loss = torch.mean(torch.clamp(nll_loss, min=0))

        recon_loss = (rw.get('direct', 0.5) * direct_loss +
                      rw.get('dist', 0.5) * distributional_loss)

        # 2. L1稀疏性损失 (作用于原始logits上)
        l1_loss = torch.mean(torch.abs(causal_logits))

        # 3. DAG损失 (鼓励无环)
        dag_loss = self._calculate_dag_constraint(causal_matrix)

        # 4. 新增：熵正则化 (鼓励概率接近0或1，惩罚不确定性)
        epsilon = 1e-8
        entropy = -causal_matrix * torch.log(causal_matrix + epsilon) - (1 - causal_matrix) * torch.log(1 - causal_matrix + epsilon)
        entropy_loss = torch.mean(entropy)

        # 5. L2 正则化 - 从模型参数中手动计算 (修复复数问题)
        l2_loss = 0
        for name, param in self.named_parameters():
            if 'weight' in name:
                if param.is_complex():
                    l2_loss += torch.sum(param.abs().pow(2))
                else:
                    l2_loss += torch.sum(param.pow(2))

        # 6. 计算加权总损失
        total_loss = (lw.get('recon_loss', 1.0) * recon_loss +
                      lw.get('l1_regularization', 0.1) * l1_loss +
                      lw.get('dag_constraint', 0.1) * dag_loss +
                      lw.get('entropy_regularization', 0.05) * entropy_loss +
                      lw.get('l2_regularization', 0.002) * l2_loss)

        # 7. 构建简化的损失字典
        loss_dict = {
            'recon_loss': recon_loss,
            'direct_loss': direct_loss,
            'distributional_loss': distributional_loss,
            'l1_regularization': l1_loss,
            'dag_constraint': dag_loss,
            'entropy_regularization': entropy_loss,
            'l2_regularization': l2_loss,
            'total': total_loss
        }
        
        return total_loss, loss_dict

    def get_causal_probabilities(self, x, n_samples=50, temperature=1.0):
        """获取因果关系概率矩阵，支持温度缩放。"""
        self.eval()
        with torch.no_grad():
            prob_samples = []
            
            for _ in range(n_samples):
                result = self.forward(x, temperature=temperature)
                prob_samples.append(result['causal_matrix'])
            
            prob_stack = torch.stack(prob_samples, dim=0)
            causal_probs = torch.mean(prob_stack, dim=0)
            
            if causal_probs.dim() == 3:
                causal_probs = causal_probs[0]
            
            return causal_probs
    
    def get_causal_graph(self, x, confidence_threshold=0.3, n_samples=50, temperature=2.0):
        """获取因果图，基于概率阈值，支持温度缩放。"""
        causal_probs = self.get_causal_probabilities(x, n_samples=n_samples, temperature=temperature)
        causal_graph = (causal_probs > confidence_threshold).float()
        return causal_graph 