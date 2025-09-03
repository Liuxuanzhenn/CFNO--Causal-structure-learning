# This file provides the implementation of the FNOBlocks and SpectralConv classes as layers of the neuraloperator.
# The code is adapted from https://github.com/neuraloperator/neuraloperator.

# MIT License

# Copyright (c) 2023 NeuralOperator developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from neuralop.utils import validate_scaling_factor
import torch
from torch import nn
from tltorch.factorized_tensors.core import FactorizedTensor
from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.resample import resample
from neuralop.layers.spectral_convolution import get_contract_fun

from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.normalization_layers import AdaIN
from neuralop.layers.fno_block import SubModule
from neuralop.layers.spectral_convolution import SubConv
from neuralop.layers.spherical_convolution import SHT

Number = Union[int, float]


class MLP(torch.nn.Module):
    # Obtain input of shape [Batch, channels, d1, ..., dn]
    def __init__(
            self,
            in_channels,
            out_channels,
            hidden_channels,
            non_linearity=F.gelu,
            dropout_rate=None,
            n_layers=2,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        if dropout_rate is not None:
            self.dropout_rate = nn.ModuleList(
                [nn.Dropout(dropout_rate) for _ in range(self.n_layers - 1)]
            )
        else:
            self.dropout_rate = None

        # Input layer
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        # Hidden layers
        for j in range(self.n_layers - 2):
            self.fcs.append(torch.nn.Linear(hidden_channels, hidden_channels))
        # Output layer
        self.fcs.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        # Reorder channel dim to last dim
        x = torch.movedim(x, 1, -1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                if self.dropout_rate is not None:
                    x = self.dropout_rate[i](x)
                x = self.non_linearity(x)

        # Return channel dim
        x = torch.movedim(x, -1, 1)
        return x


class SpectralConv(BaseSpectralConv):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            dropout_rate=None,
            max_n_modes=None,
            bias=True,
            n_layers=1,
            separable=False,
            output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
            fno_block_precision="full",
            rank=0.5,
            factorization=None,
            implementation="reconstructed",
            fixed_rank_modes=False,
            joint_factorization=False,
            decomposition_kwargs: Optional[dict] = None,
            init_std="auto",
            fft_norm="backward",
            device=None,
            dtype=None,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization
        # Dropout
        self.dropout_rate = dropout_rate
        # n_modes is the total number of modes kept along each dimension
        self.n_modes = n_modes
        self.order = len(self.n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.order, n_layers)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels)) ** 0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *max_n_modes)
        else:
            weight_shape = (in_channels, out_channels, *max_n_modes)
        self.separable = separable

        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **tensor_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **tensor_kwargs,
                    )
                    for _ in range(n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)
        self._contract = get_contract_fun(
            self.weight[0], implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

    def _get_weight(self, index):
        return self.weight[index]

    def transform(self, x, layer_index=0, output_shape=None):
        in_shape = list(x.shape[2:])

        if self.output_scaling_factor is not None and output_shape is None:
            out_shape = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(in_shape, self.output_scaling_factor[layer_index])
                ]
            )
        elif output_shape is not None:
            out_shape = output_shape
        else:
            out_shape = in_shape

        if in_shape == out_shape:
            return x
        else:
            return resample(
                x,
                1.0,
                list(range(2, x.ndim)),
                output_shape=out_shape,
            )

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int):  # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        # The last mode has a redundacy as we use real FFT
        # As a design choice we do the operation here to avoid users dealing with the +1
        n_modes[-1] = n_modes[-1] // 2 + 1
        self._n_modes = n_modes

    # Dropout mask
    def get_dropout_mask(self, weights):
        mask = torch.ones_like(weights.real)
        if self.dropout_rate is not None:
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout_rate, training=self.training
            )
        return mask

    def forward(
            self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            # 使用float32替代half，以避免ComplexHalf警告
            x = x.float()

        # 使用torch.amp.autocast代替直接调用rfftn，并确保使用float32精度
        with torch.autocast(device_type='cuda', enabled=False):
            # 确保x是float32类型，避免使用half
            x = x.to(torch.float32)
            x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
            if self.order > 1:
                x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at a different precision
            # 使用complex64替代复数half类型
            x = x.to(torch.complex64)

        # 始终使用complex64作为输出类型，避免使用complexhalf
        out_dtype = torch.complex64
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device=x.device, dtype=out_dtype)
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in
                  zip(fft_size, self.n_modes, self.max_n_modes)]
        slices_w = [slice(None), slice(None)]  # Batch_size, channels
        slices_w += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [
            slice(None, -starts[-1]) if starts[-1] else slice(None)]  # The last mode already has redundant half removed
        weight = self._get_weight(indices)[slices_w]

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x = [slice(None), slice(None)]  # Batch_size, channels
        slices_x += [slice(start // 2, -start // 2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [
            slice(None, -starts[-1]) if starts[-1] else slice(None)]  # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])])

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])

        # Apply Dropout in Frequency space
        out_fft = out_fft * self.get_dropout_mask(out_fft)
        
        # 使用torch.amp.autocast代替直接调用irfftn
        with torch.autocast(device_type='cuda', enabled=False):
            x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            Warning("A single convolution is parametrized, directly use the main class.")

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)


class FNOBlocks(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            dropout=None,
            output_scaling_factor=None,
            n_layers=1,
            max_n_modes=None,
            fno_block_precision="full",
            use_mlp=False,
            mlp_dropout=0,
            mlp_expansion=0.5,
            non_linearity=F.gelu,
            stabilizer=None,
            norm=None,
            ada_in_features=None,
            preactivation=False,
            fno_skip="linear",
            mlp_skip="soft-gating",
            separable=False,
            factorization=None,
            rank=1.0,
            conv_module=SpectralConv,
            joint_factorization=False,
            fixed_rank_modes=False,
            implementation="factorized",
            decomposition_kwargs=dict(),
            fft_norm="forward",
            **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.mlp_skip = mlp_skip
        self.use_mlp = use_mlp
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        # Dropout
        self.dropout_rate = dropout

        self.convs = conv_module(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            dropout_rate=self.dropout_rate,
            output_scaling_factor=output_scaling_factor,
            max_n_modes=max_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if use_mlp:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * mlp_expansion),
                        dropout_rate=mlp_dropout,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList(
        #         [
        #             nn.LayerNorm(elementwise_affine=False)
        #             for _ in range(n_layers*self.n_norms)
        #         ]
        #     )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, ada_in]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if self.mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.mlp[index](x) + x_skip_mlp

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class CausalConv(nn.Module):
    """
    多核因果卷积层
    
    特点:
    1. 时间优先约束: 通过左填充确保卷积只考虑历史数据
    2. 多核设计: 为每对时间序列分配独立的卷积核
    3. 自因果处理: 通过右移卷积结果避免自预测泄露
    """
    def __init__(
        self,
        num_nodes,
        kernel_size=3,
        dilation=1,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
        # 为每对节点创建独立的卷积核
        # shape: [num_nodes, num_nodes, kernel_size]
        self.weight = nn.Parameter(
            torch.randn(num_nodes, num_nodes, kernel_size) / (kernel_size ** 0.5)
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_nodes))
        else:
            self.register_parameter('bias', None)
            
        # 初始化权重
        self._reset_parameters()
        
    def _reset_parameters(self):
        """初始化卷积核参数"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, sequence_length, num_nodes]
            
        Returns:
            输出张量 [batch_size, sequence_length, num_nodes]
        """
        batch_size, seq_len, _ = x.shape
        
        # 左填充以确保因果性 (只看历史数据)
        # 填充大小 = (kernel_size - 1) * dilation
        padding_size = (self.kernel_size - 1) * self.dilation
        x_padded = F.pad(x, (0, 0, padding_size, 0))
        
        # 初始化输出
        output = torch.zeros_like(x)
        
        # 对每个目标节点应用卷积
        for i in range(self.num_nodes):
            # 对每个源节点应用独立的卷积核
            for j in range(self.num_nodes):
                # 跳过自环 (可选，取决于是否允许自因果)
                # if i == j:
                #     continue
                
                # 提取当前节点对的卷积核
                kernel = self.weight[i, j]  # [kernel_size]
                
                # 对源节点的时间序列应用1D卷积
                for t in range(seq_len):
                    # 计算卷积窗口的起始位置
                    start_idx = t
                    
                    # 应用卷积
                    for k in range(self.kernel_size):
                        # 确保索引在有效范围内
                        idx = start_idx + padding_size - k * self.dilation
                        if 0 <= idx < seq_len + padding_size:
                            output[:, t, i] += x_padded[:, idx, j] * kernel[k]
        
        # 添加偏置
        if self.bias is not None:
            output += self.bias
            
        # 应用dropout
        if self.dropout is not None:
            output = self.dropout(output)
            
        return output
    
    def get_causal_weights(self):
        """
        获取因果权重矩阵
        
        Returns:
            因果权重矩阵 [num_nodes, num_nodes]，表示节点间的因果强度
        """
        # 使用卷积核的L1范数作为因果强度指标
        causal_weights = torch.sum(torch.abs(self.weight), dim=2)
        return causal_weights
    
    def get_time_lags(self):
        """
        获取最大响应的时间滞后
        
        Returns:
            时间滞后矩阵 [num_nodes, num_nodes]，表示因果关系的时间延迟
        """
        # 找到每个卷积核的最大响应位置
        max_response_idx = torch.argmax(torch.abs(self.weight), dim=2)
        
        # 转换为时间滞后 (kernel_size - 1 - idx) * dilation
        time_lags = (self.kernel_size - 1 - max_response_idx) * self.dilation
        
        return time_lags

# 导入math模块，用于CausalConv类中的初始化
import math