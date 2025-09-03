from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


# We will use a simplified, self-contained version of SpectralConv and skip_connection
# to avoid dependency issues and ensure stability.

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, n_modes_height, n_modes_width):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_height = n_modes_height
        self.n_modes_width = n_modes_width

        self.fourier_weight = nn.ParameterList([
            nn.Parameter(torch.empty(in_channels, out_channels, n_modes_height, n_modes_width, dtype=torch.cfloat))
            for _ in range(2)  # Two weights for two parts of the spectrum
        ])

        for param in self.fourier_weight:
            nn.init.kaiming_normal_(param, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x, norm='ortho')

        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, dtype=torch.cfloat, device=x.device)

        # 动态调整模式数，以防止其超过输入数据的傅里叶域分辨率
        modes_h = min(self.n_modes_height, H)
        modes_w = min(self.n_modes_width, W // 2 + 1)

        # Top-left and top-right modes
        # 关键：同时对输入和权重进行切片，确保einsum的维度匹配
        out_ft[:, :, :modes_h, :modes_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, :modes_h, :modes_w],
            self.fourier_weight[0][:, :, :modes_h, :modes_w]
        )
        out_ft[:, :, -modes_h:, :modes_w] = torch.einsum(
            "bixy,ioxy->boxy",
            x_ft[:, :, -modes_h:, :modes_w],
            self.fourier_weight[1][:, :, :modes_h, :modes_w]
        )

        x = torch.fft.irfft2(out_ft, s=(H, W), norm='ortho')
        return x


class GatedMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=None, out_channels=None, non_linearity=F.gelu, dropout_rate=0.0):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or int(in_channels * 2)
        self.non_linearity = non_linearity
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.gate_fc = nn.Conv2d(in_channels, hidden_channels, 1)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        hidden = self.non_linearity(self.fc1(x)) * torch.sigmoid(self.gate_fc(x))
        hidden = self.dropout(hidden)
        out = self.fc2(hidden)
        return out


class FNOBlocks(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,  # expects a tuple (n_modes_height, n_modes_width)
            n_layers=1,
            use_mlp=False,
            mlp_dropout=0,
            mlp_expansion=0.5,
            non_linearity=F.gelu,
            **kwargs,  # Absorb extra params
    ):
        super().__init__()
        self.n_layers = n_layers
        self.fno_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.fno_layers.append(
                SpectralConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    n_modes_height=n_modes[0],
                    n_modes_width=n_modes[1],
                    use_mlp=use_mlp,
                    mlp_dropout=mlp_dropout,
                    mlp_expansion=mlp_expansion,
                    non_linearity=non_linearity
                )
            )

    def forward(self, x: torch.Tensor, output_layer_indices: list = None):
        if output_layer_indices:
            intermediate_outputs = {}

        for i, layer in enumerate(self.fno_layers):
            x = layer(x)
            if output_layer_indices and i in output_layer_indices:
                intermediate_outputs[i] = x

        if output_layer_indices:
            return x, intermediate_outputs

        return x


class SpectralConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes_height,
            n_modes_width,
            use_mlp=False,
            mlp_dropout=0.0,
            mlp_expansion=0.5,
            non_linearity=F.gelu,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_mlp = use_mlp

        self.conv = SpectralConv2d(in_channels, out_channels, n_modes_height, n_modes_width)
        self.w = nn.Conv2d(in_channels, out_channels, 1)  # Skip connection
        self.non_linearity = non_linearity

        if use_mlp:
            self.mlp = GatedMLP(
                in_channels=out_channels,
                hidden_channels=int(round(out_channels * mlp_expansion * 2)),
                out_channels=out_channels,
                dropout_rate=mlp_dropout,
                non_linearity=non_linearity,
            )

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = self.non_linearity(x)

        if self.use_mlp:
            x_mlp = self.mlp(x)
            x = x + x_mlp  # Second skip connection

        return x