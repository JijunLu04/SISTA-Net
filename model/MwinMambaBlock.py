import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial
from typing import Optional, Callable
from timm.layers import DropPath, to_2tuple, trunc_normal_

from typing import List, Sequence, Tuple

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref

# -----------------------------Mamba SS2D Module---------------------------------
class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=8, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=8, merge=True)  # (K=4, D, N)

        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 8

        x_hwwh = torch.stack([x.view(B, -1, L),
                              torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L),
                              torch.rot90(x, 1, dims=[2, 3]).contiguous().view(B, -1, L),
                              torch.rot90(x, -1, dims=[2, 3]).contiguous().view(B, -1, L)
                              ],dim=1).view(B, 4, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        # First 4 original ways
        y1 = out_y[:, 0]  # x.view(B, -1, L)
        y2 = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)  # transpose
        y3 = torch.rot90(out_y[:, 2].view(B, -1, W, H), -1, dims=[2, 3]).contiguous().view(B, -1, L)  # rotate 90 clockwise
        y4 = torch.rot90(out_y[:, 3].view(B, -1, W, H), 1, dims=[2, 3]).contiguous().view(B, -1,L)  # rotate 90 counterclockwise

        # Flipped ways
        y5 = torch.flip(out_y[:, 4], dims=[-1])
        y6 = torch.flip(out_y[:, 5].view(B, -1, W, H), dims=[2, 3]).contiguous().view(B, -1, L)
        y7 = torch.rot90(out_y[:, 6].view(B, -1, W, H), -1, dims=[2, 3]).flip(dims=[-1]).contiguous().view(B, -1, L)
        y8 = torch.rot90(out_y[:, 7].view(B, -1, W, H), 1, dims=[2, 3]).flip(dims=[-1]).contiguous().view(B, -1, L)

        return y1, y2, y3, y4, y5, y6, y7, y8

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4, y5, y6, y7, y8 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4 + y5 + y6 + y7 + y8
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            input_dim: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.reduction = nn.Linear(input_dim, hidden_dim, bias=False)

    def forward(self, input: torch.Tensor):
        input = self.reduction(input)
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x


def window_partition(
    x: torch.Tensor, window_size: int
) -> Tuple[torch.Tensor, int, int, int, int]:
    """Partition windows, padding spatial dims if not divisible by window_size."""
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h or pad_w:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    H_pad, W_pad = H + pad_h, W + pad_w
    x = x.view(
        B,
        H_pad // window_size,
        window_size,
        W_pad // window_size,
        window_size,
        C,
    )
    windows = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )
    return windows, H_pad, W_pad, pad_h, pad_w


def window_reverse(
    windows: torch.Tensor, window_size: int, H_pad: int, W_pad: int, pad_h: int, pad_w: int
) -> torch.Tensor:
    """Reverse window partitioning and remove spatial padding."""
    B = int(windows.shape[0] / (H_pad * W_pad / window_size / window_size))
    x = windows.view(
        B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1
    )
    x = (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H_pad, W_pad, -1)
    )
    if pad_h or pad_w:
        x = x[:, : H_pad - pad_h, : W_pad - pad_w, :].contiguous()
    return x

class WindowMambaBlock(nn.Module):
    """Helper block that runs VSSBlock inside sliding windows."""

    def __init__(
        self,
        dim: int,
        window_size: int,
        drop_path: float,
        attn_drop_rate: float,
        d_state: int,
        norm_layer: Callable[..., torch.nn.Module],
    ):
        super().__init__()
        self.window_size = window_size
        self.mamba = VSSBlock(
            input_dim=dim,
            hidden_dim=dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            attn_drop_rate=attn_drop_rate,
            d_state=d_state,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = x.permute(0, 2, 3, 1).contiguous()

        window_size = min(self.window_size, H, W)
        windows, H_pad, W_pad, pad_h, pad_w = window_partition(feat, window_size)
        windows = self.mamba(windows)
        feat = window_reverse(windows, window_size, H_pad, W_pad, pad_h, pad_w)
        feat = feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return feat


class MultiWinMambaBlocks(nn.Module):
    """Apply multiple window-based Mamba blocks with different window sizes."""

    def __init__(
        self,
        dim: int,
        window_sizes: Sequence[int],
        drop_path: Sequence[float] | float = 0.1,
        attn_drop_rate: float = 0.1,
        d_state: int = 16,
        norm_layer: Callable[..., torch.nn.Module] | None = None,
    ):
        super().__init__()
        if len(window_sizes) == 0:
            raise ValueError("window_sizes must contain at least one size")

        if isinstance(drop_path, float):
            drop_path = [drop_path] * len(window_sizes)
        assert (
            len(drop_path) == len(window_sizes)
        ), "drop_path length must match window_sizes"

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.blocks = nn.ModuleList(
            [
                WindowMambaBlock(
                    dim=dim,
                    window_size=win_size,
                    drop_path=drop_path[idx],
                    attn_drop_rate=attn_drop_rate,
                    d_state=d_state,
                    norm_layer=norm_layer,
                )
                for idx, win_size in enumerate(window_sizes)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass that keeps BCHW layout."""
        for block in self.blocks:
            x = block(x)
        return x

# Example:
# multi_win_block = MultiWinMambaBlocks(dim=64, window_sizes=[8, 16, 32])


