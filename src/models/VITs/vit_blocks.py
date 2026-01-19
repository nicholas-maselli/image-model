from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """
    Stochastic Depth per sample (when applied in main path of residual blocks).
    """
    if drop_prob <= 0.0 or (not training):
        return x
    keep_prob = 1.0 - float(drop_prob)
    # Work with broadcastable random tensor: (B, 1, 1, ...)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor = random_tensor.floor()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, drop_prob=self.drop_prob, training=self.training)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SDPA(nn.Module):
    """
    Multi-head self-attention implemented with scaled_dot_product_attention (SDPA).

    This tends to be faster than nn.MultiheadAttention on many GPUs because it can hit
    fused attention kernels when available.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")

        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(dim // num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = float(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        b, t, d = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        qkv = qkv.reshape(b, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, Hd)
        q, k, v = qkv.unbind(dim=0)  # each: (B, H, T, Hd)

        # Dropout is only applied during training
        drop_p = self.attn_drop if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)  # (B, H, T, Hd)
        out = out.transpose(1, 2).reshape(b, t, d)  # (B, T, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class EncoderBlock(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDPA(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x

class EncoderBlockLS(nn.Module):
    def __init__(self, *, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.0):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

import torch
import torch.nn as nn

from .vit_blocks import SDPA, MLP, DropPath


def _make_gn(channels: int, max_groups: int = 32) -> nn.GroupNorm:
    g = min(max_groups, channels)
    while g > 1 and (channels % g) != 0:
        g //= 2
    return nn.GroupNorm(g, channels)


class EncoderBlockLS(nn.Module):
    """Pre-norm Transformer block + optional LayerScale (helps stability when scaling depth)."""

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        layer_scale_init: float = 0.0,  # set ~1e-5 to enable
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SDPA(dim=dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim=dim, hidden_dim=int(dim * mlp_ratio), drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if layer_scale_init and layer_scale_init > 0.0:
            self.gamma1 = nn.Parameter(torch.ones(dim) * float(layer_scale_init))
            self.gamma2 = nn.Parameter(torch.ones(dim) * float(layer_scale_init))
        else:
            self.gamma1 = None
            self.gamma2 = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.attn(self.norm1(x))
        if self.gamma1 is not None:
            a = a * self.gamma1
        x = x + self.drop_path1(a)

        m = self.mlp(self.norm2(x))
        if self.gamma2 is not None:
            m = m * self.gamma2
        x = x + self.drop_path2(m)
        return x
