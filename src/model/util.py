import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from einops import rearrange


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(1, d_model, 1, 1))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps)


class Inception(nn.Module):
    def __init__(self, d_model, d_hidden=None, norm_eps=1e-6):
        super(Inception, self).__init__()
        assert d_model % 4 == 0

        if d_hidden is None:
            d_hidden = 4 * d_model

        d_conv = d_model // 4
        self.dwconv_wh = nn.Conv2d(d_conv, d_conv, kernel_size=3, padding=1, groups=d_conv)
        self.dwconv_w = nn.Conv2d(d_conv, d_conv, kernel_size=(1, 11), padding=(0, 5), groups=d_conv)
        self.dwconv_h = nn.Conv2d(d_conv, d_conv, kernel_size=(11, 1), padding=(5, 0), groups=d_conv)

        self.norm = RMSNorm(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x):
        x_id, x_wh, x_w, x_h = x.chunk(4, 1)

        x = self.norm(torch.concatenate((
            x_id,
            self.dwconv_wh(x_wh),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
        ), dim=1)).permute(0, 2, 3, 1)

        return self.ffn(x).permute(0, 3, 1, 2)


class WindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=7):
        super(WindowAttention, self).__init__()
        assert d_model % n_heads == 0
        self.scale = sqrt(d_model // n_heads)
        self.window_size = window_size
        self.n_heads = n_heads

        self.rel_pos_bias = nn.Parameter(
            torch.empty(self.n_heads, (2 * self.window_size - 1) ** 2).normal_(std=0.02)
        )

        rel_coords = torch.stack(
            torch.meshgrid([torch.arange(self.window_size)] * 2)
        )
        rel_coords = torch.flatten(rel_coords, 1)
        rel_coords = rel_coords.unsqueeze(-1) - rel_coords.unsqueeze(1)
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords += self.window_size - 1
        rel_coords[:, :, 0] *= 2 * self.window_size - 1
        self.register_buffer(
            "rel_coords",
            rel_coords.sum(-1),
            persistent=False
        )

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x):
        x = rearrange(x, "b c (H wh) (W ww) -> b H W (wh ww) c", wh=self.window_size, ww=self.window_size)

        q = rearrange(self.W_q(x), "b H W L (n x) -> b H W n L x", n=self.n_heads)
        k = rearrange(self.W_k(x), "b H W L (n x) -> b H W n L x", n=self.n_heads)
        v = rearrange(self.W_v(x), "b H W L (n x) -> b H W n L x", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale + self.rel_pos_bias[:, self.rel_coords]

        x = F.softmax(attn, dim=-1) @ v

        return self.W_o(
            rearrange(x, "b h w n (H W) c -> b (n c) (h H) (w W)", H=self.window_size, W=self.window_size)
        )


class Block(nn.Module):
    def __init__(self, d_model, n_heads, window_size=7, norm_eps=1e-6):
        super(Block, self).__init__()
        self.attn = WindowAttention(d_model, n_heads, window_size=window_size)
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)

        self.ffn = Inception(d_model, norm_eps=norm_eps)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))

        return x + self.ffn(x)
