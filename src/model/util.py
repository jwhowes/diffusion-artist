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


class RMSFiLM(nn.Module):
    def __init__(self, d_model, d_t, eps=1e-6):
        super(RMSFiLM, self).__init__()
        self.eps = eps

        self.gamma = nn.Linear(d_t, d_model, bias=False)
        self.beta = nn.Linear(d_t, d_model, bias=False)

    def forward(self, x, t):
        B = x.shape[0]

        g = rearrange(self.gamma(t), "b d -> b d 1 1")
        b = rearrange(self.beta(t), "b d -> b d 1 1")

        return g * x * torch.rsqrt(x.pow(2).mean(1, keepdim=True) + self.eps) + b


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


class FiLMInception(nn.Module):
    def __init__(self, d_model, d_t, d_hidden=None, norm_eps=1e-6):
        super(FiLMInception, self).__init__()
        assert d_model % 4 == 0

        if d_hidden is None:
            d_hidden = 4 * d_model

        d_conv = d_model // 4
        self.dwconv_wh = nn.Conv2d(d_conv, d_conv, kernel_size=3, padding=1, groups=d_conv)
        self.dwconv_w = nn.Conv2d(d_conv, d_conv, kernel_size=(1, 11), padding=(0, 5), groups=d_conv)
        self.dwconv_h = nn.Conv2d(d_conv, d_conv, kernel_size=(11, 1), padding=(5, 0), groups=d_conv)

        self.norm = RMSFiLM(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )

    def forward(self, x, t):
        x_id, x_wh, x_w, x_h = x.chunk(4, 1)

        x = self.norm(torch.concatenate((
            x_id,
            self.dwconv_wh(x_wh),
            self.dwconv_w(x_w),
            self.dwconv_h(x_h)
        ), dim=1), t).permute(0, 2, 3, 1)

        return self.ffn(x).permute(0, 3, 1, 2)


class CrossAttention2d(nn.Module):
    def __init__(self, d_model, d_cond, n_heads):
        super(CrossAttention2d, self).__init__()
        assert d_model % n_heads == 0

        self.scale = sqrt(d_model // n_heads)
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_cond, d_model, bias=False)
        self.W_v = nn.Linear(d_cond, d_model, bias=False)

        self.W_o = nn.Conv2d(d_model, d_model, kernel_size=1)

    def forward(self, x, cond, attention_mask=None):
        B, _, H, W = x.shape
        L = cond.shape[1]

        x = rearrange(x, "b c h w -> b (h w) c")

        q = rearrange(self.W_q(x), "b l (n d) -> b n l d", n=self.n_heads)
        k = rearrange(self.W_k(cond), "b l (n d) -> b n l d", n=self.n_heads)
        v = rearrange(self.W_v(cond), "b l (n d) -> b n l d", n=self.n_heads)

        attn = (q @ k.transpose(-2, -1)) / self.scale

        if attention_mask is not None:
            attn = attn + attention_mask.view(B, 1, -1, L)

        x = F.softmax(attn, dim=-1) @ v

        return self.W_o(
            rearrange(x, "b n (h w) d -> b (n d) h w", h=H, w=W)
        )


class WindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window_size=8):
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
    def __init__(self, d_model, n_heads, window_size=8, norm_eps=1e-6):
        super(Block, self).__init__()
        self.attn = WindowAttention(d_model, n_heads, window_size=window_size)
        self.attn_norm = RMSNorm(d_model, eps=norm_eps)

        self.ffn = Inception(d_model, norm_eps=norm_eps)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))

        return x + self.ffn(x)


class FiLMBlock(nn.Module):
    def __init__(self, d_model, d_t, n_heads, window_size=8, norm_eps=1e-6):
        super(FiLMBlock, self).__init__()
        self.attn = WindowAttention(d_model, n_heads, window_size=window_size)
        self.attn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)

        self.ffn = FiLMInception(d_model, d_t, norm_eps=norm_eps)

    def forward(self, x, t):
        x = x + self.attn(self.attn_norm(x, t))

        return x + self.ffn(x, t)


class FiLMCrossAttentionBlock(nn.Module):
    def __init__(self, d_model, d_t, d_cond, n_heads, window_size=8, norm_eps=1e-6):
        super(FiLMCrossAttentionBlock, self).__init__()
        self.self_attn = WindowAttention(d_model, n_heads, window_size=window_size)
        self.self_attn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)

        self.cross_attn = CrossAttention2d(d_model, d_cond, n_heads)
        self.cross_attn_norm = RMSFiLM(d_model, d_t, eps=norm_eps)

        self.ffn = FiLMInception(d_model, d_t, norm_eps=norm_eps)

    def forward(self, x, t, cond, attention_mask=None):
        x = x + self.self_attn(self.self_attn_norm(x, t))

        x = x + self.cross_attn(
            self.cross_attn_norm(x, t), cond, attention_mask
        )

        return x + self.ffn(x, t)


class SinusoidalEmbedding(nn.Module):
    def __init__(self, d_model, base=1e5):
        super(SinusoidalEmbedding, self).__init__()
        assert d_model % 2 == 0
        self.register_buffer(
            "theta",
            1 / (base ** (2 * torch.arange(d_model // 2) / d_model)),
            persistent=False
        )

    def forward(self, x):
        x = x.float().view(-1, 1) * self.theta
        B = x.shape[0]

        return torch.stack((
            x.cos(),
            x.sin()
        ), dim=-1).view(B, -1)
