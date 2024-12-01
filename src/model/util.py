import torch
import torch.nn.functional as F

from torch import nn
from math import sqrt
from einops import rearrange


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class FiLM(nn.Module):
    def __init__(self, d_model, d_t, eps=1e-6):
        super(FiLM, self).__init__()
        self.norm = LayerNorm2d(d_model, eps=eps, elementwise_affine=False)

        self.gamma = nn.Linear(d_t, d_model, bias=False)
        self.beta = nn.Linear(d_t, d_model, bias=False)

    def forward(self, x, t):
        g = rearrange(self.gamma(t), "b d -> b d 1 1")
        b = rearrange(self.beta(t), "b d -> b d 1 1")

        return g * self.norm(x) + b


class InceptionBlock(nn.Module):
    def __init__(self, d_model, d_hidden=None, norm_eps=1e-6):
        super(InceptionBlock, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        assert d_model % 4 == 0

        d_conv = d_model // 4
        self.dwconv_wh = nn.Conv2d(d_conv, d_conv, kernel_size=3, padding=1, groups=d_conv)
        self.dwconv_h = nn.Conv2d(d_conv, d_conv, kernel_size=(11, 1), padding=(5, 0), groups=d_conv)
        self.dwconv_w = nn.Conv2d(d_conv, d_conv, kernel_size=(1, 11), padding=(0, 5), groups=d_conv)

        self.norm = LayerNorm2d(d_model, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_hidden, d_model, kernel_size=1)
        )

    def forward(self, x):
        x_id, x_wh, x_h, x_w = x.chunk(4, 1)

        x = self.norm(torch.concatenate((
            x_id,
            self.dwconv_wh(x_wh),
            self.dwconv_h(x_h),
            self.dwconv_w(x_w)
        ), dim=1))

        return self.ffn(x)


class FiLMConvNeXtBlock(nn.Module):
    def __init__(self, d_model, d_t, d_hidden=None, norm_eps=1e-6):
        super(FiLMConvNeXtBlock, self).__init__()
        if d_hidden is None:
            d_hidden = 4 * d_model

        assert d_model % 4 == 0

        d_conv = d_model // 4
        self.dwconv_wh = nn.Conv2d(d_conv, d_conv, kernel_size=3, padding=1, groups=d_conv)
        self.dwconv_h = nn.Conv2d(d_conv, d_conv, kernel_size=(11, 1), padding=(5, 0), groups=d_conv)
        self.dwconv_w = nn.Conv2d(d_conv, d_conv, kernel_size=(1, 11), padding=(0, 5), groups=d_conv)

        self.norm = FiLM(d_model, d_t, eps=norm_eps)

        self.ffn = nn.Sequential(
            nn.Conv2d(d_model, d_hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(d_hidden, d_model, kernel_size=1)
        )

    def forward(self, x, t):
        x_id, x_wh, x_h, x_w = x.chunk(4, 1)

        x = self.norm(torch.concatenate((
            x_id,
            self.dwconv_wh(x_wh),
            self.dwconv_h(x_h),
            self.dwconv_w(x_w)
        ), dim=1), t)

        return self.ffn(x)


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


class FiLMCrossAttentionConvNeXtBlock(nn.Module):
    def __init__(self, d_model, d_t, d_cond, n_heads, norm_eps=1e-6):
        super(FiLMCrossAttentionConvNeXtBlock, self).__init__()
        self.cross_attn = CrossAttention2d(d_model, d_cond, n_heads)
        self.cross_attn_norm = FiLM(d_model, d_t, eps=norm_eps)

        self.ffn = FiLMConvNeXtBlock(d_model, d_t, norm_eps=norm_eps)

    def forward(self, x, t, cond, attention_mask=None):
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
