import torch

from torch import nn
from dataclasses import dataclass
from einops import rearrange

from .util import InceptionBlock


@dataclass
class DiagonalGaussian:
    mean: torch.FloatTensor
    log_var: torch.FloatTensor

    def sample(self):
        return torch.randn_like(self.mean) * (0.5 * self.log_var).exp() + self.mean

    @property
    def kl(self):
        return 0.5 * (
            self.mean.pow(2) + self.log_var.exp() - 1.0 - self.log_var
        ).mean((1, 2, 3))


@dataclass
class VAEConfig:
    d_latent: int = 4
    d_init: int = 128
    n_heads: int = 4
    n_scales: int = 3


@dataclass
class DiscriminatorConfig:
    d_model: int = 256
    n_heads: int = 4
    patch_size: int = 28
    n_blocks: int = 2


class Decoder(nn.Module):
    def __init__(self, in_channels, d_latent, d_init, n_heads, n_scales=4):
        super(Decoder, self).__init__()
        self.stem = nn.Conv2d(d_latent, d_init * (2 ** (n_scales - 1)), kernel_size=7, padding=3)

        self.mid_block = InceptionBlock(d_init * (2 ** (n_scales - 1)), n_heads)

        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for scale in range(n_scales - 2, -1, -1):
            self.up_samples.append(nn.ConvTranspose2d(
                d_init * (2 ** (scale + 1)),
                d_init * (2 ** scale),
                kernel_size=2,
                stride=2)
            )
            self.up_blocks.append(InceptionBlock(d_init * (2 ** scale), n_heads))

        self.head = nn.Conv2d(d_init, in_channels, kernel_size=1)

    def forward(self, z):
        z = self.mid_block(self.stem(z))

        for up_block, up_sample in zip(self.up_blocks, self.up_samples):
            z = up_sample(z)
            z = up_block(z)

        return self.head(z)


class Encoder(nn.Module):
    def __init__(self, in_channels, d_latent, d_init, n_heads, n_scales=4):
        super(Encoder, self).__init__()
        self.stem = nn.Conv2d(in_channels, d_init, kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for scale in range(n_scales - 1):
            self.down_blocks.append(
                InceptionBlock(d_init * (2 ** scale), n_heads)
            )
            self.down_samples.append(
                nn.Conv2d(d_init * (2 ** scale), 2 * d_init * (2 ** scale), kernel_size=2, stride=2)
            )

        self.mid_block = InceptionBlock(
            d_init * (2 ** (n_scales - 1)), n_heads
        )

        self.head = nn.Conv2d(d_init * (2 ** (n_scales - 1)), 2 * d_latent, kernel_size=1)

    def forward(self, x):
        x = self.stem(x)

        for down_block, down_sample in zip(self.down_blocks, self.down_samples):
            x = down_block(x)
            x = down_sample(x)

        x = self.mid_block(x)

        mean, log_var = self.head(x).chunk(2, 1)

        return DiagonalGaussian(mean, log_var)


class Discriminator(nn.Module):
    def __init__(self, in_channels, d_model, n_heads, patch_size=14, n_blocks=2):
        super(Discriminator, self).__init__()
        self.patch_size = patch_size

        self.stem = nn.Conv2d(in_channels, d_model, kernel_size=7, padding=3)

        blocks = [
            InceptionBlock(d_model, n_heads) for _ in range(n_blocks)
        ]
        self.blocks = nn.Sequential(*blocks)

        self.pooler = nn.AdaptiveAvgPool2d((1, 1))

        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, 1)
        )

    def forward(self, x):
        x = rearrange(x, "b c (h p1) (w p2) -> (b h w) c p1 p2", p1=self.patch_size, p2=self.patch_size)

        x = self.stem(x)
        x = self.pooler(self.blocks(x)).squeeze()

        return self.ffn(x).squeeze()
