import torch
import torch.nn.functional as F

from torch import nn
from dataclasses import dataclass
from tqdm import tqdm

from .util import FiLMConvNeXtBlock, FiLMCrossAttentionConvNeXtBlock, SinusoidalEmbedding


class LatentDiffusionWrapper(nn.Module):
    def __init__(
            self, diffusion_model, tokenizer, image_decoder,
            in_channels=3, latent_size=28
    ):
        super(LatentDiffusionWrapper, self).__init__()
        self.latent_size = latent_size
        self.in_channels = in_channels

        self.image_decoder = image_decoder
        self.image_decoder.eval()
        self.image_decoder.requires_grad_(False)

        self.diffusion_model = diffusion_model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        self.tokenizer = tokenizer

    @torch.inference_mode()
    def sample(self, title, style, genre, num_steps=200, guidance_scale=6.0, device=torch.device("cpu")):
        text = " / ".join([title, style, genre])
        if guidance_scale > 1.0:
            text = ["", text]
        
        text_encoding = self.tokenizer(text, return_tensors="pt", padding=True).to(device)

        latent = torch.randn(1, self.in_channels, self.latent_size, self.latent_size).to(device)
        self.diffusion_model.scheduler.set_timesteps(num_steps)

        cond, mask = self.diffusion_model.encode_cond(**text_encoding)
        for t in tqdm(self.diffusion_model.scheduler.timesteps):
            if guidance_scale > 1.0:
                pred = self.diffusion_model.pred_noise(
                    torch.concat([latent] * 2),
                    t.view(1).repeat(2),
                    cond,
                    mask
                )

                pred_uncond, pred_cond = pred.chunk(2)
                pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred_noise = self.diffusion_model.pred_noise(
                    latent,
                    t.view(1),
                    cond,
                    mask
                )

            latent = self.diffusion_model.scheduler.step(pred_noise, t, latent).prev_sample

        latent = latent.squeeze() / self.diffusion_model.latent_scale

        return ((self.image_decoder(latent) + 1) / 2).clamp(0.0, 1.0)


class ConditionalFiLMUNet(nn.Module):
    def __init__(
            self, in_channels, d_init, d_t, d_cond, n_heads,
            n_scales=6, n_cross_attn_scales=4
    ):
        super(ConditionalFiLMUNet, self).__init__()
        assert n_scales >= n_cross_attn_scales > 0
        self.n_scales = n_scales
        self.self_attn_scales = n_scales - n_cross_attn_scales

        self.t_model = nn.Sequential(
            SinusoidalEmbedding(d_t),
            nn.Linear(d_t, 4 * d_t),
            nn.GELU(),
            nn.Linear(4 * d_t, d_t)
        )

        self.stem = nn.Conv2d(in_channels, d_init, kernel_size=7, padding=3)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for scale in range(n_scales - 1):
            if scale < n_scales - n_cross_attn_scales:
                self.down_blocks.append(
                    FiLMConvNeXtBlock(d_init * (2 ** scale), d_t, n_heads)
                )
            else:
                self.down_blocks.append(
                    FiLMCrossAttentionConvNeXtBlock(d_init * (2 ** scale), d_t, d_cond, n_heads)
                )
            self.down_samples.append(
                nn.Conv2d(d_init * (2 ** scale), 2 * d_init * (2 ** scale), kernel_size=2, stride=2)
            )

        self.mid_block = FiLMCrossAttentionConvNeXtBlock(
            d_init * (2 ** (n_scales - 1)), d_t, d_cond, n_heads
        )

        self.up_samples = nn.ModuleList()
        self.up_combines = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for scale in range(n_scales - 2, -1, -1):
            self.up_samples.append(
                nn.ConvTranspose2d(
                    2 * d_init * (2 ** scale),
                    d_init * (2 ** scale),
                    kernel_size=2,
                    stride=2
                )
            )
            self.up_combines.append(
                nn.Conv2d(
                    2 * d_init * (2 ** scale),
                    d_init * (2 ** scale),
                    kernel_size=5,
                    padding=2
                )
            )

            if scale < n_scales - n_cross_attn_scales:
                self.up_blocks.append(FiLMConvNeXtBlock(d_init * (2 ** scale), d_t, n_heads))
            else:
                self.up_blocks.append(
                    FiLMCrossAttentionConvNeXtBlock(d_init * (2 ** scale), d_t, d_cond, n_heads)
                )

        self.head = nn.Conv2d(d_init, in_channels, kernel_size=1)

    def forward(self, x, t, cond, attention_mask=None):
        t_emb = self.t_model(t)

        x = self.stem(x)

        down_acts = []
        for scale, (down_block, down_sample) in enumerate(zip(self.down_blocks, self.down_samples)):
            if scale < self.self_attn_scales:
                x = down_block(x, t_emb)
            else:
                x = down_block(x, t_emb, cond, attention_mask)

            down_acts.append(x)
            x = down_sample(x)

        x = self.mid_block(x, t_emb, cond, attention_mask)

        for i, (up_block, up_sample, up_combine, act) in enumerate(zip(
            self.up_blocks, self.up_samples, self.up_combines, down_acts[::-1]
        )):
            x = up_sample(x)
            x = up_combine(
                torch.concatenate((
                    x, act
                ), dim=1)
            )
            if self.n_scales - i - 2 < self.self_attn_scales:
                x = up_block(x, t_emb)
            else:
                x = up_block(x, t_emb, cond, attention_mask)

        return self.head(x)


class DiffusionModel(nn.Module):
    def __init__(
            self, text_encoder, scheduler, in_channels, d_init, d_t, n_heads,
            image_encoder=None, n_scales=5, n_cross_attn_scales=3, latent_scale=1.0
    ):
        super(DiffusionModel, self).__init__()
        self.latent_scale = latent_scale

        self.scheduler = scheduler
        self.T = self.scheduler.config.num_train_timesteps

        self.image_encoder = image_encoder
        if self.image_encoder is not None:
            self.image_encoder.eval()
            self.image_encoder.requires_grad_(False)

        self.text_encoder = text_encoder
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

        self.unet = ConditionalFiLMUNet(
            in_channels=in_channels,
            d_init=d_init,
            d_t=d_t,
            d_cond=self.text_encoder.config.hidden_size,
            n_heads=n_heads,
            n_scales=n_scales,
            n_cross_attn_scales=n_cross_attn_scales
        )

    @torch.no_grad()
    def encode_cond(self, input_ids, attention_mask):
        cond = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        attention_mask = torch.zeros(attention_mask.shape, device=attention_mask.device).masked_fill(
            ~attention_mask.to(torch.bool), float('-inf')
        )

        return cond, attention_mask

    def pred_noise(self, x, t, cond, attention_mask=None):
        return self.unet(x, t, cond, attention_mask)

    def forward(self, image, input_ids, attention_mask):
        B = image.shape[0]
        cond, attention_mask = self.encode_cond(input_ids, attention_mask)

        z = self.image_encoder(image).sample() * self.latent_scale

        t = torch.randint(0, self.T, (B,), device=image.device)
        noise = torch.randn_like(z)
        noisy_z = self.scheduler.add_noise(z, noise, t)

        pred_noise = self.pred_noise(noisy_z, t, cond, attention_mask)

        return F.mse_loss(pred_noise, noise)


@dataclass
class DiffusionConfig:
    d_init: int = 128
    d_t: int = 256
    n_heads: int = 8
    n_scales: int = 3
    n_cross_attn_scales: int = 3
    vae_path: str = "vae_ckpts/checkpoint_01.pt"
    latent_scale: float = 1.0  # TODO
