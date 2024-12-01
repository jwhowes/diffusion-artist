import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from accelerate import Accelerator
from transformers import get_cosine_schedule_with_warmup

from src.model.vae import Encoder, Decoder, Discriminator, VAEConfig, DiscriminatorConfig
from src.data import ArtImageDataset, DataConfig


def train(encoder, decoder, discriminator, dataloader, kl_weight=0.01, adv_weight=0.1):
    num_epochs = 5

    opt = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=5e-5)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=5e-5)

    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )
    disc_lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    encoder, decoder, discriminator, dataloader, opt, disc_opt, lr_scheduler, disc_lr_scheduler = accelerator.prepare(
        encoder, decoder, discriminator, dataloader, opt, disc_opt, lr_scheduler, disc_lr_scheduler
    )

    encoder.train()
    decoder.train()
    discriminator.train()
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, image in enumerate(dataloader):
            opt.zero_grad()

            dist = encoder(image)
            z = dist.sample()
            recon = decoder(z)

            fake_pred = discriminator(recon)

            recon_loss = F.mse_loss(recon, image)
            kl_loss = dist.kl.mean()
            adv_loss = F.binary_cross_entropy_with_logits(fake_pred, torch.ones_like(fake_pred))

            loss = recon_loss + kl_weight * kl_loss + adv_weight * adv_loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)

            opt.step()
            lr_scheduler.step()

            disc_opt.zero_grad()
            real_pred = discriminator(image)
            fake_pred = discriminator(recon.detach())

            loss = (
                    F.binary_cross_entropy_with_logits(real_pred, torch.ones_like(real_pred)) +
                    F.binary_cross_entropy_with_logits(fake_pred, torch.zeros_like(fake_pred))
            )

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(discriminator.parameters(), 1.0)

            disc_opt.step()
            disc_lr_scheduler.step()

            if i % 25 == 0:
                print(
                    f"\t{i} / {len(dataloader)} iters.\t"
                    f"Recon Loss: {recon_loss.item():.4f}\t"
                    f"KL Loss: {kl_loss.item():.4f}\t"
                    f"Adv Loss: {adv_loss.item():.4f}"
                )

        torch.save(
            {
                "encoder": accelerator.get_state_dict(encoder),
                "decoder": accelerator.get_state_dict(decoder),
                "discriminator": accelerator.get_state_dict(discriminator)
            },
            f"vae_ckpts/checkpoint_{epoch + 1:02}.pt"
        )


if __name__ == "__main__":
    dataset = ArtImageDataset()

    encoder = Encoder(
        in_channels=DataConfig.image_channels,
        d_latent=VAEConfig.d_latent,
        d_init=VAEConfig.d_init,
        n_heads=VAEConfig.n_heads,
        n_scales=VAEConfig.n_scales
    )

    decoder = Decoder(
        in_channels=DataConfig.image_channels,
        d_latent=VAEConfig.d_latent,
        d_init=VAEConfig.d_init,
        n_heads=VAEConfig.n_heads,
        n_scales=VAEConfig.n_scales
    )

    discriminator = Discriminator(
        in_channels=DataConfig.image_channels,
        d_init=DiscriminatorConfig.d_init,
        n_heads=DiscriminatorConfig.n_heads,
        patch_size=DiscriminatorConfig.patch_size,
        n_scales=DiscriminatorConfig.n_scales
    )

    dataloader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        pin_memory=True
    )

    train(encoder, decoder, discriminator, dataloader, kl_weight=VAEConfig.kl_weight, adv_weight=VAEConfig.adv_weight)
