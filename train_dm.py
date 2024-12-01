import torch

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
from accelerate import Accelerator

from src.data import ArtConditionalDataset, DataConfig
from src.model.diffusion import DiffusionModel, DiffusionConfig
from src.model.vae import Encoder, VAEConfig


def train(model, dataloader):
    num_epochs = 5

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    lr_scheduler = get_cosine_schedule_with_warmup(
        opt,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(dataloader)
    )

    accelerator = Accelerator()
    model, dataloader, opt, lr_scheduler = accelerator.prepare(
        model, dataloader, opt, lr_scheduler
    )

    model.train()
    for epoch in range(num_epochs):
        print(f"EPOCH {epoch + 1} / {num_epochs}")
        for i, encoding in enumerate(dataloader):
            opt.zero_grad()

            loss = model(**encoding)
            accelerator.backward(loss)

            opt.step()
            lr_scheduler.step()

            if i % 25 == 0:
                print(f"\t{i} / {len(dataloader)} iters.\tLoss: {loss.item():.6f}")

        torch.save(
            accelerator.get_state_dict(model),
            f"dm_ckpts/checkpoint_{epoch + 1:02}.pt"
        )


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dataset = ArtConditionalDataset(tokenizer, p_uncond=DataConfig.p_uncond)

    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    image_encoder = Encoder(
        in_channels=DataConfig.image_channels,
        d_latent=VAEConfig.d_latent,
        d_init=VAEConfig.d_init,
        n_heads=VAEConfig.n_heads,
        n_scales=VAEConfig.n_scales
    )
    # TODO. Load ckpt

    model = DiffusionModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        scheduler=DDPMScheduler(),
        in_channels=VAEConfig.d_latent,
        d_init=DiffusionConfig.d_init,
        d_t=DiffusionConfig.d_t,
        n_heads=DiffusionConfig.n_heads,
        n_scales=DiffusionConfig.n_scales,
        n_cross_attn_scales=DiffusionConfig.n_cross_attn_scales
    )

    dataloader = DataLoader(
        dataset,
        batch_size=24,
        shuffle=True,
        pin_memory=True,
        collate_fn=dataset.collate
    )

    train(model, dataloader)
