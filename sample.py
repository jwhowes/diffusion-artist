import torch

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler

from src.model.diffusion import DiffusionModel, DDPMWrapper, DiffusionConfig
from src.data import DataConfig


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    diffusion_model = DiffusionModel(
        text_encoder=text_encoder,
        scheduler=DDPMScheduler(),
        in_channels=DataConfig.image_channels,
        d_init=DiffusionConfig.d_init,
        d_t=DiffusionConfig.d_t,
        n_heads=DiffusionConfig.n_heads,
        n_scales=DiffusionConfig.n_scales,
        n_cross_attn_scales=DiffusionConfig.n_cross_attn_scales
    )

    ckpt = torch.load("dm_ckpts/checkpoint_01.pt", weights_only=True)
    diffusion_model.load_state_dict(ckpt)
    del ckpt

    model = DDPMWrapper(
        diffusion_model, tokenizer,
        in_channels=DataConfig.image_channels,
        image_size=DataConfig.image_size,
        mean=DataConfig.mean,
        std=DataConfig.std
    )

    pred = model.sample(
        title="Cornelia Street",
        style="New Realism",
        genre="cityscape"
    )

    import matplotlib.pyplot as plt
    plt.imshow(pred.permute(1, 2, 0))
    plt.savefig("pred.png")
