import torch

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler

from src.model.diffusion import DiffusionModel, LatentDiffusionWrapper, DiffusionConfig
from src.model.vae import Encoder, Decoder, VAEConfig
from src.data import DataConfig


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

    image_decoder = Decoder(
        in_channels=DataConfig.image_channels,
        d_latent=VAEConfig.d_latent,
        d_init=VAEConfig.d_init,
        n_heads=VAEConfig.n_heads,
        n_scales=VAEConfig.n_scales
    )
    vae_ckpt = torch.load(
        DiffusionConfig.vae_path, weights_only=True, map_location=torch.device("cpu")
    )
    image_decoder.load_state_dict(vae_ckpt["decoder"])
    del vae_ckpt

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

    dm_ckpt = torch.load("dm_ckpts/checkpoint_01.pt", weights_only=True)
    diffusion_model.load_state_dict(dm_ckpt)
    del dm_ckpt

    model = LatentDiffusionWrapper(
        diffusion_model=diffusion_model,
        tokenizer=tokenizer,
        image_decoder=image_decoder,
        in_channels=DataConfig.image_channels,
        latent_size=DataConfig.image_size // (2 ** (VAEConfig.n_scales - 1))
    )

    pred = model.sample(
        title="Cornelia Street",
        style="New Realism",
        genre="cityscape"
    )

    import matplotlib.pyplot as plt
    plt.imshow(pred.permute(1, 2, 0))
    plt.savefig("pred.png")
