from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from dataclasses import dataclass


class ArtistDataset(Dataset):
    def __init__(self, split="train"):
        self.ds = load_dataset("Artificio/WikiArt", split=split)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (DataConfig.image_size, DataConfig.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
            )
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]

        return self.transform(data["image"])


@dataclass
class DataConfig:
    image_channels: int = 3
    image_size: int = 112
