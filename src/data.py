import torch

from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from dataclasses import dataclass
from transformers import BatchEncoding


class ArtImageDataset(Dataset):
    def __init__(self, split="train", norm=False):
        self.ds = load_dataset("Artificio/WikiArt", split=split)

        if norm:
            mean = (
                0.5218818187713623,
                0.4685268700122833,
                0.40528634190559387
            )

            std = (
                0.27518653869628906,
                0.26829564571380615,
                0.26563310623168945
            )
        else:
            mean = (0.5, 0.5, 0.5)
            std = (0.5, 0.5, 0.5)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(
                (DataConfig.image_size, DataConfig.image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True
            ),
            transforms.Normalize(
                mean=mean, std=std
            )
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        data = self.ds[idx]

        return self.transform(data["image"])


class ArtConditionalDataset(ArtImageDataset):
    def __init__(self, tokenizer, split="train"):
        super().__init__(split=split, norm=True)
        self.tokenizer = tokenizer

    def collate(self, batch):
        image, text = zip(*batch)

        text_encoding = self.tokenizer(text, return_tensors="pt", padding=True)

        return BatchEncoding({
            "image": torch.stack(image),
            "tokens": text_encoding["input_ids"],
            "attention_mask": text_encoding["attention_mask"]
        })

    def __getitem__(self, idx):
        data = self.ds[idx]

        image = self.transform(data["image"])

        text = " / ".join([
            data["title"],
            data["style"],
            data["genre"]
        ])

        return image, text


@dataclass
class DataConfig:
    image_channels: int = 3
    image_size: int = 64
