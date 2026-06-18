from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from .config import DCGANConfig

DatasetName = Literal["imagefolder", "lfw", "celeba"]


class FlatImageDataset(Dataset):
    def __init__(self, image_dir: str | Path, transform: transforms.Compose | None = None) -> None:
        self.image_dir = Path(image_dir)
        self.transform = transform
        patterns = ("*.jpg", "*.jpeg", "*.png")
        self.paths = sorted(path for pattern in patterns for path in self.image_dir.glob(pattern))
        if not self.paths:
            raise RuntimeError(f"No images found under {self.image_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, 0


def build_face_transforms(image_size: int = 64, train: bool = True) -> transforms.Compose:
    """DCGAN preprocessing: crop/resize RGB images and normalize to [-1, 1]."""
    ops: list[transforms.Transform] = [
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
    ]
    if train:
        ops.append(transforms.RandomHorizontalFlip(p=0.5))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return transforms.Compose(ops)


def load_face_dataset(
    root: str | Path,
    name: DatasetName = "imagefolder",
    image_size: int = 64,
    train: bool = True,
    download: bool = False,
) -> Dataset:
    """Load ImageFolder/LFW/CelebA with a shared preprocessing pipeline.

    For local custom data, use ImageFolder layout:
        root/class_name/*.jpg
    If all images are unlabeled, put them under one subdirectory, e.g. root/faces.
    """
    root = Path(root)
    transform = build_face_transforms(image_size=image_size, train=train)

    if name == "imagefolder":
        return datasets.ImageFolder(root=str(root), transform=transform)
    if name == "lfw":
        return datasets.LFWPeople(root=str(root), split="train", transform=transform, download=download)
    if name == "celeba":
        image_dir = root / "celeba" / "img_align_celeba"
        if image_dir.is_dir() and any(image_dir.glob("*.jpg")):
            return FlatImageDataset(image_dir, transform=transform)
        split = "train" if train else "test"
        return datasets.CelebA(root=str(root), split=split, target_type="attr", transform=transform, download=download)
    raise ValueError(f"Unsupported dataset: {name}")


def build_dataloader(
    dataset: Dataset,
    config: DCGANConfig = DCGANConfig(),
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    use_workers = config.num_workers > 0
    pin_memory = config.pin_memory and torch.cuda.is_available()
    kwargs = {}
    if use_workers:
        kwargs["persistent_workers"] = config.persistent_workers
        kwargs["prefetch_factor"] = config.prefetch_factor
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs,
    )


def split_dataset(dataset: Dataset, val_ratio: float = 0.1, seed: int = 42) -> tuple[Dataset, Dataset]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    val_size = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)
