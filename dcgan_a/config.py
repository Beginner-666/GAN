from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DCGANConfig:
    noise_dim: int = 128
    image_size: int = 64
    channels: int = 3
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 50
    lr: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    feature_maps_d: int = 64
    device: str = "cuda"
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50


def resolve_device(preferred: str = "cuda") -> torch.device:
    if preferred == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(preferred)
