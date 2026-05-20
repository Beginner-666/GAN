from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torchvision.models import Inception_V3_Weights, inception_v3
except ImportError:  # pragma: no cover - exercised only in minimal environments
    Inception_V3_Weights = None
    inception_v3 = None

try:
    from scipy import linalg
except ImportError:  # pragma: no cover - exercised only in minimal environments
    linalg = None


@torch.no_grad()
def collect_inception_outputs(
    images: Iterable[torch.Tensor],
    device: torch.device | str = "cuda",
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return Inception pool features and class probabilities for FID/IS.

    Input images can be normalized to [-1, 1] or [0, 1]. They are resized to
    299x299 and evaluated with ImageNet-pretrained Inception v3.
    """
    if Inception_V3_Weights is None or inception_v3 is None:
        raise ImportError("collect_inception_outputs requires torchvision. Install project dependencies from requirements-cu128.txt.")
    device = torch.device(device if torch.cuda.is_available() or str(device) == "cpu" else "cpu")
    weights = Inception_V3_Weights.DEFAULT
    model = inception_v3(weights=weights, transform_input=False).to(device)
    model.eval()

    features: list[np.ndarray] = []
    probs: list[np.ndarray] = []
    hook_values: list[torch.Tensor] = []

    def save_pool(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        hook_values.append(output.flatten(1).detach())

    handle = model.avgpool.register_forward_hook(save_pool)
    try:
        for batch in _iter_batches(images, batch_size):
            batch = _prepare_inception_batch(batch.to(device))
            logits = model(batch)
            prob = F.softmax(logits, dim=1)
            features.append(hook_values.pop().cpu().numpy())
            probs.append(prob.cpu().numpy())
    finally:
        handle.remove()

    return np.concatenate(features, axis=0), np.concatenate(probs, axis=0)


def calculate_fid(real_features: np.ndarray, fake_features: np.ndarray, eps: float = 1e-6) -> float:
    """Compute Fréchet Inception Distance from two feature matrices."""
    if linalg is None:
        raise ImportError("calculate_fid requires scipy. Install project dependencies from requirements-cu128.txt.")
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma_real.shape[0]) * eps
        covmean = linalg.sqrtm((sigma_real + offset) @ (sigma_fake + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    return float(fid)


def calculate_inception_score(probs: np.ndarray, splits: int = 10, eps: float = 1e-16) -> tuple[float, float]:
    """Compute mean/std Inception Score from class probabilities."""
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array shaped [num_images, num_classes]")
    scores = []
    split_count = min(splits, len(probs))
    for part in np.array_split(probs, split_count):
        py = np.mean(part, axis=0, keepdims=True)
        kl = part * (np.log(part + eps) - np.log(py + eps))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


def evaluate_fid_and_is(
    real_images: Iterable[torch.Tensor],
    fake_images: Iterable[torch.Tensor],
    device: torch.device | str = "cuda",
    batch_size: int | None = None,
) -> dict[str, float]:
    real_features, _ = collect_inception_outputs(real_images, device=device, batch_size=batch_size)
    fake_features, fake_probs = collect_inception_outputs(fake_images, device=device, batch_size=batch_size)
    is_mean, is_std = calculate_inception_score(fake_probs)
    return {
        "fid": calculate_fid(real_features, fake_features),
        "inception_score_mean": is_mean,
        "inception_score_std": is_std,
    }


def _prepare_inception_batch(batch: torch.Tensor) -> torch.Tensor:
    if batch.ndim != 4:
        raise ValueError("Expected image batch shaped [N, C, H, W]")
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    batch = batch.float()
    if batch.min() < 0:
        batch = (batch + 1.0) / 2.0
    batch = batch.clamp(0.0, 1.0)
    return F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)


def _iter_batches(images: Iterable[torch.Tensor], batch_size: int | None) -> Iterable[torch.Tensor]:
    for item in images:
        if item.ndim == 4:
            yield item
        elif item.ndim == 3:
            yield item.unsqueeze(0)
        else:
            raise ValueError("Each image item must be [C, H, W] or [N, C, H, W]")
