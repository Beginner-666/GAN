import torch
from torch import nn

from .config import DCGANConfig


def discriminator_bce_loss(
    discriminator: nn.Module,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    criterion: nn.Module | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the discriminator BCE loss for one DCGAN step."""
    if criterion is None:
        criterion = nn.BCELoss()

    real_pred = discriminator(real_images)
    fake_pred = discriminator(fake_images.detach())
    real_targets = torch.ones_like(real_pred)
    fake_targets = torch.zeros_like(fake_pred)

    real_loss = criterion(real_pred, real_targets)
    fake_loss = criterion(fake_pred, fake_targets)
    loss = real_loss + fake_loss
    stats = {
        "d_loss": float(loss.detach().cpu()),
        "d_loss_real": float(real_loss.detach().cpu()),
        "d_loss_fake": float(fake_loss.detach().cpu()),
        "d_real_prob": float(real_pred.detach().mean().cpu()),
        "d_fake_prob": float(fake_pred.detach().mean().cpu()),
    }
    return loss, stats


def build_discriminator_optimizer(discriminator: nn.Module, config: DCGANConfig = DCGANConfig()) -> torch.optim.Optimizer:
    return torch.optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))


def train_discriminator_step(
    discriminator: nn.Module,
    optimizer: torch.optim.Optimizer,
    real_images: torch.Tensor,
    fake_images: torch.Tensor,
    device: torch.device | str,
    criterion: nn.Module | None = None,
) -> dict[str, float]:
    """Run one production discriminator update on the selected device."""
    discriminator.train()
    real_images = real_images.to(device, non_blocking=True)
    fake_images = fake_images.to(device, non_blocking=True)

    optimizer.zero_grad(set_to_none=True)
    loss, stats = discriminator_bce_loss(discriminator, real_images, fake_images, criterion)
    loss.backward()
    optimizer.step()
    return stats
