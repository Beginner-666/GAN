from __future__ import annotations

import copy
import math

import torch
import torch.nn.functional as F
from torch import nn

from dcgan_a import DCGANConfig


class PixelNorm(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, lr_mul: float = 1.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.scale = (1.0 / math.sqrt(in_features)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.scale, self.bias * self.lr_mul)


class EqualizedConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))
        fan_in = in_channels * kernel_size * kernel_size
        self.scale = 1.0 / math.sqrt(fan_in)
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight * self.scale, self.bias, padding=self.padding)


class NoiseInjection(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, image: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn(image.size(0), 1, image.size(2), image.size(3), device=image.device, dtype=image.dtype)
        return image + self.weight * noise


class AdaIN(nn.Module):
    def __init__(self, channels: int, style_dim: int) -> None:
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        self.style = EqualizedLinear(style_dim, channels * 2)
        nn.init.zeros_(self.style.bias)
        with torch.no_grad():
            self.style.bias[:channels].fill_(1.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        style = self.style(style).view(style.size(0), 2, x.size(1), 1, 1)
        gamma, beta = style[:, 0], style[:, 1]
        return self.norm(x) * gamma + beta


class StyledConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, style_dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise = NoiseInjection(out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.adain = AdaIN(out_channels, style_dim)

    def forward(
        self,
        x: torch.Tensor,
        style: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.noise(x, noise=noise)
        x = self.activate(x)
        return self.adain(x, style)


class StyledConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, style_dim: int, upsample: bool) -> None:
        super().__init__()
        self.conv1 = StyledConv(in_channels, out_channels, style_dim, upsample=upsample)
        self.conv2 = StyledConv(out_channels, out_channels, style_dim, upsample=False)

    def forward(
        self,
        x: torch.Tensor,
        style1: torch.Tensor,
        style2: torch.Tensor | None = None,
        noise1: torch.Tensor | None = None,
        noise2: torch.Tensor | None = None,
    ) -> torch.Tensor:
        style2 = style1 if style2 is None else style2
        x = self.conv1(x, style1, noise=noise1)
        x = self.conv2(x, style2, noise=noise2)
        return x


class MappingNetwork(nn.Module):
    def __init__(self, noise_dim: int = 128, style_dim: int = 128, num_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [PixelNorm()]
        in_dim = noise_dim
        for _ in range(num_layers):
            layers.append(EqualizedLinear(in_dim, style_dim))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_dim = style_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 4:
            z = z.flatten(1)
        return self.net(z)


class StyleGANGenerator64(nn.Module):
    """StyleGAN-like generator for 64x64 RGB faces.

    Input accepts either [N, noise_dim] or [N, noise_dim, 1, 1]. Output is
    [N, 3, 64, 64] in [-1, 1], so it can reuse the existing artifact and metric
    utilities.
    """

    def __init__(
        self,
        config: DCGANConfig = DCGANConfig(),
        style_dim: int = 128,
        base_channels: int = 128,
        mapping_layers: int = 4,
    ) -> None:
        super().__init__()
        self.noise_dim = config.noise_dim
        self.style_dim = style_dim
        self.mapping = MappingNetwork(config.noise_dim, style_dim, mapping_layers)
        self.constant = nn.Parameter(torch.randn(1, base_channels, 4, 4))
        self.blocks = nn.ModuleList(
            [
                StyledConvBlock(base_channels, base_channels, style_dim, upsample=False),
                StyledConvBlock(base_channels, base_channels, style_dim, upsample=True),
                StyledConvBlock(base_channels, base_channels // 2, style_dim, upsample=True),
                StyledConvBlock(base_channels // 2, base_channels // 4, style_dim, upsample=True),
                StyledConvBlock(base_channels // 4, base_channels // 8, style_dim, upsample=True),
            ]
        )
        self.num_style_layers = len(self.blocks) * 2
        self.to_rgb = nn.Sequential(
            EqualizedConv2d(base_channels // 8, config.channels, kernel_size=1),
            nn.Tanh(),
        )

    def get_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)

    def mean_latent(self, n_latent: int = 4096, device: torch.device | None = None) -> torch.Tensor:
        device = device or self.constant.device
        noise = torch.randn(n_latent, self.noise_dim, device=device)
        return self.get_latent(noise).mean(dim=0, keepdim=True)

    def make_noise(self, batch_size: int, device: torch.device | None = None) -> list[torch.Tensor]:
        device = device or self.constant.device
        noises: list[torch.Tensor] = []
        for stage_index in range(len(self.blocks)):
            resolution = 4 * (2**stage_index)
            noises.append(torch.randn(batch_size, 1, resolution, resolution, device=device))
            noises.append(torch.randn(batch_size, 1, resolution, resolution, device=device))
        return noises

    def truncation(
        self,
        styles: torch.Tensor,
        truncation_psi: float = 1.0,
        truncation_cutoff: int | None = None,
        mean_latent: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if truncation_psi == 1.0:
            return styles
        mean_latent = self.mean_latent(device=styles.device) if mean_latent is None else mean_latent
        if styles.ndim == 2:
            return mean_latent + (styles - mean_latent) * truncation_psi
        if styles.ndim != 3:
            raise ValueError("Expected styles shaped [N, style_dim] or [N, num_layers, style_dim]")
        cutoff = self.num_style_layers if truncation_cutoff is None else max(0, min(truncation_cutoff, self.num_style_layers))
        truncated = styles.clone()
        if cutoff > 0:
            truncated[:, :cutoff] = mean_latent.unsqueeze(1) + (truncated[:, :cutoff] - mean_latent.unsqueeze(1)) * truncation_psi
        return truncated

    def _style_stack(
        self,
        style: torch.Tensor,
        mixing_style: torch.Tensor | None = None,
        style_mixing_prob: float = 0.0,
    ) -> torch.Tensor:
        styles = style.unsqueeze(1).expand(-1, self.num_style_layers, -1).contiguous()
        if mixing_style is None or style_mixing_prob <= 0.0:
            return styles
        if style.size(0) <= 1:
            return styles
        if torch.rand((), device=style.device).item() >= style_mixing_prob:
            return styles
        mixing_stack = mixing_style.unsqueeze(1).expand(-1, self.num_style_layers, -1)
        cutoff = int(torch.randint(1, self.num_style_layers, (1,), device=style.device).item())
        styles = styles.clone()
        styles[:, cutoff:] = mixing_stack[:, cutoff:]
        return styles

    def forward_from_styles(
        self,
        styles: torch.Tensor | list[torch.Tensor],
        noise: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if isinstance(styles, list):
            styles = torch.stack(styles, dim=1)
        if styles.ndim == 2:
            styles = styles.unsqueeze(1).expand(-1, self.num_style_layers, -1).contiguous()
        if styles.ndim != 3 or styles.size(1) != self.num_style_layers:
            raise ValueError(f"Expected styles shaped [N, {self.num_style_layers}, style_dim]")

        if noise is not None and len(noise) != self.num_style_layers:
            raise ValueError(f"Expected {self.num_style_layers} noise tensors or None")

        x = self.constant.repeat(styles.size(0), 1, 1, 1)
        style_index = 0
        for block_index, block in enumerate(self.blocks):
            noise1 = None if noise is None else noise[style_index]
            noise2 = None if noise is None else noise[style_index + 1]
            x = block(x, styles[:, style_index], styles[:, style_index + 1], noise1=noise1, noise2=noise2)
            style_index += 2
        return self.to_rgb(x)

    def forward_from_style(
        self,
        style: torch.Tensor,
        noise: list[torch.Tensor] | None = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: int | None = None,
    ) -> torch.Tensor:
        if style.ndim == 2:
            styles = style.unsqueeze(1).expand(-1, self.num_style_layers, -1).contiguous()
        elif style.ndim == 3:
            styles = style
        else:
            raise ValueError("Expected style shaped [N, style_dim] or [N, num_layers, style_dim]")
        styles = self.truncation(styles, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        return self.forward_from_styles(styles, noise=noise)

    def forward(
        self,
        z: torch.Tensor,
        return_style: bool = False,
        return_styles: bool = False,
        mixing_z: torch.Tensor | None = None,
        style_mixing_prob: float = 0.0,
        truncation_psi: float = 1.0,
        truncation_cutoff: int | None = None,
        noise: list[torch.Tensor] | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        style = self.get_latent(z)
        styles = self._style_stack(style, mixing_style=None if mixing_z is None else self.get_latent(mixing_z), style_mixing_prob=style_mixing_prob)
        styles = self.truncation(styles, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
        image = self.forward_from_styles(styles, noise=noise)
        if return_styles:
            return image, styles
        if return_style:
            return image, style
        return image


class MinibatchStdDev(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 1:
            std = torch.zeros(1, 1, 1, 1, device=x.device, dtype=x.dtype)
        else:
            std = x.float().std(dim=0, unbiased=False).mean().view(1, 1, 1, 1)
        std = std.to(dtype=x.dtype, device=x.device)
        std_map = std.expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std_map], dim=1)


class StyleGANDiscriminator64(nn.Module):
    """64x64 discriminator that returns logits, not probabilities."""

    def __init__(self, config: DCGANConfig = DCGANConfig(), base_channels: int = 64) -> None:
        super().__init__()
        c = config.channels
        ndf = base_channels
        self.net = nn.Sequential(
            nn.Conv2d(c, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            MinibatchStdDev(),
            nn.Conv2d(ndf * 8 + 1, ndf * 8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).view(-1)


def copy_stylegan_for_ema(generator: nn.Module) -> nn.Module:
    ema_generator = copy.deepcopy(generator)
    for parameter in ema_generator.parameters():
        parameter.requires_grad_(False)
    return ema_generator


@torch.no_grad()
def update_ema(target: nn.Module, source: nn.Module, decay: float) -> None:
    target_params = dict(target.named_parameters())
    source_params = dict(source.named_parameters())
    for name, parameter in target_params.items():
        parameter.data.mul_(decay).add_(source_params[name].data, alpha=1.0 - decay)
    target_buffers = dict(target.named_buffers())
    source_buffers = dict(source.named_buffers())
    for name, buffer in target_buffers.items():
        if name in source_buffers:
            buffer.data.copy_(source_buffers[name].data)


def build_stylegan_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    lr: float = 2e-4,
    beta1: float = 0.0,
    beta2: float = 0.99,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer_g, optimizer_d
