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


class StyledConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, style_dim: int, upsample: bool) -> None:
        super().__init__()
        self.upsample = upsample
        self.conv = EqualizedConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.noise = NoiseInjection(out_channels)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.adain = AdaIN(out_channels, style_dim)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.conv(x)
        x = self.noise(x)
        x = self.activate(x)
        return self.adain(x, style)


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
    """Lightweight StyleGAN-like generator for 64x64 RGB faces.

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
        self.to_rgb = nn.Sequential(
            EqualizedConv2d(base_channels // 8, config.channels, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, return_style: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        style = self.mapping(z)
        image = self.forward_from_style(style)
        if return_style:
            return image, style
        return image

    def forward_from_style(self, style: torch.Tensor) -> torch.Tensor:
        x = self.constant.repeat(style.size(0), 1, 1, 1)
        for block in self.blocks:
            x = block(x, style)
        return self.to_rgb(x)


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
