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


class ConstantInput(nn.Module):
    def __init__(self, channels: int, size: int = 4) -> None:
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channels, size, size))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.input.repeat(batch_size, 1, 1, 1)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        style_dim: int,
        upsample: bool = False,
        demodulate: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        self.demodulate = demodulate
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channels, in_channels, kernel_size, kernel_size))
        self.scale = 1.0 / math.sqrt(in_channels * kernel_size * kernel_size)
        self.modulation = EqualizedLinear(style_dim, in_channels)
        with torch.no_grad():
            self.modulation.bias.fill_(1.0)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        style = self.modulation(style).view(batch_size, 1, in_channels, 1, 1)
        weight = self.weight * self.scale * style
        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum(dim=(2, 3, 4)) + 1e-8)
            weight = weight * demod.view(batch_size, self.out_channels, 1, 1, 1)

        weight = weight.view(batch_size * self.out_channels, in_channels, self.kernel_size, self.kernel_size)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            height, width = x.shape[2:]
        x = x.view(1, batch_size * in_channels, height, width)
        out = F.conv2d(x, weight, padding=self.padding, groups=batch_size)
        return out.view(batch_size, self.out_channels, out.shape[2], out.shape[3])


class StyledConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, style_dim: int, upsample: bool = False) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 3, style_dim, upsample=upsample)
        self.noise = NoiseInjection(out_channels)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.activate = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor, style: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        x = self.conv(x, style)
        x = self.noise(x, noise=noise)
        x = x + self.bias
        return self.activate(x)


class ToRGB(nn.Module):
    def __init__(self, in_channels: int, style_dim: int, out_channels: int = 3) -> None:
        super().__init__()
        self.conv = ModulatedConv2d(in_channels, out_channels, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, x: torch.Tensor, style: torch.Tensor, skip: torch.Tensor | None = None) -> torch.Tensor:
        out = self.conv(x, style) + self.bias
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2, mode="bilinear", align_corners=False)
            out = out + skip
        return out


class StyleGAN2Generator64(nn.Module):
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
        self.channels = config.channels
        self.mapping = MappingNetwork(config.noise_dim, style_dim, mapping_layers)
        self.input = ConstantInput(base_channels)
        self.conv1 = StyledConv(base_channels, base_channels, style_dim, upsample=False)
        self.to_rgb1 = ToRGB(base_channels, style_dim, out_channels=config.channels)

        channel_plan = [base_channels, base_channels // 2, base_channels // 4, base_channels // 8]
        in_channels = base_channels
        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        for out_channels in channel_plan:
            self.convs.append(StyledConv(in_channels, out_channels, style_dim, upsample=True))
            self.convs.append(StyledConv(out_channels, out_channels, style_dim, upsample=False))
            self.to_rgbs.append(ToRGB(out_channels, style_dim, out_channels=config.channels))
            in_channels = out_channels

        self.num_style_layers = 1 + len(self.convs) + 1 + len(self.to_rgbs)
        self.num_noise_layers = 1 + len(self.convs)

    def get_latent(self, z: torch.Tensor) -> torch.Tensor:
        return self.mapping(z)

    def mean_latent(self, n_latent: int = 4096, device: torch.device | None = None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        z = torch.randn(n_latent, self.noise_dim, device=device)
        return self.get_latent(z).mean(dim=0, keepdim=True)

    def make_noise(self, batch_size: int, device: torch.device | None = None) -> list[torch.Tensor]:
        device = device or next(self.parameters()).device
        resolutions = [4, 8, 8, 16, 16, 32, 32, 64, 64]
        return [torch.randn(batch_size, 1, resolution, resolution, device=device) for resolution in resolutions]

    def _style_stack(
        self,
        style: torch.Tensor,
        mixing_style: torch.Tensor | None = None,
        style_mixing_prob: float = 0.0,
    ) -> torch.Tensor:
        styles = style.unsqueeze(1).expand(-1, self.num_style_layers, -1).contiguous()
        if mixing_style is None or style_mixing_prob <= 0.0 or style.size(0) <= 1:
            return styles
        if torch.rand((), device=style.device).item() >= style_mixing_prob:
            return styles
        mixing_stack = mixing_style.unsqueeze(1).expand(-1, self.num_style_layers, -1)
        cutoff = int(torch.randint(1, self.num_style_layers, (1,), device=style.device).item())
        styles = styles.clone()
        styles[:, cutoff:] = mixing_stack[:, cutoff:]
        return styles

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
        cutoff = self.num_style_layers if truncation_cutoff is None else max(0, min(truncation_cutoff, self.num_style_layers))
        truncated = styles.clone()
        if cutoff > 0:
            truncated[:, :cutoff] = mean_latent.unsqueeze(1) + (truncated[:, :cutoff] - mean_latent.unsqueeze(1)) * truncation_psi
        return truncated

    def forward_from_styles(self, styles: torch.Tensor | list[torch.Tensor], noise: list[torch.Tensor] | None = None) -> torch.Tensor:
        if isinstance(styles, list):
            styles = torch.stack(styles, dim=1)
        if styles.ndim == 2:
            styles = styles.unsqueeze(1).expand(-1, self.num_style_layers, -1).contiguous()
        if styles.ndim != 3 or styles.size(1) != self.num_style_layers:
            raise ValueError(f"Expected styles shaped [N, {self.num_style_layers}, style_dim]")
        if noise is not None and len(noise) != self.num_noise_layers:
            raise ValueError(f"Expected {self.num_noise_layers} noise tensors or None")

        batch_size = styles.size(0)
        noise_index = 0
        style_index = 0
        x = self.input(batch_size)
        x = self.conv1(x, styles[:, style_index], None if noise is None else noise[noise_index])
        noise_index += 1
        style_index += 1
        skip = self.to_rgb1(x, styles[:, style_index])
        style_index += 1

        for conv_index in range(0, len(self.convs), 2):
            x = self.convs[conv_index](x, styles[:, style_index], None if noise is None else noise[noise_index])
            noise_index += 1
            style_index += 1
            x = self.convs[conv_index + 1](x, styles[:, style_index], None if noise is None else noise[noise_index])
            noise_index += 1
            style_index += 1
            skip = self.to_rgbs[conv_index // 2](x, styles[:, style_index], skip)
            style_index += 1

        return torch.tanh(skip)

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
        mixing_style = None if mixing_z is None else self.get_latent(mixing_z)
        styles = self._style_stack(style, mixing_style=mixing_style, style_mixing_prob=style_mixing_prob)
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


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = EqualizedConv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = EqualizedConv2d(in_channels, out_channels, 3, padding=1)
        self.skip = EqualizedConv2d(in_channels, out_channels, 1)
        self.activate = nn.LeakyReLU(0.2, inplace=True)
        self.scale = 1 / math.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.avg_pool2d(self.skip(x), 2)
        x = self.activate(self.conv1(x))
        x = self.activate(self.conv2(x))
        x = F.avg_pool2d(x, 2)
        return (x + residual) * self.scale


class StyleGAN2Discriminator64(nn.Module):
    def __init__(self, config: DCGANConfig = DCGANConfig(), base_channels: int = 64) -> None:
        super().__init__()
        ndf = base_channels
        self.from_rgb = EqualizedConv2d(config.channels, ndf, 1)
        self.blocks = nn.Sequential(
            DiscriminatorBlock(ndf, ndf * 2),
            DiscriminatorBlock(ndf * 2, ndf * 4),
            DiscriminatorBlock(ndf * 4, ndf * 8),
            DiscriminatorBlock(ndf * 8, ndf * 8),
        )
        self.final = nn.Sequential(
            MinibatchStdDev(),
            EqualizedConv2d(ndf * 8 + 1, ndf * 8, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            EqualizedLinear(ndf * 8 * 4 * 4, ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedLinear(ndf * 8, 1),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.from_rgb(images), 0.2, inplace=True)
        x = self.blocks(x)
        return self.final(x).view(-1)


def copy_stylegan2_for_ema(generator: nn.Module) -> nn.Module:
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


def build_stylegan2_optimizers(
    generator: nn.Module,
    discriminator: nn.Module,
    lr: float = 2e-4,
    beta1: float = 0.0,
    beta2: float = 0.99,
) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, beta2))
    return optimizer_g, optimizer_d
