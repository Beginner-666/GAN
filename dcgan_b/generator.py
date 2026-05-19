import torch
from torch import nn

from dcgan_a import DCGANConfig


class DCGANGenerator(nn.Module):
    """DCGAN generator for 64x64 RGB images.

    Input shape:  (N, noise_dim, 1, 1)
    Output shape: (N, 3, 64, 64), values in [-1, 1]
    """

    def __init__(self, config: DCGANConfig = DCGANConfig(), feature_maps_g: int = 64) -> None:
        super().__init__()
        nz = config.noise_dim
        ngf = feature_maps_g
        c = config.channels
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def build_generator_optimizer(generator: nn.Module, config: DCGANConfig = DCGANConfig()) -> torch.optim.Optimizer:
    return torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

