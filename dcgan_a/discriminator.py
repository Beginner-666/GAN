import torch
from torch import nn

from .config import DCGANConfig


class DCGANDiscriminator(nn.Module):
    """DCGAN discriminator for 64x64 RGB images.

    Input shape:  (N, 3, 64, 64), values normalized to [-1, 1]
    Output shape: (N,), probability that each image is real
    """

    def __init__(self, config: DCGANConfig = DCGANConfig()) -> None:
        super().__init__()
        c = config.channels
        ndf = config.feature_maps_d
        self.net = nn.Sequential(
            nn.Conv2d(c, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
            nn.init.normal_(module.weight.data, 0.0, 0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.constant_(module.bias.data, 0.0)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images).view(-1)
