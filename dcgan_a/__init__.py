"""A-student modules for Task D: data, discriminator, and evaluation."""

from .config import DCGANConfig, resolve_device
from .discriminator import DCGANDiscriminator

__all__ = ["DCGANConfig", "DCGANDiscriminator", "resolve_device"]
