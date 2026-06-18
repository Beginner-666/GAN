"""B-student modules for Task D: generator, training utilities, and artifacts."""

from .generator import DCGANGenerator, build_generator_optimizer
from .stylegan import StyleGANDiscriminator64, StyleGANGenerator64, build_stylegan_optimizers, copy_stylegan_for_ema, update_ema
from .stylegan2 import StyleGAN2Discriminator64, StyleGAN2Generator64, build_stylegan2_optimizers, copy_stylegan2_for_ema

__all__ = [
    "DCGANGenerator",
    "StyleGANDiscriminator64",
    "StyleGANGenerator64",
    "StyleGAN2Discriminator64",
    "StyleGAN2Generator64",
    "build_generator_optimizer",
    "build_stylegan_optimizers",
    "build_stylegan2_optimizers",
    "copy_stylegan_for_ema",
    "copy_stylegan2_for_ema",
    "update_ema",
]
