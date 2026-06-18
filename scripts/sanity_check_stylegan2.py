from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, resolve_device
from dcgan_b.artifacts import load_checkpoint, save_checkpoint, save_generated_grid
from dcgan_b.stylegan2 import StyleGAN2Discriminator64, StyleGAN2Generator64, build_stylegan2_optimizers, copy_stylegan2_for_ema, update_ema


def path_length_regularization(fake_images: torch.Tensor, styles: torch.Tensor) -> torch.Tensor:
    noise = torch.randn_like(fake_images) / math.sqrt(fake_images.shape[2] * fake_images.shape[3])
    grad = torch.autograd.grad(
        outputs=(fake_images * noise).sum(),
        inputs=styles,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return grad.pow(2).sum(2).mean(1).sqrt().mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check StyleGAN2 modules.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="outputs/sanity_stylegan2/sample.png")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generator-channels", type=int, default=64)
    parser.add_argument("--discriminator-channels", type=int, default=32)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = DCGANConfig(batch_size=args.batch_size, num_workers=0, device=str(device))
    generator = StyleGAN2Generator64(config, base_channels=args.generator_channels).to(device)
    discriminator = StyleGAN2Discriminator64(config, base_channels=args.discriminator_channels).to(device)
    generator_ema = copy_stylegan2_for_ema(generator).to(device)
    optimizer_g, optimizer_d = build_stylegan2_optimizers(generator, discriminator)

    z = torch.randn(config.batch_size, config.noise_dim, 1, 1, device=device)
    mixing_z = torch.randn_like(z)
    fake_images, styles = generator(z, return_styles=True, mixing_z=mixing_z, style_mixing_prob=1.0)
    assert fake_images.shape == (config.batch_size, config.channels, config.image_size, config.image_size)
    assert styles.shape == (config.batch_size, generator.num_style_layers, generator.style_dim)
    assert float(fake_images.detach().min().cpu()) >= -1.05
    assert float(fake_images.detach().max().cpu()) <= 1.05

    latent = generator.get_latent(z)
    assert latent.shape == (config.batch_size, generator.style_dim)
    truncated_images = generator.forward_from_style(latent, truncation_psi=0.7, truncation_cutoff=4)
    assert truncated_images.shape == fake_images.shape

    noises = generator.make_noise(config.batch_size, device=device)
    noised_images = generator.forward_from_styles(styles, noise=noises)
    assert noised_images.shape == fake_images.shape

    real_images = torch.randn_like(fake_images).clamp(-1, 1)
    real_logits = discriminator(real_images)
    fake_logits = discriminator(fake_images.detach())
    assert real_logits.shape == (config.batch_size,)
    assert fake_logits.shape == (config.batch_size,)

    d_loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
    optimizer_d.zero_grad(set_to_none=True)
    d_loss.backward()
    optimizer_d.step()

    fake_logits_for_g = discriminator(fake_images)
    g_loss = F.softplus(-fake_logits_for_g).mean() + 0.1 * path_length_regularization(fake_images, styles)
    optimizer_g.zero_grad(set_to_none=True)
    g_loss.backward()
    optimizer_g.step()
    update_ema(generator_ema, generator, decay=0.5)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "stylegan2_sanity.pt"
        save_checkpoint(checkpoint_path, 1, generator, discriminator, optimizer_g, optimizer_d, config, z, generator_ema=generator_ema)
        reloaded = StyleGAN2Generator64(config, base_channels=args.generator_channels).to(device)
        load_checkpoint(checkpoint_path, reloaded, device=device, use_ema=True)
        reloaded.eval()
        reloaded_images = reloaded(z)
        assert reloaded_images.shape == fake_images.shape

    save_generated_grid(generator_ema, z, args.output, nrow=2)
    print("device:", device)
    print("fake_shape:", tuple(fake_images.shape))
    print("style_shape:", tuple(styles.shape))
    print("num_style_layers:", generator.num_style_layers)
    print("d_loss:", round(float(d_loss.detach().cpu()), 4))
    print("g_loss:", round(float(g_loss.detach().cpu()), 4))
    print("sample:", args.output)


if __name__ == "__main__":
    main()
