import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, resolve_device
from dcgan_b.artifacts import save_generated_grid
from dcgan_b.stylegan import StyleGANDiscriminator64, StyleGANGenerator64, build_stylegan_optimizers


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check lightweight StyleGAN bonus modules.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="outputs/sanity_stylegan/sample.png")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--generator-channels", type=int, default=64)
    parser.add_argument("--discriminator-channels", type=int, default=32)
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = DCGANConfig(batch_size=args.batch_size, num_workers=0, device=str(device))
    generator = StyleGANGenerator64(config, base_channels=args.generator_channels).to(device)
    discriminator = StyleGANDiscriminator64(config, base_channels=args.discriminator_channels).to(device)
    optimizer_g, optimizer_d = build_stylegan_optimizers(generator, discriminator)

    z = torch.randn(config.batch_size, config.noise_dim, 1, 1, device=device)
    fake_images, styles = generator(z, return_style=True)
    assert fake_images.shape == (config.batch_size, config.channels, config.image_size, config.image_size)
    assert styles.shape == (config.batch_size, 128)
    assert float(fake_images.detach().min().cpu()) >= -1.05
    assert float(fake_images.detach().max().cpu()) <= 1.05

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
    g_loss = F.softplus(-fake_logits_for_g).mean()
    optimizer_g.zero_grad(set_to_none=True)
    g_loss.backward()
    optimizer_g.step()

    save_generated_grid(generator, z, args.output, nrow=2)
    print("device:", device)
    print("fake_shape:", tuple(fake_images.shape))
    print("style_shape:", tuple(styles.shape))
    print("d_loss:", round(float(d_loss.detach().cpu()), 4))
    print("g_loss:", round(float(g_loss.detach().cpu()), 4))
    print("sample:", args.output)


if __name__ == "__main__":
    main()
