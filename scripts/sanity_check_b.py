import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, DCGANDiscriminator, resolve_device
from dcgan_a.losses import build_discriminator_optimizer, train_discriminator_step
from dcgan_b import DCGANGenerator, build_generator_optimizer
from dcgan_b.artifacts import save_generated_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check B-student DCGAN modules.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output", default="outputs/sanity_b/sample.png")
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = DCGANConfig(batch_size=4, num_workers=0, device=str(device))
    generator = DCGANGenerator(config).to(device)
    discriminator = DCGANDiscriminator(config).to(device)
    optimizer_g = build_generator_optimizer(generator, config)
    optimizer_d = build_discriminator_optimizer(discriminator, config)
    criterion = nn.BCELoss()

    z = torch.randn(config.batch_size, config.noise_dim, 1, 1, device=device)
    fake_images = generator(z)
    assert fake_images.shape == (config.batch_size, config.channels, config.image_size, config.image_size)
    assert float(fake_images.detach().min().cpu()) >= -1.05
    assert float(fake_images.detach().max().cpu()) <= 1.05

    real_images = torch.randn_like(fake_images).clamp(-1, 1)
    d_stats = train_discriminator_step(discriminator, optimizer_d, real_images, fake_images, device, criterion)

    optimizer_g.zero_grad(set_to_none=True)
    pred_fake = discriminator(fake_images)
    g_loss = criterion(pred_fake, torch.ones_like(pred_fake))
    g_loss.backward()
    optimizer_g.step()

    save_generated_grid(generator, z, args.output, nrow=2)
    print("device:", device)
    print("fake_shape:", tuple(fake_images.shape))
    print("d_loss:", round(d_stats["d_loss"], 4))
    print("g_loss:", round(float(g_loss.detach().cpu()), 4))
    print("sample:", args.output)


if __name__ == "__main__":
    main()

