import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, DCGANDiscriminator, resolve_device
from dcgan_a.data import build_dataloader, load_face_dataset
from dcgan_a.losses import build_discriminator_optimizer, train_discriminator_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal discriminator-only training example.")
    parser.add_argument("--data-root", required=True, help="Dataset root path.")
    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = DCGANConfig(batch_size=args.batch_size, num_workers=args.num_workers, device=str(device))
    dataset = load_face_dataset(
        root=args.data_root,
        name=args.dataset,
        image_size=config.image_size,
        train=True,
        download=args.download,
    )
    dataloader = build_dataloader(dataset, config)

    discriminator = DCGANDiscriminator(config).to(device)
    optimizer = build_discriminator_optimizer(discriminator, config)

    for step, (real_images, _targets) in enumerate(dataloader, start=1):
        fake_images = torch.randn_like(real_images).clamp(-1, 1)
        stats = train_discriminator_step(discriminator, optimizer, real_images, fake_images, device)
        if step == 1 or step % config.log_interval == 0:
            print(f"step={step} d_loss={stats['d_loss']:.4f} real={stats['d_real_prob']:.4f} fake={stats['d_fake_prob']:.4f}")
        if step >= args.steps:
            break


if __name__ == "__main__":
    main()
