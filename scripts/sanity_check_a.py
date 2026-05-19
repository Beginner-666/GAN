import os
import sys
import argparse
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, DCGANDiscriminator, resolve_device
from dcgan_a.losses import build_discriminator_optimizer, train_discriminator_step
from dcgan_a.metrics import calculate_fid, calculate_inception_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanity-check A-student DCGAN modules.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Preferred runtime device.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    config = DCGANConfig(batch_size=4, num_workers=0, device=str(device))
    discriminator = DCGANDiscriminator(config).to(device)
    real = torch.randn(config.batch_size, config.channels, config.image_size, config.image_size).clamp(-1, 1)
    fake = torch.randn_like(real).clamp(-1, 1)

    output = discriminator(real.to(device))
    assert output.shape == (config.batch_size,)

    optimizer = build_discriminator_optimizer(discriminator, config)
    stats = train_discriminator_step(discriminator, optimizer, real, fake, device)

    real_features = torch.randn(16, 32).numpy()
    fake_features = torch.randn(16, 32).numpy()
    probs = torch.softmax(torch.randn(16, 10), dim=1).numpy()
    try:
        fid = calculate_fid(real_features, fake_features)
        fid_text = round(fid, 4)
    except ImportError as exc:
        fid_text = f"skipped ({exc})"
    is_mean, is_std = calculate_inception_score(probs, splits=4)

    print("device:", device)
    print("discriminator_output_shape:", tuple(output.shape))
    print("loss_stats:", stats)
    print("dummy_fid:", fid_text)
    print("dummy_is:", round(is_mean, 4), round(is_std, 4))


if __name__ == "__main__":
    main()
