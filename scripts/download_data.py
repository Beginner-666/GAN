import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a.data import load_face_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify face datasets for Task D.")
    parser.add_argument("--dataset", required=True, choices=["lfw", "celeba"], help="Dataset to download.")
    parser.add_argument("--data-root", default="data", help="Root directory for datasets.")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--split", default="train", choices=["train", "test"], help="Split used for verification.")
    args = parser.parse_args()

    train = args.split == "train"
    root = Path(args.data_root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"dataset={args.dataset}", flush=True)
    print(f"data_root={root.resolve()}", flush=True)
    print("status=downloading_or_loading", flush=True)
    print("note=torchvision will print the download progress when files are not present", flush=True)
    dataset = load_face_dataset(
        root=root,
        name=args.dataset,
        image_size=args.image_size,
        train=train,
        download=True,
    )
    print("status=verifying", flush=True)
    image, target = dataset[0]
    print(f"dataset={args.dataset}")
    print(f"num_samples={len(dataset)}")
    print(f"first_image_shape={tuple(image.shape)}")
    print(f"first_target_type={type(target).__name__}")
    print("status=ok")


if __name__ == "__main__":
    main()
