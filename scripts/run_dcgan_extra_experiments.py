import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dcgan_a import DCGANConfig, resolve_device
from dcgan_a.metrics import evaluate_fid_and_is
from dcgan_b import DCGANGenerator
from dcgan_b.artifacts import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run extra DCGAN experiments for epoch ablation and mode collapse analysis.")
    parser.add_argument("--data-root", default="data/celeba_imagefolder")
    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-images", type=int, default=2048)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--output-root", default="outputs/dcgan_extra")
    parser.add_argument("--checkpoint-root", default="checkpoints/dcgan_extra")
    parser.add_argument("--log-root", default="logs/dcgan_extra")
    parser.add_argument("--skip-epoch-training", action="store_true")
    parser.add_argument("--skip-mode-collapse-training", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    log_root = Path(args.log_root)
    for path in (output_root, checkpoint_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    epoch_results = run_epoch_ablation(args, output_root, checkpoint_root, log_root)
    mode_result = run_mode_collapse(args, output_root, checkpoint_root, log_root)
    write_combined_notes(output_root, epoch_results, mode_result)


def run_epoch_ablation(args: argparse.Namespace, output_root: Path, checkpoint_root: Path, log_root: Path) -> list[dict[str, Any]]:
    output_dir = output_root / "epoch_ablation_bs32_g128_d64_e100"
    checkpoint_dir = checkpoint_root / "epoch_ablation_bs32_g128_d64_e100"
    log_path = log_root / "epoch_ablation_train.log"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_epoch_training and not (checkpoint_dir / "dcgan_latest.pt").exists():
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "train_dcgan.py"),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--epochs",
            "100",
            "--batch-size",
            "32",
            "--num-workers",
            str(args.num_workers),
            "--lr",
            "0.0002",
            "--feature-maps-g",
            "128",
            "--feature-maps-d",
            "64",
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--sample-every",
            "10",
            "--checkpoint-every",
            "10",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        run_logged(cmd, log_path)

    rows: list[dict[str, Any]] = []
    for epoch in (10, 30, 50, 100):
        checkpoint = checkpoint_dir / f"dcgan_epoch_{epoch:04d}.pt"
        if not checkpoint.exists() and epoch == 100:
            checkpoint = checkpoint_dir / "dcgan_latest.pt"
        row = evaluate_dcgan_checkpoint(
            args=args,
            checkpoint=checkpoint,
            feature_maps_g=128,
            noise_dim=128,
            output_dir=output_dir,
            name=f"epoch_{epoch:04d}",
        )
        row.update(
            {
                "experiment": "epoch_ablation",
                "epoch": epoch,
                "batch_size": 32,
                "lr": 0.0002,
                "feature_maps_g": 128,
                "feature_maps_d": 64,
                "checkpoint": str(checkpoint),
            }
        )
        rows.append(row)

    write_csv(output_root / "epoch_ablation_summary.csv", rows)
    write_epoch_ablation_plots(output_root, rows)
    return rows


def run_mode_collapse(args: argparse.Namespace, output_root: Path, checkpoint_root: Path, log_root: Path) -> dict[str, Any]:
    output_dir = output_root / "mode_collapse_forced"
    checkpoint_dir = checkpoint_root / "mode_collapse_forced"
    log_path = log_root / "mode_collapse_train.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_mode_collapse_training and not (checkpoint_dir / "dcgan_latest.pt").exists():
        cmd = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts" / "train_dcgan_mode_collapse.py"),
            "--dataset",
            args.dataset,
            "--data-root",
            args.data_root,
            "--epochs",
            "10",
            "--batch-size",
            "32",
            "--num-workers",
            str(args.num_workers),
            "--noise-dim",
            "8",
            "--lr-g",
            "1e-5",
            "--lr-d",
            "1e-3",
            "--feature-maps-g",
            "32",
            "--feature-maps-d",
            "128",
            "--d-steps",
            "5",
            "--output-dir",
            str(output_dir),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--sample-every",
            "1",
            "--checkpoint-every",
            "5",
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        run_logged(cmd, log_path)

    checkpoint = checkpoint_dir / "dcgan_latest.pt"
    row = evaluate_dcgan_checkpoint(
        args=args,
        checkpoint=checkpoint,
        feature_maps_g=32,
        noise_dim=8,
        output_dir=output_dir,
        name="mode_collapse_forced",
    )
    row.update(
        {
            "experiment": "mode_collapse_forced",
            "epoch": 10,
            "batch_size": 32,
            "lr_g": 1e-5,
            "lr_d": 1e-3,
            "feature_maps_g": 32,
            "feature_maps_d": 128,
            "noise_dim": 8,
            "d_steps": 5,
            "checkpoint": str(checkpoint),
        }
    )
    write_csv(output_root / "mode_collapse_summary.csv", [row])
    write_mode_collapse_plot(output_root, row)
    return row


def evaluate_dcgan_checkpoint(
    args: argparse.Namespace,
    checkpoint: Path,
    feature_maps_g: int,
    noise_dim: int,
    output_dir: Path,
    name: str,
) -> dict[str, Any]:
    if not checkpoint.exists():
        return {
            "name": name,
            "eval_status": "missing_checkpoint",
            "fid": "",
            "inception_score_mean": "",
            "inception_score_std": "",
        }

    device = resolve_device(args.device)
    config = DCGANConfig(
        noise_dim=noise_dim,
        batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        device=str(device),
    )
    from dcgan_a.data import build_dataloader, load_face_dataset

    dataset = load_face_dataset(args.data_root, args.dataset, image_size=config.image_size, train=False, download=False)
    dataloader = build_dataloader(dataset, config, shuffle=True, drop_last=False)
    generator = DCGANGenerator(config, feature_maps_g=feature_maps_g).to(device)
    load_checkpoint(checkpoint, generator, device=device)
    generator.eval()

    real_batches: list[torch.Tensor] = []
    fake_batches: list[torch.Tensor] = []
    collected = 0
    with torch.no_grad():
        for real_images, _targets in dataloader:
            current = min(real_images.size(0), args.eval_images - collected)
            if current <= 0:
                break
            real = real_images[:current]
            z = torch.randn(current, noise_dim, 1, 1, device=device)
            fake = generator(z).cpu()
            real_batches.append(real)
            fake_batches.append(fake)
            collected += current
            if collected >= args.eval_images:
                break

    metrics = evaluate_fid_and_is(real_batches, fake_batches, device=device, batch_size=args.eval_batch_size)
    result = {
        "name": name,
        "eval_status": "ok",
        "num_images": collected,
        "fid": metrics["fid"],
        "inception_score_mean": metrics["inception_score_mean"],
        "inception_score_std": metrics["inception_score_std"],
    }
    eval_path = output_dir / f"{name}_eval_fid_is.txt"
    eval_path.write_text(
        "\n".join(
            [
                f"num_images={collected}",
                f"fid={metrics['fid']:.6f}",
                f"inception_score_mean={metrics['inception_score_mean']:.6f}",
                f"inception_score_std={metrics['inception_score_std']:.6f}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"evaluated {name}: fid={metrics['fid']:.6f} is={metrics['inception_score_mean']:.6f}", flush=True)
    return result


def run_logged(cmd: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("running:", " ".join(cmd), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_combined_notes(output_root: Path, epoch_results: list[dict[str, Any]], mode_result: dict[str, Any]) -> None:
    lines = [
        "# DCGAN Extra Experiments Summary",
        "",
        "## Epoch Ablation",
        "",
        "Use `epoch_ablation_summary.csv` to report how FID/IS changes from 10 to 100 epochs.",
        "",
        "## Mode Collapse Analysis",
        "",
        "Use `mode_collapse_summary.csv`, `mode_collapse_forced/samples/latest.png`, and",
        "`mode_collapse_forced/loss.svg` to discuss discriminator-dominant training and reduced diversity.",
        "",
        "## Key Files",
        "",
        "- `epoch_ablation_summary.csv`",
        "- `epoch_ablation_fid.png`",
        "- `epoch_ablation_is.png`",
        "- `mode_collapse_summary.csv`",
        "- `mode_collapse_fid_compare.png`",
        "- `epoch_ablation_bs32_g128_d64_e100/samples/`",
        "- `mode_collapse_forced/samples/`",
    ]
    output_root.joinpath("README_EXTRA_EXPERIMENTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_epoch_ablation_plots(output_root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip epoch ablation plots", flush=True)
        return

    valid = [row for row in rows if isinstance(row.get("fid"), float)]
    if not valid:
        return
    epochs = [int(row["epoch"]) for row in valid]
    fids = [float(row["fid"]) for row in valid]
    is_means = [float(row["inception_score_mean"]) for row in valid]

    plt.figure(figsize=(7, 4.2), dpi=160)
    plt.plot(epochs, fids, marker="o", linewidth=2.2, color="#2563eb")
    for x, y in zip(epochs, fids):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("FID (lower is better)")
    plt.title("DCGAN Epoch Ablation: FID")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "epoch_ablation_fid.png")
    plt.close()

    plt.figure(figsize=(7, 4.2), dpi=160)
    plt.plot(epochs, is_means, marker="o", linewidth=2.2, color="#dc2626")
    for x, y in zip(epochs, is_means):
        plt.text(x, y, f"{y:.2f}", ha="center", va="bottom", fontsize=8)
    plt.xlabel("Epoch")
    plt.ylabel("Inception Score")
    plt.title("DCGAN Epoch Ablation: Inception Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_root / "epoch_ablation_is.png")
    plt.close()


def write_mode_collapse_plot(output_root: Path, mode_row: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skip mode collapse plot", flush=True)
        return

    baseline_csv = output_root / "epoch_ablation_summary.csv"
    if not baseline_csv.exists() or not isinstance(mode_row.get("fid"), float):
        return

    baseline_fid = None
    with baseline_csv.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("epoch") == "50" and row.get("fid"):
                baseline_fid = float(row["fid"])
                break
    if baseline_fid is None:
        return

    names = ["Normal DCGAN\n50 epochs", "Forced unstable\nsetting"]
    values = [baseline_fid, float(mode_row["fid"])]
    colors = ["#16a34a", "#dc2626"]
    plt.figure(figsize=(6.5, 4.2), dpi=160)
    bars = plt.bar(names, values, color=colors, width=0.55)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel("FID (lower is better)")
    plt.title("Mode Collapse / Unstable Training Comparison")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_root / "mode_collapse_fid_compare.png")
    plt.close()


if __name__ == "__main__":
    main()
