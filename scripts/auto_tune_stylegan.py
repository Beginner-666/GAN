import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sequential StyleGAN tuning, evaluation, interpolation, and optional final training.")
    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--data-root", default="data/celeba_imagefolder")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--eval-images", type=int, default=4096)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--fixed-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/stylegan_autotune")
    parser.add_argument("--checkpoint-root", default="checkpoints/stylegan_autotune")
    parser.add_argument("--log-root", default="logs/stylegan_autotune")
    parser.add_argument("--candidate-json", default="", help="Optional JSON file containing a list of candidate dicts.")
    parser.add_argument("--max-batches", type=int, default=0, help="Optional per-epoch batch limit for smoke tests.")
    parser.add_argument("--run-final", action="store_true", help="Run a longer final training after selecting the best FID candidate.")
    return parser.parse_args()


def default_candidates() -> list[dict[str, Any]]:
    return [
        {
            "name": "bs32_lr2e4_r1_10",
            "batch_size": 32,
            "lr": 2e-4,
            "beta1": 0.0,
            "beta2": 0.99,
            "generator_channels": 128,
            "discriminator_channels": 64,
            "style_dim": 128,
            "mapping_layers": 4,
            "r1_gamma": 10.0,
            "r1_every": 16,
        },
        {
            "name": "bs32_lr1e4_r1_10",
            "batch_size": 32,
            "lr": 1e-4,
            "beta1": 0.0,
            "beta2": 0.99,
            "generator_channels": 128,
            "discriminator_channels": 64,
            "style_dim": 128,
            "mapping_layers": 4,
            "r1_gamma": 10.0,
            "r1_every": 16,
        },
        {
            "name": "bs64_lr2e4_r1_10",
            "batch_size": 64,
            "lr": 2e-4,
            "beta1": 0.0,
            "beta2": 0.99,
            "generator_channels": 128,
            "discriminator_channels": 64,
            "style_dim": 128,
            "mapping_layers": 4,
            "r1_gamma": 10.0,
            "r1_every": 16,
        },
        {
            "name": "bs32_lr2e4_r1_5",
            "batch_size": 32,
            "lr": 2e-4,
            "beta1": 0.0,
            "beta2": 0.99,
            "generator_channels": 128,
            "discriminator_channels": 64,
            "style_dim": 128,
            "mapping_layers": 4,
            "r1_gamma": 5.0,
            "r1_every": 16,
        },
    ]


def load_candidates(path: str) -> list[dict[str, Any]]:
    if not path:
        return default_candidates()
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("candidate JSON must contain a list of objects")
    return data


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    checkpoint_root = Path(args.checkpoint_root)
    log_root = Path(args.log_root)
    for path in (output_root, checkpoint_root, log_root):
        path.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.candidate_json)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        result = run_candidate(args, candidate, output_root, checkpoint_root, log_root)
        results.append(result)
        write_summary(output_root, results)

    winner = choose_winner(results)
    write_winner(output_root, winner)
    print(f"winner={winner['name']} fid={winner.get('fid')} is_mean={winner.get('inception_score_mean')}", flush=True)
    print_final_command(args, winner, output_root, checkpoint_root, log_root)

    if args.run_final:
        final_result = run_final(args, winner, output_root, checkpoint_root, log_root)
        write_final(output_root, final_result)


def run_candidate(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    output_root: Path,
    checkpoint_root: Path,
    log_root: Path,
) -> dict[str, Any]:
    name = str(candidate["name"])
    output_dir = output_root / name
    checkpoint_dir = checkpoint_root / name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "stylegan_latest.pt"

    if checkpoint.exists():
        print(f"skip_training_existing_checkpoint={checkpoint}", flush=True)
    else:
        run_logged(
            train_command(args, candidate, args.tune_epochs, output_dir, checkpoint_dir),
            log_root / f"{name}.log",
        )

    eval_result = evaluate_checkpoint(args, candidate, checkpoint, output_dir, log_root / f"{name}_eval.log")
    interpolate_checkpoint(args, candidate, checkpoint, output_dir / "interpolation.png", log_root / f"{name}_interpolate.log")
    return result_row(args, candidate, name, "tune", args.tune_epochs, output_dir, checkpoint, eval_result)


def run_final(
    args: argparse.Namespace,
    winner: dict[str, Any],
    output_root: Path,
    checkpoint_root: Path,
    log_root: Path,
) -> dict[str, Any]:
    candidate = dict(winner)
    candidate["name"] = f"final_{winner['name']}_e{args.final_epochs}"
    output_dir = output_root / candidate["name"]
    checkpoint_dir = checkpoint_root / candidate["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "stylegan_latest.pt"

    if checkpoint.exists():
        print(f"skip_final_existing_checkpoint={checkpoint}", flush=True)
    else:
        run_logged(
            train_command(args, candidate, args.final_epochs, output_dir, checkpoint_dir),
            log_root / f"{candidate['name']}.log",
        )

    eval_result = evaluate_checkpoint(args, candidate, checkpoint, output_dir, log_root / f"{candidate['name']}_eval.log")
    interpolate_checkpoint(args, candidate, checkpoint, output_dir / "interpolation.png", log_root / f"{candidate['name']}_interpolate.log")
    return result_row(args, candidate, str(candidate["name"]), "final", args.final_epochs, output_dir, checkpoint, eval_result)


def result_row(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    name: str,
    phase: str,
    epochs: int,
    output_dir: Path,
    checkpoint: Path,
    eval_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "name": name,
        "phase": phase,
        "epochs": epochs,
        "batch_size": int(candidate["batch_size"]),
        "lr": float(candidate["lr"]),
        "beta1": float(candidate.get("beta1", 0.0)),
        "beta2": float(candidate.get("beta2", 0.99)),
        "generator_channels": int(candidate.get("generator_channels", 128)),
        "discriminator_channels": int(candidate.get("discriminator_channels", 64)),
        "style_dim": int(candidate.get("style_dim", 128)),
        "mapping_layers": int(candidate.get("mapping_layers", 4)),
        "r1_gamma": float(candidate.get("r1_gamma", 10.0)),
        "r1_every": int(candidate.get("r1_every", 16)),
        "eval_images": int(args.eval_images),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        **eval_result,
    }


def train_command(args: argparse.Namespace, candidate: dict[str, Any], epochs: int, output_dir: Path, checkpoint_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_stylegan.py"),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(candidate["batch_size"]),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(candidate["lr"]),
        "--beta1",
        str(candidate.get("beta1", 0.0)),
        "--beta2",
        str(candidate.get("beta2", 0.99)),
        "--style-dim",
        str(candidate.get("style_dim", 128)),
        "--mapping-layers",
        str(candidate.get("mapping_layers", 4)),
        "--generator-channels",
        str(candidate.get("generator_channels", 128)),
        "--discriminator-channels",
        str(candidate.get("discriminator_channels", 64)),
        "--r1-gamma",
        str(candidate.get("r1_gamma", 10.0)),
        "--r1-every",
        str(candidate.get("r1_every", 16)),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--sample-every",
        str(args.sample_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--fixed-samples",
        str(args.fixed_samples),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.max_batches:
        cmd.extend(["--max-batches", str(args.max_batches)])
    return cmd


def evaluate_checkpoint(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    checkpoint: Path,
    output_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    if not checkpoint.exists():
        return {"eval_status": "missing_checkpoint", "fid": "", "inception_score_mean": "", "inception_score_std": ""}

    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "evaluate_stylegan.py"),
        "--checkpoint",
        str(checkpoint),
        "--dataset",
        args.dataset,
        "--data-root",
        args.data_root,
        "--num-images",
        str(args.eval_images),
        "--batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--style-dim",
        str(candidate.get("style_dim", 128)),
        "--mapping-layers",
        str(candidate.get("mapping_layers", 4)),
        "--generator-channels",
        str(candidate.get("generator_channels", 128)),
        "--device",
        args.device,
    ]
    completed = run_logged(cmd, log_path, check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = parse_eval_metrics(text)
    (output_dir / f"eval_fid_is_{args.eval_images}.txt").write_text(text, encoding="utf-8")
    return {
        "eval_status": "ok" if completed.returncode == 0 else f"failed_{completed.returncode}",
        "fid": metrics.get("fid", ""),
        "inception_score_mean": metrics.get("inception_score_mean", ""),
        "inception_score_std": metrics.get("inception_score_std", ""),
    }


def interpolate_checkpoint(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    checkpoint: Path,
    output_path: Path,
    log_path: Path,
) -> None:
    if not checkpoint.exists():
        return
    cmd = [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "interpolate_stylegan.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
        "--style-dim",
        str(candidate.get("style_dim", 128)),
        "--mapping-layers",
        str(candidate.get("mapping_layers", 4)),
        "--generator-channels",
        str(candidate.get("generator_channels", 128)),
        "--device",
        args.device,
    ]
    run_logged(cmd, log_path, check=False)


def run_logged(cmd: list[str], log_path: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
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
    completed = subprocess.CompletedProcess(cmd, returncode)
    if check and returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
    return completed


def parse_eval_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, value = line.strip().split("=", 1)
        if key in {"fid", "inception_score_mean", "inception_score_std"}:
            try:
                metrics[key] = float(value)
            except ValueError:
                pass
    return metrics


def choose_winner(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in results if isinstance(row.get("fid"), float)]
    if valid:
        return min(valid, key=lambda row: float(row["fid"]))
    raise RuntimeError("No candidate produced a valid FID. Check evaluation logs under the log root.")


def write_summary(output_root: Path, results: list[dict[str, Any]]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    fields = [
        "name",
        "phase",
        "epochs",
        "batch_size",
        "lr",
        "beta1",
        "beta2",
        "generator_channels",
        "discriminator_channels",
        "style_dim",
        "mapping_layers",
        "r1_gamma",
        "r1_every",
        "eval_images",
        "eval_status",
        "fid",
        "inception_score_mean",
        "inception_score_std",
        "output_dir",
        "checkpoint",
    ]
    with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_winner(output_root: Path, winner: dict[str, Any]) -> None:
    (output_root / "winner.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")


def write_final(output_root: Path, final_result: dict[str, Any]) -> None:
    (output_root / "final_result.json").write_text(json.dumps(final_result, indent=2), encoding="utf-8")


def print_final_command(args: argparse.Namespace, winner: dict[str, Any], output_root: Path, checkpoint_root: Path, log_root: Path) -> None:
    final_name = f"final_{winner['name']}_e{args.final_epochs}"
    output_dir = output_root / final_name
    checkpoint_dir = checkpoint_root / final_name
    log_path = log_root / f"{final_name}.log"
    cmd = train_command(args, winner, args.final_epochs, output_dir, checkpoint_dir)
    cmd[0] = "python"
    command_text = " \\\n+  ".join(cmd) + f" \\\n+  2>&1 | tee {log_path}"
    helper_path = output_root / "final_train_command.sh"
    helper_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + command_text + "\n", encoding="utf-8")
    print("final_training_command:", flush=True)
    print(command_text, flush=True)
    print(f"final_training_command_saved={helper_path}", flush=True)


if __name__ == "__main__":
    main()
