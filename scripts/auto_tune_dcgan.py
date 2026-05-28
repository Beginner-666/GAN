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
    parser = argparse.ArgumentParser(description="Run sequential DCGAN tuning, evaluation, and final training.")
    parser.add_argument("--dataset", default="imagefolder", choices=["imagefolder", "lfw", "celeba"])
    parser.add_argument("--data-root", default="data/celeba_imagefolder")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tune-epochs", type=int, default=20)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--eval-images", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--sample-every", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="outputs/celeba_autotune")
    parser.add_argument("--checkpoint-root", default="checkpoints/celeba_autotune")
    parser.add_argument("--log-root", default="logs/celeba_autotune")
    parser.add_argument(
        "--candidate-json",
        default="",
        help="Optional JSON file containing a list of candidate dicts. If omitted, built-in candidates are used.",
    )
    parser.add_argument(
        "--skip-final",
        action="store_true",
        help="Only run tuning and evaluation; do not launch the final longer training run.",
    )
    return parser.parse_args()


def default_candidates() -> list[dict[str, Any]]:
    return [
        {"name": "baseline_bs128_lr2e4_g64_d64", "batch_size": 128, "lr": 2e-4, "feature_maps_g": 64, "feature_maps_d": 64},
        {"name": "stable_bs128_lr1e4_g64_d64", "batch_size": 128, "lr": 1e-4, "feature_maps_g": 64, "feature_maps_d": 64},
        {"name": "smallbatch_bs64_lr2e4_g64_d64", "batch_size": 64, "lr": 2e-4, "feature_maps_g": 64, "feature_maps_d": 64},
        {"name": "largerg_bs64_lr2e4_g128_d64", "batch_size": 64, "lr": 2e-4, "feature_maps_g": 128, "feature_maps_d": 64},
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

    if not args.skip_final:
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
    log_path = log_root / f"{name}.log"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = checkpoint_dir / "dcgan_latest.pt"
    if checkpoint.exists():
        print(f"skip_training_existing_checkpoint={checkpoint}", flush=True)
    else:
        train_cmd = train_command(
            args=args,
            candidate=candidate,
            epochs=args.tune_epochs,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        run_logged(train_cmd, log_path)

    eval_result = evaluate_checkpoint(args, candidate, checkpoint, output_dir, log_root / f"{name}_eval.log")
    interpolate_checkpoint(args, candidate, checkpoint, output_dir / "interpolation.png", log_root / f"{name}_interpolate.log")
    return {
        "name": name,
        "phase": "tune",
        "epochs": args.tune_epochs,
        "batch_size": int(candidate["batch_size"]),
        "lr": float(candidate["lr"]),
        "feature_maps_g": int(candidate["feature_maps_g"]),
        "feature_maps_d": int(candidate["feature_maps_d"]),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        **eval_result,
    }


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
    checkpoint = checkpoint_dir / "dcgan_latest.pt"

    if checkpoint.exists():
        print(f"skip_final_existing_checkpoint={checkpoint}", flush=True)
    else:
        train_cmd = train_command(
            args=args,
            candidate=candidate,
            epochs=args.final_epochs,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir,
        )
        run_logged(train_cmd, log_root / f"{candidate['name']}.log")

    eval_result = evaluate_checkpoint(args, candidate, checkpoint, output_dir, log_root / f"{candidate['name']}_eval.log")
    interpolate_checkpoint(args, candidate, checkpoint, output_dir / "interpolation.png", log_root / f"{candidate['name']}_interpolate.log")
    return {
        "name": candidate["name"],
        "phase": "final",
        "epochs": args.final_epochs,
        "batch_size": int(candidate["batch_size"]),
        "lr": float(candidate["lr"]),
        "feature_maps_g": int(candidate["feature_maps_g"]),
        "feature_maps_d": int(candidate["feature_maps_d"]),
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        **eval_result,
    }


def train_command(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    epochs: int,
    output_dir: Path,
    checkpoint_dir: Path,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        str(PROJECT_ROOT / "scripts" / "train_dcgan.py"),
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
        "--feature-maps-g",
        str(candidate["feature_maps_g"]),
        "--feature-maps-d",
        str(candidate["feature_maps_d"]),
        "--output-dir",
        str(output_dir),
        "--checkpoint-dir",
        str(checkpoint_dir),
        "--sample-every",
        str(args.sample_every),
        "--checkpoint-every",
        str(args.checkpoint_every),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]


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
        str(PROJECT_ROOT / "scripts" / "evaluate_dcgan.py"),
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
        "--feature-maps-g",
        str(candidate["feature_maps_g"]),
        "--device",
        args.device,
    ]
    completed = run_logged(cmd, log_path, check=False)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    metrics = parse_eval_metrics(text)
    eval_path = output_dir / "eval_fid_is.txt"
    eval_path.write_text(text, encoding="utf-8")
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
        str(PROJECT_ROOT / "scripts" / "interpolate.py"),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output_path),
        "--feature-maps-g",
        str(candidate["feature_maps_g"]),
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
    summary_json = output_root / "summary.json"
    summary_csv = output_root / "summary.csv"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    fields = [
        "name",
        "phase",
        "epochs",
        "batch_size",
        "lr",
        "feature_maps_g",
        "feature_maps_d",
        "eval_status",
        "fid",
        "inception_score_mean",
        "inception_score_std",
        "output_dir",
        "checkpoint",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_winner(output_root: Path, winner: dict[str, Any]) -> None:
    (output_root / "winner.json").write_text(json.dumps(winner, indent=2), encoding="utf-8")


def write_final(output_root: Path, final_result: dict[str, Any]) -> None:
    (output_root / "final_result.json").write_text(json.dumps(final_result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
