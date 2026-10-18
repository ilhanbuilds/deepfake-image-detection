"""Robustness yalnızca --run-evaluation ile çalışır; eğitim içermez."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Sabit V2 modelinde 8 fotoğraf kalitesi koşulu."
    )
    parser.add_argument("--run-evaluation", action="store_true", required=True)
    parser.add_argument(
        "--data-dir", type=Path, default=PROJECT_ROOT / "data/real_vs_fake"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best_model_v2.pth",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=PROJECT_ROOT / "outputs/metrics/final_test_metrics_v2.json",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="cuda")
    args = parser.parse_args()
    from src.robustness import run_robustness

    run_robustness(
        args.data_dir,
        args.checkpoint,
        args.reference,
        args.output_dir,
        PROJECT_ROOT / "docs/EXPERIMENTS.md",
        args.device,
        args.batch_size,
        args.seed,
    )


if __name__ == "__main__":
    main()
