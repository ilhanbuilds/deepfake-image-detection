"""Verify recorded runs and protect existing artifacts; never reads test video bytes."""

import argparse
import json
from pathlib import Path

import torch

from src.video_manifest import load_manifest
from src.video_pipeline import file_sha256
from src.video_spatial_evaluate import (
    spatial_source_metrics,
    validate_recorded_predictions,
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protected = read(
        Path(
            "outputs/video_finetune_v1/balanced_readiness_20261012/preservation_before.json"
        )
    )
    # Hash existing paths afresh, including the now completed balanced experiment.
    paths = {Path(name) for name in protected}
    for folder in (
        args.balanced_run,
        args.reference_run,
        Path("src"),
        Path("tests"),
        Path("docs"),
    ):
        paths.update(
            p.resolve()
            for p in folder.rglob("*")
            if p.is_file()
            and "__pycache__" not in p.parts
            and "video_spatial" not in p.name
            and p.name != "VIDEO_SPATIAL_V1.md"
        )
    snapshot = {str(p.resolve()): file_sha256(p) for p in paths if p.is_file()}
    write(args.output_dir / "preservation_before.json", snapshot)
    manifest = load_manifest(args.manifest)
    fixed = read(args.balanced_run / "best_fixed_evaluation.json")
    comparison = read(args.balanced_run / "comparison.json")
    history = read(args.balanced_run / "history.json")
    audit = read(args.balanced_run / "roster_audit.json")
    checkpoint = args.balanced_run / "checkpoints/best_video.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    best = max(history, key=lambda r: r["validation_metrics"]["macro_f1"])
    if (
        not (
            saved["epoch"]
            == saved["best_epoch"]
            == fixed["epoch"]
            == best["epoch"]
            == comparison["best_epoch"]
        )
        or saved["manifest_fingerprint"] != manifest["fingerprint"]
        or file_sha256(checkpoint) != fixed["checkpoint_sha256"]
        or file_sha256(checkpoint)
        != file_sha256(args.balanced_run / f"checkpoints/epoch_{best['epoch']:03d}.pth")
    ):
        raise ValueError("Balanced best/checksum/manifest uyuşmuyor.")
    verified = {
        name: validate_recorded_predictions(
            fixed["results"][name], audit["splits"][name]["accepted_records"]
        )
        for name in ("train", "val")
    }
    if fixed["results"]["val"]["metrics"] != best["validation_metrics"]:
        raise ValueError("Best fixed validation/history uyuşmuyor.")
    last = history[-1]
    if (
        comparison["current_epoch"] != last["epoch"]
        or comparison["current_validation"]["macro_f1"]
        != last["validation_metrics"]["macro_f1"]
    ):
        raise ValueError("comparison.json son epoch ile uyuşmuyor.")
    natural_history = read(args.reference_run / "history.json")
    reference_epoch = max(
        natural_history, key=lambda r: r["validation_metrics"]["macro_f1"]
    )["epoch"]
    reference_eval = read(
        args.reference_run / f"epoch_{reference_epoch:03d}_coverage.json"
    )["validation"]
    control = validate_recorded_predictions(
        reference_eval, audit["splits"]["val"]["accepted_records"]
    )
    if (
        file_sha256(args.reference_run / "checkpoints/best_video.pth")
        != fixed["reference"]["checkpoint_sha256"]
    ):
        raise ValueError("Referans checkpoint hash değişmiş.")
    epochs = []
    union = {name: set() for name in ("REAL", "FAKE")}
    for row in history:
        sampling = row["training_sampling"]["by_class"]
        for name, values in sampling.items():
            union[name].update(values["draws_per_video"])
        evaluated = read(args.balanced_run / f"epoch_{row['epoch']:03d}_coverage.json")[
            "validation"
        ]
        epochs.append(
            dict(
                epoch=row["epoch"],
                validation_metrics=row["validation_metrics"],
                source_pairs=spatial_source_metrics(evaluated["predictions"]),
                sampling={
                    name: {k: v for k, v in values.items() if k != "draws_per_video"}
                    for name, values in sampling.items()
                },
                clipping=row["training_numerics"]["by_class"],
                epoch_seconds=row["epoch_duration"],
            )
        )
    result = dict(
        verified=True,
        balanced_best_epoch=best["epoch"],
        reference_best_epoch=reference_epoch,
        checkpoint_sha256=file_sha256(checkpoint),
        manifest_fingerprint=manifest["fingerprint"],
        comparison_json_scope="latest_epoch_3_not_best_epoch_1",
        balanced_fixed=verified,
        natural_validation=control,
        epochs=epochs,
        three_epoch_unique={
            name: dict(
                unique=len(paths),
                available=audit["splits"]["train"]["accepted_counts"][name],
            )
            for name, paths in union.items()
        },
        validation_deltas=fixed["validation_deltas"],
        interpretation=dict(
            observed="Best adds 3 correct Celeb-real, loses 4 YouTube-real and 41 FAKE; later checkpoints move toward REAL predictions.",
            hypotheses=[
                "Replacement repeats minority videos and limits fake diversity per epoch.",
                "Different class draw ordering, clipping and AdamW dynamics can change decision scores despite similar intended class weighting.",
                "Identity/source/crop shift may dominate class imbalance; one seed does not establish causality.",
            ],
            limitations=[
                "Sampling and loss changed jointly; their separate causal effects are not identified.",
                "Natural historical ONLINE class clipping rates were not logged; fixed-model probes are not directly comparable.",
                "No threshold tuning, official test evaluation or new training was performed.",
            ],
        ),
        official_test_video_content_accessed=False,
    )
    write(args.output_dir / "balanced_analysis.json", result)
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "verified",
                    "balanced_best_epoch",
                    "reference_best_epoch",
                    "three_epoch_unique",
                    "validation_deltas",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--balanced-run",
        type=Path,
        default=Path("outputs/video_finetune_v1/balanced_20261012_221120_937"),
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("outputs/video_finetune_v1/identity_continue_20261012_210232"),
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/video_identity_v1/manifest.json")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    analyze(parser.parse_args())
