"""Explicit small smoke OR explicit train/val feature cache; never trains a model."""

import argparse
import json
import random
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import torch
from PIL import Image

from src.dataset import seed_everything
from src.model import resolve_device
from src.video_evaluate import coverage
from src.video_manifest import DEFAULT_MANIFEST, load_manifest
from src.video_model import VideoClassifier
from src.video_pipeline import FacePipeline, FeatureCache, VideoSettings


def select_smoke(records: list[dict], per_class: int = 2) -> list[dict]:
    if not 1 <= per_class <= 3:
        raise ValueError("Smoke örneklemi sınıf başına 1-3 video olmalı.")
    selected = []
    for label in (0, 1):
        selected.extend([r for r in records if r["label"] == label][:per_class])
    if len(selected) != 2 * per_class:
        raise ValueError("Smoke için iki sınıftan yeterli video yok.")
    return selected


def select_balanced_sample(
    records: list[dict], per_class: int = 20, seed: int = 42
) -> list[dict]:
    if not 1 <= per_class <= 20:
        raise ValueError("Küçük cache testi sınıf başına 1–20 video olmalı.")
    rng = random.Random(seed)
    selected = []
    for label in (0, 1):
        candidates = sorted(
            (r for r in records if r["label"] == label), key=lambda r: r["path"]
        )
        if len(candidates) < per_class:
            raise ValueError(
                "Küçük cache testi için iki sınıftan yeterli train videosu yok."
            )
        selected.extend(rng.sample(candidates, per_class))
    rng.shuffle(selected)
    return selected


def estimate_preprocessing(
    timings: list[dict], counts: dict, startup_seconds: float
) -> dict | None:
    durations = {
        name: [
            row["elapsed_seconds"]
            for row in timings
            if row["label"] == label and not row["cache_hit"]
        ]
        for label, name in enumerate(("REAL", "FAKE"))
    }
    if any(not values for values in durations.values()):
        return None
    means = {name: sum(values) / len(values) for name, values in durations.items()}
    seconds = startup_seconds + sum(means[name] * counts[name] for name in means)
    return dict(
        method="new-cache per-class mean seconds weighted by unchanged manifest counts",
        target_counts=counts,
        seconds_per_video_by_class=means,
        fresh_sample_counts={name: len(values) for name, values in durations.items()},
        estimated_seconds=seconds,
        estimated_minutes=seconds / 60,
        limitation="Küçük örneklem tahmini; video uzunluğu, decode, yüz hataları, disk ve GPU yükü değişebilir. Cache-hit süreleri kullanılmaz. Python import süresi dahil değil.",
    )


def make_parser():
    parser = argparse.ArgumentParser(
        description="Açık video smoke/cache; test ayrımına erişmez."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument(
        "--sample-cache",
        action="store_true",
        help="Yalnız train'den seed 42 ile 20 REAL + 20 FAKE cache/benchmark.",
    )
    mode.add_argument("--cache-split", choices=("train", "val"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--per-class", type=int)
    parser.add_argument("--image-size", type=int, choices=(160, 192), default=160)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()
    if args.cache_split and args.per_class is not None:
        parser.error("--per-class yalnız --smoke veya --sample-cache ile kullanılır.")
    command_started = perf_counter()
    seed_everything(42)
    manifest = load_manifest(args.manifest)
    settings = VideoSettings(image_size=args.image_size)
    pipeline = FacePipeline(settings)
    device = resolve_device(args.device)
    model = VideoClassifier(pretrained=True).to(device).eval()
    cache = FeatureCache(
        args.cache_dir, pipeline.signature, model.backbone_fingerprint()
    )
    root = Path(manifest["dataset_root"])
    records = (
        select_smoke(
            manifest["splits"]["train"], 2 if args.per_class is None else args.per_class
        )
        if args.smoke
        else select_balanced_sample(
            manifest["splits"]["train"],
            20 if args.per_class is None else args.per_class,
        )
        if args.sample_cache
        else manifest["splits"][args.cache_split]
    )
    directory = (
        Path("outputs/video_v1")
        / ("smoke" if args.smoke else "cache_reports")
        / uuid4().hex
    )
    directory.mkdir(parents=True, exist_ok=False)
    reports = []
    timings = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = perf_counter()
    startup_seconds = started - command_started
    try:
        for index, record in enumerate(records, 1):
            video_started = perf_counter()
            if not args.smoke and cache.path(record).exists():
                item = cache.read(record, root)
                reports.append(item["report"])
                timings.append(
                    dict(
                        path=record["path"],
                        label=record["label"],
                        cache_hit=True,
                        elapsed_seconds=perf_counter() - video_started,
                    )
                )
                print(
                    f"Video {index}/{len(records)}: {record['path']} cache verified",
                    flush=True,
                )
                continue
            frames, report, crops = pipeline.read(root / record["path"])
            report.update(relative_path=record["path"], label=record["label"])
            features = None
            if frames is not None:
                # FP32 features make persistent cache reproducible across raw/cached runs.
                features = model.extract_features(frames.to(device)).detach().cpu()
                if not torch.isfinite(features).all():
                    raise RuntimeError(
                        f"Sonlu olmayan backbone özellikleri: {record['path']}"
                    )
                report["feature_shape"] = list(features.shape)
                if args.smoke:
                    sample = directory / f"sample_{index:02d}"
                    sample.mkdir()
                    for frame_index, crop in zip(
                        report["valid_indices"], crops, strict=True
                    ):
                        crop.resize(
                            (settings.image_size, settings.image_size),
                            Image.Resampling.BILINEAR,
                        ).save(sample / f"frame_{frame_index:06d}.png")
                    torch.save(features, sample / "features.pt")
            if not args.smoke:
                cache.write(record, root, features, report)
            reports.append(report)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(
                dict(
                    path=record["path"],
                    label=record["label"],
                    cache_hit=False,
                    elapsed_seconds=perf_counter() - video_started,
                )
            )
            print(
                f"Video {index}/{len(records)}: {record['path']} {report['status']} ({report['valid_frames']}/6)",
                flush=True,
            )
    except (torch.cuda.OutOfMemoryError, KeyboardInterrupt) as error:
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        (directory / "interrupted.json").write_text(
            json.dumps(
                dict(reason=type(error).__name__, coverage=coverage(reports)),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            f"İşlem kesildi; tamamlanan cache korunur. Rapor: {directory}. CUDA OOM için CPU kullanın."
        ) from error
    result = dict(
        mode="smoke"
        if args.smoke
        else "sample_cache"
        if args.sample_cache
        else "cache",
        split="train" if args.smoke or args.sample_cache else args.cache_split,
        manifest_fingerprint=manifest["fingerprint"],
        pipeline=pipeline.signature,
        backbone_sha256=model.backbone_fingerprint(),
        pretrained="ImageNet IMAGENET1K_V1",
        elapsed_seconds=perf_counter() - started,
        startup_seconds=startup_seconds,
        seed=42,
        selected_counts={
            name: sum(r["label"] == label for r in records)
            for label, name in enumerate(("REAL", "FAKE"))
        },
        timings=timings,
        fresh_videos=sum(not row["cache_hit"] for row in timings),
        reused_videos=sum(row["cache_hit"] for row in timings),
        train_preprocessing_estimate=estimate_preprocessing(
            timings, manifest["split_counts"]["train"], startup_seconds
        )
        if not args.smoke
        else None,
        coverage=coverage(reports),
        cache_directory=str(cache.directory) if not args.smoke else None,
    )
    if device.type == "cuda":
        result["gpu"] = torch.cuda.get_device_name(device)
        result["peak_cuda_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
        result["peak_cuda_reserved_mib"] = torch.cuda.max_memory_reserved() / 1024**2
    (directory / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rapor: {directory / 'report.json'}; eğitim yapılmadı.")
    if not result["coverage"]["evaluated_videos"]:
        raise SystemExit("Hiç video kabul edilmedi; yüz/decode raporunu inceleyin.")


if __name__ == "__main__":
    main()
