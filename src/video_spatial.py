"""Separate bounded spatial experiment; natural videos, weighted video BCE."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import torch

from src.dataset import seed_everything
from src.model import resolve_device
from src.train import (
    EarlyStopping,
    atomic_copy,
    atomic_save,
    calculate_timing,
    capture_rng,
    restore_rng,
    select_resume,
)
from src.video_cache import audit_feature_cache
from src.video_evaluate import coverage, evaluate_video
from src.video_finetune import validate_resume as validate_gru_reference
from src.video_finetune_data import CropDriftError, VerifiedImageDataset
from src.video_pipeline import (
    FacePipeline,
    FeatureCache,
    FeatureDataset,
    VideoSettings,
    file_sha256,
    make_video_loader,
)
from src.video_spatial_evaluate import summarize_evaluation
from src.video_spatial_model import (
    PIPELINE,
    SpatialVideoClassifier,
    make_spatial_optimizer,
)
from src.video_manifest import load_manifest
from src.video_train import VideoNumericalError, train_epoch, training_class_balance


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def prepare_context(args):
    seed_everything(42)
    manifest = load_manifest(args.manifest)
    if manifest.get("schema") != "celeb_df_identity_components_v1":
        raise ValueError("Mevcut identity-aware manifest gerekli.")
    settings = VideoSettings()
    pipeline = FacePipeline(settings)
    model = SpatialVideoClassifier(pretrained=True, backbone_mode=args.backbone_mode)
    backbone_hash = model.backbone_fingerprint()
    cache = FeatureCache(args.cache_dir, pipeline.signature, backbone_hash)
    audit = audit_feature_cache(manifest, cache)
    balance = training_class_balance(audit["splits"]["train"]["accepted_records"])
    reference_dir = args.reference_run.resolve()
    control_config = read_json(reference_dir / "config.json")
    reference_path = reference_dir / "checkpoints/best_video.pth"
    reference_state = torch.load(reference_path, map_location="cpu", weights_only=True)
    validate_gru_reference(reference_state, manifest, control_config)
    history = read_json(reference_dir / "history.json")
    best = max(history, key=lambda r: r["validation_metrics"]["macro_f1"])
    if (
        [r["epoch"] for r in history] != [1, 2, 3]
        or reference_state["epoch"] != best["epoch"]
        or reference_state["best_epoch"] != best["epoch"]
        or control_config.get("sampling", "natural") != "natural"
        or balance["counts"] != {"REAL": 417, "FAKE": 1806}
        or audit["splits"]["val"]["accepted"] != 602
    ):
        raise ValueError("Natural+weighted üç epoch kontrol/roster/best uyuşmuyor.")
    matched = dict(
        settings=asdict(settings),
        pipeline=pipeline.signature,
        initial_backbone_sha256=backbone_hash,
        cache_fingerprint=audit["cache_fingerprint"],
        batch_size=1,
        frames=6,
        precision="fp32",
        seed=42,
        workers=0,
        backbone_lr=1e-5,
        head_lr=3e-4,
        weight_decay=0.01,
        optimizer="AdamW",
        gradient_clip=1.0,
        patience=3,
        selection="validation_macro_f1",
        threshold=0.5,
        bn_running_stats="all_fixed",
    )
    if (
        any(control_config.get(k) != v for k, v in matched.items())
        or control_config["training_class_balance"]["counts"] != balance["counts"]
        or control_config["training_class_balance"]["class_weights"]
        != balance["class_weights"]
        or control_config.get("initialization") != "imagenet_b0_v1_random_gru_head"
    ):
        raise ValueError(
            "ImageNet başlangıcı, preprocessing, roster veya referans hyperparameter uyuşmuyor."
        )
    reference_eval = read_json(
        reference_dir / f"epoch_{best['epoch']:03d}_coverage.json"
    )["validation"]
    if reference_eval["metrics"] != best["validation_metrics"]:
        raise ValueError("Referans history/validation uyuşmuyor.")
    cached = args.backbone_mode == "frozen"
    config = dict(
        **matched,
        architecture="EfficientNet-B0 masked feature mean Linear1280->1",
        backbone_mode=args.backbone_mode,
        stage="features[8]_and_linear_head"
        if not cached
        else "frozen_backbone_linear_head",
        initialization="imagenet_b0_v1_fresh_linear_head",
        sampling="natural_without_replacement",
        loss="accepted_train_class_weighted_video_bce",
        training_class_balance=balance,
        input="frozen_imagenet_features" if cached else "fresh_normalized_face_images",
        feature_cache_used_as_input=cached,
        cache_dir=str(args.cache_dir.resolve()),
        maximum_epochs=3,
        optimizer_steps_per_epoch=2223,
        reference=dict(
            checkpoint=str(reference_path),
            sha256=file_sha256(reference_path),
            epoch=best["epoch"],
            validation=summarize_evaluation(reference_eval),
        ),
        comparison_scope="aggregation_architecture_and_capacity_changed_same_partial_backbone"
        if not cached
        else "diagnostic_frozen_representation_not_direct_partial_GRU_ablation",
        device_type=resolve_device(args.device).type,
    )
    root = Path(manifest["dataset_root"])
    datasets = {
        name: FeatureDataset(audit["splits"][name]["accepted_records"], cache, root)
        if cached
        else VerifiedImageDataset(
            root, audit["splits"][name]["accepted_records"], settings, cache
        )
        for name in ("train", "val")
    }
    return manifest, model, audit, config, datasets


def validate_resume(saved, manifest, config):
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "scaler_state_dict",
        "rng_state",
        "early_stopping",
        "history",
        "epoch",
        "best_epoch",
        "best_macro_f1",
    }
    if (
        saved.get("pipeline") != PIPELINE
        or saved.get("class_names") != ["REAL", "FAKE"]
        or saved.get("manifest_fingerprint") != manifest["fingerprint"]
        or saved.get("config") != config
        or not required.issubset(saved)
    ):
        raise ValueError("Spatial resume checkpoint/config/split/state uyumsuz.")


def fixed_evaluation(model, datasets, device, cached):
    return {
        name: evaluate_video(
            model, make_video_loader(dataset, 1), device, cached, False
        )
        for name, dataset in datasets.items()
    }


def run(args):
    if not 1 <= args.epochs <= 3:
        raise ValueError("Spatial kontrollü bütçe en fazla 3 epoch.")
    out = args.run_dir.resolve()
    allowed = Path("outputs/video_spatial_v1").resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError("Ayrı video_spatial_v1 alt dizini gerekli; eski run korunur.")
    checkpoint_dir = out / "checkpoints"
    resume = select_resume(checkpoint_dir, args.resume)
    if args.prepare and args.resume:
        raise ValueError("Prepare resume veya epoch yapmaz.")
    if args.resume and resume is None:
        raise FileNotFoundError("Spatial resume checkpoint eksik.")
    if resume and resume.resolve().parent != checkpoint_dir:
        raise ValueError("Resume aynı spatial run/checkpoints içinde olmalı.")
    if resume is None and out.exists():
        raise FileExistsError("Mevcut çıktı korunuyor; yeni run-dir kullanın.")
    manifest, model, audit, config, datasets = prepare_context(args)
    device = resolve_device(args.device)
    if args.prepare and device.type != "cuda":
        raise ValueError("Prepare gerçek CUDA smoke gerektirir.")
    model.to(device)
    loaders = dict(
        train=make_video_loader(datasets["train"], 1, shuffle=True),
        val=make_video_loader(datasets["val"], 1),
    )
    cached = config["feature_cache_used_as_input"]
    weights = torch.tensor(
        [
            config["training_class_balance"]["class_weights"][n]
            for n in ("REAL", "FAKE")
        ],
        device=device,
    )
    optimizer = make_spatial_optimizer(model)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    if args.prepare:
        from src.video_spatial_smoke import cuda_smoke

        report = cuda_smoke(model, datasets, device, optimizer, scaler, weights, cached)
        out.mkdir(parents=True, exist_ok=False)
        write_json(out / "smoke.json", report)
        write_json(out / "config.json", config)
        write_json(out / "roster_audit.json", audit)
        print(json.dumps(report, ensure_ascii=False), flush=True)
        return
    history, best_epoch, best_score, start = [], 0, -1.0, 1
    stopper = EarlyStopping(patience=3)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        validate_resume(saved, manifest, config)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        restore_rng(saved["rng_state"], loaders["train"], loaders["val"])
        stopper = EarlyStopping(**saved["early_stopping"])
        history, best_epoch, best_score, start = (
            saved["history"],
            saved["best_epoch"],
            saved["best_macro_f1"],
            saved["epoch"] + 1,
        )
        atomic_copy(
            checkpoint_dir / f"epoch_{best_epoch:03d}.pth",
            checkpoint_dir / "best_video.pth",
        )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    write_json(out / "roster_audit.json", audit)
    last_completed = start - 1
    try:
        for epoch in range(start, args.epochs + 1):
            if stopper.bad_epochs >= 3:
                break
            destination = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            if destination.exists():
                raise FileExistsError("Mevcut spatial epoch korunuyor.")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            began = perf_counter()
            trained = train_epoch(
                model,
                loaders["train"],
                device,
                optimizer,
                scaler,
                cached,
                False,
                epoch,
                weights,
                track_video_diagnostics=True,
            )
            if trained["optimizer_steps"] != 2223:
                raise RuntimeError("Spatial epoch 2223 optimizer adımı gerektirir.")
            optimizer.zero_grad(set_to_none=True)
            evaluated = evaluate_video(model, loaders["val"], device, cached, False)
            for name, result in (("train", trained), ("val", evaluated)):
                result["coverage"] = coverage(
                    result["coverage"]["video_reports"]
                    + audit["splits"][name]["rejected_reports"]
                )
            summary = summarize_evaluation(evaluated)
            score = summary["macro_f1"]
            improved = score > best_score
            if improved:
                best_epoch, best_score = epoch, score
            stopped = stopper.update(score)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = dict(
                epoch=epoch,
                train_video_loss=trained["video_loss"],
                val_video_loss=evaluated["video_loss"],
                training_metrics=trained["metrics"],
                training_metrics_scope=trained["metrics_scope"],
                training_numerics=trained["numerics"],
                optimizer_steps=trained["optimizer_steps"],
                validation_metrics=evaluated["metrics"],
                validation_sources=summary["source_pairs"],
                **calculate_timing(
                    history,
                    perf_counter() - began,
                    0 if stopped else args.epochs - epoch,
                ),
            )
            if device.type == "cuda":
                row.update(
                    peak_cuda_allocated_mib=torch.cuda.max_memory_allocated(device)
                    / 1024**2,
                    peak_cuda_reserved_mib=torch.cuda.max_memory_reserved(device)
                    / 1024**2,
                )
            history.append(row)
            atomic_save(
                dict(
                    pipeline=PIPELINE,
                    class_names=["REAL", "FAKE"],
                    config=config,
                    manifest_fingerprint=manifest["fingerprint"],
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    scaler_state_dict=scaler.state_dict(),
                    rng_state=capture_rng(loaders["train"], loaders["val"]),
                    early_stopping=asdict(stopper),
                    history=history,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_macro_f1=best_score,
                ),
                destination,
            )
            last_completed = epoch
            if improved:
                atomic_copy(destination, checkpoint_dir / "best_video.pth")
            atomic_copy(destination, checkpoint_dir / "last_video.pth")
            write_json(out / "history.json", history)
            write_json(out / f"epoch_{epoch:03d}_evaluation.json", evaluated)
            print(
                json.dumps(
                    {k: v for k, v in row.items() if k != "training_numerics"},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if stopped:
                break
        best_path = checkpoint_dir / "best_video.pth"
        fixed_path = (
            out / f"best_fixed_epoch_{best_epoch:03d}_after_{len(history):03d}.json"
        )
        if not fixed_path.exists():
            best = torch.load(best_path, map_location="cpu", weights_only=True)
            model.load_state_dict(best["model_state_dict"])
            fixed = fixed_evaluation(model, datasets, device, cached)
            write_json(
                fixed_path,
                dict(
                    epoch=best_epoch,
                    checkpoint_sha256=file_sha256(best_path),
                    scope="fixed_best_checkpoint_no_optimizer_steps",
                    results=fixed,
                    summary={
                        name: summarize_evaluation(r) for name, r in fixed.items()
                    },
                    reference=config["reference"],
                    completed_training_epochs=len(history),
                ),
            )
        result = read_json(fixed_path)
        if result["checkpoint_sha256"] != file_sha256(best_path):
            raise ValueError(
                "Mevcut fixed spatial raporu farklı checkpoint için; korunuyor."
            )
        write_json(out / "best_fixed_evaluation.json", result)
    except (
        torch.cuda.OutOfMemoryError,
        KeyboardInterrupt,
        VideoNumericalError,
        CropDriftError,
    ) as error:
        optimizer.zero_grad(set_to_none=True)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        write_json(
            out / "interrupted.json",
            dict(
                reason=type(error).__name__,
                details=str(error),
                last_completed_epoch=last_completed,
                recovery="Aynı run-dir/config ile --resume auto; tamamlanmış eğitimde yalnız eksik fixed evaluation tekrar edilir.",
            ),
        )
        raise


def make_parser():
    parser = argparse.ArgumentParser(
        description="Spatial-only masked mean video baseline; explicit train/prepare."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-training", action="store_true")
    parser.add_argument(
        "--accept-filename-families", action="store_true", required=True
    )
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/video_identity_v1/manifest.json")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        default=Path("outputs/video_finetune_v1/identity_continue_20261012_210232"),
    )
    parser.add_argument(
        "--backbone-mode", choices=("frozen", "partial"), default="partial"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--resume", nargs="?", const="auto")
    return parser


if __name__ == "__main__":
    run(make_parser().parse_args())
