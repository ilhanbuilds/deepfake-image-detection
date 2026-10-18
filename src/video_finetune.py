"""Explicit, separate FP32 partial fine-tuning; prepare never runs an epoch."""

import argparse
import json
import math
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
from src.video_finetune_data import CropDriftError, VerifiedImageDataset
from src.video_finetune_model import PIPELINE, PartialVideoClassifier, make_optimizer
from src.video_manifest import DEFAULT_MANIFEST, load_manifest
from src.video_sampling import comparison_metrics, make_balanced_video_loader
from src.video_pipeline import (
    FacePipeline,
    FeatureCache,
    VideoSettings,
    file_sha256,
    make_video_loader,
)
from src.video_train import (
    VideoNumericalError,
    build_experiment_record,
    train_epoch,
    training_class_balance,
)

DEFAULT_INITIAL = Path(
    "outputs/video_v1/gru_v2_hidden64_fp32/checkpoints/best_video.pth"
)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def load_balanced_reference(args, manifest, config):
    """Require the same initialization, roster and optimization budget as control."""
    if (
        not args.fresh_head
        or not 1 <= args.epochs <= 3
        or manifest.get("schema") != "celeb_df_identity_components_v1"
        or config["training_class_balance"]["counts"] != {"REAL": 417, "FAKE": 1806}
        or args.reference_run is None
    ):
        raise ValueError(
            "Balanced karşılaştırma: identity manifest, fresh head, 2223 kabul edilmiş train, "
            "en fazla 3 epoch ve --reference-run gerekli."
        )
    reference_dir = args.reference_run.resolve()
    reference_config = json.loads(
        (reference_dir / "config.json").read_text(encoding="utf-8")
    )
    # The only changed factor is the balancing POLICY (sampler + loss together).
    differences = [
        key for key, value in reference_config.items() if config.get(key) != value
    ]
    if differences or reference_config.get("sampling", "natural") != "natural":
        raise ValueError(f"Referans ayarları eşleşmiyor: {differences}")
    checkpoint = reference_dir / "checkpoints/best_video.pth"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    validate_resume(saved, manifest, reference_config)
    history = json.loads((reference_dir / "history.json").read_text(encoding="utf-8"))
    selected = max(history, key=lambda row: row["validation_metrics"]["macro_f1"])
    if saved["epoch"] != saved["best_epoch"] or selected["epoch"] != saved["epoch"]:
        raise ValueError("Referans best checkpoint validation Macro-F1 ile eşleşmiyor.")
    if len(history) != 3 or [row["epoch"] for row in history] != [1, 2, 3]:
        raise ValueError("Referans aynı üç epoch karşılaştırma bütçesini içermeli.")
    evaluated = json.loads(
        (reference_dir / f"epoch_{saved['epoch']:03d}_coverage.json").read_text(
            encoding="utf-8"
        )
    )["validation"]
    if evaluated["metrics"] != selected["validation_metrics"]:
        raise ValueError("Referans validation metrikleri/history uyuşmuyor.")
    return dict(
        run_dir=str(reference_dir),
        checkpoint=str(checkpoint),
        checkpoint_sha256=file_sha256(checkpoint),
        epoch=saved["epoch"],
        balancing="natural_sampling_weighted_bce",
        validation=comparison_metrics(evaluated),
        training_metrics_scope="online_training_batches_not_fixed_evaluation",
    )


def make_finetune_loaders(datasets, audit, balanced):
    train = (
        make_balanced_video_loader(
            datasets["train"], audit["splits"]["train"]["accepted_records"]
        )
        if balanced
        else make_video_loader(datasets["train"], 1, shuffle=True)
    )
    return dict(train=train, val=make_video_loader(datasets["val"], 1))


def evaluate_balanced_best(run_dir, model, datasets, device, checkpoint_dir, reference):
    """Natural full rosters, fixed selected checkpoint; resume evaluation without training."""
    checkpoint = checkpoint_dir / "best_video.pth"
    checksum = file_sha256(checkpoint)
    destination = run_dir / "best_fixed_evaluation.json"
    if destination.exists():
        completed = json.loads(destination.read_text(encoding="utf-8"))
        if completed.get("checkpoint_sha256") != checksum:
            raise ValueError(
                "Mevcut fixed değerlendirme farklı checkpoint için; korunuyor."
            )
        return
    selected = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(selected["model_state_dict"])
    began = perf_counter()
    try:
        fixed = {
            name: evaluate_video(
                model, make_video_loader(datasets[name], 1), device, False, False
            )
            for name in ("train", "val")
        }
    except (torch.cuda.OutOfMemoryError, KeyboardInterrupt, CropDriftError) as error:
        write_json(
            run_dir / "evaluation_interrupted.json",
            dict(
                reason=type(error).__name__,
                recovery="Aynı komuta --resume auto ekleyin: eğitim tamamlandıysa yalnız fixed best evaluation yeniden çalışır.",
                optimizer_steps=0,
            ),
        )
        raise
    current = comparison_metrics(fixed["val"])
    write_json(
        destination,
        dict(
            scope="fixed_best_checkpoint_eval_no_optimizer_steps",
            epoch=selected["epoch"],
            checkpoint_sha256=checksum,
            reference=reference,
            results=fixed,
            comparison={
                name: comparison_metrics(value) for name, value in fixed.items()
            },
            validation_deltas={
                key: current[key] - reference["validation"][key]
                for key in (
                    "macro_f1",
                    "roc_auc",
                    "balanced_accuracy",
                    "accuracy",
                    "real_recall",
                    "fake_recall",
                    "celeb_real_recall",
                    "youtube_real_recall",
                )
            },
            evaluation_seconds=perf_counter() - began,
        ),
    )


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
        raise ValueError("Partial fine-tuning resume pipeline/config/state uyumsuz.")


def prepare_context(args):
    seed_everything(42)
    manifest = load_manifest(args.manifest)
    settings = VideoSettings()
    pipeline = FacePipeline(settings)
    model = PartialVideoClassifier(pretrained=True, hidden_size=64)
    initial_backbone = model.backbone_fingerprint()
    fresh = getattr(args, "fresh_head", False)
    if manifest.get("schema") == "celeb_df_identity_components_v1" and not fresh:
        raise ValueError(
            "Yeni identity split eski GRU/head ile başlatılamaz: yeni validation'a maruz kalmış ağırlıklar. --fresh-head kullanın."
        )
    initial = None
    if not fresh:
        initial = torch.load(
            args.initial_checkpoint, map_location="cpu", weights_only=True
        )
        if (
            initial.get("pipeline") != "video_v1_frozen_gru"
            or initial.get("manifest_fingerprint") != manifest["fingerprint"]
            or initial.get("class_names") != ["REAL", "FAKE"]
            or initial.get("config", {}).get("hidden_size") != 64
            or initial.get("config", {}).get("pipeline") != pipeline.signature
        ):
            raise ValueError(
                "Başlangıç yalnız aynı pipeline/split için frozen GRU hidden64 olmalı."
            )
        model.load_state_dict(initial["model_state_dict"])
        if model.backbone_fingerprint() != initial_backbone:
            raise ValueError(
                "Başlangıç backbone ImageNet B0 V1 ağırlıklarıyla aynı değil."
            )
    cache = FeatureCache(args.cache_dir, pipeline.signature, initial_backbone)
    audit = audit_feature_cache(manifest, cache)
    balance = training_class_balance(audit["splits"]["train"]["accepted_records"])
    balance["source"] = "accepted_train_crop_roster_verified_against_live_decode"
    config = dict(
        settings=asdict(settings),
        pipeline=pipeline.signature,
        hidden_size=64,
        batch_size=1,
        frames=6,
        precision="fp32",
        seed=42,
        workers=0,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=0.01,
        patience=3,
        selection="validation_macro_f1",
        threshold=0.5,
        stage="features[8]_and_gru_head",
        bn_running_stats="all_fixed",
        input="fresh_normalized_face_images",
        feature_cache_used_as_input=False,
        roster_cache_dir=str(args.cache_dir.resolve()),
        cache_fingerprint=audit["cache_fingerprint"],
        initial_checkpoint=str(args.initial_checkpoint.resolve()) if initial else None,
        initial_checkpoint_sha256=file_sha256(args.initial_checkpoint)
        if initial
        else None,
        initial_checkpoint_epoch=initial["epoch"] if initial else None,
        initial_backbone_sha256=initial_backbone,
        training_class_balance=balance,
        optimizer="AdamW",
        gradient_clip=1.0,
    )
    if fresh:
        config["initialization"] = "imagenet_b0_v1_random_gru_head"
    root = Path(manifest["dataset_root"])
    datasets = {
        name: VerifiedImageDataset(
            root, audit["splits"][name]["accepted_records"], settings, cache
        )
        for name in ("train", "val")
    }
    return manifest, model, pipeline, audit, balance, config, datasets


def run_training(args):
    run_dir = args.run_dir.resolve()
    allowed = Path("outputs/video_finetune_v1").resolve()
    if not run_dir.is_relative_to(allowed) or run_dir == allowed:
        raise ValueError(
            "Yeni fine-tuning run-dir outputs/video_finetune_v1 altında ayrı bir dizin olmalı."
        )
    checkpoint_dir = run_dir / "checkpoints"
    resume_path = select_resume(checkpoint_dir, args.resume)
    if args.resume and resume_path is None:
        raise FileNotFoundError("Tamamlanmış partial fine-tuning epoch bulunamadı.")
    if resume_path is not None and resume_path.resolve().parent != checkpoint_dir:
        raise ValueError(
            "Resume yalnız aynı fine-tuning run-dir/checkpoints içinde olabilir."
        )
    if resume_path is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Mevcut run korunuyor; yeni run-dir veya --resume auto kullanın."
        )
    manifest, model, _, audit, balance, config, datasets = prepare_context(args)
    device = resolve_device(args.device)
    config["device_type"] = device.type
    balanced = getattr(args, "sampling", "natural") == "balanced"
    reference = load_balanced_reference(args, manifest, config) if balanced else None
    if balanced:
        config.update(
            sampling="balanced_weighted_random_sampler",
            sampling_replacement=True,
            sampler_rng="shared_train_loader_generator_seed42_checkpointed",
            samples_per_epoch=2223,
            maximum_epochs=3,
            loss="unweighted_video_bce",
            loss_class_weights=None,
            reference=reference,
        )
    saved = (
        torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_path
        else None
    )
    if saved is not None:
        validate_resume(saved, manifest, config)
        model.load_state_dict(saved["model_state_dict"])
    model.to(device)
    loaders = make_finetune_loaders(datasets, audit, balanced)
    optimizer = make_optimizer(model, args.backbone_lr, args.head_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    weights = (
        None
        if balanced
        else torch.tensor(
            [balance["class_weights"][name] for name in ("REAL", "FAKE")], device=device
        )
    )
    stopper, history, best_epoch, best_f1, start = (
        EarlyStopping(patience=3),
        [],
        0,
        -1.0,
        1,
    )
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        stopper = EarlyStopping(**saved["early_stopping"])
        history, best_epoch, best_f1, start = (
            saved["history"],
            saved["best_epoch"],
            saved["best_macro_f1"],
            saved["epoch"] + 1,
        )
        restore_rng(saved["rng_state"], loaders["train"], loaders["val"])
        selected = checkpoint_dir / f"epoch_{best_epoch:03d}.pth"
        if not selected.is_file():
            raise FileNotFoundError("Best partial epoch checkpoint eksik.")
        if stopper.bad_epochs >= 3 or start > args.epochs:
            if balanced:
                evaluate_balanced_best(
                    run_dir, model, datasets, device, checkpoint_dir, reference
                )
            print("Tamamlanmış run; yeni epoch başlamadı.")
            return
        atomic_copy(selected, checkpoint_dir / "best_video.pth")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "roster_audit.json", audit)
    write_json(run_dir / "config.json", config)
    last_completed = start - 1
    try:
        for epoch in range(start, args.epochs + 1):
            destination = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            if destination.exists():
                raise FileExistsError("Mevcut epoch korunuyor; --resume auto kullanın.")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            began = perf_counter()
            trained = train_epoch(
                model,
                loaders["train"],
                device,
                optimizer,
                scaler,
                False,
                False,
                epoch,
                weights,
                track_video_diagnostics=True,
            )
            if balanced and trained["optimizer_steps"] != 2223:
                raise RuntimeError(
                    "Balanced epoch tam 2223 optimizer adımı gerektirir."
                )
            optimizer.zero_grad(set_to_none=True)
            evaluated = evaluate_video(model, loaders["val"], device, False, False)
            for name, result in (("train", trained), ("val", evaluated)):
                result["coverage"] = coverage(
                    result["coverage"]["video_reports"]
                    + audit["splits"][name]["rejected_reports"]
                )
            if evaluated["metrics"] is None or evaluated["metrics"]["roc_auc"] is None:
                raise ValueError("Validation iki sınıfı içermiyor.")
            score = evaluated["metrics"]["macro_f1"]
            improved = score > best_f1
            if improved:
                best_epoch, best_f1 = epoch, score
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
                training_sampling=trained["sampling"],
                optimizer_steps=trained["optimizer_steps"],
                validation_metrics=evaluated["metrics"],
                validation_sources=comparison_metrics(evaluated)["sources"],
                train_evaluated=trained["coverage"]["evaluated_videos"],
                val_evaluated=evaluated["coverage"]["evaluated_videos"],
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
                    best_macro_f1=best_f1,
                    best_epoch=best_epoch,
                    epoch=epoch,
                ),
                destination,
            )
            last_completed = epoch
            if improved:
                atomic_copy(destination, checkpoint_dir / "best_video.pth")
            atomic_copy(destination, checkpoint_dir / "last_video.pth")
            write_json(
                run_dir / f"epoch_{epoch:03d}_coverage.json",
                dict(train=trained["coverage"], validation=evaluated),
            )
            write_json(run_dir / "history.json", history)
            if balanced:
                current = comparison_metrics(evaluated)
                write_json(
                    run_dir / "comparison.json",
                    dict(
                        reference=reference,
                        current_epoch=epoch,
                        current_validation=current,
                        best_epoch=best_epoch,
                        best_macro_f1=best_f1,
                        deltas={
                            key: current[key] - reference["validation"][key]
                            for key in (
                                "macro_f1",
                                "roc_auc",
                                "balanced_accuracy",
                                "accuracy",
                                "real_recall",
                                "fake_recall",
                                "celeb_real_recall",
                                "youtube_real_recall",
                            )
                        },
                    ),
                )
            # Full per-video counters remain in JSON; keep console output readable.
            console = {
                k: v
                for k, v in row.items()
                if k not in {"training_numerics", "training_sampling"}
            }
            console["clipping_by_class"] = trained["numerics"]["by_class"]
            console["sampling_by_class"] = {
                name: {k: v for k, v in values.items() if k != "draws_per_video"}
                for name, values in trained["sampling"]["by_class"].items()
            }
            print(json.dumps(console, ensure_ascii=False), flush=True)
            if stopped:
                break
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
            run_dir / "interrupted.json",
            dict(
                reason=type(error).__name__,
                details=str(error),
                last_completed_epoch=last_completed,
                recovery="Tamamlanmayan epoch tekrar çalıştırılır: aynı run-dir --resume auto. Crop drift çözülmeden devam etme. Frozen feature cache ile fine-tune yapılamaz.",
            ),
        )
        raise
    record = build_experiment_record(
        run_dir,
        manifest,
        config,
        audit,
        history,
        best_epoch,
        "completed",
        "Yalnız son feature bloğu ve GRU/head; resmi test kullanılmadı.",
    )
    record.update(
        id=record["id"].replace("video-v1", "video-partial-v1"),
        name="Video kısmi fine-tuning V1",
        model="ImageNet EfficientNet-B0 features[8] + GRU64",
        objective="Gerçek yüz görüntüleri; frozen prefix, sabit BN, son blok + GRU/head. Validation Macro-F1 seçimi.",
    )
    record["observations"] = [
        "Eski cache yalnız kabul/ret roster ve crop tutarlılık denetiminde; model girdisi canlı decode edilen görüntülerdir.",
        "Train loss unweighted ve sampling balanced; validation doğal roster."
        if balanced
        else "Train loss class-weighted; validation loss unweighted. Train metrikleri online batch tahminleridir.",
        "Filename aileleri splitler arasında ayrı; gerçek identity/donor ayrımı doğrulanmadı.",
    ]
    record["files"] = [
        str(run_dir / name)
        for name in ("history.json", "roster_audit.json", "config.json")
    ]
    write_json(run_dir / "experiment_record.json", record)
    if balanced:
        evaluate_balanced_best(
            run_dir, model, datasets, device, checkpoint_dir, reference
        )


def make_parser():
    parser = argparse.ArgumentParser(
        description="Ayrı kısmi EfficientNet B0 + GRU64; FP32, batch1, 6 kare, patience3."
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--prepare",
        action="store_true",
        help="20+20 crop kalite raporu ve küçük CUDA smoke; tam epoch yok.",
    )
    modes.add_argument("--run-training", action="store_true")
    modes.add_argument(
        "--balanced-smoke",
        action="store_true",
        help="Yalnız sekiz gerçek CUDA optimizer adımı; checkpoint/tam epoch yok.",
    )
    parser.add_argument(
        "--accept-filename-families", action="store_true", required=True
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument(
        "--fresh-head",
        action="store_true",
        help="ImageNet backbone + sıfırdan GRU/head. Yeni identity split için zorunlu; eski learned video ağırlıkları yüklenmez.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument(
        "--run-dir", type=Path, default=Path("outputs/video_finetune_v1/benchmark_fp32")
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", nargs="?", const="auto")
    parser.add_argument(
        "--sampling",
        choices=("natural", "balanced"),
        default="natural",
        help="Balanced: kabul edilmiş train'den replacement; UNWEIGHTED video BCE.",
    )
    parser.add_argument(
        "--reference-run",
        type=Path,
        help="Balanced deneyinin değişmeyen ayarlarını ve validation karşılaştırmasını doğrular.",
    )
    return parser


def main():
    args = make_parser().parse_args()
    if not 1 <= args.epochs <= 999 or any(
        not math.isfinite(v) or v <= 0 for v in (args.backbone_lr, args.head_lr)
    ):
        raise ValueError("Epoch 1–999 ve pozitif sonlu LR gerekli.")
    if args.balanced_smoke:
        from src.video_balanced_smoke import run_balanced_smoke

        run_balanced_smoke(args)
    elif args.prepare:
        if args.sampling == "balanced":
            raise ValueError("Balanced hazırlık için --balanced-smoke kullanın.")
        if args.resume:
            raise ValueError("Hazırlık modu resume veya tam eğitim yapmaz.")
        from src.video_crop_quality import prepare_quality_and_smoke

        prepare_quality_and_smoke(args)
    else:
        run_training(args)


if __name__ == "__main__":
    main()
