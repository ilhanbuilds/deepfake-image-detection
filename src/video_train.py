"""Frozen cache GRU: FP32 by default, opt-in AMP, checked video-level updates."""

import argparse
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

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
from src.video_evaluate import calculate_video_metrics, coverage, evaluate_video
from src.video_manifest import DEFAULT_MANIFEST, digest_json, load_manifest
from src.video_model import VideoClassifier
from src.video_sampling import gradient_summary, sampling_summary
from src.video_pipeline import (
    FacePipeline,
    FeatureCache,
    FeatureDataset,
    VideoSettings,
    feature_batch,
    make_video_loader,
)


def training_class_balance(records: list[dict]) -> dict:
    if any(record["label"] not in (0, 1) for record in records):
        raise ValueError("Train video etiketi 0/1 olmalı.")
    counts = {
        name: sum(record["label"] == label for record in records)
        for label, name in enumerate(("REAL", "FAKE"))
    }
    if not all(counts.values()):
        raise ValueError("Class-weighted loss için train iki sınıfı içermeli.")
    total = sum(counts.values())
    return dict(
        counts=counts,
        fractions={name: count / total for name, count in counts.items()},
        class_weights={name: total / (2 * count) for name, count in counts.items()},
        formula="N_train / (2 * N_train_class)",
        source="accepted_train_cache_only",
    )


def weighted_video_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != labels.shape or logits.ndim != 1:
        raise ValueError("Her video için bir logit ve bir etiket gerekli.")
    if not torch.isfinite(logits).all() or not torch.isfinite(labels).all():
        raise ValueError("Video logits/etiket NaN/Inf içeriyor.")
    if torch.any((labels != 0) & (labels != 1)):
        raise ValueError("Video etiketleri 0/1 olmalı.")
    weights = None
    if class_weights is not None:
        if (
            class_weights.shape != (2,)
            or not torch.isfinite(class_weights).all()
            or torch.any(class_weights <= 0)
        ):
            raise ValueError("REAL/FAKE loss ağırlıkları pozitif ve sonlu olmalı.")
        weights = class_weights.to(device=labels.device, dtype=torch.float32)[
            labels.long()
        ]
    # FP32 BCE stays stable even when called inside a CUDA autocast context.
    with torch.amp.autocast(logits.device.type, enabled=False):
        return nn.functional.binary_cross_entropy_with_logits(
            logits.float(), labels.float(), weight=weights
        )


class VideoNumericalError(RuntimeError):
    def __init__(self, stage: str, details: dict):
        self.details = dict(stage=stage, **details)
        super().__init__(
            "Video NaN/Inf; optimizer adımı uygulanmadı. "
            + json.dumps(self.details, ensure_ascii=False)
            + " AMP kullanıyorsanız yeni run-dir ile --precision fp32 kullanın."
        )


def checked_clip_gradients(parameters, details):
    parameters = [p for p in parameters if p.requires_grad and p.grad is not None]
    try:
        # PyTorch returns the TOTAL NORM BEFORE modifying the gradients.
        before = nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        after = torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(p.grad.detach().float()) for p in parameters]
            )
        )
    except RuntimeError as error:
        raise VideoNumericalError("gradient_norm", details) from error
    if not torch.isfinite(after) or after.item() > 1.0001:
        raise VideoNumericalError(
            "gradient_norm_after_clip", dict(**details, post_clip_norm=after.item())
        )
    return before.item(), after.item()


def train_video_batch(
    model, batch, device, optimizer, scaler, cached, amp, class_weights=None
) -> dict:
    """Exactly one checked update; never clip or step non-finite gradients."""
    optimizer.zero_grad(set_to_none=True)
    features, lengths = feature_batch(model, batch, device, cached, amp)
    details = dict(
        paths=[item["record"]["path"] for item in batch["items"]],
        lengths=lengths.tolist(),
        precision="amp" if amp else "fp32",
        loss_scale=scaler.get_scale(),
    )
    if not torch.isfinite(features).all():
        raise VideoNumericalError("features", details)
    with torch.amp.autocast(device.type, enabled=amp):
        logits = model(features, lengths)
    if not torch.isfinite(logits).all():
        raise VideoNumericalError("logits", details)
    loss = weighted_video_loss(logits, batch["labels"].to(device), class_weights)
    details.update(logits=logits.detach().float().cpu().tolist(), loss=loss.item())
    if not torch.isfinite(loss):
        raise VideoNumericalError("loss", details)
    scaler.scale(loss).backward()
    # Required order: scaled backward -> unscale -> finite check -> clip -> step.
    scaler.unscale_(optimizer)
    invalid = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and parameter.grad is not None
        and not torch.isfinite(parameter.grad).all()
    ]
    if invalid:
        optimizer.zero_grad(set_to_none=True)
        raise VideoNumericalError(
            "gradients", dict(**details, invalid_parameters=invalid)
        )
    try:
        before, after = checked_clip_gradients(model.parameters(), details)
    except VideoNumericalError:
        optimizer.zero_grad(set_to_none=True)
        raise
    scaler.step(optimizer)
    scaler.update()
    return dict(
        loss=loss.item(),
        logits=logits.detach().float(),
        gradient_norm=before,
        gradient_norm_pre_clip=before,
        gradient_norm_post_clip=after,
        paths=details["paths"],
        loss_scale=details["loss_scale"],
        lengths=details["lengths"],
        finite=True,
    )


def validate_resume(saved: dict, manifest: dict, config: dict) -> None:
    if (
        saved.get("pipeline") != "video_v1_frozen_gru"
        or saved.get("class_names") != ["REAL", "FAKE"]
        or saved.get("manifest_fingerprint") != manifest["fingerprint"]
        or saved.get("config") != config
    ):
        raise ValueError("Video resume pipeline/dataset/config uyumsuz.")
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
    if not required.issubset(saved):
        raise ValueError("Video resume durumu eksik.")


def train_epoch(
    model,
    loader,
    device,
    optimizer,
    scaler,
    cached: bool,
    amp: bool,
    epoch: int,
    class_weights: torch.Tensor | None = None,
    track_video_diagnostics: bool = False,
) -> dict:
    if track_video_diagnostics and loader.batch_size != 1:
        raise ValueError("Sınıf bazlı clipping denetimi batch size 1 gerektirir.")
    model.train()
    loss_sum, count = 0.0, 0
    reports, labels_seen = [], set()
    targets, predictions, scores = [], [], []
    max_gradient_norm = 0.0
    max_post_clip_norm = 0.0
    peak_gradient_batch = None
    draws, gradient_rows = [], []
    optimizer_steps = 0
    for index, batch in enumerate(loader, 1):
        reports.extend(batch["reports"])
        if not batch["items"]:
            continue
        result = train_video_batch(
            model, batch, device, optimizer, scaler, cached, amp, class_weights
        )
        optimizer_steps += 1
        if track_video_diagnostics:
            record = batch["items"][0]["record"]
            draws.append(record)
            gradient_rows.append(
                dict(
                    path=record["path"],
                    label=record["label"],
                    pre_clip=result["gradient_norm_pre_clip"],
                    post_clip=result["gradient_norm_post_clip"],
                )
            )
        if result["gradient_norm"] > max_gradient_norm:
            max_gradient_norm = result["gradient_norm"]
            peak_gradient_batch = dict(
                paths=result["paths"],
                lengths=result["lengths"],
                loss=result["loss"],
                logits=result["logits"].cpu().tolist(),
                labels=batch["labels"].tolist(),
            )
        max_post_clip_norm = max(max_post_clip_norm, result["gradient_norm_post_clip"])
        probabilities = result["logits"].sigmoid().cpu().tolist()
        targets.extend(int(label) for label in batch["labels"].tolist())
        scores.extend(probabilities)
        predictions.extend(int(score >= 0.5) for score in probabilities)
        loss_sum += result["loss"] * len(batch["items"])
        count += len(batch["items"])
        labels_seen.update(batch["labels"].tolist())
        if index % 50 == 0:
            print(
                f"Epoch {epoch}: batch {index}/{len(loader)}, video loss {loss_sum / count:.5f}",
                flush=True,
            )
    if labels_seen != {0, 1}:
        raise ValueError("Yüz/decode reddinden sonra train iki sınıfı içermiyor.")
    output = dict(
        video_loss=loss_sum / count,
        coverage=coverage(reports),
        metrics=calculate_video_metrics(targets, predictions, scores),
        metrics_scope="online_training_batches",
        optimizer_steps=optimizer_steps,
        numerics=dict(
            precision="amp" if amp else "fp32",
            max_gradient_norm=max_gradient_norm,
            max_gradient_norm_pre_clip=max_gradient_norm,
            max_gradient_norm_post_clip=max_post_clip_norm,
            gradient_clip_max_norm=1.0,
            gradient_norm_scope="before_clipping",
            peak_gradient_batch=peak_gradient_batch,
            final_loss_scale=scaler.get_scale(),
            finite_checks_passed=True,
        ),
    )
    if track_video_diagnostics:
        records = getattr(loader.dataset, "records", None)
        if records is None:
            records = [item["record"] for item in loader.dataset]
        output["sampling"] = sampling_summary(records, draws)
        output["numerics"]["by_class"] = gradient_summary(gradient_rows)
        output["numerics"]["video_gradient_norms"] = gradient_rows
    return output


def build_experiment_record(
    run_dir: Path,
    manifest: dict,
    config: dict,
    cache_audit: dict,
    history: list[dict],
    best_epoch: int,
    status: str,
    conclusion: str,
) -> dict:
    best_row = next((row for row in history if row["epoch"] == best_epoch), None)
    return dict(
        id=f"video-v1-{digest_json(str(run_dir))[:16]}-{len(history):03d}",
        name="Video Modeli V1 frozen GRU",
        date=datetime.now(timezone(timedelta(hours=3))).isoformat(timespec="seconds"),
        model="ImageNet EfficientNet-B0 frozen + GRU",
        status=status,
        checkpoint=str(run_dir / "checkpoints/best_video.pth") if best_row else None,
        dataset=manifest["dataset_root"],
        counts={
            **{
                name: {
                    k: v
                    for k, v in subset.items()
                    if k not in {"accepted_records", "rejected_reports"}
                }
                for name, subset in cache_audit["splits"].items()
            },
            "official_test": "erişilmedi",
        },
        objective="Kabul edilmiş video cache üzerinde GRU/head; yalnız validation macro-F1 ile model seçimi.",
        settings=config,
        results=dict(
            history=history,
            best_epoch=best_epoch if best_row else None,
            best_validation_metrics=best_row["validation_metrics"]
            if best_row
            else None,
            selection_metric="validation_macro_f1",
        ),
        duration_seconds=sum(row["epoch_duration"] for row in history),
        observations=[
            "Train metrikleri epoch içindeki online batch tahminleridir; validation sabit epoch sonu modelinden hesaplanır.",
            "Reddedilen cache kayıtları loss ve metriklere girmez; kapsam ayrı loglanır.",
            "Kaynak aileleri filename çıkarımıdır; identity-disjoint değildir. Resmi test model seçiminde kullanılmadı.",
        ],
        conclusion=conclusion,
        files=[
            str(run_dir / filename)
            for filename in (
                "history.json",
                "cache_audit.json",
                "experiment_record.json",
            )
        ],
    )


def make_parser():
    parser = argparse.ArgumentParser(
        description="Video V1 frozen GRU; fine-tuning otomatik başlatılmaz."
    )
    parser.add_argument("--run-training", action="store_true", required=True)
    parser.add_argument(
        "--accept-filename-families",
        action="store_true",
        required=True,
        help="Metadata ile doğrulanmamış dosya adı aileleri ve identity overlap sınırını kabul eder.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--run-dir", type=Path, default=Path("outputs/video_v1/training")
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("outputs/video_v1/cache")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--image-size", type=int, choices=(160, 192), default=160)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("fp32", "amp"),
        default="fp32",
        help="Frozen cache GRU için güvenilir varsayılan FP32; AMP isteğe bağlıdır.",
    )
    parser.add_argument("--resume", nargs="?", const="auto")
    return parser


def run_training(args) -> None:
    run_started = perf_counter()
    if (
        not 1 <= args.epochs <= 999
        or args.hidden_size < 1
        or args.patience < 1
        or not math.isfinite(args.learning_rate)
        or args.learning_rate <= 0
    ):
        raise ValueError(
            "Epoch 1–999; hidden-size/patience/LR pozitif ve sonlu olmalı."
        )
    seed_everything(42)
    manifest = load_manifest(args.manifest)
    settings = VideoSettings(image_size=args.image_size)
    pipeline = FacePipeline(settings)
    device = resolve_device(args.device)
    precision = getattr(args, "precision", "fp32")
    if precision not in {"fp32", "amp"} or (
        precision == "amp" and device.type != "cuda"
    ):
        raise ValueError("Precision fp32 veya CUDA amp olmalı.")
    amp = precision == "amp"
    run_dir = args.run_dir.resolve()
    checkpoint_dir = run_dir / "checkpoints"
    resume_path = select_resume(checkpoint_dir, args.resume)
    if args.resume and resume_path is None:
        raise FileNotFoundError("Resume için tamamlanmış video epoch bulunamadı.")
    if resume_path is None and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Mevcut çıktı korunuyor; yeni --run-dir veya --resume auto kullanın."
        )
    if resume_path is not None and resume_path.resolve().parent != checkpoint_dir:
        raise ValueError("Resume aynı video run-dir/checkpoints içinde olmalı.")
    if args.cache_dir is None:
        raise ValueError("Video GRU eğitimi için tamamlanmış feature cache gerekli.")
    saved = (
        torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_path
        else None
    )
    if saved is not None and (
        saved.get("pipeline") != "video_v1_frozen_gru"
        or saved.get("class_names") != ["REAL", "FAKE"]
    ):
        raise ValueError("Video resume checkpoint gerekli.")
    model = VideoClassifier(pretrained=saved is None, hidden_size=args.hidden_size).to(
        device
    )
    if saved is not None:
        model.load_state_dict(saved["model_state_dict"])
    root = Path(manifest["dataset_root"])
    cache = FeatureCache(
        args.cache_dir, pipeline.signature, model.backbone_fingerprint()
    )
    cache_audit = audit_feature_cache(manifest, cache)
    balance = training_class_balance(cache_audit["splits"]["train"]["accepted_records"])
    config = dict(
        settings=asdict(settings),
        pipeline=pipeline.signature,
        hidden_size=args.hidden_size,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        seed=42,
        workers=0,
        amp=amp,
        precision=precision,
        bce_precision="fp32",
        grad_scaler_init_scale=1024.0 if amp else None,
        device_type=device.type,
        threshold=0.5,
        stage="frozen_backbone",
        selection="validation_macro_F1",
        optimizer="AdamW",
        cache_dir=str(args.cache_dir.resolve()),
        cache_fingerprint=cache_audit["cache_fingerprint"],
        loss="class_weighted_video_bce_v1",
        training_class_balance=balance,
    )
    if resume_path is not None:
        validate_resume(saved, manifest, config)
    # Official test and quarantine records are NEVER used to construct training loaders.
    datasets = {
        name: FeatureDataset(
            cache_audit["splits"][name]["accepted_records"], cache, root
        )
        for name in ("train", "val")
    }
    loaders = {
        name: make_video_loader(dataset, args.batch_size, shuffle=name == "train")
        for name, dataset in datasets.items()
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp, init_scale=1024.0)
    class_weights = torch.tensor(
        [balance["class_weights"][name] for name in ("REAL", "FAKE")],
        device=device,
        dtype=torch.float32,
    )
    stopper = EarlyStopping(patience=args.patience)
    history, best_macro_f1, best_epoch, start = [], -1.0, 0, 1
    best_path = checkpoint_dir / "best_video.pth"
    if saved is not None:
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scaler.load_state_dict(saved["scaler_state_dict"])
        stopper = EarlyStopping(**saved["early_stopping"])
        history, best_macro_f1, best_epoch, start = (
            saved["history"],
            saved["best_macro_f1"],
            saved["best_epoch"],
            saved["epoch"] + 1,
        )
        restore_rng(saved["rng_state"], loaders["train"], loaders["val"])
        # Recover interrupted best/last alias writes from immutable epoch checkpoints.
        selected = checkpoint_dir / f"epoch_{best_epoch:03d}.pth"
        if not selected.is_file():
            raise FileNotFoundError("Validation best epoch checkpoint eksik.")
        if saved["epoch"] == best_epoch:
            atomic_copy(resume_path, best_path)
        elif not best_path.is_file():
            atomic_copy(selected, best_path)
        del saved
        if stopper.bad_epochs >= stopper.patience or start > args.epochs:
            print("Run tamamlanmış; yeni epoch başlatılmadı.")
            return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "cache_audit.json").write_text(
        json.dumps(cache_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Run: {run_dir}; AMP={amp}; cache={bool(cache)}; resmi test kullanılmaz.",
        flush=True,
    )
    print(
        json.dumps(dict(training_class_balance=balance), ensure_ascii=False), flush=True
    )
    last_completed_epoch = start - 1
    preparation_seconds = perf_counter() - run_started
    try:
        for epoch in range(start, args.epochs + 1):
            epoch_path = checkpoint_dir / f"epoch_{epoch:03d}.pth"
            if epoch_path.exists():
                raise FileExistsError("Yeni epoch korunuyor; --resume auto kullanın.")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            started = perf_counter()
            trained = train_epoch(
                model,
                loaders["train"],
                device,
                optimizer,
                scaler,
                bool(cache),
                amp,
                epoch,
                class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            evaluated = evaluate_video(model, loaders["val"], device, bool(cache), amp)
            for name, result in (("train", trained), ("val", evaluated)):
                result["coverage"] = coverage(
                    result["coverage"]["video_reports"]
                    + cache_audit["splits"][name]["rejected_reports"]
                )
            if evaluated["metrics"] is None or evaluated["metrics"]["roc_auc"] is None:
                raise ValueError(
                    "Yüz/decode reddinden sonra validation iki sınıfı içermiyor."
                )
            macro_f1 = evaluated["metrics"]["macro_f1"]
            improved = macro_f1 > best_macro_f1
            if improved:
                best_macro_f1, best_epoch = macro_f1, epoch
            stopped = stopper.update(macro_f1)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            row = dict(
                epoch=epoch,
                train_video_loss=trained["video_loss"],
                val_video_loss=evaluated["video_loss"],
                validation_metrics=evaluated["metrics"],
                training_metrics=trained["metrics"],
                training_metrics_scope=trained["metrics_scope"],
                training_numerics=trained.get("numerics"),
                train_evaluated=trained["coverage"]["evaluated_videos"],
                val_evaluated=evaluated["coverage"]["evaluated_videos"],
                **calculate_timing(
                    history,
                    perf_counter() - started,
                    0 if stopped else args.epochs - epoch,
                ),
            )
            if device.type == "cuda":
                row["peak_cuda_allocated_mib"] = (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                )
                row["peak_cuda_reserved_mib"] = (
                    torch.cuda.max_memory_reserved(device) / 1024**2
                )
            history.append(row)
            state = dict(
                pipeline="video_v1_frozen_gru",
                class_names=["REAL", "FAKE"],
                config=config,
                manifest_fingerprint=manifest["fingerprint"],
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                scaler_state_dict=scaler.state_dict(),
                rng_state=capture_rng(loaders["train"], loaders["val"]),
                early_stopping=asdict(stopper),
                history=history,
                best_macro_f1=best_macro_f1,
                best_epoch=best_epoch,
                epoch=epoch,
            )
            atomic_save(state, epoch_path)
            last_completed_epoch = epoch
            if improved:
                atomic_copy(epoch_path, best_path)
            atomic_copy(epoch_path, checkpoint_dir / "last_video.pth")
            (run_dir / f"epoch_{epoch:03d}_coverage.json").write_text(
                json.dumps(
                    dict(train=trained["coverage"], validation=evaluated),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if stopped:
                print("Early stopping: validation macro-F1 iyileşmedi.")
                break
    except (
        torch.cuda.OutOfMemoryError,
        KeyboardInterrupt,
        VideoNumericalError,
    ) as error:
        optimizer.zero_grad(set_to_none=True)
        if isinstance(error, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache()
        if isinstance(error, VideoNumericalError):
            (run_dir / "numerical_error.json").write_text(
                json.dumps(error.details, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (run_dir / "interrupted.json").write_text(
            json.dumps(
                dict(
                    reason=type(error).__name__,
                    last_completed_epoch=last_completed_epoch,
                    recovery="Tamamlanmamış epoch atılır; aynı run-dir ile --resume auto. OOM için yeni run-dir, batch 1 veya feature cache.",
                ),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        raise RuntimeError(
            "Video eğitimi kesildi; tamamlanan epoch'lar korunur. interrupted.json kaydını inceleyin."
        ) from error
    record = build_experiment_record(
        run_dir,
        manifest,
        config,
        cache_audit,
        history,
        best_epoch,
        "completed",
        "Frozen GRU run tamamlandı; resmi test ve fotoğraf modelleri kullanılmadı/değiştirilmedi.",
    )
    record["run_timing"] = dict(
        preparation_seconds=preparation_seconds,
        invocation_seconds=perf_counter() - run_started,
        python_import_time_included=False,
        resumed_from_epoch=start - 1,
    )
    (run_dir / "experiment_record.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"Validation best: {best_path}; resmi test değerlendirilmedi; fine-tuning başlatılmadı."
    )


def main() -> None:
    run_training(make_parser().parse_args())


if __name__ == "__main__":
    main()
