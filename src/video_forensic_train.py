"""Opt-in SBI adaptation with streamed frame backward and epoch-safe resume.

Uses weak video labels for equally weighted per-frame CE. This objective is not
the same as CE on a mean video score; validation always uses the latter mean.
No image V3 weights or historical output paths are writable through this CLI.
"""

import argparse
import copy
from importlib.metadata import version
import math
import os
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.dataset import seed_everything
from src.train import (
    EarlyStopping,
    atomic_copy,
    atomic_save,
    capture_rng,
    restore_rng,
    select_resume,
)
from src.video_forensic import (
    ROOT,
    SBIClassifier,
    SBIFaces,
    accepted_roster,
    frame_scores,
    score_faces,
    summarize,
    write_json,
)
from src.video_pipeline import file_sha256, make_video_loader
from src.video_manifest import digest_json
from src.video_train import training_class_balance

PIPELINE = "sbi_tail_adaptation_v1"


def configure_stage(model, epoch):
    model.requires_grad_(False)
    model.net._fc.requires_grad_(True)
    if epoch >= 2:
        for block in model.net._blocks[24:]:
            block.requires_grad_(True)
        model.net._conv_head.requires_grad_(True)
        model.net._bn1.requires_grad_(True)
    # Eval does not disable gradients. Fixed BN and disabled dropout/drop-connect
    # protect small batches and keep frozen layers deterministic.
    model.eval()


def optimizer_for(model):
    tail = [p for block in model.net._blocks[24:] for p in block.parameters()]
    tail += list(model.net._conv_head.parameters()) + list(model.net._bn1.parameters())
    return torch.optim.AdamW(
        [dict(params=tail, lr=1e-5), dict(params=model.net._fc.parameters(), lr=1e-4)],
        weight_decay=0.01,
    )


def train_group(model, optimizer, examples, weights, device):
    """One optimizer update, exact mean of video-balanced frame CE gradients."""
    if not examples:
        raise ValueError("Empty accumulation group.")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for pixels, report, label in examples:
        if report["status"] != "accepted":
            raise ValueError("Rejected training video: stop, never silently skip.")
        selected = [
            max(
                frame["faces"],
                key=lambda f: (
                    (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1])
                ),
            )["crop_index"]
            for frame in report["frames"]
        ]
        loss_value = 0.0
        for index in selected:
            image = (
                pixels[index : index + 1].to(device=device, dtype=torch.float32) / 255
            )
            logits = model(image)
            loss = nn.functional.cross_entropy(
                logits, torch.tensor([label], device=device)
            )
            loss = loss * weights[label] / len(selected) / len(examples)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite frame loss; update aborted.")
            loss.backward()
            loss_value += float(loss.detach()) * len(examples)
        losses.append(loss_value)
    norm = nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return dict(
        loss=float(np.mean(losses)), gradient_norm=float(norm), videos=len(examples)
    )


def validate_state(saved, config):
    required = {
        "model_state_dict",
        "optimizer_state_dict",
        "rng_state",
        "history",
        "epoch",
        "early_stopping",
        "best_epoch",
        "best_score",
        "validation",
    }
    if (
        saved.get("pipeline") != PIPELINE
        or saved.get("config") != config
        or not required.issubset(saved)
    ):
        raise ValueError("Resume pipeline/config/data/state mismatch.")
    if config.get("frames") == 12:
        history = saved["history"]
        epoch = saved["epoch"]
        if (
            not isinstance(epoch, int)
            or not 1 <= epoch <= config["epochs"]
            or [h["epoch"] for h in history] != list(range(1, epoch + 1))
            or saved["validation"]["metrics"] != history[-1]["validation_metrics"]
        ):
            raise ValueError("Resume epoch/history/validation mismatch.")
        stopper = EarlyStopping(patience=config["patience"])
        for h in history:
            if stopper.bad_epochs >= stopper.patience:
                raise ValueError("Checkpoint continued past early stopping.")
            stopper.update(h["validation_metrics"]["macro_f1"])
        best = max(history, key=lambda h: h["validation_metrics"]["macro_f1"])
        if (
            saved["early_stopping"] != asdict(stopper)
            or saved["best_epoch"] != best["epoch"]
            or saved["best_score"] != stopper.best_score
        ):
            raise ValueError("Resume best epoch/early stopping state mismatch.")


def prepare_config(args, manifest, audit, extractor):
    return dict(
        pipeline=PIPELINE,
        seed=args.seed,
        epochs=args.epochs,
        accumulation=4,
        frame_batch_size=1,
        precision="fp32",
        warmup_epochs=1,
        trainable_blocks="24: plus projection and fc from epoch 2",
        backbone_lr=1e-5,
        head_lr=1e-4,
        weight_decay=0.01,
        patience=3,
        manifest_fingerprint=manifest["fingerprint"],
        cache_fingerprint=audit["cache_fingerprint"],
        initial_sha256=file_sha256(args.checkpoint),
        preprocessing=extractor.signature,
        threshold=0.5,
        selection="validation_macro_f1",
        objective="class_weighted_mean_frame_CE_equal_video_weight",
        training_faces="largest bbox per valid frame",
        inference_faces="max fake score per valid frame",
        bn_dropout_dropconnect="eval throughout",
        class_names=["REAL", "FAKE"],
        torch=str(torch.__version__),
        numpy=str(np.__version__),
        device="cuda",
        implementation_sha256={
            str(path): file_sha256(path)
            for path in (
                Path("src/video_forensic.py"),
                Path(__file__),
                Path("src/video_cache.py"),
                Path("src/train.py"),
                Path("src/video_evaluate.py"),
            )
        },
        runtime_versions={
            name: version(name)
            for name in (
                "torch",
                "torchvision",
                "numpy",
                "efficientnet-pytorch",
                "retinaface-pytorch",
                "albumentations",
                "opencv-python-headless",
            )
        },
    )


def load_example(extractor, root, record, audit):
    pixels, report = extractor.read(root, record, audit)
    if report["status"] != "accepted":
        raise ValueError(
            f"Accepted cohort lost video: {record['path']}; {report['failures']}"
        )
    return pixels, report, record["label"]


def evaluate(model, extractor, root, records, audit):
    rows = []
    for record in records:
        pixels, report, _ = load_example(extractor, root, record, audit)
        scores = score_faces(model, pixels, "cuda")
        frames = frame_scores(scores, report)
        score = float(np.mean(frames))
        rows.append(
            dict(
                path=record["path"],
                target=record["label"],
                status="accepted",
                fake_probability=score,
                prediction=int(score >= 0.5),
                frame_scores=frames,
                report=report,
            )
        )
    return dict(**summarize(rows, len(records)), rows=rows)


def smoke(args, model, optimizer, extractor, root, train, val, audit, config):
    selected = [
        r
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for r in [r for r in train if r["path"].startswith(source + "/")][:2]
    ]
    decode_started = perf_counter()
    examples = [load_example(extractor, root, r, audit) for r in selected]
    if args.frames == 12 and any(len(e[1]["frames"]) != 12 for e in examples):
        raise ValueError("Smoke must exercise twelve actual frames per video.")
    decode_seconds = perf_counter() - decode_started
    # Two REAL and two FAKE videos, with naturally variable valid frame counts.
    pilot_examples = [examples[i] for i in (0, 2, 4, 5)]
    weights = torch.tensor(
        [training_class_balance(train)["class_weights"][n] for n in ("REAL", "FAKE")],
        device="cuda",
    )
    stages = []
    initial = copy.deepcopy(model.state_dict())
    for epoch in (1, 2):
        configure_stage(model, epoch)
        before = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = perf_counter()
        result = train_group(model, optimizer, pilot_examples, weights, "cuda")
        torch.cuda.synchronize()
        seconds = perf_counter() - began
        changed = [
            n
            for n, p in model.state_dict().items()
            if not torch.equal(p.cpu(), before[n])
        ]
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        if any(p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("Smoke parameters must remain FP32.")
        if not changed or any(n not in trainable for n in changed):
            raise RuntimeError("Frozen parameters or BN buffers changed.")
        if epoch == 2 and not any(n.startswith("net._blocks.") for n in changed):
            raise RuntimeError("No actual MBConv backward update.")
        stages.append(
            dict(
                epoch=epoch,
                trainable_parameters=sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                seconds=seconds,
                frames=sum(len(e[1]["frames"]) for e in pilot_examples),
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
                changed_parameters=changed,
                **result,
            )
        )
    # Round-trip a real local checkpoint and replay the same next CUDA update.
    # This file is diagnostic only and cannot pass validate_state as a run checkpoint.
    diagnostic_path = args.run_dir / "resume_probe.pth"
    train_loader = make_video_loader([], 1, seed=args.seed)
    val_loader = make_video_loader([], 1, seed=args.seed)
    atomic_save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng(train_loader, val_loader),
        ),
        diagnostic_path,
    )
    expected_order = torch.randperm(16, generator=train_loader.generator)
    first = train_group(model, optimizer, pilot_examples, weights, "cuda")
    expected = {n: p.detach().cpu().clone() for n, p in model.state_dict().items()}
    probe = torch.load(diagnostic_path, map_location="cpu", weights_only=True)
    model.load_state_dict(probe["model"])
    optimizer.load_state_dict(probe["optimizer"])
    restore_rng(probe["rng"], train_loader, val_loader)
    restored_order = torch.randperm(16, generator=train_loader.generator)
    second = train_group(model, optimizer, pilot_examples, weights, "cuda")
    if (
        not torch.equal(expected_order, restored_order)
        or first != second
        or any(
            not torch.equal(p.cpu(), expected[n]) for n, p in model.state_dict().items()
        )
    ):
        raise RuntimeError("CUDA optimizer/RNG checkpoint replay differed.")
    # Restore smoke state in memory; no pilot checkpoint is a training initializer.
    model.load_state_dict(initial)
    torch.cuda.synchronize()
    began = perf_counter()
    baseline_checks = []
    eval_examples = examples
    if args.frames == 12:
        eval_records = smoke_records(val, per_source=1)
        eval_examples = [load_example(extractor, root, r, audit) for r in eval_records]
        expected_rows = {r["path"]: r for r in extractor.baseline["rows"]}
    for pixels, report, _ in eval_examples:
        scores = score_faces(model, pixels, "cuda")
        if args.frames == 12:
            expected = expected_rows[report["path"]]
            actual = frame_scores(scores, report)
            delta = max(
                abs(a - b)
                for a, b in zip(actual, expected["frame_scores"], strict=True)
            )
            if delta > 1e-6:
                raise ValueError("12-frame zero-shot smoke score drift.")
            baseline_checks.append(
                dict(
                    path=report["path"],
                    max_frame_score_delta=delta,
                    crop_pixels_and_bboxes_equal=True,
                )
            )
    torch.cuda.synchronize()
    eval_seconds = perf_counter() - began
    seconds_per_train = stages[-1]["seconds"] / 4 + decode_seconds / len(examples)
    seconds_per_eval = eval_seconds / len(eval_examples)
    epoch_estimate = seconds_per_train * len(train) + seconds_per_eval * len(val)
    # Disk I/O and future decoding are deliberately separate from GPU estimate.
    return dict(
        config=config,
        stages=stages,
        sampled_train_videos=[r["path"] for r in selected],
        crop_preparation_seconds=decode_seconds,
        train_videos=len(train),
        validation_videos=len(val),
        cached_epoch_estimate_minutes=epoch_estimate / 60,
        full_budget_estimate_hours=epoch_estimate * args.epochs / 3600,
        validation_inference_seconds_per_video=seconds_per_eval,
        timing_scope="small bounded pilot; cached train reads included; 12-frame validation uses three cached validation videos; excludes initial crop extraction, preflight and checkpoint I/O",
        all_gradients_finite=True,
        frozen_and_bn_unchanged=True,
        training_epoch_started=False,
        optimizer_updates=4,
        checkpoint_replay_bitwise_equal=True,
        baseline_checks=baseline_checks,
        frame_batch_size=1,
        accumulation_videos=4,
        precision="fp32",
        device=torch.cuda.get_device_name(),
        model_restored=True,
        official_test_accessed=False,
    )


def smoke_records(records, per_source=2):
    return [
        r
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for r in [r for r in records if r["path"].startswith(source + "/")][:per_source]
    ]


def main():
    parser = argparse.ArgumentParser(
        description="SBI prepare/smoke or explicit long adaptation; no automatic training."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run-training", action="store_true")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "FFraw.tar")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, choices=(6, 12), default=6)
    parser.add_argument("--resume", choices=("auto",))
    args = parser.parse_args()
    if not 1 <= args.epochs <= 30 or (args.prepare and args.resume):
        raise ValueError("Invalid bounded epoch budget or prepare/resume combination.")
    out = args.run_dir.resolve()
    allowed = (ROOT / ("finetune12" if args.frames == 12 else "finetune")).resolve()
    if not out.is_relative_to(allowed) or out == allowed:
        raise ValueError(f"New directory under {allowed} required.")
    if out.exists() and not args.resume:
        raise FileExistsError("Existing run protected; use resume or a new directory.")
    from src.video_sbi12 import exclusive_lock

    with exclusive_lock(allowed / ("." + out.name + ".lock")):
        execute(args, out)


def execute(args, out):
    resume_path = select_resume(out / "checkpoints", args.resume)
    if args.resume and resume_path is None and args.frames == 6:
        raise FileNotFoundError("No completed epoch; existing outputs are protected.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(args.seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    manifest, audit, train = accepted_roster("train")
    _, _, val = accepted_roster("val")
    # Re-audit train/val source and cache bytes before any long training/resume.
    from src.video_cache import audit_feature_cache
    from src.video_pipeline import FeatureCache

    signature = audit["cache_signature"]
    root = Path(manifest["dataset_root"])
    if args.frames == 12:
        from src.video_sbi12 import (
            CHECKPOINT_SHA,
            WORK,
            TwelveFrameCache,
            exclusive_lock,
        )

        if file_sha256(args.checkpoint) != CHECKPOINT_SHA:
            raise ValueError(
                "12-frame adaptation requires the official FF-raw checkpoint."
            )
        extractor = TwelveFrameCache(manifest, audit, train, val)
        if args.prepare:
            with exclusive_lock(WORK / "cache.lock"):
                for record in smoke_records(train) + smoke_records(val, per_source=1):
                    result = extractor.prepare(root, record, audit)
                    print(dict(smoke_cache=result), flush=True)
                extractor.release_detector()
        else:
            extractor.preflight(root, train + val, audit)
    else:
        current = audit_feature_cache(
            manifest,
            FeatureCache(
                Path("outputs/video_v1/cache"),
                signature["pipeline"],
                signature["backbone_sha256"],
            ),
        )
        if current["cache_fingerprint"] != audit["cache_fingerprint"]:
            raise ValueError("Historical cache changed.")
        extractor = SBIFaces("cuda")
    model = SBIClassifier(args.checkpoint).cuda().eval()
    if model.adaptation_config is not None:
        raise ValueError(
            "Fresh adaptation must start from official SBI; use --resume for an existing run."
        )
    optimizer = optimizer_for(model)
    config = prepare_config(args, manifest, audit, extractor)
    if args.frames == 12:
        from src.video_sbi12 import BASELINE

        config.update(
            frames=12,
            baseline_sha256=file_sha256(BASELINE),
            baseline_path=str(BASELINE),
            train_roster_sha256=digest_json(train),
            val_roster_sha256=digest_json(val),
        )
        config["implementation_sha256"].update(
            {
                str(path): file_sha256(path)
                for path in (
                    Path("src/video_sbi12.py"),
                    Path("src/video_frame_study.py"),
                    Path("src/video_spatial_evaluate.py"),
                    Path("src/video_train.py"),
                )
            }
        )
        validate_fresh_resume(out, resume_path, config)
    out.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        try:
            report = smoke(
                args, model, optimizer, extractor, root, train, val, audit, config
            )
            if args.frames == 12:
                report["cache_pilot"] = extractor.prepared
                cold = [r for r in extractor.prepared if not r["reused"]]
                report["cache_estimate_hours"] = (
                    float(np.mean([r["seconds"] for r in cold]))
                    * (len(train) + len(val))
                    / 3600
                    if cold
                    else None
                )
                report["cache_estimate_gib"] = (
                    float(np.mean([r["bytes"] for r in extractor.prepared]))
                    * (len(train) + len(val))
                    / 1024**3
                )
                report["checkpoint_budget_gib"] = (
                    (out / "resume_probe.pth").stat().st_size
                    * (args.epochs + 2)
                    / 1024**3
                )
            write_json(out / "smoke.json", report)
            print(
                {k: v for k, v in report.items() if k not in ("config", "stages")},
                flush=True,
            )
        except BaseException as error:
            write_json(
                out / "interrupted.json",
                dict(
                    reason=type(error).__name__,
                    message=str(error),
                    training_epoch_started=False,
                ),
            )
            raise
        return
    # CPU-only loaders provide checkpointed independent shuffle generators.
    train_loader = make_video_loader([], 1, seed=args.seed)
    val_loader = make_video_loader([], 1, seed=args.seed)
    weights = torch.tensor(
        [training_class_balance(train)["class_weights"][n] for n in ("REAL", "FAKE")],
        device="cuda",
    )
    history, start, best_epoch, best_score = [], 1, 0, -math.inf
    stopper = EarlyStopping(patience=3)
    if resume_path:
        saved = torch.load(resume_path, map_location="cpu", weights_only=True)
        validate_state(saved, config)
        model.load_state_dict(saved["model_state_dict"])
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        history, start = saved["history"], saved["epoch"] + 1
        best_epoch, best_score = saved["best_epoch"], saved["best_score"]
        stopper = EarlyStopping(**saved["early_stopping"])
        restore_rng(saved["rng_state"], train_loader, val_loader)
        best = out / "checkpoints" / f"epoch_{best_epoch:03d}.pth"
        atomic_copy(best, out / "checkpoints/best_video.pth")
        atomic_copy(resume_path, out / "checkpoints/last_video.pth")
        write_json(out / "history.json", history)
        evaluation_path = out / f"epoch_{saved['epoch']:03d}_validation.json"
        write_json(evaluation_path, saved["validation"])
        write_comparison(out, config, history, best_epoch, best_score)
    write_json(out / "config.json", config)
    if stopper.bad_epochs >= stopper.patience:
        print("Early stopping already reached; no extra epoch.")
        return
    last_completed = start - 1
    try:
        for epoch in range(start, args.epochs + 1):
            configure_stage(model, epoch)
            began = perf_counter()
            order = torch.randperm(
                len(train), generator=train_loader.generator
            ).tolist()
            losses = []
            for offset in range(0, len(order), 4):
                examples = [
                    load_example(extractor, root, train[i], audit)
                    for i in order[offset : offset + 4]
                ]
                result = train_group(model, optimizer, examples, weights, "cuda")
                losses.append((result["loss"], result["videos"]))
                if offset % 100 == 0:
                    print(
                        f"epoch={epoch} videos={min(offset + 4, len(train))}/{len(train)} loss={result['loss']:.5f} elapsed={perf_counter() - began:.1f}s",
                        flush=True,
                    )
            evaluated = evaluate(model, extractor, root, val, audit)
            if args.frames == 12:
                from src.video_sbi12 import comparison

                evaluated["zero_shot_12_comparison"] = comparison(
                    evaluated, extractor.baseline
                )
            score = evaluated["metrics"]["macro_f1"]
            improved = score > best_score
            if improved:
                best_score, best_epoch = score, epoch
            stopped = stopper.update(score)
            row = dict(
                epoch=epoch,
                train_loss=sum(loss * count for loss, count in losses) / len(train),
                validation_metrics=evaluated["metrics"],
                sources=evaluated["sources"],
                seconds=perf_counter() - began,
            )
            if args.frames == 12:
                row["zero_shot_12_comparison"] = evaluated["zero_shot_12_comparison"]
            history.append(row)
            row["eta_minutes"] = (
                float(np.mean([h["seconds"] for h in history]))
                * (args.epochs - epoch)
                / 60
            )
            checkpoint = out / "checkpoints" / f"epoch_{epoch:03d}.pth"
            if checkpoint.exists():
                raise FileExistsError("Epoch checkpoint protected.")
            atomic_save(
                dict(
                    pipeline=PIPELINE,
                    config=config,
                    model_state_dict=model.state_dict(),
                    optimizer_state_dict=optimizer.state_dict(),
                    rng_state=capture_rng(train_loader, val_loader),
                    early_stopping=asdict(stopper),
                    history=history,
                    epoch=epoch,
                    best_epoch=best_epoch,
                    best_score=best_score,
                    validation=evaluated,
                ),
                checkpoint,
            )
            last_completed = epoch
            if improved:
                atomic_copy(checkpoint, out / "checkpoints/best_video.pth")
            atomic_copy(checkpoint, out / "checkpoints/last_video.pth")
            write_json(out / "history.json", history)
            write_json(out / f"epoch_{epoch:03d}_validation.json", evaluated)
            write_comparison(out, config, history, best_epoch, best_score)
            print(row, flush=True)
            if stopped:
                break
    except BaseException as error:
        optimizer.zero_grad(set_to_none=True)
        write_json(
            out / "interrupted.json",
            dict(
                reason=type(error).__name__,
                message=str(error),
                last_completed_epoch=last_completed,
                recovery="Same command plus --resume auto replays incomplete epoch; no exact mid-epoch resume.",
            ),
        )
        raise


def write_comparison(out, config, history, best_epoch, best_score):
    if config.get("frames") != 12:
        return
    write_json(
        out / "baseline_comparison.json",
        dict(
            baseline_path=config["baseline_path"],
            baseline_sha256=config["baseline_sha256"],
            selected_epoch=best_epoch,
            selected_macro_f1=best_score,
            epochs=[
                dict(epoch=h["epoch"], **h["zero_shot_12_comparison"]) for h in history
            ],
        ),
    )


def validate_fresh_resume(out, resume_path, config):
    """An interrupted first epoch restarts only from the same official config."""
    if resume_path is not None or not out.exists():
        return
    from src.video_forensic import read_json

    if not (out / "config.json").is_file() or read_json(out / "config.json") != config:
        raise ValueError(
            "Existing run has no compatible epoch/config; use a new run directory."
        )
    if any((out / name).exists() for name in ("smoke.json", "resume_probe.pth")):
        raise ValueError("Diagnostic smoke directory cannot become a training run.")


if __name__ == "__main__":
    main()
