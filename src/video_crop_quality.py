"""Bounded train-only crop contact sheets + actual image CUDA optimizer smoke."""

import json
from collections import Counter
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.video_finetune import prepare_context, write_json
from src.video_finetune_model import make_optimizer
from src.video_pipeline import file_sha256, video_collate
from src.video_preprocess import select_balanced_sample
from src.video_train import train_video_batch


def image_statistics(crop, cv):
    gray = cv.cvtColor(np.asarray(crop), cv.COLOR_RGB2GRAY)
    return dict(
        size=list(crop.size),
        sharpness=float(cv.Laplacian(gray, cv.CV_64F).var()),
        brightness=float(gray.mean()),
    )


def create_sheets(rows, output):
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = (
        ImageFont.truetype(str(font_path), 15)
        if font_path.exists()
        else ImageFont.load_default()
    )
    files = []
    for start in range(0, len(rows), 8):
        sheet = Image.new("RGB", (1020, 8 * 192 + 40), "#20242a")
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (12, 10),
            "Existing Haar / IoU pipeline | Train only | 160x160 input | Missing frames shown explicitly",
            fill="white",
            font=font,
        )
        for offset, row in enumerate(rows[start : start + 8]):
            top = 40 + offset * 192
            label = "REAL" if row["record"]["label"] == 0 else "FAKE"
            draw.text(
                (12, top),
                f"{start + offset + 1:02d}  {label}  {row['record']['path']}  valid {row['report']['valid_frames']}/6",
                fill="#ffdd88",
                font=font,
            )
            crops = dict(zip(row["report"]["valid_indices"], row["crops"]))
            failures = {r["index"]: r["reason"] for r in row["report"]["failed_frames"]}
            for column, index in enumerate(row["report"]["selected_indices"]):
                left = 12 + column * 168
                if index in crops:
                    sheet.paste(
                        crops[index].resize((160, 160), Image.Resampling.BILINEAR),
                        (left, top + 23),
                    )
                else:
                    draw.rectangle(
                        (left, top + 23, left + 159, top + 182), fill="#5e2830"
                    )
                    draw.text(
                        (left + 4, top + 55),
                        failures.get(index, "decode failed").replace("_", "\n"),
                        fill="white",
                        font=font,
                    )
                draw.text(
                    (left + 3, top + 25),
                    f"frame {index}",
                    fill="#00ffff",
                    font=font,
                    stroke_width=1,
                    stroke_fill="black",
                )
        name = f"crop_sheet_{start // 8 + 1:02d}.png"
        sheet.save(output / name)
        files.append(name)
    return files


def cuda_image_smoke(model, rows, balance, config):
    if not torch.cuda.is_available():
        raise RuntimeError("Gerçek CUDA smoke gerekli; CUDA bulunamadı.")
    device = torch.device("cuda")
    model.to(device).train()
    optimizer = make_optimizer(model, config["backbone_lr"], config["head_lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    weights = torch.tensor(
        [balance["class_weights"][n] for n in ("REAL", "FAKE")], device=device
    )
    before = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    # Four per class, real decoded crops truncated to exercise lengths 3/4/5/6.
    selected = [
        r
        for label in (0, 1)
        for r in rows
        if r["record"]["label"] == label and len(r["values"]) == 6
    ]
    selected = [
        r
        for label in (0, 1)
        for r in [r for r in selected if r["record"]["label"] == label][:4]
    ]
    if len(selected) != 8:
        raise ValueError("CUDA smoke için sınıf başına dört 6-kareli örnek gerekli.")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    results = []
    for index, row in enumerate(selected):
        length = 3 + index % 4
        item = dict(
            record=row["record"], values=row["values"][:length], report=row["report"]
        )
        began = perf_counter()
        result = train_video_batch(
            model,
            video_collate([item]),
            device,
            optimizer,
            scaler,
            False,
            False,
            weights,
        )
        torch.cuda.synchronize()
        # Last block must actually participate in the update, not merely be unfrozen.
        if any(
            p.grad is None or not torch.isfinite(p.grad).all()
            for p in model.backbone[0][-1].parameters()
        ):
            raise RuntimeError("Son feature bloğuna sonlu gradient ulaşmadı.")
        results.append(
            dict(
                path=row["record"]["path"],
                label=row["record"]["label"],
                length=length,
                loss=result["loss"],
                gradient_norm=result["gradient_norm"],
                seconds=perf_counter() - began,
            )
        )
    after = model.state_dict()
    changed = [
        name
        for name, value in before.items()
        if not torch.equal(value, after[name].detach().cpu())
    ]
    prefix_changed = [
        n
        for n in changed
        if n.startswith("backbone.") and not n.startswith("backbone.0.8.")
    ]
    bn_changed = [
        n
        for n in changed
        if n.endswith(("running_mean", "running_var", "num_batches_tracked"))
    ]
    if (
        prefix_changed
        or bn_changed
        or not all(
            any(n.startswith(prefix) for n in changed)
            for prefix in ("backbone.0.8.", "gru.", "head.")
        )
    ):
        raise RuntimeError(
            "Son blok/GRU/head güncellemesi veya frozen BN/prefix koruması başarısız."
        )
    model.eval()
    evaluation_times, mask_errors = [], []
    with torch.no_grad():
        for row in selected[:4]:
            began = perf_counter()
            features = model.extract_features(row["values"].to(device))
            model(features[None], torch.tensor([6]))
            torch.cuda.synchronize()
            evaluation_times.append(perf_counter() - began)
            for length in range(3, 7):
                padded = features.clone()
                padded[length:] = float("nan")
                expected = model(features[None, :length], torch.tensor([length]))
                observed = model(padded[None], torch.tensor([length]))
                mask_errors.append((expected - observed).abs().max().item())
    if max(mask_errors) > 1e-5:
        raise RuntimeError("3–6 kare padding masking başarısız.")
    return dict(
        device=torch.cuda.get_device_name(),
        device_total_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2,
        updates=results,
        precision="fp32",
        batch_size=1,
        real_images=True,
        feature_cache_input=False,
        lengths_description="Real decoded 6-frame videos truncated to 3/4/5/6 for mask diagnostics; not new preprocessing decisions.",
        frozen_prefix_unchanged=not prefix_changed,
        all_bn_running_stats_unchanged=not bn_changed,
        changed_parameters=changed,
        all_loss_gradient_finite=True,
        padding_max_logit_error=max(mask_errors),
        train_step_seconds=mean(r["seconds"] for r in results[2:]),
        validation_forward_seconds=mean(evaluation_times),
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        checkpoint_saved=False,
    )


def prepare_quality_and_smoke(args):
    existing = Path("outputs/video_finetune_v1/current_preparation.txt")
    output = (
        Path(existing.read_text(encoding="utf-8"))
        if existing.is_file()
        else Path("outputs/video_finetune_v1/preparation") / uuid4().hex
    )
    if (output / "report.json").exists():
        output = Path("outputs/video_finetune_v1/preparation") / uuid4().hex
    output.mkdir(parents=True, exist_ok=True)
    manifest, model, pipeline, audit, balance, config, datasets = prepare_context(args)
    candidates = audit["splits"]["train"]["accepted_records"]
    sample = select_balanced_sample(candidates, per_class=20, seed=42)
    if Counter(r["label"] for r in sample) != {0: 20, 1: 20}:
        raise ValueError("Crop raporu tam 20 REAL + 20 FAKE gerektirir.")
    rows = []
    for index, record in enumerate(sample, 1):
        began = perf_counter()
        values, report, crops = pipeline.read(
            Path(manifest["dataset_root"]) / record["path"]
        )
        seconds = perf_counter() - began
        datasets["train"].verify(record, values, report)
        rows.append(
            dict(
                record=record,
                values=values,
                report=report,
                crops=crops,
                decode_crop_seconds=seconds,
                crop_statistics=[image_statistics(crop, pipeline.cv) for crop in crops],
            )
        )
        print(
            f"Crop quality {index}/40: {record['path']} {report['valid_frames']}/6",
            flush=True,
        )
    sheets = create_sheets(rows, output)
    write_json(output / "roster_audit.json", audit)
    write_json(output / "config.json", config)
    smoke = cuda_image_smoke(model, rows, balance, config)
    # GPU training estimate uses six-frame examples; short prefix timings are excluded.
    six_times = [r["seconds"] for r in smoke["updates"] if r["length"] == 6]
    train_step = mean(six_times)
    decode_means = {
        label: mean(
            r["decode_crop_seconds"] for r in rows if r["record"]["label"] == label
        )
        for label in (0, 1)
    }
    decode_total = sum(
        audit["splits"][name]["accepted_counts"][cls] * decode_means[label]
        for name in ("train", "val")
        for label, cls in enumerate(("REAL", "FAKE"))
    )
    estimate = (
        decode_total
        + audit["splits"]["train"]["accepted"] * train_step
        + audit["splits"]["val"]["accepted"] * smoke["validation_forward_seconds"]
    )
    report = dict(
        schema="video_partial_finetune_preparation_v1",
        official_test_decoded_or_evaluated=False,
        manifest_fingerprint=manifest["fingerprint"],
        cross_split_families=audit["cross_split_families"],
        identity_overlap_counts=manifest["identity_overlap_counts"],
        source_identity_verified=False,
        source_family_limitation=manifest["limitation"],
        training_class_balance=balance,
        accepted_counts={
            name: audit["splits"][name]["accepted_counts"] for name in ("train", "val")
        },
        valid_frame_histograms={
            name: audit["splits"][name]["valid_frame_histogram"]
            for name in ("train", "val")
        },
        quality_sample=dict(
            split="train",
            classes={"REAL": 20, "FAKE": 20},
            seed=42,
            pool="historically accepted, fully audited crop roster; all 40 fresh decode decisions verified",
            valid_frames=sum(r["report"]["valid_frames"] for r in rows),
            requested_frames=240,
            sheets=sheets,
        ),
        videos=[
            {k: v for k, v in row.items() if k not in {"values", "crops"}}
            for row in rows
        ],
        cuda_smoke=smoke,
        one_epoch_estimate=dict(
            seconds=estimate,
            minutes=estimate / 60,
            range_minutes=[estimate / 60 * 0.6, estimate / 60 * 1.6],
            decode_crop_seconds_by_class=decode_means,
            six_frame_train_step_seconds=train_step,
            validation_forward_seconds=smoke["validation_forward_seconds"],
            scope="2253 accepted train + 572 accepted validation, fresh decode every epoch; startup audit/checkpoint I/O extra; extrapolation from 40 crops/8 updates, not a measured epoch",
        ),
    )
    snapshot = output / "preservation_before.json"
    if snapshot.exists():
        hashes = json.loads(snapshot.read_text(encoding="utf-8"))
        changed = [
            path
            for path, sha in hashes.items()
            if not Path(path).is_file() or file_sha256(Path(path)) != sha
        ]
        report["preservation"] = dict(files_checked=len(hashes), changed=changed)
        if changed:
            raise RuntimeError(f"Korunan dosyalar değişti: {changed}")
    write_json(output / "report.json", report)
    summary = [
        "# Video kısmi fine-tuning hazırlığı",
        "",
        "Yalnız mevcut kabul edilmiş train roster'ından seed 42 ile 20 REAL + 20 FAKE. "
        "Aynı Haar/IoU pipeline yeniden decode edildi; hiçbir resmi test videosu decode edilmedi.",
        "",
        f"Geçerli kare: {report['quality_sample']['valid_frames']}/240. "
        "Başarısız kareler kırmızı hücrelerde gösterilir. Görseller görsel inceleme içindir; otomatik kimlik doğrulaması değildir.",
        "",
        f"CUDA: {smoke['device']}, 8 gerçek görüntü optimizer adımı; sonlu loss/gradient, sabit frozen prefix/BN, "
        f"3–6 kare NaN padding farkı {smoke['padding_max_logit_error']:.8g}. "
        f"Peak allocated {smoke['peak_allocated_mib']:.2f} MiB / reserved {smoke['peak_reserved_mib']:.2f} MiB "
        "(CUDA context/driver dahil değil). Smoke ağırlıkları kaydedilmedi.",
        "",
        f"Bir epoch tahmini {estimate / 60:.1f} dakika (yaklaşık {estimate / 60 * 0.6:.1f}–{estimate / 60 * 1.6:.1f}); "
        "2253 train + 572 validation. Tam epoch ölçülmedi; başlangıç audit/checkpoint I/O ayrıca.",
        "",
        "REAL loss ağırlığı %.8f, FAKE %.8f; yalnız kabul edilmiş train. "
        "Eski feature tensorleri model girdisine girmez; cache sadece kabul/ret ve crop tutarlılık metadata denetimidir."
        % tuple(balance["class_weights"][n] for n in ("REAL", "FAKE")),
        "",
        "Bilinen filename kaynak aileleri train/validation arasında çakışmıyor; gerçek source/donor/identity izolasyonu doğrulanmadı.",
        "",
    ]
    for name in sheets:
        summary.extend([f"![{name}]({(output / name).resolve().as_posix()})", ""])
    summary.extend(
        [
            "Ayrıntılı video/crop ölçümleri ve CUDA adımları: [report.json](report.json).",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "cuda_smoke": smoke,
                "one_epoch_estimate": report["one_epoch_estimate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
