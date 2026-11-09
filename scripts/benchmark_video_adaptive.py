"""Explicit offline simulation, bounded CUDA smoke, or preselected raw-video pilot."""

import argparse
import hashlib
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.dataset import seed_everything
from src.runtime_provenance import verify_implementations
from src.video_adaptive import (
    CHECKPOINT,
    CHECKPOINT_SHA,
    MODES,
    PAIRED,
    RUN,
    WORK,
    AdaptiveVideo,
    authorize_input,
    load_saved,
    method_summary,
    offline_rows,
    uncertain,
    verified_source,
)
from src.video_forensic import (
    SBIClassifier,
    SBIFaces,
    accepted_roster,
    read_json,
    write_json,
)
from src.video_frame_study import STUDY, process_frame
from src.video_manifest import digest_json
from src.video_pipeline import file_sha256

LIMITATION = "Exploratory candidate |p6-.5|<.10 was chosen using this validation. Not independent evidence; no reoptimization. Person/source independence remains unverified."


def pilot_plan(records):
    chosen = [
        r
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for r in sorted(
            [r for r in records if r["path"].startswith(source + "/")],
            key=lambda r: hashlib.sha256(r["path"].encode()).hexdigest(),
        )[:3]
    ]
    if len(chosen) != 9:
        raise ValueError("Pilot needs three videos from each source.")
    return dict(
        checkpoint_sha256=CHECKPOINT_SHA,
        candidate_margin=0.10,
        decision_threshold=0.5,
        selection="three per source by SHA256(path), before new inference; no score/error selection",
        records=chosen,
        records_sha256=digest_json(chosen),
        maximum_repeats=5,
        order="rotate 6/12/adaptive by (repeat+video_index)%3, independent runs, no cross-method frame cache",
        limitation=LIMITATION,
    )


def load_engine():
    began = perf_counter()
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Only the verified epoch-6 checkpoint is permitted.")
    config = read_json(RUN / "config.json")
    verify_implementations(
        config["implementation_sha256"], Path(__file__).resolve().parents[1]
    )
    if config["frames"] != 12 or config["threshold"] != 0.5:
        raise ValueError("Training/preprocessing provenance changed.")
    if not torch.cuda.is_available():
        raise RuntimeError("This real-inference prototype requires CUDA.")
    model = SBIClassifier(CHECKPOINT).cuda().eval()
    extractor = SBIFaces("cuda")
    original = read_json(STUDY / "full.json")["provenance"]["preprocessing"]
    if extractor.signature != original or model.adaptation_config != config:
        raise ValueError("Model or official preprocessing mismatch.")
    torch.cuda.synchronize()
    load_seconds = perf_counter() - began
    began = perf_counter()
    with torch.inference_mode():
        for _ in range(3):
            model(torch.zeros((1, 3, 380, 380), device="cuda", dtype=torch.float32))
        extractor.detector.predict_jsons(np.full((480, 640, 3), 127, dtype=np.uint8))
    torch.cuda.synchronize()
    warmup_seconds = perf_counter() - began

    @torch.inference_mode()
    def processor(frame, index):
        # Reuse the already verified implementation, including official crop,
        # half-largest-area multiface retention, FP32 /255 and max face score.
        return process_frame(extractor, model, frame, index)

    return AdaptiveVideo(extractor.cv, processor), dict(
        model_and_detector_load_seconds=load_seconds,
        synthetic_warmup_seconds=warmup_seconds,
        classifier_warmup_calls=3,
        detector_warmup_calls=1,
        gpu=torch.cuda.get_device_name(),
        total_gpu_mib=torch.cuda.get_device_properties(0).total_memory / 1024**2,
        precision="FP32",
        frame_batch_size=1,
        preprocessing=extractor.signature,
        timing_scope="Video open, sequential decode, detector, crop, classifier, gate and optional second decode included. Model load, synthetic warmup, source-integrity preflight and output JSON writing reported separately/excluded.",
        memory_scope="Max of synchronized per-frame PyTorch peaks; model+detector resident. No whole-video/raw-extra-frame buffering. CPU decoded frame and temporary crops only.",
    )


def run_one(engine, source, record, count, mode):
    torch.cuda.synchronize()
    before = source.stat()
    row = engine.predict(source, mode, expected_frame_count=count)
    torch.cuda.synchronize()
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during inference.")
    row.update(path=record["path"], target=record["label"], mode=mode)
    return row


def compare_recorded(row, record, paired):
    saved = next(r for r in paired["paired_rows"] if r["path"] == record["path"])
    scores = {int(k): v for k, v in saved["frame_probabilities"].items()}
    study = read_json(STUDY / "frames" / (digest_json(record) + ".json"))
    crops = {f["index"]: f for f in study["frames"]}
    deltas = []
    if (
        row["status"] != "accepted"
        or row["warnings"]
        or row["expanded"] != uncertain(saved["score_6"])
    ):
        raise ValueError("Smoke coverage/routing differs from offline reference.")
    for frame in row["frames"]:
        if frame["status"] != "accepted":
            raise ValueError("Smoke frame rejected.")
        deltas.append(abs(frame["fake_probability"] - scores[frame["index"]]))
        old = crops[frame["index"]]
        if len(frame["faces"]) != len(old["faces"]):
            raise ValueError("Multiface count differs from frame-study.")
        for a, b in zip(frame["faces"], old["faces"], strict=True):
            if any(a[k] != b[k] for k in ("bbox", "crop_size", "crop_sha256")):
                raise ValueError("Official crop bbox/pixels changed.")
    if max(deltas) > 1e-6:
        raise ValueError("Fine-tuned per-frame CUDA score drift.")
    expected = saved["score_12"] if row["expanded"] else saved["score_6"]
    if abs(row["fake_probability"] - expected) > 1e-6:
        raise ValueError("Adaptive score differs from offline reference.")
    if row["processed_frames"] != len(set(f["index"] for f in row["frames"])):
        raise ValueError("Repeated or unaccounted frame processing.")
    return dict(
        path=record["path"],
        max_frame_score_delta=max(deltas),
        original_crop_pixels_and_bboxes_equal=True,
        unique_processed_frames=row["processed_frames"],
        expanded=row["expanded"],
        decoder_passes=row["decode_passes"],
    )


def offline(out, records):
    paired = load_saved()
    if {r["path"]: r["label"] for r in records} != {
        r["path"]: r["target"] for r in paired["paired_rows"]
    }:
        raise ValueError("Offline validation roster differs.")
    rows = offline_rows(paired)
    methods = {mode: method_summary(value) for mode, value in rows.items()}
    plan = pilot_plan(records)
    write_json(out / "pilot_plan.json", plan)
    write_json(
        out / "offline.json",
        dict(
            checkpoint_sha256=CHECKPOINT_SHA,
            limitation=LIMITATION,
            methods=methods,
            rows=rows,
            measured_latency=False,
            gpu_inference=False,
            input_sha256=file_sha256(PAIRED),
        ),
    )
    lines = [
        "# SBI adaptif 6→12: offline prototip",
        "",
        LIMITATION,
        "",
        "0,10 eşiği bu validation bulgularından seçildi. Sonuç bağımsız başarı kanıtı değildir.",
        "",
        "| Metrik | Sabit 6 | Sabit 12 | Adaptif |",
        "|---|---:|---:|---:|",
    ]
    for key in ("accuracy", "roc_auc", "macro_f1", "balanced_accuracy"):
        lines.append(
            "| "
            + key
            + " | "
            + " | ".join(f"{methods[m]['metrics'][key]:.8f}" for m in MODES)
            + " |"
        )
    for label in ("REAL", "FAKE"):
        lines.append(
            "| "
            + label
            + " recall | "
            + " | ".join(
                f"{methods[m]['metrics']['per_class'][label]['recall']:.8f}"
                for m in MODES
            )
            + " |"
        )
    for key in (
        "false_alarms",
        "mean_processed_frames",
        "median_processed_frames",
        "requested_twelve_rate",
    ):
        lines.append(
            "| " + key + " | " + " | ".join(str(methods[m][key]) for m in MODES) + " |"
        )
    lines += [
        "",
        "Ortalama kare sayısı hesaplama bütçesidir; gerçek gecikme/hız kazancı değildir. Offline simülasyonda video decode, detector, GPU inference veya yeni cache yoktur. Karar eşiği 0,5; gate sınırları 0,4 ve 0,6 hariçtir.",
        "",
        "Pilot planı aynı klasörde pilot_plan.json: üç kaynak başına SHA256(path) ile üç video; dokuz video yeni inference öncesi kilitlenir. Smoke ayrı, bilinen iki branch örneğini kullanır ve tarafsız pilot yerine geçmez.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(methods, flush=True)


def smoke(out, engine, root, audit, records):
    paired = load_saved()
    selected_paths = ("Celeb-real/id45_0001.mp4", "Celeb-synthesis/id47_id42_0001.mp4")
    selected = [next(r for r in records if r["path"] == p) for p in selected_paths]
    write_json(
        out / "smoke_plan.json",
        dict(
            records=selected,
            selection="known expansion and early-stop cases; branch coverage, not unbiased accuracy/timing pilot",
        ),
    )
    rows, checks = [], []
    for record in selected:
        source, count = verified_source(root, record, audit)
        row = run_one(engine, source, record, count, "adaptive")
        checks.append(compare_recorded(row, record, paired))
        rows.append(row)
    write_json(
        out / "smoke.json",
        dict(
            checks=checks,
            rows=rows,
            summary=method_summary(rows),
            training_started=False,
            long_benchmark_started=False,
            official_test_accessed=False,
        ),
    )
    print(checks, flush=True)


def benchmark(out, engine, root, audit, records, plan_path, repeats):
    plan = read_json(plan_path)
    if plan != pilot_plan(records):
        raise ValueError("Pilot plan differs from the preselected fixed protocol.")
    write_json(out / "pilot_plan.json", plan)
    tick = perf_counter()
    inputs = [(r, *verified_source(root, r, audit)) for r in plan["records"]]
    preflight_seconds = perf_counter() - tick
    rows = []
    for repeat in range(repeats):
        for index, (record, source, count) in enumerate(inputs):
            offset = (repeat + index) % 3
            order = MODES[offset:] + MODES[:offset]
            for mode in order:
                row = run_one(engine, source, record, count, mode)
                row["repeat"] = repeat + 1
                rows.append(row)
                write_json(
                    out / "progress.json",
                    dict(
                        completed_calls=len(rows),
                        expected_calls=repeats * 9 * 3,
                        last=dict(path=record["path"], mode=mode, repeat=repeat + 1),
                    ),
                )
                print(
                    f"repeat={repeat + 1} mode={mode} frames={row['processed_frames']} seconds={row['latency_seconds']:.3f} {record['path']}",
                    flush=True,
                )
    # Accuracy is computed once per video. Repeats only characterize timing.
    summaries = {}
    for mode in MODES:
        all_runs = [r for r in rows if r["mode"] == mode]
        first = [r for r in all_runs if r["repeat"] == 1]
        by_path = {r["path"]: r for r in first}
        for row in all_runs:
            old = by_path[row["path"]]
            if (
                row["status"] != old["status"]
                or row["prediction"] != old["prediction"]
                or row["selected_indices"] != old["selected_indices"]
                or (
                    row["fake_probability"] is not None
                    and abs(row["fake_probability"] - old["fake_probability"]) > 1e-6
                )
            ):
                raise ValueError("Benchmark repeat changed predictions/routing.")
        summary = method_summary(first)
        timings = method_summary(all_runs)
        summary.update(
            latency=timings["latency"],
            gpu_memory=timings["gpu_memory"],
            independent_videos=len(first),
            measured_runs=len(all_runs),
        )
        summaries[mode] = summary
    write_json(
        out / "benchmark.json",
        dict(
            methods=summaries,
            rows=rows,
            source_preflight_seconds=preflight_seconds,
            limitation=LIMITATION,
            repeat_note="Timing repeats of nine videos, not additional independent accuracy samples. Warm filesystem/allocator history may affect timings.",
            official_test_accessed=False,
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--offline", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--predict", type=Path, metavar="VIDEO")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--plan", type=Path, default=WORK / "offline_20261015/pilot_plan.json"
    )
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=3)
    args = parser.parse_args()
    out = args.run_dir.resolve()
    if not out.is_relative_to(WORK.resolve()) or out == WORK.resolve():
        raise ValueError("New run directory under outputs/video_adaptive_v1 required.")
    out.mkdir(parents=True, exist_ok=False)
    manifest, audit, records = accepted_roster("val")
    root = Path(manifest["dataset_root"])
    protected_paths = [
        CHECKPOINT,
        RUN / "checkpoints/epoch_006.pth",
        RUN / "epoch_006_validation.json",
        RUN / "config.json",
        Path("app.py"),
        Path("outputs/video_identity_v1/manifest.json"),
        Path(manifest["parent_manifest"]),
        STUDY / "full.json",
        PAIRED,
        Path("src/video_forensic.py"),
        Path("src/video_frame_study.py"),
        Path("src/video_sbi12.py"),
    ]
    protected = {str(p): file_sha256(p) for p in protected_paths}
    write_json(out / "protected_before.json", protected)
    write_json(
        out / "provenance.json",
        dict(
            checkpoint=str(CHECKPOINT.resolve()),
            checkpoint_sha256=CHECKPOINT_SHA,
            candidate="abs(p6-.5)<.10; strict central interval (.4,.6)",
            limitation=LIMITATION,
            implementation_sha256={
                str(p): file_sha256(p)
                for p in (Path(__file__), Path("src/video_adaptive.py"))
            },
            official_test_accessed=False,
        ),
    )
    try:
        if args.offline:
            offline(out, records)
        else:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            seed_everything(42)
            torch.set_num_threads(4)
            torch.use_deterministic_algorithms(True)
            # Reject unapproved dataset paths before initializing/decoding anything.
            predicted_record = None
            if args.predict:
                video = args.predict.resolve()
                _, _, train = accepted_roster("train")
                predicted_record = authorize_input(video, root, train + records)
            if args.benchmark and read_json(args.plan) != pilot_plan(records):
                raise ValueError(
                    "Run the offline protocol preparation first; pilot plan mismatch."
                )
            engine, environment = load_engine()
            write_json(out / "environment.json", environment)
            if args.smoke:
                smoke(out, engine, root, audit, records)
            elif args.benchmark:
                benchmark(out, engine, root, audit, records, args.plan, args.repeats)
            else:
                if predicted_record is not None:
                    video, count = verified_source(root, predicted_record, audit)
                else:
                    count = None
                row = engine.predict(video, expected_frame_count=count)
                write_json(out / "prediction.json", row)
                print({k: v for k, v in row.items() if k != "frames"}, flush=True)
    except BaseException as error:
        write_json(
            out / "interrupted.json",
            dict(reason=type(error).__name__, message=str(error)),
        )
        raise
    finally:
        after = {str(p): file_sha256(p) for p in protected_paths}
        write_json(
            out / "preservation.json", dict(unchanged=after == protected, hashes=after)
        )
        if after != protected:
            raise RuntimeError("Protected inputs changed.")


if __name__ == "__main__":
    main()
