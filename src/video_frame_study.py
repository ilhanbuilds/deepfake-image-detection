"""Paired, nested SBI frame-budget study on the locked accepted validation roster.

Each frame is detected/classified once across budgets. Baseline is rerun from video
before extensions; subsequent phases reuse sealed frame-score caches, not crops.
"""

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from src.video_forensic import (
    ROOT,
    SBIClassifier,
    SBIFaces,
    accepted_roster,
    read_json,
    score_faces,
    summarize,
    validate_crops,
    write_json,
)
from src.video_manifest import digest_json
from src.video_pipeline import file_sha256, sample_indices

STUDY = Path("outputs/video_frames_v1")
BUDGETS = (6, 12, 24, 32)
CHECKPOINT_SHA = "e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2"


def nested_indices(frame_count, original):
    """Bisect the widest gap; break equal-gap ties by earliest time, floor midpoint.

    Never duplicate a short video's frames. Return the available unique set and
    let the caller report a shortfall for every requested budget.
    """
    if original != sample_indices(frame_count, 6):
        raise ValueError("Historical six indices differ from the original sampler.")
    selected = list(original)
    result = {6: list(selected)}
    for budget in BUDGETS[1:]:
        while len(selected) < min(budget, frame_count):
            left, right = max(
                zip(selected[:-1], selected[1:], strict=True),
                key=lambda pair: (pair[1] - pair[0], -pair[0]),
            )
            selected.append((left + right) // 2)
            selected.sort()
        result[budget] = list(selected)
    return result


def validate_cache(cached, record, signature):
    payload = {k: v for k, v in cached.items() if k != "integrity_sha256"}
    if (
        cached.get("integrity_sha256") != digest_json(payload)
        or cached.get("record") != record
        or cached.get("signature") != signature
    ):
        raise ValueError("Frame cache integrity/record/provenance mismatch.")
    expected = nested_indices(cached["frame_count"], cached["original_indices"])
    if cached["sets"] != {str(k): v for k, v in expected.items()}:
        raise ValueError("Frame cache nested sampling mismatch.")
    frames = cached["frames"]
    indices = [f["index"] for f in frames]
    required = expected[32 if cached["expanded"] else 6]
    if indices != required:
        raise ValueError("Missing, duplicate or unexpected frame cache entries.")
    for frame in frames:
        if frame["status"] == "accepted":
            scores = [face["fake_probability"] for face in frame["faces"]]
            if (
                not scores
                or not np.isfinite(scores).all()
                or any(s < 0 or s > 1 for s in scores)
                or frame["fake_probability"] != max(scores)
            ):
                raise ValueError("Invalid frame probabilities/multiface aggregation.")
        elif frame["status"] != "failed" or not frame.get("reason"):
            raise ValueError("Unreported frame failure.")


def save_cache(path, cached):
    cached.pop("integrity_sha256", None)
    cached["integrity_sha256"] = digest_json(cached)
    write_json(path, cached)


def condition_row(cached, budget):
    indices = cached["sets"][str(budget)]
    frames = [f for f in cached["frames"] if f["index"] in indices]
    if [f["index"] for f in frames] != indices:
        raise ValueError("Condition not yet computed; no partial-cohort comparison.")
    valid = [f for f in frames if f["status"] == "accepted"]
    baseline_pass = cached["passes"][0]
    # All conditions reach the same last frame. Charge one measured baseline
    # sequential decode pass, plus the measured processing of selected frames.
    seconds = baseline_pass["decode_seconds"] + sum(
        f["processing_seconds"] for f in frames
    )
    row = dict(
        path=cached["record"]["path"],
        target=cached["record"]["label"],
        status="accepted" if len(valid) >= 3 else "rejected",
        selected_indices=indices,
        valid_indices=[f["index"] for f in valid],
        requested_frames=budget,
        unavailable_frames=budget - len(indices),
        failed_frames=[
            dict(index=f["index"], reason=f["reason"])
            for f in frames
            if f["status"] == "failed"
        ],
        valid_face_frames=len(valid),
        retained_faces=sum(len(f["faces"]) for f in valid),
        frame_scores=[f["fake_probability"] for f in valid],
        attributed_seconds=seconds,
        peak_allocated_mib=max(f.get("peak_allocated_mib", 0) for f in frames),
        peak_reserved_mib=max(f.get("peak_reserved_mib", 0) for f in frames),
    )
    if valid and row["status"] == "accepted":
        probability = float(np.mean(row["frame_scores"]))
        row.update(fake_probability=probability, prediction=int(probability >= 0.5))
    return row


def evaluate_conditions(caches, budgets=BUDGETS):
    result = {}
    for budget in budgets:
        rows = [condition_row(cached, budget) for cached in caches]
        result[str(budget)] = dict(
            **summarize(rows, len(caches)),
            rows=rows,
            valid_face_frames=sum(r["valid_face_frames"] for r in rows),
            retained_faces=sum(r["retained_faces"] for r in rows),
            failed_frames=sum(len(r["failed_frames"]) for r in rows),
            unavailable_frames=sum(r["unavailable_frames"] for r in rows),
            attributed_seconds=sum(r["attributed_seconds"] for r in rows),
            mean_seconds=float(np.mean([r["attributed_seconds"] for r in rows])),
            peak_allocated_mib=max(r["peak_allocated_mib"] for r in rows),
            peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows),
        )
    return result


def pilot_records(records):
    # Nine per source, chosen by path hash, independent of prediction/label errors.
    return [
        r
        for source in ("Celeb-real", "YouTube-real", "Celeb-synthesis")
        for r in sorted(
            [r for r in records if r["path"].startswith(source + "/")],
            key=lambda r: hashlib.sha256(r["path"].encode()).hexdigest(),
        )[:9]
    ]


def process_frame(extractor, model, frame, index):
    cv = extractor.cv
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    faces = [f for f in extractor.detector.predict_jsons(rgb) if len(f["bbox"]) == 4]
    areas = [
        (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in faces
    ]
    retained = []
    for face, area in zip(faces, areas, strict=True):
        if area <= 0 or area < max(areas) / 2:
            continue
        crop = extractor.crop_face(
            rgb,
            None,
            np.asarray(face["bbox"]).reshape(2, 2),
            False,
            crop_by_bbox=True,
            only_img=True,
            phase="test",
        )
        if not crop.size:
            continue
        pixels = cv.resize(crop, (380, 380)).transpose(2, 0, 1).copy()
        gray = cv.cvtColor(crop, cv.COLOR_RGB2GRAY)
        quality = float(np.log1p(cv.Laplacian(gray, cv.CV_64F).var()))
        score = score_faces(model, torch.from_numpy(pixels)[None], "cuda")[0]
        retained.append(
            dict(
                bbox=face["bbox"],
                crop_size=list(crop.shape[:2]),
                sharpness=quality,
                crop_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                fake_probability=score,
            )
        )
    torch.cuda.synchronize()
    result = dict(
        index=index,
        faces=retained,
        processing_seconds=perf_counter() - started,
        peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
    )
    if retained:
        result.update(
            status="accepted",
            fake_probability=max(f["fake_probability"] for f in retained),
        )
    else:
        result.update(status="failed", reason="no_valid_face")
    return result


def decode_selected(source, indices, frame_count, extractor, model):
    """Sequential grab, retrieve only new frames, explicit failures without replacement."""
    began = perf_counter()
    decode_seconds = 0.0
    frames = []
    cap = extractor.cv.VideoCapture(str(source))
    selected = set(indices)
    try:
        if not cap.isOpened():
            raise RuntimeError("video_open_failed")
        if int(cap.get(extractor.cv.CAP_PROP_FRAME_COUNT)) != frame_count:
            raise ValueError("Decoder frame count changed from historical cache.")
        decode_seconds += perf_counter() - began
        for index in range(indices[-1] + 1):
            tick = perf_counter()
            grabbed = cap.grab()
            ok, frame = (
                cap.retrieve() if grabbed and index in selected else (False, None)
            )
            decode_seconds += perf_counter() - tick
            if not grabbed:
                frames.extend(
                    dict(
                        index=i,
                        status="failed",
                        reason="decode_failed",
                        processing_seconds=0.0,
                    )
                    for i in indices
                    if i >= index
                )
                break
            if index not in selected:
                continue
            if not ok or frame is None:
                frames.append(
                    dict(
                        index=index,
                        status="failed",
                        reason="decode_failed",
                        processing_seconds=0.0,
                    )
                )
            else:
                frames.append(process_frame(extractor, model, frame, index))
    finally:
        cap.release()
    return frames, dict(
        indices=indices,
        decode_seconds=decode_seconds,
        wall_seconds=perf_counter() - began,
    )


def compare_baseline(cached, historical, pixels, report):
    validate_crops(pixels, report)
    if (
        report != historical["report"]
        or cached["original_indices"] != report["selected_indices"]
    ):
        raise ValueError("Historical SBI report/index mismatch.")
    old_frames = {f["index"]: f for f in report["frames"]}
    differences = []
    for frame in cached["frames"]:
        old = old_frames.get(frame["index"])
        if (
            old is None
            or frame["status"] != "accepted"
            or len(old["faces"]) != len(frame["faces"])
        ):
            raise ValueError("Historical six detector coverage/face count changed.")
        for new, face in zip(frame["faces"], old["faces"], strict=True):
            index = face["crop_index"]
            if (
                new["bbox"] != face["bbox"]
                or new["crop_size"] != face["crop_size"]
                or new["crop_sha256"]
                != hashlib.sha256(pixels[index].numpy().tobytes()).hexdigest()
            ):
                raise ValueError("Historical six detector bbox/crop pixels changed.")
            differences.append(
                abs(new["fake_probability"] - historical["face_scores"][index])
            )
    maximum = max(differences)
    if maximum > 1e-6:
        raise ValueError(f"Historical six classifier score drift: {maximum}")
    return dict(
        bboxes_equal=True, crop_pixels_equal=True, max_face_score_absolute_delta=maximum
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("baseline", "pilot", "full"))
    args = parser.parse_args()
    manifest, audit, records = accepted_roster("val")
    historical_path = ROOT / "sbi_ffraw_validation.json"
    historical = read_json(historical_path)
    expected = {r["path"]: r["label"] for r in records}
    if (
        len(historical["rows"]) != len(expected)
        or {r["path"]: r["target"] for r in historical["rows"]} != expected
    ):
        raise ValueError("Historical baseline cohort differs from accepted validation.")
    if file_sha256(ROOT / "FFraw.tar") != CHECKPOINT_SHA:
        raise ValueError("Only the unchanged official FFraw checkpoint is allowed.")
    if (
        args.phase != "baseline"
        and not read_json(STUDY / "baseline.json")["baseline_verified"]
    ):
        raise ValueError("Reproduce baseline before extensions.")
    if args.phase == "full" and not (STUDY / "pilot.json").exists():
        raise ValueError("Pilot must finish before the full extension.")
    output = STUDY / (args.phase + ".json")
    if output.exists():
        raise FileExistsError("Completed results protected; inspect the saved result.")
    torch.set_num_threads(4)
    model = SBIClassifier(ROOT / "FFraw.tar").cuda().eval()
    extractor = SBIFaces("cuda")
    if (
        historical["provenance"]["preprocessing"] != extractor.signature
        or historical["provenance"]["manifest_fingerprint"] != manifest["fingerprint"]
    ):
        raise ValueError("Historical preprocessing/manifest changed.")
    signature = dict(
        schema="sbi_nested_frames_v1",
        checkpoint_sha256=CHECKPOINT_SHA,
        manifest_fingerprint=manifest["fingerprint"],
        roster_sha256=digest_json(records),
        historical_result_sha256=file_sha256(historical_path),
        preprocessing=extractor.signature,
        budgets=list(BUDGETS),
        sampler="historical_rounded_linspace6_then_widest_gap_floor_midpoint_earliest_tie",
        precision="fp32",
        threshold=0.5,
        aggregation="maximum retained face then arithmetic mean of valid frames",
        training_steps=0,
        test_accessed=False,
        quarantine_accessed=False,
    )
    if (
        args.phase != "baseline"
        and read_json(STUDY / "baseline.json")["provenance"] != signature
    ):
        raise ValueError("Baseline provenance no longer matches.")
    chosen = pilot_records(records) if args.phase == "pilot" else records
    old_rows = {r["path"]: r for r in historical["rows"]}
    root = Path(manifest["dataset_root"]).resolve()
    caches = []
    began = perf_counter()
    newly_computed = 0
    for number, record in enumerate(chosen, 1):
        cache_path = STUDY / "frames" / (digest_json(record) + ".json")
        source = (root / record["path"]).resolve()
        if not source.is_relative_to(root):
            raise ValueError("Video escaped dataset root.")
        cached = read_json(cache_path) if cache_path.exists() else None
        if cached is not None:
            validate_cache(cached, record, signature)
            if file_sha256(source) != cached["source_sha256"]:
                raise ValueError("Source bytes changed since cached inference.")
        if cached is None:
            if args.phase != "baseline":
                raise ValueError("Baseline cache missing; no silent new cohort.")
            old = torch.load(
                Path(audit["cache_directory"]) / (digest_json(record) + ".pt"),
                map_location="cpu",
                weights_only=True,
            )
            if (
                old["record"] != record
                or old["signature"] != audit["cache_signature"]
                or file_sha256(source) != old["source_sha256"]
            ):
                raise ValueError("Historical feature cache/source mismatch.")
            count = old["report"]["frame_count"]
            original = old["report"]["selected_indices"]
            sets = nested_indices(count, original)
            crop_path = (
                ROOT
                / "sbi_crops"
                / (
                    digest_json(dict(record=record, signature=extractor.signature))
                    + ".pt"
                )
            )
            old_crop = torch.load(crop_path, map_location="cpu", weights_only=True)
            if (
                old_crop["signature"] != extractor.signature
                or old_crop["source_sha256"] != old["source_sha256"]
            ):
                raise ValueError("Historical SBI crop provenance mismatch.")
            frames, timing = decode_selected(source, original, count, extractor, model)
            cached = dict(
                record=record,
                signature=signature,
                source_sha256=old["source_sha256"],
                frame_count=count,
                original_indices=original,
                sets={str(k): v for k, v in sets.items()},
                frames=frames,
                passes=[timing],
                expanded=False,
            )
            cached["baseline_check"] = compare_baseline(
                cached, old_rows[record["path"]], old_crop["crops"], old_crop["report"]
            )
            save_cache(cache_path, cached)
            newly_computed += len(frames)
        if args.phase != "baseline" and not cached["expanded"]:
            extra = sorted(set(cached["sets"]["32"]) - set(cached["original_indices"]))
            frames, timing = (
                decode_selected(source, extra, cached["frame_count"], extractor, model)
                if extra
                else ([], dict(indices=[], decode_seconds=0.0, wall_seconds=0.0))
            )
            cached["frames"] = sorted(
                cached["frames"] + frames, key=lambda f: f["index"]
            )
            cached["passes"].append(timing)
            cached["expanded"] = True
            save_cache(cache_path, cached)
            newly_computed += len(frames)
        validate_cache(cached, record, signature)
        caches.append(cached)
        write_json(
            STUDY / (args.phase + ".progress.json"),
            dict(
                phase=args.phase,
                completed=number,
                expected=len(chosen),
                new_frames=newly_computed,
                seconds=perf_counter() - began,
                last_path=record["path"],
            ),
        )
        if number % 10 == 0 or number == len(chosen):
            print(
                f"{args.phase}: {number}/{len(chosen)}; new frames={newly_computed}; elapsed={perf_counter() - began:.1f}s",
                flush=True,
            )
    methods = evaluate_conditions(caches, (6,) if args.phase == "baseline" else BUDGETS)
    baseline_verified = (
        methods["6"]["metrics"] == historical["metrics"]
        if args.phase != "pilot"
        else True
    )
    result = dict(
        provenance=signature,
        phase=args.phase,
        methods=methods,
        baseline_verified=baseline_verified,
        max_baseline_face_score_delta=max(
            c["baseline_check"]["max_face_score_absolute_delta"] for c in caches
        ),
        newly_computed_frames=newly_computed,
        phase_wall_seconds=perf_counter() - began,
        measured_inference_pass_seconds=sum(
            p["wall_seconds"] for c in caches for p in c["passes"]
        ),
        extra_decode_seconds=sum(
            p["decode_seconds"] for c in caches for p in c["passes"][1:]
        ),
        timing_note="Per-budget times are attributed: one measured baseline decode pass plus selected frames' measured processing. Model startup, verification, JSON IO excluded. Actual experiment uses a second decode pass for extensions; no selected frame is retrieved/classified twice. Phase wall may include cache reuse; inference-pass time sums original measured passes.",
    )
    write_json(output, result)
    if not baseline_verified:
        raise ValueError("Reproduced six-frame metrics differ; extensions blocked.")
    print({k: v["metrics"] for k, v in methods.items()}, flush=True)


if __name__ == "__main__":
    main()
