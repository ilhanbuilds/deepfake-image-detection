"""Verify and remeasure exceptional frame latencies without changing study scores.

Raw caches/results stay intact. One warm-up and three measured repeats per frame
provide a separately labelled timing sensitivity estimate, not a new evaluation.
"""

from pathlib import Path

import numpy as np
import torch

from src.video_forensic import (
    ROOT,
    SBIClassifier,
    SBIFaces,
    accepted_roster,
    read_json,
    write_json,
)
from src.video_frame_study import CHECKPOINT_SHA, STUDY, decode_selected, validate_cache
from src.video_manifest import digest_json
from src.video_pipeline import file_sha256


def compare_frame(original, repeated):
    if original["status"] != repeated["status"]:
        raise ValueError("Timing recheck changed frame coverage.")
    before, after = original.get("faces", []), repeated.get("faces", [])
    if len(before) != len(after):
        raise ValueError("Timing recheck changed face count.")
    deltas = []
    for a, b in zip(before, after, strict=True):
        if any(a[k] != b[k] for k in ("bbox", "crop_size", "crop_sha256")):
            raise ValueError("Timing recheck changed bbox/crop pixels.")
        deltas.append(abs(a["fake_probability"] - b["fake_probability"]))
    maximum = max(deltas, default=0.0)
    if maximum > 1e-6:
        raise ValueError("Timing recheck changed classifier scores.")
    return maximum


def main():
    output = STUDY / "timing_recheck.json"
    if output.exists():
        raise FileExistsError("Completed timing recheck protected.")
    full = read_json(STUDY / "full.json")
    manifest, _, records = accepted_roster("val")
    if (
        full["provenance"]["roster_sha256"] != digest_json(records)
        or full["provenance"]["manifest_fingerprint"] != manifest["fingerprint"]
        or file_sha256(ROOT / "FFraw.tar") != CHECKPOINT_SHA
    ):
        raise ValueError("Timing recheck provenance changed.")
    flagged = []
    for record in records:
        cached = read_json(STUDY / "frames" / (digest_json(record) + ".json"))
        validate_cache(cached, record, full["provenance"])
        for frame in cached["frames"]:
            if frame["processing_seconds"] > 5.0:
                flagged.append((cached, frame))
    if not flagged:
        raise ValueError("No exceptional frame latency to recheck.")
    torch.set_num_threads(4)
    model = SBIClassifier(ROOT / "FFraw.tar").cuda().eval()
    extractor = SBIFaces("cuda")
    if extractor.signature != full["provenance"]["preprocessing"]:
        raise ValueError("Timing recheck preprocessing changed.")
    root = Path(manifest["dataset_root"]).resolve()
    rows = []
    for cached, original in flagged:
        source = (root / cached["record"]["path"]).resolve()
        if (
            not source.is_relative_to(root)
            or file_sha256(source) != cached["source_sha256"]
        ):
            raise ValueError("Timing recheck source changed.")
        measurements, differences = [], []
        for repeat in range(4):
            frames, timing = decode_selected(
                source, [original["index"]], cached["frame_count"], extractor, model
            )
            differences.append(compare_frame(original, frames[0]))
            measurements.append(
                dict(
                    repeat=repeat,
                    warmup=repeat == 0,
                    processing_seconds=frames[0]["processing_seconds"],
                    pass_seconds=timing["wall_seconds"],
                )
            )
        rows.append(
            dict(
                path=cached["record"]["path"],
                index=original["index"],
                raw_processing_seconds=original["processing_seconds"],
                measured_repeats=measurements,
                median_remeasured_seconds=float(
                    np.median([m["processing_seconds"] for m in measurements[1:]])
                ),
                max_face_score_absolute_delta=max(differences),
                affected_budgets=[
                    int(n)
                    for n, indices in cached["sets"].items()
                    if original["index"] in indices
                ],
            )
        )
    write_json(
        output,
        dict(
            full_result_sha256=file_sha256(STUDY / "full.json"),
            flagged_threshold_seconds=5.0,
            warmup_repeats=1,
            measured_repeats=3,
            frames=rows,
            original_scores_and_timing_unchanged=True,
            interpretation="Supplementary sensitivity estimate substitutes only the flagged frame's processing duration with its separately remeasured median. Never changes any probability, prediction, cohort or raw time. A nearby sleep/wake event is recorded; it does not establish the entire pause's cause.",
        ),
    )
    print(rows, flush=True)


if __name__ == "__main__":
    main()
