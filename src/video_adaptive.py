"""Fixed exploratory SBI 6->12 cascade; separate from all legacy inference paths.

Raw videos use at most one decoded frame at a time and a second sequential pass
only for expansion. The per-call score cache prevents repeated detector/model
work, including retries of failed frames. No historical cache is written.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from src.video_forensic import read_json, summarize
from src.video_frame_study import nested_indices
from src.video_pipeline import file_sha256, sample_indices

WORK = Path("outputs/video_adaptive_v1")
RUN = Path("outputs/video_research_v1/finetune12/main_20261014")
CHECKPOINT = RUN / "checkpoints/best_video.pth"
CHECKPOINT_SHA = "a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c"
PAIRED = Path("outputs/sbi_epoch6_frames_6_vs_12_20261015/analysis.json")
MARGIN = 0.10
MODES = ("6", "12", "adaptive")


def uncertain(score):
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Invalid gate probability.")
    # Algebraically |p-.5|<.10. Central-interval form preserves strict .4/.6
    # boundary exclusion without binary subtraction round-off at those values.
    return 0.5 - MARGIN < score < 0.5 + MARGIN


def mean_valid(frames, indices):
    probabilities = [
        frames[i]["fake_probability"]
        for i in indices
        if frames[i]["status"] == "accepted"
    ]
    if not probabilities:
        return None
    if not np.isfinite(probabilities).all() or any(
        not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError("Invalid frame probability.")
    return float(np.mean(probabilities))


def offline_rows(paired):
    """Use recorded epoch-6 frames only; no model or adaptive threshold sweep."""
    if (
        paired["provenance"]["checkpoint_sha256"] != CHECKPOINT_SHA
        or paired["provenance"]["epoch"] != 6
        or len(paired["paired_rows"]) != 602
        or len({r["path"] for r in paired["paired_rows"]}) != 602
    ):
        raise ValueError("Pinned complete epoch-6 paired records required.")
    result = {mode: [] for mode in MODES}
    for row in paired["paired_rows"]:
        six, twelve = row["indices_6"], row["indices_12"]
        scores = {
            int(index): value for index, value in row["frame_probabilities"].items()
        }
        if (
            len(six) != 6
            or len(set(six)) != 6
            or len(twelve) != 12
            or len(set(twelve)) != 12
            or not set(six) <= set(twelve)
            or set(scores) != set(twelve)
            or not np.isfinite(list(scores.values())).all()
            or any(not 0 <= p <= 1 for p in scores.values())
        ):
            raise ValueError("Incomplete six/twelve frame scores.")
        p6 = float(np.mean([scores[i] for i in six]))
        p12 = float(np.mean([scores[i] for i in twelve]))
        if p6 != row["score_6"] or p12 != row["score_12"]:
            raise ValueError("Recorded six/twelve mean mismatch.")
        expand = uncertain(p6)
        for mode in MODES:
            expanded = mode == "adaptive" and expand
            selected = twelve if mode == "12" or expanded else six
            score = p12 if mode == "12" or expanded else p6
            result[mode].append(
                dict(
                    path=row["path"],
                    target=row["target"],
                    status="accepted",
                    fake_probability=score,
                    prediction=int(score >= 0.5),
                    p6=p6,
                    expanded=expanded,
                    expansion_reason="margin" if expanded else None,
                    processed_frames=len(selected),
                    attempted_frames=len(selected),
                    valid_face_frames=len(selected),
                    selected_indices=selected,
                    frame_scores=[scores[i] for i in selected],
                    latency_seconds=None,
                    peak_allocated_mib=None,
                    peak_reserved_mib=None,
                )
            )
    for mode in ("6", "12"):
        actual = summarize(result[mode], 602)
        if any(
            actual[key] != paired["methods"][mode][key]
            for key in ("metrics", "sources", "coverage")
        ):
            raise ValueError("Recorded fixed reference did not reproduce exactly.")
    return result


def method_summary(rows):
    summary = summarize(rows, len(rows))
    summary.update(
        false_alarms=sum(
            r["status"] == "accepted" and r["target"] == 0 and r["prediction"] == 1
            for r in rows
        ),
        mean_processed_frames=float(np.mean([r["processed_frames"] for r in rows])),
        median_processed_frames=float(np.median([r["processed_frames"] for r in rows])),
        expansion_rate=sum(r["expanded"] for r in rows) / len(rows),
        requested_twelve_rate=sum(len(r["selected_indices"]) == 12 for r in rows)
        / len(rows),
        partial_videos=sum(bool(r.get("warnings")) for r in rows),
    )
    measured = [r for r in rows if r["latency_seconds"] is not None]
    summary["latency"] = (
        None
        if not measured
        else dict(
            mean_seconds=float(np.mean([r["latency_seconds"] for r in measured])),
            median_seconds=float(np.median([r["latency_seconds"] for r in measured])),
            p95_seconds=float(
                np.quantile([r["latency_seconds"] for r in measured], 0.95)
            ),
            videos=len(measured),
        )
    )
    summary["gpu_memory"] = (
        None
        if not measured
        else dict(
            peak_allocated_mib=max(r["peak_allocated_mib"] for r in measured),
            peak_reserved_mib=max(r["peak_reserved_mib"] for r in measured),
            scope="PyTorch process allocator; reserved shares allocator history, not isolated total GPU memory.",
        )
    )
    return summary


class AdaptiveVideo:
    """Injected official frame processor keeps the control flow testable on CPU."""

    def __init__(self, cv, processor):
        self.cv, self.processor = cv, processor

    def _pass(self, cap, indices, frames, counters):
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate indices requested.")
        requested = set(indices) - frames.keys()
        if not requested:
            return
        for index in range(max(requested) + 1):
            began = perf_counter()
            grabbed = cap.grab()
            counters["decoded_grabs"] += 1
            ok, frame = (
                cap.retrieve() if grabbed and index in requested else (False, None)
            )
            counters["decode_seconds"] += perf_counter() - began
            if not grabbed:
                for remaining in sorted(requested):
                    if remaining >= index:
                        frames[remaining] = dict(
                            index=remaining,
                            status="failed",
                            reason="decode_failed",
                            processing_attempted=False,
                            processing_seconds=0.0,
                        )
                break
            if index not in requested:
                continue
            counters["retrieved_frames"] += 1
            if not ok or frame is None:
                frames[index] = dict(
                    index=index,
                    status="failed",
                    reason="retrieve_failed",
                    processing_attempted=False,
                    processing_seconds=0.0,
                )
                continue
            measured = self.processor(frame, index)
            if measured["index"] != index or measured["status"] not in (
                "accepted",
                "failed",
            ):
                raise ValueError("Frame processor returned invalid identity/status.")
            if measured["status"] == "accepted" and (
                not measured.get("faces")
                or not np.isfinite(measured["fake_probability"])
                or not 0 <= measured["fake_probability"] <= 1
            ):
                raise ValueError("Invalid processed face score.")
            frames[index] = dict(measured, processing_attempted=True)

    def predict(self, source, mode="adaptive", expected_frame_count=None):
        if mode not in MODES:
            raise ValueError("Unknown inference mode.")
        began = perf_counter()
        counters = dict(
            decode_seconds=0.0, decoded_grabs=0, retrieved_frames=0, decode_passes=1
        )
        tick = perf_counter()
        cap = self.cv.VideoCapture(str(source))
        opened = cap.isOpened()
        count = int(cap.get(self.cv.CAP_PROP_FRAME_COUNT)) if opened else 0
        counters["decode_seconds"] += perf_counter() - tick
        frames, six, twelve, selected = {}, [], [], []
        expanded, p6, reason = False, None, None
        try:
            if not opened:
                reason = "video_open_failed"
            elif count < 6:
                reason = "too_short_for_six_unique_historical_frames"
            elif expected_frame_count is not None and count != expected_frame_count:
                raise ValueError("Video frame count changed.")
            else:
                sets = nested_indices(count, sample_indices(count, 6))
                six, twelve = sets[6], sets[12]
                selected = twelve if mode == "12" else six
                self._pass(cap, selected, frames, counters)
        finally:
            cap.release()
        if six:
            p6 = mean_valid(frames, six)
        if mode == "adaptive" and six:
            incomplete = any(frames[i]["status"] != "accepted" for i in six)
            expanded = incomplete or p6 is None or uncertain(p6)
            if expanded:
                reason = "incomplete_initial_six" if incomplete else "margin"
                selected = twelve
                extras = sorted(set(twelve) - frames.keys())
                if extras:
                    counters["decode_passes"] += 1
                    tick = perf_counter()
                    cap = self.cv.VideoCapture(str(source))
                    reopened = cap.isOpened()
                    counters["decode_seconds"] += perf_counter() - tick
                    try:
                        if not reopened:
                            for i in extras:
                                frames[i] = dict(
                                    index=i,
                                    status="failed",
                                    reason="reopen_failed",
                                    processing_attempted=False,
                                    processing_seconds=0.0,
                                )
                        else:
                            self._pass(cap, extras, frames, counters)
                    finally:
                        cap.release()
        valid = [frames[i] for i in selected if frames[i]["status"] == "accepted"]
        accepted = len(valid) >= 3
        failures = [
            dict(index=i, reason=frames[i]["reason"])
            for i in selected
            if frames[i]["status"] == "failed"
        ]
        warnings = []
        if count >= 6 and count < 12:
            warnings.append("fewer_than_twelve_unique_frames")
        if failures:
            warnings.append("missing_face_or_decode_failure_no_imputation")
        score = mean_valid(frames, selected) if accepted else None
        return dict(
            status="accepted" if accepted else "rejected",
            fake_probability=score,
            prediction=int(score >= 0.5) if accepted else None,
            rejection_reason=None
            if accepted
            else reason
            if not six
            else "fewer_than_three_valid_face_frames",
            p6=p6,
            expanded=expanded,
            expansion_reason=reason if expanded else None,
            frame_count=count,
            selected_indices=selected,
            historical_six=six,
            available_twelve=twelve,
            valid_indices=[f["index"] for f in valid],
            frame_scores=[f["fake_probability"] for f in valid],
            processed_frames=sum(f["processing_attempted"] for f in frames.values()),
            attempted_frames=len(frames),
            valid_face_frames=len(valid),
            failures=failures,
            warnings=warnings,
            budget_shortfall=max(0, 12 - count),
            classifier_face_calls=sum(len(f.get("faces", [])) for f in frames.values()),
            frames=[frames[i] for i in sorted(frames)],
            latency_seconds=perf_counter() - began,
            frame_processing_seconds=sum(
                f["processing_seconds"] for f in frames.values()
            ),
            peak_allocated_mib=max(
                [f.get("peak_allocated_mib", 0.0) for f in frames.values()] + [0.0]
            ),
            peak_reserved_mib=max(
                [f.get("peak_reserved_mib", 0.0) for f in frames.values()] + [0.0]
            ),
            **counters,
        )


def verified_source(root, record, audit):
    """Only caller-selected accepted train/val records; no dataset inventory scan."""
    import torch
    from src.video_manifest import digest_json

    source = (root / record["path"]).resolve()
    if not source.is_relative_to(root.resolve()):
        raise ValueError("Source escaped dataset root.")
    historical = torch.load(
        Path(audit["cache_directory"]) / (digest_json(record) + ".pt"),
        map_location="cpu",
        weights_only=True,
    )
    if (
        historical["record"] != record
        or historical["signature"] != audit["cache_signature"]
        or file_sha256(source) != historical["source_sha256"]
    ):
        raise ValueError("Source/cache provenance changed.")
    return source, historical["report"]["frame_count"]


def authorize_input(video, root, records):
    video, root = video.resolve(), root.resolve()
    known = {(root / r["path"]).resolve(): r for r in records}
    if video.is_relative_to(root) and video not in known:
        raise ValueError(
            "Only accepted train/val videos may be opened inside dataset root; test/quarantine locked."
        )
    return known.get(video)


def load_saved():
    paired = read_json(PAIRED)
    if file_sha256(CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("Pinned epoch-6 checkpoint changed.")
    saved = read_json(RUN / "epoch_006_validation.json")
    original = {row["path"]: row for row in saved["rows"]}
    if len(original) != 602 or len(paired["paired_rows"]) != 602:
        raise ValueError("Incomplete recorded epoch-6 validation.")
    for row in paired["paired_rows"]:
        recorded = original[row["path"]]
        frames = dict(
            zip(
                [f["index"] for f in recorded["report"]["frames"]],
                recorded["frame_scores"],
                strict=True,
            )
        )
        if (
            frames != {int(k): v for k, v in row["frame_probabilities"].items()}
            or row["target"] != recorded["target"]
            or row["score_12"] != recorded["fake_probability"]
        ):
            raise ValueError("Paired frame scores differ from epoch-6 validation.")
    return paired
