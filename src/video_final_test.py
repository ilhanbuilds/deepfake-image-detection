"""One frozen final evaluation with shared, sealed, resumable frame scores.

Importing this module performs no data access or inference. The official test
runner calls it only after the fail-closed leakage audit has succeeded.
"""

from pathlib import Path
from time import perf_counter

import numpy as np

from src.video_adaptive import AdaptiveVideo, mean_valid, method_summary, uncertain
from src.video_forensic import read_json, write_json
from src.video_frame_study import nested_indices
from src.video_manifest import digest_json
from src.video_pipeline import sample_indices


def sealed_write(path, payload):
    if path.exists():
        if sealed_read(path) != payload:
            raise FileExistsError("Committed cache/result is immutable.")
        return
    write_json(path, dict(payload=payload, integrity_sha256=digest_json(payload)))


def sealed_read(path):
    saved = read_json(path)
    if saved["integrity_sha256"] != digest_json(saved["payload"]):
        raise ValueError("Final-test cache/result integrity mismatch.")
    return saved["payload"]


class ScoreCache:
    def __init__(self, directory, protocol_sha, video_sha, path):
        self.directory = directory
        self.binding = dict(
            protocol_sha256=protocol_sha, source_sha256=video_sha, path=path
        )

    def load(self):
        frames = {}
        for path in sorted(self.directory.glob("*.json")):
            saved = sealed_read(path)
            if saved["binding"] != self.binding:
                raise ValueError("Final frame cache protocol/source/video mismatch.")
            frame = saved["frame"]
            index = frame["index"]
            if path.stem != f"{index:06d}" or index in frames:
                raise ValueError("Duplicate or renamed final frame cache entry.")
            self.validate(frame)
            frames[index] = frame
        return frames

    @staticmethod
    def validate(frame):
        if not isinstance(frame["index"], int) or frame["index"] < 0:
            raise ValueError("Invalid cached frame index.")
        if frame["status"] == "accepted":
            scores = [f["fake_probability"] for f in frame["faces"]]
            if (
                not scores
                or not np.isfinite(scores).all()
                or any(not 0 <= p <= 1 for p in scores)
                or frame["fake_probability"] != max(scores)
            ):
                raise ValueError("Invalid cached face/frame probabilities.")
        elif frame["status"] != "failed" or not frame.get("reason"):
            raise ValueError("Unreported cached frame failure.")

    def save(self, frame):
        self.validate(frame)
        sealed_write(
            self.directory / f"{frame['index']:06d}.json",
            dict(binding=self.binding, frame=frame),
        )


def shared_video(engine, source, cache):
    """Six-first shared schedule; each distinct frame's GPU work is committed once.

    Fixed12 intentionally shares the two-stage schedule. Completion times are
    actual endpoints in this schedule, NOT independent deployment benchmarks.
    """
    began = perf_counter()
    frames = cache.load()
    resumed_partial = bool(frames)
    new_calls = []

    def process(frame, index):
        if index in frames:
            raise RuntimeError("Cached frame was unnecessarily reinferred.")
        result = dict(engine.processor(frame, index), processing_attempted=True)
        cache.save(result)
        new_calls.append(index)
        return result

    controller = AdaptiveVideo(engine.cv, process)
    counters = dict(
        decode_seconds=0.0, decoded_grabs=0, retrieved_frames=0, decode_passes=1
    )
    tick = perf_counter()
    cap = engine.cv.VideoCapture(str(source))
    opened = cap.isOpened()
    count = int(cap.get(engine.cv.CAP_PROP_FRAME_COUNT)) if opened else 0
    counters["decode_seconds"] += perf_counter() - tick
    six, twelve = [], []
    try:
        if opened and count >= 6:
            sets = nested_indices(count, sample_indices(count, 6))
            six, twelve = sets[6], sets[12]
            if not set(frames) <= set(twelve):
                raise ValueError("Cached indices differ from frozen nested sampler.")
            controller._pass(cap, six, frames, counters)
    finally:
        cap.release()
    for frame in frames.values():
        cache.save(frame)
    six_elapsed = perf_counter() - began
    p6 = mean_valid(frames, six) if six else None
    missing = any(frames[i]["status"] != "accepted" for i in six)
    expand = bool(six) and (missing or p6 is None or uncertain(p6))
    extra = sorted(set(twelve) - frames.keys())
    if extra:
        counters["decode_passes"] += 1
        tick = perf_counter()
        cap = engine.cv.VideoCapture(str(source))
        reopened = cap.isOpened()
        counters["decode_seconds"] += perf_counter() - tick
        try:
            if reopened:
                if int(cap.get(engine.cv.CAP_PROP_FRAME_COUNT)) != count:
                    raise ValueError(
                        "Source frame count changed between decode passes."
                    )
                controller._pass(cap, extra, frames, counters)
            else:
                for i in extra:
                    frames[i] = dict(
                        index=i,
                        status="failed",
                        reason="reopen_failed",
                        processing_attempted=False,
                        processing_seconds=0.0,
                    )
        finally:
            cap.release()
    for frame in frames.values():
        cache.save(frame)
    full_elapsed = perf_counter() - began
    rows = {}
    for mode in ("6", "12", "adaptive"):
        indices = twelve if mode == "12" or mode == "adaptive" and expand else six
        valid = [frames[i] for i in indices if frames[i]["status"] == "accepted"]
        accepted = len(valid) >= 3
        score = mean_valid(frames, indices) if accepted else None
        completion = (
            full_elapsed
            if mode == "12" or mode == "adaptive" and expand
            else six_elapsed
        )
        warnings = []
        failures = [
            dict(index=i, reason=frames[i]["reason"])
            for i in indices
            if frames[i]["status"] != "accepted"
        ]
        if failures:
            warnings.append("missing_frames_no_imputation")
        if 6 <= count < 12:
            warnings.append("fewer_than_twelve_unique_frames")
        rows[mode] = dict(
            status="accepted" if accepted else "rejected",
            fake_probability=score,
            prediction=int(score >= 0.5) if accepted else None,
            rejection_reason=None
            if accepted
            else "video_open_failed"
            if not opened
            else "too_short_for_historical_six"
            if count < 6
            else "fewer_than_three_valid_frames",
            p6=p6,
            expanded=mode == "adaptive" and expand,
            expansion_reason=("incomplete_initial_six" if missing else "margin")
            if mode == "adaptive" and expand
            else None,
            selected_indices=indices,
            valid_indices=[f["index"] for f in valid],
            frame_scores=[f["fake_probability"] for f in valid],
            warnings=warnings,
            failures=failures,
            processed_frames=sum(frames[i]["processing_attempted"] for i in indices),
            classifier_face_calls=sum(len(frames[i].get("faces", [])) for i in indices),
            latency_seconds=None,  # never mislabel a shared schedule as independent runtime
            shared_completion_seconds=None if resumed_partial else completion,
            selected_frame_processing_seconds=sum(
                frames[i]["processing_seconds"] for i in indices
            ),
            peak_allocated_mib=max(
                [frames[i].get("peak_allocated_mib", 0.0) for i in indices] + [0.0]
            ),
            peak_reserved_mib=max(
                [frames[i].get("peak_reserved_mib", 0.0) for i in indices] + [0.0]
            ),
        )
    return dict(
        rows=rows,
        frames=[frames[i] for i in sorted(frames)],
        source_frame_count=count,
        newly_inferred_frames=len(new_calls),
        resumed_partial_frame_cache=resumed_partial,
        actual_current_attempt_wall_seconds=full_elapsed,
        shared_completion_timing_valid=not resumed_partial,
        counters=counters,
        timing_scope="Measured completion endpoints in the shared six-first/two-pass schedule. Not independent one-pass fixed12 or standalone adaptive speed. Partial-cache resume invalidates comparable completion latency.",
    )


def aggregate(records, targets, videos):
    methods = {}
    shared_paths = {
        r["path"]
        for r in records
        if all(
            videos[r["path"]]["rows"][m]["status"] == "accepted"
            for m in ("6", "12", "adaptive")
        )
    }
    for mode in ("6", "12", "adaptive"):
        rows = [
            dict(
                videos[r["path"]]["rows"][mode],
                path=r["path"],
                target=targets[r["path"]],
            )
            for r in records
        ]
        summary = method_summary(rows)
        timed = [
            r["shared_completion_seconds"]
            for r in rows
            if r["shared_completion_seconds"] is not None
        ]
        summary.update(
            actual_shared_completion_timing=dict(
                mean_seconds=float(np.mean(timed)) if timed else None,
                median_seconds=float(np.median(timed)) if timed else None,
                total_seconds=float(sum(timed)),
                measured_videos=len(timed),
                omitted_partial_resume_videos=len(rows) - len(timed),
                scope="Actual common-schedule completion, not independent per-method inference benchmark.",
            ),
            selected_frame_compute_seconds=float(
                sum(r["selected_frame_processing_seconds"] for r in rows)
            ),
            peak_allocated_mib=max(r["peak_allocated_mib"] for r in rows),
            peak_reserved_mib=max(r["peak_reserved_mib"] for r in rows),
        )
        common = [r for r in rows if r["path"] in shared_paths]
        summary["common_accepted_cohort"] = method_summary(common) if common else None
        methods[mode] = dict(summary=summary, rows=rows)
    return dict(
        methods=methods,
        official_attempted=len(records),
        common_accepted_count=len(shared_paths),
        actual_shared_execution_seconds=float(
            sum(v["actual_current_attempt_wall_seconds"] for v in videos.values())
        ),
        resumed_partial_videos=sum(
            v["resumed_partial_frame_cache"] for v in videos.values()
        ),
        note="Shared execution counts GPU work once. Do not sum the three completion times as wall runtime. Interrupted pre-resume attempt time is not reconstructed as a cold run.",
    )


def result_path(directory, record):
    return directory / "videos" / (digest_json(record) + ".json")


def evaluate(protocol, protocol_sha, records, targets, hashes, out, engine_factory):
    engine = None
    videos = {}
    root = Path(protocol["official"]["root"])
    from src.video_final_protocol import safe_source

    for index, record in enumerate(records, 1):
        binding = dict(
            protocol_sha256=protocol_sha,
            source_sha256=hashes[record["path"]],
            path=record["path"],
        )
        path = result_path(out, record)
        if path.exists():
            saved = sealed_read(path)
            if saved["binding"] != binding:
                raise ValueError("Committed video result source/protocol mismatch.")
            video = saved["video"]
        else:
            if engine is None:
                engine, environment = engine_factory()
                write_json(out / "environment.json", environment)
            source = safe_source(root, record)
            before = source.stat()
            cache = ScoreCache(
                out / "frames" / digest_json(record),
                protocol_sha,
                hashes[record["path"]],
                record["path"],
            )
            video = shared_video(engine, source, cache)
            after = source.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError("Test source changed during processing.")
            sealed_write(path, dict(binding=binding, video=video))
        videos[record["path"]] = video
        write_json(
            out / "progress.json",
            dict(
                committed_videos=index, total=len(records), protocol_sha256=protocol_sha
            ),
        )
        print(
            f"final video={index}/{len(records)} committed; no per-video performance displayed",
            flush=True,
        )
    result = aggregate(records, targets, videos)
    result.update(
        protocol_sha256=protocol_sha,
        checkpoint_sha256=protocol["checkpoint"]["sha256"],
        decision=protocol["decision"],
        leakage_audit_passed=True,
        interpretation="Frozen one-shot evaluation only. Do not retune using these results. Real person/source independence and cross-dataset generalization are not proven by the official split alone.",
    )
    sealed_write(out / "final_results.json", result)
    return result
