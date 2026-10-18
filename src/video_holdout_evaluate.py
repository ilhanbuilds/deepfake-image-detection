"""Shared inference with explicit video-holdout interpretation and full coverage."""

import csv
import io
from pathlib import Path

from src.video_final_protocol import safe_source
from src.video_final_test import (
    ScoreCache,
    aggregate,
    result_path,
    sealed_read,
    sealed_write,
    shared_video,
)
from src.video_forensic import write_json
from src.video_manifest import digest_json


def summarize_holdout(protocol, protocol_sha, records, targets, videos):
    result = aggregate(records, targets, videos)
    for data in result["methods"].values():
        rows, summary = data["rows"], data["summary"]
        summary["mean_valid_face_frames"] = sum(
            len(r["valid_indices"]) for r in rows
        ) / len(rows)
        summary["mean_selected_unique_frames"] = sum(
            len(r["selected_indices"]) for r in rows
        ) / len(rows)
        summary["coverage_by_source"] = {}
        for source in sorted({r["path"].split("/")[0] for r in rows}):
            subset = [r for r in rows if r["path"].startswith(source + "/")]
            accepted = [r for r in subset if r["status"] == "accepted"]
            summary["coverage_by_source"][source] = dict(
                attempted=len(subset),
                accepted=len(accepted),
                rejected=len(subset) - len(accepted),
                rejected_paths=[r["path"] for r in subset if r["status"] != "accepted"],
            )
        correct = sum(
            r["status"] == "accepted" and r["prediction"] == r["target"] for r in rows
        )
        summary["correct_over_all_attempts_rejections_unsuccessful"] = correct / len(
            rows
        )
        summary["metric_denominator"] = (
            "Classification metrics on accepted videos; coverage, rejections and common accepted cohort separately reported."
        )
    result.update(
        schema="sbi_video_holdout_results_v1",
        protocol_sha256=protocol_sha,
        checkpoint_sha256=protocol["checkpoint"]["sha256"],
        decision=protocol["decision"],
        independence_policy=protocol["independence_policy"],
        limitations=protocol["limitations"],
        decode_counters={
            key: sum(v["counters"][key] for v in videos.values())
            for key in (
                "decode_seconds",
                "decoded_grabs",
                "retrieved_frames",
                "decode_passes",
            )
        },
        new_frame_inferences_in_committed_video_attempts=sum(
            v["newly_inferred_frames"] for v in videos.values()
        ),
        cohort_definition="All 518 official paths attempted for all methods; no score-based subset or fallback imputation.",
        one_decision_evaluation=True,
        retuning_allowed=False,
    )
    return result


def evaluate(protocol, protocol_sha, records, targets, hashes, out, engine_factory):
    if set(targets) != {r["path"] for r in records}:
        raise ValueError("Target/cohort mismatch")
    engine, videos = None, {}
    root = Path(protocol["official"]["root"])
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
                raise ValueError("Committed video source/protocol mismatch")
            video = saved["video"]
        else:
            if engine is None:
                engine, environment = engine_factory()
                sealed_write(
                    out
                    / "environments"
                    / f"{len(list((out / 'environments').glob('*.json'))):04d}.json",
                    environment,
                )
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
                raise ValueError("Test video changed during processing")
            sealed_write(path, dict(binding=binding, video=video))
        videos[record["path"]] = video
        write_json(
            out / "progress.json",
            dict(
                committed_videos=index, total=len(records), protocol_sha256=protocol_sha
            ),
        )
        print(
            f"video-holdout video={index}/{len(records)} committed; performance withheld until completion",
            flush=True,
        )
    result = summarize_holdout(protocol, protocol_sha, records, targets, videos)
    sealed_write(out / "final_results.json", result)
    return result


def export_report(out, result, preflight):
    """Reproducible exports after sealed results; no extra inference or selection."""
    fields = (
        "method",
        "path",
        "target",
        "status",
        "fake_probability",
        "prediction",
        "processed_frames",
        "expanded",
        "rejection_reason",
    )
    path = out / "predictions.csv"
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for mode, data in result["methods"].items():
        for row in data["rows"]:
            writer.writerow(
                {key: mode if key == "method" else row[key] for key in fields}
            )
    expected_csv = stream.getvalue().encode("utf-8-sig")
    if path.exists():
        if path.read_bytes() != expected_csv:
            raise ValueError("Completed prediction CSV changed")
    else:
        temporary = path.with_suffix(".csv.tmp")
        temporary.write_bytes(expected_csv)
        temporary.replace(path)
    lines = [
        "# SBI video-holdout: one frozen decision evaluation",
        "",
        "Metrics below use accepted videos. All official attempts, method-specific rejections, common-cohort metrics, confusion matrices, per-source results and cost details are in final_results.json.",
        "",
        "| Method | Attempted | Accepted | Accuracy | ROC-AUC | Macro-F1 | REAL recall | FAKE recall | Mean frames |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, data in result["methods"].items():
        s = data["summary"]
        m = s["metrics"]
        values = [m.get(k) if m else None for k in ("accuracy", "roc_auc", "macro_f1")]
        recalls = [m["per_class"][c]["recall"] if m else None for c in ("REAL", "FAKE")]
        formatted = ["NA" if v is None else f"{v:.8f}" for v in values + recalls]
        lines.append(
            f"| {mode} | {s['attempted']} | {s['evaluated']} | "
            + " | ".join(formatted)
            + f" | {s['mean_processed_frames']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Identity overlap is reported, not a blocking video-overlap condition:",
            "",
            json_block(preflight["metadata"]["identity_overlap"]),
            "",
            "Timing: shared six-first/two-pass completion endpoints; no independent per-method speedup claim. Model load/warmup: environments/. Actual invocation wall and preflight duration: attempts/. Consolidated measured cost and hard-interruption gaps: execution_cost.json.",
            "",
            "Frozen protocol SHA-256: " + result["protocol_sha256"],
            "",
            *["- " + text for text in result["limitations"]],
        ]
    )
    report = out / "REPORT.md"
    expected = "\n".join(lines) + "\n"
    if report.exists():
        if report.read_text(encoding="utf-8") != expected:
            raise ValueError("Completed report changed")
    else:
        report.open("x", encoding="utf-8").write(expected)


def json_block(value):
    import json

    return "```json\n" + json.dumps(value, indent=2) + "\n```"
