"""Explicit, separately frozen VIDEO-HOLDOUT evaluation; never runs by default."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.video_final_protocol import (
    LeakageBlocked,
    claim_once,
    test_targets_after_audit,
    update_once,
)
from src.video_final_test import sealed_read, sealed_write
from src.video_forensic import read_json, write_json
from src.video_holdout_evaluate import evaluate, export_report
from src.video_holdout_protocol import (
    BASE,
    PROTOCOL_DIR,
    load_protocol,
    preflight,
    prepare,
    snapshot_metadata_check,
    verify_passed_preflight,
)
from src.video_sbi12 import exclusive_lock


def factory():
    import torch
    from scripts.benchmark_video_adaptive import load_engine
    from src.dataset import seed_everything

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    return load_engine()


def write_cost_summary(out):
    attempts = [read_json(p) for p in sorted((out / "attempts").glob("*.json"))]
    environments = [
        sealed_read(p) for p in sorted((out / "environments").glob("*.json"))
    ]
    sealed_write(
        out / "execution_cost.json",
        dict(
            recorded_attempts=len(attempts),
            measured_attempt_wall_seconds=sum(
                a.get("actual_attempt_wall_seconds", 0.0) for a in attempts
            ),
            measured_preflight_seconds=sum(
                a.get("preflight_seconds", 0.0) for a in attempts
            ),
            attempts_without_completed_wall_measurement=sum(
                "actual_attempt_wall_seconds" not in a for a in attempts
            ),
            recorded_model_loads=len(environments),
            model_and_detector_load_seconds=sum(
                e.get("model_and_detector_load_seconds", 0.0) for e in environments
            ),
            synthetic_warmup_seconds=sum(
                e.get("synthetic_warmup_seconds", 0.0) for e in environments
            ),
            scope="Invocation wall includes audit, model load/warmup, shared decode/inference and output writes. Hard-killed attempts may be unmeasured; never reconstruct them as a cold-run benchmark. Do not sum method completion times as total runtime.",
        ),
    )


def run(protocol, sha):
    registry, out = Path(protocol["once_registry"]), Path(protocol["run_directory"])
    with exclusive_lock(BASE / "official_video_holdout.lock"):
        if not claim_once(registry, sha, out):
            if (out / "final_results.json").exists():
                write_cost_summary(out)
            print(
                f"Evaluation already complete; no audit/inference repeated: {out / 'final_results.json'}"
            )
            return
        out.mkdir(parents=True, exist_ok=True)
        attempt = out / "attempts" / (uuid4().hex + ".json")
        began = perf_counter()
        info = dict(
            protocol_sha256=sha,
            started_utc=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        write_json(attempt, info)
        try:
            tick = perf_counter()
            records, hashes = preflight(protocol, out)
            passed = verify_passed_preflight(out, protocol)
            info["preflight_seconds"] = perf_counter() - tick
            # Target conversion is intentionally delayed until the separate gate passed.
            targets = test_targets_after_audit(protocol, records, passed=True)
            update_once(registry, "inference")
            if (out / "final_results.json").exists():
                result = sealed_read(out / "final_results.json")
                if result["protocol_sha256"] != sha:
                    raise ValueError("Sealed results use another protocol")
            else:
                result = evaluate(protocol, sha, records, targets, hashes, out, factory)
            export_report(out, result, passed)
            update_once(registry, "complete")
            info["status"] = "complete"
            print(
                {m: d["summary"]["metrics"] for m, d in result["methods"].items()},
                flush=True,
            )
        except LeakageBlocked:
            update_once(registry, "blocked_overlap")
            info["status"] = "blocked_overlap"
            raise
        except BaseException as error:
            update_once(registry, "interrupted")
            info.update(
                status="interrupted", error=type(error).__name__, message=str(error)
            )
            raise
        finally:
            info["actual_attempt_wall_seconds"] = perf_counter() - began
            write_json(attempt, info)
            if info["status"] == "complete":
                write_cost_summary(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-protocol", action="store_true")
    action.add_argument("--check-protocol", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run-final-test", action="store_true")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_DIR / "protocol.json")
    parser.add_argument("--protocol-sha256")
    parser.add_argument("--resume", choices=("auto",))
    args = parser.parse_args()
    if args.prepare_protocol:
        if (
            args.protocol_sha256
            or args.resume
            or args.protocol != PROTOCOL_DIR / "protocol.json"
        ):
            parser.error(
                "Preparation has one fixed separate destination and no test flags"
            )
        print("Separate VIDEO-HOLDOUT protocol SHA-256: " + prepare())
        return
    if not args.protocol_sha256:
        parser.error("An externally pinned --protocol-sha256 is required")
    if args.run_final_test != (args.resume == "auto"):
        parser.error("Only --run-final-test requires/allows --resume auto")
    protocol = load_protocol(args.protocol, args.protocol_sha256)
    if args.check_protocol:
        print(snapshot_metadata_check(protocol))
    elif args.preflight:
        with exclusive_lock(BASE / "official_video_holdout.lock"):
            preflight(protocol, BASE / "preflight_checks")
        print(
            "Video-holdout path/family/hash preflight passed; no targets, model or inference"
        )
    else:
        try:
            run(protocol, args.protocol_sha256)
        except LeakageBlocked as error:
            print(f"BLOCKED BEFORE EVALUATION: {error}", flush=True)
            raise SystemExit(3) from error


if __name__ == "__main__":
    main()
