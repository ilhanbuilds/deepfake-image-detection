"""Prepare a frozen protocol OR explicitly claim/resume its single final test.

Never defaults to official-list access, test labels, video decoding or GPU load.
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from src.video_final_protocol import (
    BASE,
    PROTOCOL_DIR,
    LeakageBlocked,
    claim_once,
    leakage_audit,
    load_protocol,
    prepare,
    test_targets_after_audit,
    update_once,
)
from src.video_final_test import evaluate
from src.video_forensic import write_json
from src.video_sbi12 import exclusive_lock


def factory():
    import torch
    from src.dataset import seed_everything
    from scripts.benchmark_video_adaptive import load_engine

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    seed_everything(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    return load_engine()


def run(protocol, sha):
    registry = Path(protocol["once_registry"])
    out = Path(protocol["run_directory"])
    with exclusive_lock(BASE / "official_final.lock"):
        if not claim_once(registry, sha, out):
            print(
                f"Final evaluation already complete; no audit/inference repeated. Result: {out / 'final_results.json'}"
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
            records, hashes = leakage_audit(protocol, out)
            targets = test_targets_after_audit(protocol, records, passed=True)
            update_once(registry, "inference")
            result = evaluate(protocol, sha, records, targets, hashes, out, factory)
            update_once(registry, "complete")
            info["status"] = "complete"
            print(
                {method: data["summary"] for method, data in result["methods"].items()},
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
            write_json(out / "interrupted.json", info)
            raise
        finally:
            info["actual_attempt_wall_seconds"] = perf_counter() - began
            write_json(attempt, info)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare-protocol", action="store_true")
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
            raise ValueError(
                "Preparation uses the single fixed protocol directory; no test flags."
            )
        sha = prepare()
        print(
            f"Frozen protocol SHA-256: {sha}; test list, labels and videos not accessed."
        )
    else:
        if not args.protocol_sha256 or args.resume != "auto":
            parser.error(
                "Explicit --protocol-sha256 and --resume auto required for final test."
            )
        protocol = load_protocol(args.protocol, args.protocol_sha256)
        try:
            run(protocol, args.protocol_sha256)
        except LeakageBlocked as error:
            print(f"BLOCKED BEFORE EVALUATION: {error}", flush=True)
            raise SystemExit(3) from error


if __name__ == "__main__":
    main()
