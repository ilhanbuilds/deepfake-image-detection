"""Frozen final-test protocol and fail-closed leakage gate.

Preparation reads development metadata only and hashes manifests as opaque files.
The official list, labels and video bytes are accessed only by the explicit final
command, never by prepare(). Code/asset/protocol drift invalidates that command.
"""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path, PurePosixPath
import platform
import re

from src.video_adaptive import CHECKPOINT, CHECKPOINT_SHA, RUN
from src.video_forensic import ROOT, read_json, write_json
from src.video_manifest import FOLDERS, digest_json, family_info
from src.video_pipeline import file_sha256

BASE = Path("outputs/video_final_v1")
PROTOCOL_DIR = BASE / "protocol_20261018"
MANIFEST = Path("outputs/video_identity_v1/manifest.json")
PARENT = Path("outputs/video_v1/manifest.json")
AUDIT = ROOT / "roster_audit.json"
SOURCES = (
    "src/video_final_protocol.py",
    "src/video_final_test.py",
    "scripts/run_video_final_test.py",
    "scripts/benchmark_video_adaptive.py",
    "src/video_adaptive.py",
    "src/video_forensic.py",
    "src/video_frame_study.py",
    "src/video_pipeline.py",
    "src/video_manifest.py",
    "src/video_evaluate.py",
    "src/video_spatial_evaluate.py",
    "src/video_sampling.py",
    "src/evaluate.py",
    "src/dataset.py",
    "src/video_sbi12.py",
)
PACKAGES = (
    "torch",
    "torchvision",
    "numpy",
    "scikit-learn",
    "efficientnet-pytorch",
    "retinaface-pytorch",
    "albumentations",
    "opencv-python-headless",
)


class LeakageBlocked(RuntimeError):
    pass


def development_prefix(path):
    """Pinned pretty-JSON layout: stop BEFORE the first official test record.

    Closing the train/val-only prefix produces valid JSON without reading the
    test array, its labels, quarantine, or test counts later in the manifest.
    """
    prefix = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith('    "test":'):
                value = json.loads("".join(prefix) + '    "__not_read__": []}}')
                if set(value["splits"]) != {"train", "val", "__not_read__"}:
                    raise ValueError("Unexpected manifest layout.")
                del value["splits"]["__not_read__"]
                return value
            prefix.append(line)
    raise ValueError("Cannot safely parse development-only manifest prefix.")


def runtime_versions():
    return dict(
        python=platform.python_version(),
        packages={name: version(name) for name in PACKAGES},
    )


def prepare(directory=PROTOCOL_DIR):
    if directory.exists():
        raise FileExistsError("Frozen protocol directory already exists; no overwrite.")
    dev = development_prefix(MANIFEST)
    config, audit = read_json(RUN / "config.json"), read_json(AUDIT)
    if (
        file_sha256(CHECKPOINT) != CHECKPOINT_SHA
        or config["frames"] != 12
        or config["threshold"] != 0.5
    ):
        raise ValueError("Pinned epoch-6 configuration changed.")
    if config["manifest_fingerprint"] != audit["manifest_fingerprint"]:
        raise ValueError("Development metadata changed.")
    for split in ("train", "val"):
        raw = {r["path"]: r for r in dev["splits"][split]}
        accepted = audit["splits"][split]["accepted_records"]
        if (
            any(raw.get(r["path"]) != r for r in accepted)
            or digest_json(accepted) != config[f"{split}_roster_sha256"]
        ):
            raise ValueError("Model development cohort changed.")
    protected = {
        str(p): file_sha256(p)
        for p in (
            CHECKPOINT,
            RUN / "config.json",
            MANIFEST,
            PARENT,
            AUDIT,
            Path("app.py"),
            Path("outputs/checkpoints/best_model_v3_robust.pth"),
            ROOT / "retinaface.pth",
            ROOT / "upstream/src_inference_preprocess.py",
        )
    }
    protocol = dict(
        schema="sbi_final_official_518_v1",
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=dict(
            path=str(CHECKPOINT.resolve()),
            sha256=CHECKPOINT_SHA,
            epoch=6,
            model="SBI EfficientNet-B4",
        ),
        decision=dict(
            methods=["6", "12", "adaptive"],
            threshold=0.5,
            gate_margin=0.10,
            gate="0.4 < p6 < 0.6, equivalent to abs(p6-.5)<.10; strict endpoints",
            aggregation="maximum retained-face FAKE probability per frame, then arithmetic mean of valid frames",
            sampler="historical rounded linspace six, then widest-gap floor midpoint earliest tie to twelve",
        ),
        preprocessing=dict(
            resolution=380,
            pixels="RGB uint8 to FP32 /255; no ImageNet normalization",
            precision="FP32",
            face_batch_size=1,
            detector="official RetinaFace max_size=2048",
            crop="pinned official SBI bbox crop phase=test",
            faces="area >= half-largest, max face probability",
            minimum_valid_frames=3,
            under_six_unique="reject",
            six_to_eleven="unique available frames, explicit shortfall",
            missing_initial_frame="force adaptive expansion, no retry/imputation",
            rejection="preserve attempted denominator, report accepted and common-cohort metrics",
        ),
        official=dict(
            root=dev["dataset_root"],
            list_name="List_of_testing_videos.txt",
            expected_count=518,
            list_sha256=dev["official_sha256"],
            identity_manifest=str(MANIFEST.resolve()),
            risk_policy="block on any known path, canonical path, filename family, any-role identity code, or byte-hash overlap; no bypass or filtering",
        ),
        development_rosters=dev["splits"],
        development_manifest_fingerprint=config["manifest_fingerprint"],
        feature_cache=dict(
            directory=audit["cache_directory"],
            signature=audit["cache_signature"],
            fingerprint=audit["cache_fingerprint"],
        ),
        run_directory=str((BASE / "final_test_20261018").resolve()),
        once_registry=str((BASE / "official_518_once.json").resolve()),
        source_sha256={name: file_sha256(Path(name)) for name in SOURCES},
        protected_sha256=protected,
        runtime=runtime_versions(),
        evaluation=dict(
            shared_frame_cache="atomic sealed per-frame JSON bound to protocol SHA, source SHA, path and index",
            schedule="six first, then all extra frames once for fixed twelve; adaptive reuses the appropriate prefix",
            timing="actual shared-schedule completion times, not independent one-pass fixed12/standalone adaptive latency benchmarks",
            resume_timing="partial-cache resumed videos have no comparable original completion latency; record actual current attempt wall separately",
            final_results="one sealed result set; completion forbids new evaluation; no tuning flags",
        ),
        limitations=[
            "Gate .10 and epoch were selected on validation; test must not retune them.",
            "Filename codes do not guarantee real person/source independence; re-encoded/shared clips and pretraining overlaps remain unverified.",
            "Existing historical identity audit already reports official-test identity-code overlap; the command must stop if confirmed.",
        ],
        prepared_without_test_list_labels_or_video_access=True,
    )
    encoded = (
        json.dumps(
            protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode()
    import hashlib

    sha = hashlib.sha256(encoded).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "protocol.json").open("xb") as handle:
        handle.write(encoded)
    with (directory / "protocol.sha256").open("x", encoding="ascii") as handle:
        handle.write(sha + "\n")
    write_json(
        directory / "preparation.json",
        dict(
            protocol_sha256=sha,
            test_list_opened=False,
            test_videos_opened=False,
            test_labels_inspected=False,
            model_forward_calls=0,
            immutability="Exclusive creation plus externally pinned SHA-256. Any config/code/runtime drift is rejected; not a claim of OS-level write-once storage.",
        ),
    )
    return sha


def load_protocol(path, expected_sha):
    if file_sha256(path) != expected_sha:
        raise ValueError("Frozen protocol hash mismatch.")
    p = read_json(path)
    if (
        p["schema"] != "sbi_final_official_518_v1"
        or p["checkpoint"]["sha256"] != CHECKPOINT_SHA
        or p["checkpoint"]["epoch"] != 6
        or p["official"]["expected_count"] != 518
        or p["decision"]["methods"] != ["6", "12", "adaptive"]
        or p["decision"]["threshold"] != 0.5
        or p["decision"]["gate_margin"] != 0.10
    ):
        raise ValueError("Unsupported final-test protocol.")
    if p["runtime"] != runtime_versions():
        raise ValueError("Frozen runtime changed; resume/evaluation refused.")
    for filename, sha in {**p["source_sha256"], **p["protected_sha256"]}.items():
        if file_sha256(Path(filename)) != sha:
            raise ValueError(f"Frozen implementation/asset changed: {filename}")
    return p


def official_paths(protocol):
    """Called only after explicit --run-final-test; no target decoding here."""
    path = Path(protocol["official"]["root"]) / protocol["official"]["list_name"]
    if file_sha256(path) != protocol["official"]["list_sha256"]:
        raise ValueError("Official list changed.")
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"[01] ([^\s]+\.mp4)", line)
        if not match:
            raise ValueError("Invalid official list syntax.")
        relative = match[1]
        posix = PurePosixPath(relative)
        if (
            len(posix.parts) != 2
            or posix.is_absolute()
            or ".." in posix.parts
            or posix.parts[0] not in FOLDERS
        ):
            raise ValueError("Unsafe official video path.")
        family, codes = family_info(relative)
        records.append(dict(path=relative, family=family, identities=codes))
    if len(records) != protocol["official"]["expected_count"] or len(
        {r["path"] for r in records}
    ) != len(records):
        raise ValueError(
            "Exact unique official 518-video roster required; no subset selection."
        )
    # Membership only; target fields are not used by the leakage gate.
    locked = read_json(Path(protocol["official"]["identity_manifest"]))
    known = {r["path"]: r for r in locked["splits"]["test"]}
    if {r["path"] for r in records} != set(known):
        raise ValueError("Official roster differs from the frozen manifest.")
    for record in records:
        old = known[record["path"]]
        if (
            old["family"] != record["family"]
            or old["identities"] != record["identities"]
        ):
            raise ValueError("Official filename metadata changed.")
        record.update(expected_size=old["size"], expected_mtime_ns=old["mtime_ns"])
    return records


def safe_source(root, record):
    source = (root / record["path"]).resolve()
    if (
        not source.is_relative_to(root.resolve())
        or (root / record["path"]).is_symlink()
    ):
        raise ValueError("Unsafe dataset source path.")
    return source


def metadata_overlap(records, development, root):
    result = {}
    for split, known in development.items():
        paths = {r["path"] for r in known}
        canonical = {str(safe_source(root, r)).casefold() for r in known}
        families = {r["family"] for r in known}
        codes = {code for r in known for code in r["identities"]}
        result[split] = dict(
            paths=sorted(r["path"] for r in records if r["path"] in paths),
            canonical_paths=sorted(
                r["path"]
                for r in records
                if str(safe_source(root, r)).casefold() in canonical
            ),
            families=sorted({r["family"] for r in records} & families),
            any_role_identity_codes=sorted(
                {code for r in records for code in r["identities"]} & codes
            ),
        )
    return result


def leakage_audit(protocol, out):
    """Any metadata risk blocks BEFORE MP4 reads, target conversion or GPU load."""
    import torch

    records = official_paths(protocol)
    root = Path(protocol["official"]["root"])
    overlap = metadata_overlap(records, protocol["development_rosters"], root)
    if any(values for split in overlap.values() for values in split.values()):
        write_json(
            out / "leakage_blocked.json",
            dict(
                stage="metadata",
                overlap=overlap,
                model_loaded=False,
                test_mp4_opened=False,
                test_targets_converted=False,
                reason="Known development/test membership, family or identity-code overlap; no independent identity claim and no override.",
            ),
        )
        raise LeakageBlocked(
            "Known train/validation-to-test overlap; final inference not started. See leakage_blocked.json."
        )
    started = datetime.now(timezone.utc).isoformat()
    memberships, source_hashes, cache_signature = {}, {}, []
    cache = protocol["feature_cache"]
    for split, rows in protocol["development_rosters"].items():
        for record in rows:
            source = safe_source(root, record)
            cache_path = Path(cache["directory"]) / (digest_json(record) + ".pt")
            old = torch.load(cache_path, map_location="cpu", weights_only=True)
            if old["record"] != record or old["signature"] != cache["signature"]:
                raise ValueError("Frozen development feature metadata changed.")
            sha = file_sha256(source)
            if sha != old["source_sha256"]:
                raise ValueError("Development source video changed since training.")
            memberships.setdefault(sha, []).append(
                dict(split=split, path=record["path"])
            )
            cache_signature.append((split, record, file_sha256(cache_path)))
    if digest_json(cache_signature) != cache["fingerprint"]:
        raise ValueError("Historical development cache fingerprint changed.")
    duplicates, within_test = [], {}
    for record in records:
        source = safe_source(root, record)
        before = source.stat()
        if "expected_size" in record and (before.st_size, before.st_mtime_ns) != (
            record["expected_size"],
            record["expected_mtime_ns"],
        ):
            raise ValueError("Official video inventory changed from frozen manifest.")
        sha = file_sha256(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Official video changed during hash audit.")
        source_hashes[record["path"]] = sha
        if sha in memberships:
            duplicates.append(
                dict(test_path=record["path"], sha256=sha, development=memberships[sha])
            )
        within_test.setdefault(sha, []).append(record["path"])
    internal = [paths for paths in within_test.values() if len(paths) > 1]
    if duplicates or internal:
        write_json(
            out / "leakage_blocked.json",
            dict(
                stage="hash",
                development_duplicates=duplicates,
                identical_test_groups=internal,
                model_loaded=False,
                test_targets_converted=False,
            ),
        )
        raise LeakageBlocked(
            "Exact source-byte duplicates found; no inference started."
        )
    result = dict(
        passed=True,
        official_count=len(records),
        metadata_overlap=overlap,
        source_hashes=source_hashes,
        started_utc=started,
        limitation="Byte hashes do not rule out re-encoded/shared segments or aliases for the same person; pretraining person/source overlap unverified.",
        protocol_digest=digest_json(protocol),
    )
    audit_path = out / "leakage_passed.json"
    if audit_path.exists():
        previous = read_json(audit_path)
        if previous.get("integrity_sha256") != digest_json(
            {k: v for k, v in previous.items() if k != "integrity_sha256"}
        ):
            raise ValueError("Committed leakage-audit integrity mismatch.")
        if any(
            previous[key] != result[key]
            for key in (
                "protocol_digest",
                "source_hashes",
                "metadata_overlap",
                "official_count",
            )
        ):
            raise ValueError(
                "Frozen test source bytes/protocol changed; resume refused."
            )
    else:
        result["integrity_sha256"] = digest_json(result)
        write_json(audit_path, result)
    return records, source_hashes


def test_targets_after_audit(protocol, records, passed):
    if not passed:
        raise LeakageBlocked("No targets before successful overlap audit.")
    path = Path(protocol["official"]["root"]) / protocol["official"]["list_name"]
    if file_sha256(path) != protocol["official"]["list_sha256"]:
        raise ValueError("Official list changed after audit.")
    targets = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        value, relative = line.split(" ")
        targets[relative] = 1 - int(value)
        if targets[relative] != FOLDERS[relative.split("/")[0]]:
            raise ValueError("Official label/folder mapping mismatch.")
    if set(targets) != {r["path"] for r in records}:
        raise ValueError("Official label roster changed.")
    return targets


def claim_once(registry, protocol_sha, run_directory):
    if registry.exists():
        state = read_json(registry)
        check_registry(state)
        if state["protocol_sha256"] != protocol_sha or state["run_directory"] != str(
            run_directory
        ):
            raise ValueError(
                "Official test already claimed by another protocol/run; no second evaluation."
            )
        if state["status"] == "complete":
            return False
        if state["status"] == "blocked_overlap":
            raise LeakageBlocked(
                "This frozen run was already blocked by overlap; no evaluation or bypass."
            )
        if state["status"] not in ("started", "inference", "interrupted"):
            raise ValueError("Unknown final-test registry state.")
        return True
    state = dict(
        protocol_sha256=protocol_sha,
        run_directory=str(run_directory),
        status="started",
        claimed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    state["integrity_sha256"] = digest_json(state)
    write_json(registry, state)
    return True


def update_once(registry, status):
    value = read_json(registry)
    check_registry(value)
    del value["integrity_sha256"]
    value["status"] = status
    value["integrity_sha256"] = digest_json(value)
    write_json(registry, value)


def check_registry(value):
    if value.get("integrity_sha256") != digest_json(
        {k: v for k, v in value.items() if k != "integrity_sha256"}
    ):
        raise ValueError("Final-test registry integrity mismatch.")
