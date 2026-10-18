"""Separate video-holdout policy; the older identity-blocking protocol is immutable."""

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

from src.video_final_protocol import (
    LeakageBlocked,
    check_registry,
    metadata_overlap,
    runtime_versions,
    safe_source,
)
from src.video_final_test import sealed_read, sealed_write
from src.video_forensic import read_json
from src.video_manifest import digest_json, family_info
from src.video_pipeline import file_sha256

BASE = Path("outputs/video_holdout_v1")
PROTOCOL_DIR = BASE / "protocol_20261018"
LEGACY = Path("outputs/video_final_v1/protocol_20261018/protocol.json")
LEGACY_SHA = "a14a2ae31482caf1b57afb4bc621f9f84a0c3a5b06d822c90e4b339d4a9f7eab"
HASH_AUDIT = Path("outputs/sbi_test_metadata_audit_20261018")
SOURCE_AUDIT = Path("outputs/sbi_source_family_audit_20261018")
NEW_SOURCES = (
    "src/video_holdout_protocol.py",
    "src/video_holdout_evaluate.py",
    "scripts/run_video_holdout_test.py",
    "tests/test_video_holdout.py",
)
SCHEMA = "sbi_official_518_video_holdout_v1"
POLICY = dict(
    blocking=("paths", "canonical_paths", "families", "sha256"),
    identity_codes="report_only_not_a_blocker",
    carrier_mapping="Celeb-synthesis/idA_idB_NNNN.mp4 -> Celeb-real/idA_NNNN.mp4",
    mapping_evidence="Supported by six preselected development triplets only; no official filename-order specification or test-frame verification.",
)
LIMITATIONS = [
    "Within-Celeb-DF video holdout; not identity-disjoint or cross-dataset evaluation.",
    "Carrier naming is supported by six development examples, not pixel-certified for every video; official filename-order specification was not found.",
    "Re-encoded/shared segments, common original uploads, donor training clips, YouTube identities and pretraining overlaps remain unverified.",
    "Epoch 6, twelve frames and adaptive margin .10 were developed/selected on validation; no test-based tuning is permitted.",
    "Accepted-cohort metrics exclude rejections; all 518 attempts, method-specific exclusions and the common accepted cohort must be reported.",
    "Shared six-first/two-pass completion times are measured endpoints, not independent deployment speed benchmarks for the three methods.",
]


def metadata_record(path):
    if not re.fullmatch(
        r"(?:Celeb-real|Celeb-synthesis|YouTube-real)/[^/\\\s]+\.mp4", path
    ):
        raise ValueError("Unsafe holdout filename")
    family, identities = family_info(path)
    return dict(path=path, family=family, identities=identities)


def assert_legacy_unused(protocol):
    """Read only: never claim, unlock, update or bypass the older registry."""
    registry = Path(protocol["legacy_once_registry"])
    run = Path(protocol["legacy_run_directory"])
    if any(
        (run / name).exists()
        for name in ("environment.json", "frames", "videos", "final_results.json")
    ):
        raise ValueError(
            "Older protocol has inference artifacts; no second decision evaluation"
        )
    if registry.exists():
        state = read_json(registry)
        check_registry(state)
        if state["status"] != "blocked_overlap":
            raise ValueError("Older evaluation already claimed; new evaluation refused")


def snapshot_metadata_check(protocol):
    """No official list, MP4, labels, decoder or model access."""
    test = protocol["test_roster"]
    dev = protocol["development_rosters"]
    root = Path(protocol["official"]["root"])
    for rows in (test, *dev.values()):
        for record in rows:
            if any(record[k] != v for k, v in metadata_record(record["path"]).items()):
                raise ValueError("Stored family/identity differs from filename")
    overlaps = metadata_overlap(test, dev, root)
    development_overlap = metadata_overlap(dev["val"], {"train": dev["train"]}, root)
    fatal = any(
        value
        for comparison in (overlaps, development_overlap)
        for split in comparison.values()
        for key, value in split.items()
        if key in ("paths", "canonical_paths", "families")
    )
    identity = {}
    for split, rows in {**dev, "combined": dev["train"] + dev["val"]}.items():
        known = {i for r in rows for i in r["identities"]}
        shared = sorted({i for r in test for i in r["identities"]} & known)
        identity[split] = dict(
            shared_codes=shared,
            shared_code_count=len(shared),
            test_videos_with_shared_code=sum(
                bool(set(r["identities"]) & known) for r in test
            ),
        )
    return dict(
        blocking_metadata_found=fatal,
        test_development=overlaps,
        train_validation=development_overlap,
        identity_overlap=identity,
        identity_overlap_is_blocking=False,
    )


def prepare(directory=PROTOCOL_DIR):
    if directory.exists():
        raise FileExistsError("Separate frozen protocol already exists; no overwrite")
    if file_sha256(LEGACY) != LEGACY_SHA:
        raise ValueError("Older frozen protocol changed")
    old = read_json(LEGACY)
    protected = {
        **old["protected_sha256"],
        **old["source_sha256"],
        str(LEGACY): LEGACY_SHA,
    }
    for filename in ("protocol.sha256", "preparation.json"):
        path = LEGACY.parent / filename
        protected[str(path)] = file_sha256(path)
    # Audit snapshots are verified as opaque files before using their neutral metadata.
    for folder in (HASH_AUDIT, SOURCE_AUDIT):
        seal_path = folder / "artifact_sha256.json"
        protected[str(seal_path)] = file_sha256(seal_path)
        for filename, expected in read_json(seal_path).items():
            protected[str(folder / filename)] = expected
    for filename, expected in protected.items():
        if file_sha256(Path(filename)) != expected:
            raise ValueError(f"Protected/audit file changed: {filename}")
    config = read_json(Path(old["checkpoint"]["path"]).parent.parent / "config.json")
    audit = read_json(Path("outputs/video_research_v1/roster_audit.json"))
    development = {s: audit["splits"][s]["accepted_records"] for s in ("train", "val")}
    for split, expected_count in (("train", 2223), ("val", 602)):
        if (
            len(development[split]) != expected_count
            or digest_json(development[split])
            != config["preprocessing"]["roster_hashes"][split]
        ):
            raise ValueError("Used epoch-6 development roster changed")
    indexed = {}
    with (HASH_AUDIT / "source_hashes.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            key = row["split"], row["path"]
            if key in indexed or not re.fullmatch("[a-f0-9]{64}", row["sha256"]):
                raise ValueError("Invalid historical source-hash index")
            indexed[key] = row
    source_hashes = {}
    for split, records in development.items():
        for record in records:
            entry = indexed[split, record["path"]]
            if int(entry["size"]) != record["size"]:
                raise ValueError("Development size differs from verified hash audit")
            source_hashes[record["path"]] = entry["sha256"]
    test = [
        dict(metadata_record(path), size=int(row["size"]), source_sha256=row["sha256"])
        for (split, path), row in indexed.items()
        if split == "test"
    ]
    if len(test) != 518 or len({r["path"] for r in test}) != 518:
        raise ValueError("Exact official filename snapshot required")
    source_evidence = read_json(SOURCE_AUDIT / "source_family_recount.json")
    if source_evidence["confirmed_cross_split_source_families_found"] != 0:
        raise LeakageBlocked("Prior source audit reports a shared source family")
    protocol = dict(
        schema=SCHEMA,
        locked_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=old["checkpoint"],
        decision=old["decision"],
        preprocessing=old["preprocessing"],
        official={
            k: old["official"][k]
            for k in ("root", "list_name", "list_sha256", "expected_count")
        },
        independence_policy=POLICY,
        development_rosters=development,
        development_source_sha256=source_hashes,
        test_roster=test,
        source_sha256={name: file_sha256(Path(name)) for name in NEW_SOURCES},
        protected_sha256=protected,
        runtime=runtime_versions(),
        limitations=LIMITATIONS,
        run_directory=str((BASE / "final_test_20261018").resolve()),
        once_registry=str((BASE / "official_518_once.json").resolve()),
        legacy_once_registry=old["once_registry"],
        legacy_run_directory=old["run_directory"],
        legacy_protocol_sha256=LEGACY_SHA,
        evaluation=old["evaluation"],
        prepared_without_new_official_list_or_video_access=True,
    )
    assert_legacy_unused(protocol)
    metadata = snapshot_metadata_check(protocol)
    if metadata["blocking_metadata_found"]:
        raise LeakageBlocked(
            "Snapshot path/carrier-family overlap; protocol not created"
        )
    encoded = (
        json.dumps(
            protocol, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    ).encode()
    checksum = hashlib.sha256(encoded).hexdigest()
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "protocol.json").open("xb").write(encoded)
    (directory / "protocol.sha256").open("x", encoding="ascii").write(checksum + "\n")
    sealed_write(
        directory / "preparation.json",
        dict(
            protocol_sha256=checksum,
            metadata_snapshot_check=metadata,
            new_official_list_opened=False,
            test_labels_interpreted=False,
            test_video_bytes_read=False,
            model_loaded=False,
            note="Exclusive creation plus externally pinned SHA; detects drift, not OS-level write-once storage.",
        ),
    )
    return checksum


def load_protocol(path, expected_sha):
    if (
        not re.fullmatch("[a-f0-9]{64}", expected_sha)
        or file_sha256(path) != expected_sha
    ):
        raise ValueError("Video-holdout protocol SHA-256 mismatch")
    p = read_json(path)
    if (
        p["schema"] != SCHEMA
        or p["legacy_protocol_sha256"] != LEGACY_SHA
        or p["independence_policy"] != json.loads(json.dumps(POLICY))
        or p["checkpoint"]["epoch"] != 6
        or p["checkpoint"]["sha256"]
        != "a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c"
        or p["decision"]["methods"] != ["6", "12", "adaptive"]
        or p["decision"]["threshold"] != 0.5
        or p["decision"]["gate_margin"] != 0.10
        or p["official"]["expected_count"] != 518
        or len(p["test_roster"]) != 518
        or p["run_directory"] != str((BASE / "final_test_20261018").resolve())
        or p["once_registry"] != str((BASE / "official_518_once.json").resolve())
    ):
        raise ValueError("Unsupported or altered video-holdout policy")
    if p["runtime"] != runtime_versions():
        raise ValueError("Frozen runtime drift; evaluation/resume refused")
    if file_sha256(Path(p["checkpoint"]["path"])) != p["checkpoint"]["sha256"]:
        raise ValueError("Pinned epoch-6 checkpoint changed")
    for filename, expected in {**p["source_sha256"], **p["protected_sha256"]}.items():
        if file_sha256(Path(filename)) != expected:
            raise ValueError(f"Frozen implementation/asset changed: {filename}")
    assert_legacy_unused(p)
    return p


def official_filenames(protocol):
    """Hash list as opaque bytes, then discard each label field without conversion."""
    path = Path(protocol["official"]["root"]) / protocol["official"]["list_name"]
    if file_sha256(path) != protocol["official"]["list_sha256"]:
        raise ValueError("Official list changed")
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"[^\s]+ ([^\s]+\.mp4)", line)
        if not match:
            raise ValueError("Invalid official filename row")
        metadata_record(match[1])
        names.append(match[1])
    if names != [r["path"] for r in protocol["test_roster"]] or len(set(names)) != 518:
        raise ValueError("Official 518 filenames/order changed; no subsets")
    return protocol["test_roster"]


def preflight(protocol, out):
    """Read-only media hashes; no target conversion, decoder, GPU or registry claim."""
    records = official_filenames(protocol)
    metadata = snapshot_metadata_check(protocol)
    if metadata["blocking_metadata_found"]:
        sealed_write(
            out / "preflight_blocked.json", dict(stage="metadata", metadata=metadata)
        )
        raise LeakageBlocked("Known path or carrier source-family overlap")
    root = Path(protocol["official"]["root"])
    hashes, membership = {}, development_hash_membership(protocol)
    changed, duplicate, seen_test = [], [], {}
    for split, rows in (*protocol["development_rosters"].items(), ("test", records)):
        for record in rows:
            source = safe_source(root, record)
            before = source.stat()
            actual = file_sha256(source)
            after = source.stat()
            expected = (
                record["source_sha256"]
                if split == "test"
                else protocol["development_source_sha256"][record["path"]]
            )
            if (
                (before.st_size, before.st_mtime_ns)
                != (after.st_size, after.st_mtime_ns)
                or before.st_size != record["size"]
                or actual != expected
            ):
                changed.append(dict(split=split, path=record["path"]))
            if split == "test":
                hashes[record["path"]] = actual
                if actual in membership:
                    duplicate.append(
                        dict(test_path=record["path"], development=membership[actual])
                    )
                if actual in seen_test:
                    duplicate.append(
                        dict(test_path=record["path"], other_test=seen_test[actual])
                    )
                seen_test[actual] = record["path"]
    dev_hashes = [
        {protocol["development_source_sha256"][r["path"]] for r in rows}
        for rows in protocol["development_rosters"].values()
    ]
    if changed or duplicate or dev_hashes[0] & dev_hashes[1]:
        sealed_write(
            out / "preflight_blocked.json",
            dict(
                stage="hash",
                changed_sources=changed,
                identical_sources=duplicate,
                train_validation_identical_hashes=sorted(dev_hashes[0] & dev_hashes[1]),
                model_loaded=False,
                test_labels_interpreted=False,
            ),
        )
        raise LeakageBlocked("Source drift or identical content; no targets/inference")
    result = dict(
        passed=True,
        protocol_digest=digest_json(protocol),
        metadata=metadata,
        source_hashes=hashes,
        attempted_count=len(records),
        model_loaded=False,
        test_labels_interpreted=False,
        limitations=protocol["limitations"],
    )
    sealed_write(out / "preflight_passed.json", result)
    return records, hashes


def development_hash_membership(protocol):
    membership = {}
    for split, rows in protocol["development_rosters"].items():
        for record in rows:
            sha = protocol["development_source_sha256"][record["path"]]
            membership.setdefault(sha, []).append(
                dict(split=split, path=record["path"])
            )
    return membership


def verify_passed_preflight(out, protocol):
    result = sealed_read(out / "preflight_passed.json")
    if not result["passed"] or result["protocol_digest"] != digest_json(protocol):
        raise LeakageBlocked("Successful matching video-holdout preflight required")
    return result
