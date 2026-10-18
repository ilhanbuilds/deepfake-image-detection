"""Synthetic media bytes and injected processors only; no real test/labels/GPU."""

import csv
import copy
import json
from pathlib import Path
import sys

import pytest

from scripts import run_video_holdout_test as driver
from src import video_holdout_protocol as policy
from src.video_final_protocol import LeakageBlocked, claim_once, update_once
from src.video_final_test import ScoreCache, sealed_read, shared_video
from src.video_forensic import read_json, write_json
from src.video_holdout_evaluate import evaluate, export_report, summarize_holdout
from src.video_manifest import digest_json
from src.video_pipeline import file_sha256
from tests.test_video_final_test import Engine


def record(path, data=b"synthetic"):
    import hashlib

    return dict(
        policy.metadata_record(path),
        size=len(data),
        source_sha256=hashlib.sha256(data).hexdigest(),
    )


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    dev = {
        "train": [record("Celeb-real/id4_0001.mp4", b"train")],
        "val": [record("Celeb-real/id8_0001.mp4", b"val")],
    }
    tests = [record("Celeb-real/id4_0002.mp4", b"test")]
    for r in dev["train"] + dev["val"] + tests:
        path = tmp_path / r["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(
            {"id4_0001.mp4": b"train", "id8_0001.mp4": b"val", "id4_0002.mp4": b"test"}[
                path.name
            ]
        )
    p = dict(
        official=dict(root=str(tmp_path)),
        development_rosters=dev,
        test_roster=tests,
        development_source_sha256={
            r["path"]: r["source_sha256"] for rows in dev.values() for r in rows
        },
        limitations=["synthetic"],
        checkpoint=dict(sha256="fixed"),
        decision=dict(threshold=0.5),
        independence_policy={"scope": "synthetic"},
        once_registry=str(tmp_path / "once.json"),
        run_directory=str(tmp_path / "run"),
    )
    monkeypatch.setattr(policy, "official_filenames", lambda _: tests)
    return p


def test_identity_overlap_reports_without_blocking(synthetic, tmp_path):
    metadata = policy.snapshot_metadata_check(synthetic)
    assert not metadata["blocking_metadata_found"]
    assert metadata["identity_overlap"]["train"]["shared_codes"] == ["id4"]
    assert metadata["identity_overlap"]["combined"]["test_videos_with_shared_code"] == 1
    policy.preflight(synthetic, tmp_path / "checks")
    passed = policy.verify_passed_preflight(tmp_path / "checks", synthetic)
    assert passed["passed"] and not passed["test_labels_interpreted"]
    assert not Path(synthetic["once_registry"]).exists()


@pytest.mark.parametrize(
    "test_path",
    [
        "Celeb-real/id4_0001.mp4",
        "Celeb-synthesis/id4_id7_0001.mp4",
        "Celeb-real/ID4_0001.mp4",
    ],
)
def test_path_or_carrier_blocks_before_media_hash(
    synthetic, tmp_path, monkeypatch, test_path
):
    if "ID4" in test_path:
        # Canonical case aliases cannot be hidden behind identity parsing.
        with pytest.raises(ValueError):
            policy.metadata_record(test_path)
        return
    synthetic["test_roster"][:] = [record(test_path)]
    monkeypatch.setattr(
        policy,
        "file_sha256",
        lambda p: pytest.fail("media hash read after metadata block"),
    )
    with pytest.raises(LeakageBlocked, match="carrier"):
        policy.preflight(synthetic, tmp_path / "checks")


def test_exact_hash_duplicate_blocks_despite_different_filename(synthetic, tmp_path):
    target = synthetic["test_roster"][0]
    (tmp_path / target["path"]).write_bytes(b"train")
    target.update(
        size=5,
        source_sha256=synthetic["development_rosters"]["train"][0]["source_sha256"],
    )
    with pytest.raises(LeakageBlocked, match="identical"):
        policy.preflight(synthetic, tmp_path / "checks")
    assert sealed_read(tmp_path / "checks/preflight_blocked.json")["identical_sources"]


def test_source_change_and_resume_are_fail_closed(synthetic, tmp_path):
    out = tmp_path / "checks"
    policy.preflight(synthetic, out)
    original = (out / "preflight_passed.json").read_bytes()
    policy.preflight(synthetic, out)
    assert (out / "preflight_passed.json").read_bytes() == original
    (tmp_path / synthetic["test_roster"][0]["path"]).write_bytes(b"changed")
    with pytest.raises(LeakageBlocked):
        policy.preflight(synthetic, out)


def test_train_validation_source_overlap_blocks(synthetic):
    synthetic["development_rosters"]["val"] = [
        record("Celeb-synthesis/id4_id9_0001.mp4")
    ]
    assert policy.snapshot_metadata_check(synthetic)["blocking_metadata_found"]


def test_blocked_driver_never_reads_targets_or_loads_model(
    synthetic, tmp_path, monkeypatch
):
    monkeypatch.setattr(driver, "BASE", tmp_path / "locks")
    synthetic["test_roster"][:] = [record("Celeb-synthesis/id4_id9_0001.mp4")]
    monkeypatch.setattr(
        driver,
        "test_targets_after_audit",
        lambda *a, **k: pytest.fail("test labels read"),
    )
    monkeypatch.setattr(driver, "factory", lambda: pytest.fail("model loaded"))
    with pytest.raises(LeakageBlocked):
        driver.run(synthetic, "synthetic-sha")
    assert read_json(Path(synthetic["once_registry"]))["status"] == "blocked_overlap"


def test_completed_driver_no_second_audit_or_inference(
    synthetic, tmp_path, monkeypatch
):
    monkeypatch.setattr(driver, "BASE", tmp_path / "locks")
    claim_once(
        Path(synthetic["once_registry"]), "sha", Path(synthetic["run_directory"])
    )
    update_once(Path(synthetic["once_registry"]), "complete")
    monkeypatch.setattr(
        driver, "preflight", lambda *a: pytest.fail("completed test reopened")
    )
    driver.run(synthetic, "sha")


def test_old_registry_is_read_only_and_prior_inference_blocks(tmp_path):
    registry, run = tmp_path / "old.json", tmp_path / "oldrun"
    p = dict(legacy_once_registry=str(registry), legacy_run_directory=str(run))
    policy.assert_legacy_unused(p)
    claim_once(registry, "old", run)
    original = registry.read_bytes()
    with pytest.raises(ValueError, match="Older evaluation"):
        policy.assert_legacy_unused(p)
    assert registry.read_bytes() == original
    update_once(registry, "blocked_overlap")
    original = registry.read_bytes()
    policy.assert_legacy_unused(p)
    assert registry.read_bytes() == original
    (run / "frames").mkdir(parents=True)
    with pytest.raises(ValueError, match="inference artifacts"):
        policy.assert_legacy_unused(p)


def test_shared_scores_resume_all_methods_and_rejection_reporting(synthetic, tmp_path):
    test = synthetic["test_roster"][0]
    engine = Engine(0.8, fail_after=4)
    cache = ScoreCache(tmp_path / "partial", "sha", "video", test["path"])
    with pytest.raises(KeyboardInterrupt):
        shared_video(engine, Path("synthetic"), cache)
    resumed = Engine(0.8)
    video = shared_video(resumed, Path("synthetic"), cache)
    assert len(resumed.calls) == 8 and not set(engine.calls) & set(resumed.calls)
    targets = {test["path"]: 0}
    video["rows"]["6"].update(status="rejected", fake_probability=None, prediction=None)
    result = summarize_holdout(synthetic, "sha", [test], targets, {test["path"]: video})
    assert result["official_attempted"] == 1 and result["common_accepted_count"] == 0
    summary = result["methods"]["6"]["summary"]
    assert summary["coverage_by_source"]["Celeb-real"]["rejected"] == 1
    assert summary["correct_over_all_attempts_rejections_unsuccessful"] == 0
    assert summary["actual_shared_completion_timing"]["measured_videos"] == 0
    out = tmp_path / "result"
    out.mkdir()
    export_report(out, result, {"metadata": {"identity_overlap": {}}})
    assert (
        len(list(csv.DictReader((out / "predictions.csv").open(encoding="utf-8-sig"))))
        == 3
    )
    assert "NA" in (out / "REPORT.md").read_text()
    (out / "predictions.csv").write_bytes(b"tampered export")
    with pytest.raises(ValueError, match="CSV changed"):
        export_report(out, result, {"metadata": {"identity_overlap": {}}})


def test_complete_video_replay_never_initializes_model(synthetic, tmp_path):
    records = synthetic["test_roster"]
    target = {records[0]["path"]: 0}
    hashes = {records[0]["path"]: records[0]["source_sha256"]}
    engine = Engine(0.2)
    out = tmp_path / "evaluated"
    first = evaluate(
        synthetic, "sha", records, target, hashes, out, lambda: (engine, {})
    )
    assert len(engine.calls) == len(set(engine.calls)) == 12
    second = evaluate(
        synthetic,
        "sha",
        records,
        target,
        hashes,
        out,
        lambda: pytest.fail("model initialized on replay"),
    )
    assert first == second


def test_successful_driver_seals_results_and_does_not_retune(
    synthetic, tmp_path, monkeypatch
):
    monkeypatch.setattr(driver, "BASE", tmp_path / "locks")
    engine = Engine(0.2)
    monkeypatch.setattr(
        driver, "factory", lambda: (engine, {"model_and_detector_load_seconds": 0.0})
    )
    monkeypatch.setattr(
        driver, "test_targets_after_audit", lambda p, r, passed: {r[0]["path"]: 0}
    )
    driver.run(synthetic, "sha")
    out = Path(synthetic["run_directory"])
    assert len(engine.calls) == 12
    assert read_json(Path(synthetic["once_registry"]))["status"] == "complete"
    saved = sealed_read(out / "final_results.json")
    assert saved["retuning_allowed"] is False and saved["official_attempted"] == 1
    assert sealed_read(out / "execution_cost.json")["recorded_model_loads"] == 1
    original = (out / "final_results.json").read_bytes()
    monkeypatch.setattr(
        driver, "preflight", lambda *a: pytest.fail("second decision evaluation")
    )
    driver.run(synthetic, "sha")
    assert (out / "final_results.json").read_bytes() == original


def test_protocol_hash_policy_runtime_and_code_drift_fail_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(policy, "BASE", tmp_path / "base")
    monkeypatch.setattr(policy, "runtime_versions", lambda: dict(synthetic=True))
    code = tmp_path / "code.py"
    code.write_text("original")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"opaque synthetic checkpoint")
    checkpoint_sha = "a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c"
    actual_sha = file_sha256
    monkeypatch.setattr(
        policy,
        "file_sha256",
        lambda path: checkpoint_sha if path == checkpoint else actual_sha(path),
    )
    p = dict(
        schema=policy.SCHEMA,
        legacy_protocol_sha256=policy.LEGACY_SHA,
        independence_policy=json.loads(json.dumps(policy.POLICY)),
        checkpoint=dict(path=str(checkpoint), epoch=6, sha256=checkpoint_sha),
        decision=dict(methods=["6", "12", "adaptive"], threshold=0.5, gate_margin=0.1),
        official=dict(expected_count=518),
        test_roster=[{}] * 518,
        runtime=dict(synthetic=True),
        source_sha256={str(code): actual_sha(code)},
        protected_sha256={},
        run_directory=str((policy.BASE / "final_test_20261018").resolve()),
        once_registry=str((policy.BASE / "official_518_once.json").resolve()),
        legacy_once_registry=str(tmp_path / "old.json"),
        legacy_run_directory=str(tmp_path / "oldrun"),
    )
    path = tmp_path / "protocol.json"
    write_json(path, p)
    sha = actual_sha(path)
    assert policy.load_protocol(path, sha) == p
    code.write_text("changed")
    with pytest.raises(ValueError, match="implementation"):
        policy.load_protocol(path, sha)
    code.write_text("original")
    monkeypatch.setattr(policy, "runtime_versions", lambda: dict(synthetic=False))
    with pytest.raises(ValueError, match="runtime"):
        policy.load_protocol(path, sha)
    monkeypatch.setattr(policy, "runtime_versions", lambda: dict(synthetic=True))
    for key, value in (("threshold", 0.6), ("gate_margin", 0.2)):
        changed = copy.deepcopy(p)
        changed["decision"][key] = value
        write_json(path, changed)
        with pytest.raises(ValueError, match="SHA-256"):
            policy.load_protocol(path, sha)
        with pytest.raises(ValueError, match="altered"):
            policy.load_protocol(path, actual_sha(path))


def test_official_filenames_does_not_convert_or_count_labels(tmp_path, monkeypatch):
    # Undo fixture-free call: these are synthetic 518 filenames, not official records.
    records = [record(f"YouTube-real/{i:05d}.mp4") for i in range(518)]
    listing = tmp_path / "list.txt"
    listing.write_text("\n".join("DISCARDED " + r["path"] for r in records))
    p = dict(
        official=dict(
            root=str(tmp_path), list_name=listing.name, list_sha256=file_sha256(listing)
        ),
        test_roster=records,
    )
    assert policy.official_filenames(p) == records
    records.reverse()
    with pytest.raises(ValueError, match="order"):
        policy.official_filenames(p)


def test_prepare_never_opens_dataset_and_preserves_legacy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    legacy = old_dir / "protocol.json"
    monkeypatch.setattr(policy, "LEGACY", legacy)
    monkeypatch.setattr(policy, "NEW_SOURCES", ("new.py",))
    Path("new.py").write_text("synthetic source")
    for name in ("protocol.sha256", "preparation.json"):
        (old_dir / name).write_text("opaque legacy file")
    rows = {
        s: [record(f"YouTube-real/{offset + i:05d}.mp4") for i in range(n)]
        for s, offset, n in (
            ("train", 0, 2223),
            ("val", 3000, 602),
            ("test", 5000, 518),
        )
    }
    audit = dict(splits={s: dict(accepted_records=rows[s]) for s in ("train", "val")})
    write_json(Path("outputs/video_research_v1/roster_audit.json"), audit)
    checkpoint = tmp_path / "run/checkpoints/best.pth"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"opaque checkpoint")
    write_json(
        checkpoint.parent.parent / "config.json",
        dict(
            preprocessing=dict(
                roster_hashes={s: digest_json(rows[s]) for s in ("train", "val")}
            )
        ),
    )
    hash_dir, source_dir = tmp_path / "hashes", tmp_path / "source"
    hash_dir.mkdir()
    source_dir.mkdir()
    monkeypatch.setattr(policy, "HASH_AUDIT", hash_dir)
    monkeypatch.setattr(policy, "SOURCE_AUDIT", source_dir)
    with (hash_dir / "source_hashes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("split", "path", "size", "sha256"))
        writer.writeheader()
        for split, records in rows.items():
            for r in records:
                writer.writerow(
                    dict(
                        split=split,
                        path=r["path"],
                        size=r["size"],
                        sha256=r["source_sha256"],
                    )
                )
    write_json(
        source_dir / "source_family_recount.json",
        dict(confirmed_cross_split_source_families_found=0),
    )
    for folder in (hash_dir, source_dir):
        write_json(
            folder / "artifact_sha256.json",
            {p.name: file_sha256(p) for p in folder.iterdir()},
        )
    old = dict(
        protected_sha256={},
        source_sha256={},
        checkpoint=dict(path=str(checkpoint)),
        decision={},
        preprocessing={},
        official=dict(
            root=str(tmp_path / "NEVER_OPEN"),
            list_name="list.txt",
            list_sha256="opaque",
            expected_count=518,
        ),
        once_registry=str(old_dir / "once.json"),
        run_directory=str(old_dir / "run"),
        evaluation={},
    )
    write_json(legacy, old)
    monkeypatch.setattr(policy, "LEGACY_SHA", file_sha256(legacy))
    monkeypatch.setattr(policy, "runtime_versions", lambda: dict(synthetic=True))
    before = {p: p.read_bytes() for p in old_dir.iterdir()}
    sha = policy.prepare(tmp_path / "new_protocol")
    assert sha == file_sha256(tmp_path / "new_protocol/protocol.json")
    prep = sealed_read(tmp_path / "new_protocol/preparation.json")
    assert not prep["new_official_list_opened"] and not prep["test_video_bytes_read"]
    assert all(path.read_bytes() == data for path, data in before.items())
    with pytest.raises(FileExistsError):
        policy.prepare(tmp_path / "new_protocol")


@pytest.mark.parametrize(
    "args", [[], ["--run-final-test"], ["--check-protocol", "--resume", "auto"]]
)
def test_cli_requires_explicit_action_and_pinned_sha(args, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["holdout"] + args)
    monkeypatch.setattr(
        driver,
        "load_protocol",
        lambda *a: pytest.fail("data accessed with invalid CLI"),
    )
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
