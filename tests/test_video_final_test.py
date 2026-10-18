"""Synthetic final-test fixtures only: never opens the real official list/videos."""

import copy
from pathlib import Path
import sys

import pytest
import torch

import src.video_final_protocol as protocol_module
import scripts.run_video_final_test as driver
from src.video_final_protocol import (
    LeakageBlocked,
    claim_once,
    development_prefix,
    leakage_audit,
    metadata_overlap,
    test_targets_after_audit as read_targets,
    update_once,
)
from src.video_final_test import (
    ScoreCache,
    aggregate,
    sealed_read,
    sealed_write,
    shared_video,
)
from src.video_forensic import read_json, write_json
from src.video_manifest import digest_json, family_info
from src.video_pipeline import file_sha256, sample_indices


def record(path):
    family, codes = family_info(path)
    return dict(path=path, family=family, identities=codes)


def test_prepare_prefix_never_reads_or_parses_test_records(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        '{\n  "dataset_root": "no-videos",\n  "official_sha256": "opaque",\n  "splits": {\n    "train": [],\n    "val": [],\n    "test": [\nTHIS IS DELIBERATELY NOT JSON AND MUST NOT BE PARSED',
        encoding="utf-8",
    )
    result = development_prefix(path)
    assert result["splits"] == dict(train=[], val=[])
    assert result["official_sha256"] == "opaque"


@pytest.mark.parametrize(
    "test_path", ["Celeb-real/id0_0001.mp4", "Celeb-synthesis/id2_id0_0009.mp4"]
)
def test_identity_any_role_overlap_blocks_before_video_hash_or_targets(
    tmp_path, monkeypatch, test_path
):
    dev = dict(train=[record("Celeb-synthesis/id0_id1_0001.mp4")], val=[])
    test = [record(test_path)]
    protocol = dict(official=dict(root=str(tmp_path)), development_rosters=dev)
    monkeypatch.setattr(protocol_module, "official_paths", lambda _: test)
    monkeypatch.setattr(
        protocol_module,
        "file_sha256",
        lambda _: pytest.fail("video/hash accessed after identity overlap"),
    )
    with pytest.raises(LeakageBlocked):
        leakage_audit(protocol, tmp_path)
    blocked = read_json(tmp_path / "leakage_blocked.json")
    assert blocked["stage"] == "metadata" and blocked["test_mp4_opened"] is False
    assert not blocked["test_targets_converted"] and not blocked["model_loaded"]


def test_path_family_and_canonical_overlap_are_reported(tmp_path):
    dev = dict(train=[record("Celeb-real/id0_0001.mp4")], val=[])
    test = [
        record("Celeb-real/id0_0001.mp4"),
        record("Celeb-synthesis/id0_id2_0001.mp4"),
    ]
    result = metadata_overlap(test, dev, tmp_path)["train"]
    assert result["paths"] == [test[0]["path"]]
    assert result["canonical_paths"] == [test[0]["path"]]
    assert result["families"] == ["celeb:id0:0001"]
    assert result["any_role_identity_codes"] == ["id0"]


def test_source_byte_duplicates_block_after_metadata_pass(tmp_path, monkeypatch):
    development = dict(
        train=[record("YouTube-real/00001.mp4")],
        val=[record("Celeb-real/id0_0001.mp4")],
    )
    tests = [record("YouTube-real/00002.mp4")]
    cache = tmp_path / "cache"
    cache.mkdir()
    fingerprint = []
    for split, rows in development.items():
        for r in rows:
            source = tmp_path / r["path"]
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(
                b"same-source" if split == "train" else b"different-source"
            )
            path = cache / (digest_json(r) + ".pt")
            torch.save(
                dict(record=r, signature={}, source_sha256=file_sha256(source)), path
            )
            fingerprint.append((split, r, file_sha256(path)))
    (tmp_path / tests[0]["path"]).write_bytes(b"same-source")
    p = dict(
        official=dict(root=str(tmp_path)),
        development_rosters=development,
        feature_cache=dict(
            directory=str(cache), signature={}, fingerprint=digest_json(fingerprint)
        ),
    )
    monkeypatch.setattr(protocol_module, "official_paths", lambda _: tests)
    with pytest.raises(LeakageBlocked, match="duplicates"):
        leakage_audit(p, tmp_path)
    blocked = read_json(tmp_path / "leakage_blocked.json")
    assert blocked["stage"] == "hash" and len(blocked["development_duplicates"]) == 1
    assert not blocked["model_loaded"] and not blocked["test_targets_converted"]


def test_no_target_read_without_passed_audit():
    with pytest.raises(LeakageBlocked):
        read_targets({}, [], False)


def test_once_registry_resume_completion_and_different_protocol(tmp_path):
    registry = tmp_path / "once.json"
    run = tmp_path / "run"
    assert claim_once(registry, "pinned", run)
    update_once(registry, "interrupted")
    assert claim_once(registry, "pinned", run)
    with pytest.raises(ValueError, match="another protocol"):
        claim_once(registry, "new", run)
    update_once(registry, "complete")
    assert claim_once(registry, "pinned", run) is False
    before = registry.read_bytes()
    assert (
        claim_once(registry, "pinned", run) is False and registry.read_bytes() == before
    )


def test_blocked_registry_has_no_bypass_and_tampering_fails(tmp_path):
    path = tmp_path / "registry.json"
    claim_once(path, "p", tmp_path / "run")
    update_once(path, "blocked_overlap")
    with pytest.raises(LeakageBlocked):
        claim_once(path, "p", tmp_path / "run")
    saved = read_json(path)
    saved["status"] = "started"
    write_json(path, saved)
    with pytest.raises(ValueError, match="integrity"):
        claim_once(path, "p", tmp_path / "run")


def test_driver_block_never_calls_targets_or_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "BASE", tmp_path / "locks")

    def blocked(protocol, out):
        raise LeakageBlocked("synthetic overlap")

    monkeypatch.setattr(driver, "leakage_audit", blocked)
    monkeypatch.setattr(
        driver,
        "test_targets_after_audit",
        lambda *a, **k: pytest.fail("targets read after blocked audit"),
    )
    monkeypatch.setattr(
        driver, "factory", lambda: pytest.fail("GPU loaded after blocked audit")
    )
    p = dict(
        once_registry=str(tmp_path / "once.json"), run_directory=str(tmp_path / "run")
    )
    with pytest.raises(LeakageBlocked):
        driver.run(p, "sha")
    assert read_json(tmp_path / "once.json")["status"] == "blocked_overlap"
    assert not (tmp_path / "run/final_results.json").exists()


def test_driver_completed_run_does_no_further_audit_or_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "BASE", tmp_path / "locks")
    p = dict(
        once_registry=str(tmp_path / "once.json"), run_directory=str(tmp_path / "run")
    )
    claim_once(Path(p["once_registry"]), "sha", Path(p["run_directory"]))
    update_once(Path(p["once_registry"]), "complete")
    monkeypatch.setattr(
        driver, "leakage_audit", lambda *a: pytest.fail("reopened completed test")
    )
    driver.run(p, "sha")


class CV:
    CAP_PROP_FRAME_COUNT = 1

    def __init__(self, count=315):
        self.count = count
        self.captures = []

    def VideoCapture(self, path):
        cap = Capture(self.count)
        self.captures.append(cap)
        return cap


class Capture:
    def __init__(self, count):
        self.count = count
        self.index = -1
        self.closed = False

    def isOpened(self):
        return True

    def get(self, key):
        return self.count

    def grab(self):
        self.index += 1
        return self.index < self.count

    def retrieve(self):
        return True, self.index

    def release(self):
        self.closed = True


class Engine:
    def __init__(self, score=0.5, fail_after=None):
        self.cv = CV()
        self.score = score
        self.calls = []
        self.fail_after = fail_after

    def processor(self, frame, index):
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise KeyboardInterrupt("synthetic interrupted GPU call")
        self.calls.append(index)
        return dict(
            index=index,
            status="accepted",
            faces=[dict(fake_probability=self.score)],
            fake_probability=self.score,
            processing_seconds=0.01,
            peak_allocated_mib=1.0,
            peak_reserved_mib=2.0,
        )


def test_shared_cache_three_methods_only_twelve_unique_calls(tmp_path):
    engine = Engine()
    cache = ScoreCache(tmp_path / "cache", "p", "video", "source.mp4")
    result = shared_video(engine, Path("synthetic"), cache)
    assert (
        len(engine.calls)
        == len(set(engine.calls))
        == result["newly_inferred_frames"]
        == 12
    )
    assert engine.calls[:6] == sample_indices(315, 6)
    assert result["rows"]["6"]["processed_frames"] == 6
    assert result["rows"]["12"]["processed_frames"] == 12
    assert result["rows"]["adaptive"]["expanded"]
    assert (
        result["rows"]["6"]["shared_completion_seconds"]
        <= result["rows"]["12"]["shared_completion_seconds"]
    )
    assert all(row["latency_seconds"] is None for row in result["rows"].values())
    assert all(cap.closed for cap in engine.cv.captures)


def test_confident_adaptive_uses_six_completion_while_fixed12_extends(tmp_path):
    engine = Engine(0.8)
    result = shared_video(
        engine, Path("synthetic"), ScoreCache(tmp_path / "cache", "p", "v", "source")
    )
    assert not result["rows"]["adaptive"]["expanded"]
    assert result["rows"]["adaptive"]["processed_frames"] == 6
    assert (
        result["rows"]["adaptive"]["shared_completion_seconds"]
        == result["rows"]["6"]["shared_completion_seconds"]
    )
    assert len(engine.calls) == 12


def test_partial_frame_resume_reuses_commits_and_marks_timing_not_comparable(tmp_path):
    cache = ScoreCache(tmp_path / "cache", "p", "v", "source")
    broken = Engine(fail_after=4)
    with pytest.raises(KeyboardInterrupt):
        shared_video(broken, Path("synthetic"), cache)
    assert len(cache.load()) == 4
    resumed = Engine()
    result = shared_video(resumed, Path("synthetic"), cache)
    assert len(resumed.calls) == 8 and not set(broken.calls) & set(resumed.calls)
    assert (
        result["resumed_partial_frame_cache"]
        and not result["shared_completion_timing_valid"]
    )
    assert all(
        row["shared_completion_seconds"] is None for row in result["rows"].values()
    )


def test_cache_binding_integrity_and_immutable_commit(tmp_path):
    cache = ScoreCache(tmp_path, "p", "v", "source")
    frame = dict(
        index=0,
        status="accepted",
        faces=[dict(fake_probability=0.2)],
        fake_probability=0.2,
        processing_attempted=True,
        processing_seconds=0.1,
    )
    cache.save(frame)
    cache.save(copy.deepcopy(frame))
    with pytest.raises(ValueError, match="mismatch"):
        ScoreCache(tmp_path, "different", "v", "source").load()
    with pytest.raises(FileExistsError):
        cache.save(
            dict(frame, fake_probability=0.3, faces=[dict(fake_probability=0.3)])
        )
    path = tmp_path / "000000.json"
    value = read_json(path)
    value["payload"]["frame"]["fake_probability"] = 0.9
    write_json(path, value)
    with pytest.raises(ValueError, match="integrity"):
        cache.load()


def test_sealed_video_result_does_not_overwrite(tmp_path):
    path = tmp_path / "result.json"
    sealed_write(path, dict(score=0.2))
    assert sealed_read(path) == dict(score=0.2)
    with pytest.raises(FileExistsError):
        sealed_write(path, dict(score=0.3))


def test_aggregate_keeps_attempted_denominator_and_explicit_shared_timing(tmp_path):
    rows = [
        record("Celeb-real/id0_0001.mp4"),
        record("YouTube-real/00001.mp4"),
        record("Celeb-synthesis/id1_id2_0001.mp4"),
    ]
    targets = {r["path"]: int(r["path"].startswith("Celeb-synthesis")) for r in rows}
    videos = {}
    for i, r in enumerate(rows):
        videos[r["path"]] = shared_video(
            Engine(0.8 if targets[r["path"]] else 0.2),
            Path("synthetic"),
            ScoreCache(tmp_path / str(i), "p", str(i), r["path"]),
        )
    videos[rows[0]["path"]]["rows"]["6"].update(
        status="rejected", fake_probability=None, prediction=None
    )
    result = aggregate(rows, targets, videos)
    assert result["official_attempted"] == 3 and result["common_accepted_count"] == 2
    assert result["methods"]["6"]["summary"]["attempted"] == 3
    assert result["methods"]["6"]["summary"]["evaluated"] == 2
    assert result["methods"]["12"]["summary"]["evaluated"] == 3
    assert result["methods"]["6"]["summary"]["latency"] is None


def test_protocol_prepare_and_external_hash_detect_config_and_code_drift(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    for name in protocol_module.SOURCES:
        path = Path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source")
    for path in (
        protocol_module.CHECKPOINT,
        protocol_module.PARENT,
        Path("app.py"),
        Path("outputs/checkpoints/best_model_v3_robust.pth"),
        protocol_module.ROOT / "retinaface.pth",
        protocol_module.ROOT / "upstream/src_inference_preprocess.py",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"opaque synthetic fixture")
    manifest = protocol_module.MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        '{\n  "dataset_root": "NEVER_OPEN_THIS_DATASET",\n  "official_sha256": "opaque",\n  "splits": {\n    "train": [],\n    "val": [],\n    "test": [\nDO NOT READ OR PARSE TEST RECORDS',
        encoding="utf-8",
    )
    config = dict(
        frames=12,
        threshold=0.5,
        manifest_fingerprint="m",
        train_roster_sha256=digest_json([]),
        val_roster_sha256=digest_json([]),
    )
    write_json(protocol_module.RUN / "config.json", config)
    write_json(
        protocol_module.AUDIT,
        dict(
            manifest_fingerprint="m",
            splits={s: dict(accepted_records=[]) for s in ("train", "val")},
            cache_directory="not-opened",
            cache_signature={},
            cache_fingerprint="frozen",
        ),
    )
    actual_sha = file_sha256
    monkeypatch.setattr(
        protocol_module,
        "file_sha256",
        lambda p: (
            protocol_module.CHECKPOINT_SHA
            if p == protocol_module.CHECKPOINT
            else actual_sha(p)
        ),
    )
    monkeypatch.setattr(
        protocol_module, "runtime_versions", lambda: dict(synthetic=True)
    )
    directory = tmp_path / "frozen"
    sha = protocol_module.prepare(directory)
    path = directory / "protocol.json"
    protocol_module.load_protocol(path, sha)
    assert read_json(directory / "preparation.json")["test_labels_inspected"] is False
    with pytest.raises(FileExistsError):
        protocol_module.prepare(directory)
    code = Path(protocol_module.SOURCES[0])
    code.write_text("changed implementation")
    with pytest.raises(ValueError, match="implementation"):
        protocol_module.load_protocol(path, sha)
    code.write_text("synthetic source")
    value = read_json(path)
    value["decision"]["gate_margin"] = 0.2
    write_json(path, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        protocol_module.load_protocol(path, sha)


def test_cli_no_action_or_unpinned_final_cannot_access_data(monkeypatch):
    for args in ([], ["--run-final-test"]):
        monkeypatch.setattr(sys, "argv", ["final"] + args)
        with pytest.raises(SystemExit) as error:
            driver.main()
        assert error.value.code == 2


def test_passing_hash_audit_pins_test_bytes_and_refuses_changed_resume(
    tmp_path, monkeypatch
):
    dev = record("YouTube-real/00001.mp4")
    target = record("YouTube-real/00002.mp4")
    cache = tmp_path / "cache"
    cache.mkdir()
    dev_source = tmp_path / dev["path"]
    dev_source.parent.mkdir(parents=True)
    dev_source.write_bytes(b"synthetic development")
    test_source = tmp_path / target["path"]
    test_source.write_bytes(b"synthetic unused test fixture")
    path = cache / (digest_json(dev) + ".pt")
    torch.save(
        dict(record=dev, signature={}, source_sha256=file_sha256(dev_source)), path
    )
    p = dict(
        official=dict(root=str(tmp_path)),
        development_rosters=dict(train=[dev], val=[]),
        feature_cache=dict(
            directory=str(cache),
            signature={},
            fingerprint=digest_json([("train", dev, file_sha256(path))]),
        ),
    )
    monkeypatch.setattr(protocol_module, "official_paths", lambda _: [target])
    leakage_audit(p, tmp_path)
    committed = (tmp_path / "leakage_passed.json").read_bytes()
    leakage_audit(p, tmp_path)
    assert (tmp_path / "leakage_passed.json").read_bytes() == committed
    test_source.write_bytes(b"changed source, must never be accepted on resume")
    with pytest.raises(ValueError, match="source bytes/protocol changed"):
        leakage_audit(p, tmp_path)


def test_completed_video_resume_does_not_load_model_or_reinfer_frames(tmp_path):
    from src.video_final_test import evaluate as evaluate_final

    records = [
        record("Celeb-real/id0_0001.mp4"),
        record("YouTube-real/00001.mp4"),
        record("Celeb-synthesis/id1_id2_0001.mp4"),
    ]
    labels = {r["path"]: int(r["path"].startswith("Celeb-synthesis")) for r in records}
    hashes = {r["path"]: "synthetic-byte-sha" for r in records}
    for r in records:
        source = tmp_path / r["path"]
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(b"fake decoder input")
    p = dict(
        official=dict(root=str(tmp_path)),
        checkpoint=dict(sha256="fixed"),
        decision=dict(threshold=0.5),
    )
    engine = Engine(0.8)
    result = evaluate_final(
        p, "protocol", records, labels, hashes, tmp_path / "out", lambda: (engine, {})
    )
    assert len(engine.calls) == 36 and result["official_attempted"] == 3
    before = len(engine.calls)
    replay = evaluate_final(
        p,
        "protocol",
        records,
        labels,
        hashes,
        tmp_path / "out",
        lambda: pytest.fail("model initialized on complete-video resume"),
    )
    assert replay == result and len(engine.calls) == before
