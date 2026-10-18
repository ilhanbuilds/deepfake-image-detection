"""12-frame cache, accumulation, resume and baseline contracts; synthetic inputs."""

import copy
from dataclasses import asdict
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.train import EarlyStopping, atomic_save, select_resume
from src.video_forensic import summarize, write_json
from src.video_forensic_train import (
    PIPELINE,
    main as training_main,
    train_group,
    validate_fresh_resume,
    validate_state,
)
from src.video_frame_study import nested_indices
from src.video_manifest import digest_json
from src.video_pipeline import sample_indices
from src.video_sbi12 import (
    TwelveFrameCache,
    comparison,
    exclusive_lock,
    main as cache_main,
    pixel_hash,
    read_crops,
    save_crops,
    validate_payload,
)


def fixture_cache():
    indices = nested_indices(315, sample_indices(315, 6))[12]
    record = dict(path="Celeb-real/example.mp4", label=0)
    pixels = torch.zeros((12, 3, 380, 380), dtype=torch.uint8)
    report = dict(
        path=record["path"],
        selected_indices=indices,
        status="accepted",
        failures=[],
        frames=[
            dict(index=i, faces=[dict(crop_index=j, bbox=[0, 0, 80, 80])])
            for j, i in enumerate(indices)
        ],
    )
    signature = dict(frames=12)
    historical = dict(
        source_sha256="source",
        report=dict(frame_count=315, selected_indices=sample_indices(315, 6)),
    )
    cached = dict(
        record=record,
        signature=signature,
        source_sha256="source",
        crops=pixels,
        pixels_sha256=pixel_hash(pixels),
        report=report,
    )
    cached["integrity"] = digest_json({k: v for k, v in cached.items() if k != "crops"})
    return cached, record, signature, historical


def reseal(cached):
    cached["integrity"] = digest_json(
        {k: v for k, v in cached.items() if k not in ("crops", "integrity")}
    )


def test_lossless_atomic_cache_roundtrip(tmp_path):
    cached, record, signature, old = fixture_cache()
    path = tmp_path / "cache.pt"
    save_crops(cached, path)
    loaded = read_crops(path)
    assert torch.equal(loaded["crops"], cached["crops"])
    validate_payload(loaded, record, signature, old)
    assert path.stat().st_size < cached["crops"].numel()


@pytest.mark.parametrize(
    "mutation",
    ["pixel", "source", "record", "indices", "missing", "duplicate", "six_signature"],
)
def test_cache_tampering_and_wrong_budget_fail(mutation):
    cached, record, signature, old = fixture_cache()
    if mutation == "pixel":
        cached["crops"][0, 0, 0, 0] = 1
    elif mutation == "source":
        cached["source_sha256"] = "other"
    elif mutation == "record":
        cached["record"] = dict(record, label=1)
    elif mutation == "indices":
        cached["report"]["selected_indices"][1] += 1
    elif mutation == "missing":
        cached["report"]["frames"].pop()
        cached["crops"] = cached["crops"][:-1]
        cached["pixels_sha256"] = pixel_hash(cached["crops"])
    elif mutation == "duplicate":
        cached["report"]["failures"] = [dict(index=0, reason="decode_failed")]
    else:
        cached["signature"] = dict(frames=6)
    reseal(cached)
    with pytest.raises(ValueError):
        validate_payload(cached, record, signature, old)


def test_interrupted_cache_write_preserves_existing(tmp_path, monkeypatch):
    cached, *_ = fixture_cache()
    path = tmp_path / "cache.pt"
    save_crops(cached, path)
    original = path.read_bytes()

    def interrupted(value, target):
        Path(target).write_bytes(b"partial")
        raise OSError("simulated interruption")

    monkeypatch.setattr(torch, "save", interrupted)
    with pytest.raises(OSError):
        save_crops(cached, path)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


def test_cache_reuse_does_not_create_detector_or_rewrite(tmp_path, monkeypatch):
    cached, record, signature, old = fixture_cache()
    cache = TwelveFrameCache.__new__(TwelveFrameCache)
    cache.directory, cache.signature = tmp_path, signature
    cache.validation_paths, cache.prepared, cache.extractor = set(), [], None
    monkeypatch.setattr(cache, "historical", lambda *_: old)
    monkeypatch.setattr(cache, "verify_source", lambda *_: Path("unused"))
    path = cache.path(record)
    save_crops(cached, path)
    before = path.read_bytes(), path.stat().st_mtime_ns
    result = cache.prepare(tmp_path, record, {})
    assert result["reused"] and cache.extractor is None
    assert before == (path.read_bytes(), path.stat().st_mtime_ns)


def test_missing_cache_training_read_cannot_preprocess(tmp_path, monkeypatch):
    cache = TwelveFrameCache.__new__(TwelveFrameCache)
    cache.directory = tmp_path
    monkeypatch.setattr(cache, "historical", lambda *_: {})
    with pytest.raises(FileNotFoundError, match="Prepare"):
        cache.read(tmp_path, dict(path="example"), {})


@pytest.mark.parametrize("videos", [1, 3, 4])
def test_twelve_frame_accumulation_matches_full_objective_including_partial_group(
    videos,
):
    torch.manual_seed(42)
    model = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
    reference = copy.deepcopy(model)
    examples = []
    for i in range(videos):
        pixels = torch.randint(0, 256, (12, 3, 2, 2), dtype=torch.uint8)
        report = dict(
            status="accepted",
            frames=[
                dict(faces=[dict(crop_index=j, bbox=[0, 0, 2, 2])]) for j in range(12)
            ],
        )
        examples.append((pixels, report, i % 2))
    weights = torch.tensor([2.665, 0.615])
    expected_optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    loss = sum(
        nn.functional.cross_entropy(
            reference(p.float() / 255), torch.full((12,), label, dtype=torch.long)
        )
        * weights[label]
        / videos
        for p, _, label in examples
    )
    loss.backward()
    nn.utils.clip_grad_norm_(reference.parameters(), 1, error_if_nonfinite=True)
    expected_optimizer.step()
    result = train_group(
        model, torch.optim.SGD(model.parameters(), lr=0.01), examples, weights, "cpu"
    )
    assert result["loss"] == pytest.approx(loss.item(), abs=1e-6)
    for actual, expected in zip(
        model.parameters(), reference.parameters(), strict=True
    ):
        assert actual.dtype == torch.float32
        torch.testing.assert_close(actual, expected)


def state_fixture(scores=(0.7, 0.8, 0.8, 0.79, 0.78)):
    config = dict(frames=12, patience=3, epochs=8)
    stopper = EarlyStopping(patience=3)
    history = []
    for epoch, score in enumerate(scores, 1):
        stopper.update(score)
        history.append(dict(epoch=epoch, validation_metrics=dict(macro_f1=score)))
    state = dict(
        pipeline=PIPELINE,
        config=config,
        model_state_dict={},
        optimizer_state_dict={},
        rng_state={},
        history=history,
        epoch=len(scores),
        early_stopping=asdict(stopper),
        best_epoch=scores.index(max(scores)) + 1,
        best_score=max(scores),
        validation=dict(metrics=history[-1]["validation_metrics"]),
    )
    return config, state


def test_early_stopping_ties_resume_and_epoch_atomic_selection(tmp_path):
    config, state = state_fixture()
    validate_state(state, config)
    assert state["early_stopping"]["bad_epochs"] == 3
    restored = EarlyStopping(**state["early_stopping"])
    assert restored.bad_epochs >= restored.patience
    atomic_save(state, tmp_path / "epoch_005.pth")
    (tmp_path / "epoch_006.pth.tmp").write_bytes(b"partial")
    selected = select_resume(tmp_path, "auto")
    assert selected.name == "epoch_005.pth"
    validate_state(torch.load(selected, weights_only=True), config)
    earlier = EarlyStopping(patience=3)
    earlier.update(0.7)
    earlier.update(0.6)
    restarted = EarlyStopping(**asdict(earlier))
    assert not restarted.update(0.8) and restarted.bad_epochs == 0


@pytest.mark.parametrize(
    "field", ["epoch", "best_epoch", "best_score", "early_stopping", "config"]
)
def test_resume_rejects_corrupt_control_state(field):
    config, state = state_fixture()
    state[field] = (
        dict(frames=6)
        if field == "config"
        else ({} if field == "early_stopping" else -1)
    )
    with pytest.raises(ValueError):
        validate_state(state, config)


def test_first_epoch_resume_requires_identical_config_and_rejects_smoke(tmp_path):
    config = dict(frames=12)
    with pytest.raises(ValueError):
        validate_fresh_resume(tmp_path, None, config)
    write_json(tmp_path / "config.json", config)
    validate_fresh_resume(tmp_path, None, config)
    with pytest.raises(ValueError):
        validate_fresh_resume(tmp_path, None, dict(frames=6))
    (tmp_path / "smoke.json").write_text("{}")
    with pytest.raises(ValueError, match="Diagnostic"):
        validate_fresh_resume(tmp_path, None, config)


def test_baseline_comparison_metrics_sources_and_cohort():
    rows = [
        dict(
            path=source + "/a",
            target=label,
            status="accepted",
            prediction=label,
            fake_probability=prob,
        )
        for source, label, prob in (
            ("Celeb-real", 0, 0.2),
            ("YouTube-real", 0, 0.3),
            ("Celeb-synthesis", 1, 0.8),
        )
    ]
    baseline = dict(**summarize(rows, len(rows)), rows=rows)
    assert all(value == 0 for value in comparison(baseline, baseline)["delta"].values())
    invalid = dict(baseline, rows=rows[:-1])
    with pytest.raises(ValueError, match="complete cohort"):
        comparison(invalid, baseline)


@pytest.mark.parametrize("entry", [cache_main, training_main])
def test_explicit_action_required_without_any_data_access(entry, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["command"])
    with pytest.raises(SystemExit) as error:
        entry()
    assert error.value.code == 2


def test_run_lock_is_released_after_interruption(tmp_path):
    path = tmp_path / "run.lock"
    with pytest.raises(KeyboardInterrupt):
        with exclusive_lock(path):
            with pytest.raises(RuntimeError, match="Another process"):
                with exclusive_lock(path):
                    pytest.fail("second writer admitted")
            raise KeyboardInterrupt
    with exclusive_lock(path):
        pass


@pytest.mark.parametrize("crash", ["before_commit", "after_commit"])
def test_epoch_driver_resume_replays_interrupted_epoch_and_rebuilds_reports(
    tmp_path, monkeypatch, crash
):
    """Exercise the production epoch/checkpoint loop on a tiny CPU model."""
    import src.video_forensic_train as training
    import src.video_sbi12 as caching
    from src.video_forensic import read_json

    monkeypatch.chdir(tmp_path)
    records = [
        dict(path=s + "/a", label=label)
        for s, label in (("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1))
    ]
    rows = [
        dict(
            path=r["path"],
            target=r["label"],
            status="accepted",
            prediction=r["label"],
            fake_probability=0.8 if r["label"] else 0.2,
        )
        for r in records
    ]
    baseline = dict(**summarize(rows, len(rows)), rows=rows)

    class Cache:
        signature = dict(frames=12, resolution=380)

        def __init__(self, *args):
            self.baseline = baseline

        def preflight(self, *args):
            pass

        def read(self, root, record, audit):
            return torch.full(
                (12, 3, 2, 2), 50 + record["label"] * 100, dtype=torch.uint8
            ), dict(
                status="accepted",
                frames=[
                    dict(faces=[dict(crop_index=i, bbox=[0, 0, 2, 2])])
                    for i in range(12)
                ],
            )

    class Model(nn.Sequential):
        adaptation_config = None

        def __init__(self, path):
            super().__init__(nn.Flatten(), nn.Linear(12, 2))

        def cuda(self):
            return self

    monkeypatch.setattr(caching, "TwelveFrameCache", Cache)
    monkeypatch.setattr(
        training,
        "accepted_roster",
        lambda split: (
            dict(dataset_root=str(tmp_path), fingerprint="fixed"),
            dict(cache_signature={}),
            records,
        ),
    )
    monkeypatch.setattr(training, "SBIClassifier", Model)
    monkeypatch.setattr(training, "file_sha256", lambda _: caching.CHECKPOINT_SHA)
    monkeypatch.setattr(
        training,
        "prepare_config",
        lambda args, *_: dict(
            frames=12, epochs=args.epochs, patience=3, implementation_sha256={}
        ),
    )
    monkeypatch.setattr(training, "configure_stage", lambda model, epoch: model.eval())
    monkeypatch.setattr(
        training,
        "optimizer_for",
        lambda model: torch.optim.AdamW(model.parameters(), lr=0.01),
    )
    # Only the driver device is substituted; the actual loss, optimizer, RNG and
    # checkpoint functions execute. Real CUDA behavior is checked by --prepare.
    real_tensor = torch.tensor
    monkeypatch.setattr(
        torch,
        "tensor",
        lambda *args, **kwargs: real_tensor(*args, **dict(kwargs, device="cpu")),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    original_group = train_group
    calls = []

    def cpu_group(model, optimizer, examples, weights, device):
        calls.append(len(examples))
        return original_group(model, optimizer, examples, weights, "cpu")

    monkeypatch.setattr(training, "train_group", cpu_group)
    evaluations = []

    def evaluate(*args):
        evaluations.append(1)
        return copy.deepcopy(baseline)

    monkeypatch.setattr(training, "evaluate", evaluate)

    def arguments(out):
        return SimpleNamespace(
            frames=12,
            resume="auto",
            seed=42,
            prepare=False,
            epochs=8,
            checkpoint=Path("FFraw.tar"),
            run_dir=out,
        )

    reference = tmp_path / "reference"
    training.execute(arguments(reference), reference)
    assert len(calls) == len(evaluations) == 4  # first best then three ties
    expected = torch.load(reference / "checkpoints/epoch_004.pth", weights_only=True)

    interrupted = tmp_path / "interrupted"
    real_copy = training.atomic_copy

    def crash_after_commit(source, destination):
        if source.name == "epoch_002.pth" and crash == "after_commit":
            raise KeyboardInterrupt("checkpoint committed, report not yet written")
        real_copy(source, destination)

    def crash_before_commit(*args):
        result = evaluate(*args)
        if len(evaluations) == 6:
            raise KeyboardInterrupt("incomplete second epoch")
        return result

    if crash == "before_commit":
        monkeypatch.setattr(training, "evaluate", crash_before_commit)
    monkeypatch.setattr(training, "atomic_copy", crash_after_commit)
    with pytest.raises(KeyboardInterrupt):
        training.execute(arguments(interrupted), interrupted)
    assert (interrupted / "checkpoints/epoch_002.pth").is_file() == (
        crash == "after_commit"
    )
    monkeypatch.setattr(training, "evaluate", evaluate)
    monkeypatch.setattr(training, "atomic_copy", real_copy)
    training.execute(arguments(interrupted), interrupted)
    actual = torch.load(interrupted / "checkpoints/epoch_004.pth", weights_only=True)
    for key in expected["model_state_dict"]:
        assert torch.equal(
            expected["model_state_dict"][key], actual["model_state_dict"][key]
        )
    assert expected["early_stopping"] == actual["early_stopping"]
    assert actual["best_epoch"] == 1
    assert (interrupted / "epoch_002_validation.json").is_file()
    count = len(calls)
    (interrupted / "baseline_comparison.json").unlink()
    training.execute(arguments(interrupted), interrupted)
    assert len(calls) == count  # stopped run cannot accidentally train again
    assert len(read_json(interrupted / "baseline_comparison.json")["epochs"]) == 4
    assert all(count == 3 for count in calls)  # final partial accumulation group
