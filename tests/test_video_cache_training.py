"""Accepted-cache training safety; synthetic tensors only, no dataset training."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch

from src.experiment_log import append_experiment
from src.video_cache import audit_feature_cache
from src.video_evaluate import calculate_video_metrics, coverage
from src.video_model import VideoClassifier
from src.video_pipeline import FeatureCache, feature_batch, make_video_loader
from src.video_train import run_training, training_class_balance


@pytest.fixture
def cached_inventory(tmp_path):
    cache = FeatureCache(
        tmp_path / "cache",
        dict(settings=dict(frames=6, min_valid_frames=3)),
        "synthetic_backbone",
    )
    splits = dict(train=[], val=[], test=[dict(path="never_open_test.mp4")])
    for split, lengths in (("train", [3, 4, 5, 6, 2]), ("val", [3, 6])):
        for index, length in enumerate(lengths):
            record = dict(
                path=f"{split}_{index}.mp4", label=index % 2, family=f"{split}_{index}"
            )
            (tmp_path / record["path"]).write_bytes(b"synthetic source")
            report = dict(
                path=record["path"],
                status="accepted" if length >= 3 else "rejected",
                valid_frames=length,
                selected_indices=list(range(6)),
                valid_indices=list(range(length)),
                failed_frames=[
                    dict(index=i, reason="face_not_detected") for i in range(length, 6)
                ],
            )
            values = torch.randn(length, 1280) if length >= 3 else None
            cache.write(record, tmp_path, values, report)
            splits[split].append(record)
    manifest = dict(dataset_root=str(tmp_path), fingerprint="synthetic", splits=splits)
    return manifest, cache


def test_audit_filters_rejections_and_never_reads_test(cached_inventory, monkeypatch):
    manifest, cache = cached_inventory
    read = cache.read
    accessed = []

    def checked_read(record, root):
        assert record not in manifest["splits"]["test"]
        accessed.append(record["path"])
        return read(record, root)

    monkeypatch.setattr(cache, "read", checked_read)
    report = audit_feature_cache(manifest, cache)
    train, val = report["splits"]["train"], report["splits"]["val"]
    assert train["accepted"] == 4 and train["rejected"] == 1
    assert train["accepted_counts"] == dict(REAL=2, FAKE=2)
    assert train["valid_frame_histogram"] == {3: 1, 4: 1, 5: 1, 6: 1}
    assert val["accepted_counts"] == dict(REAL=1, FAKE=1)
    assert len(accessed) == 7 and report["official_test_accessed"] is False
    assert report["cross_split_paths"] == report["cross_split_families"] == []
    assert len(report["cache_fingerprint"]) == 64
    # A rejected REAL changes raw counts but must not affect training loss weights.
    assert training_class_balance(train["accepted_records"])["class_weights"] == {
        "REAL": 1.0,
        "FAKE": 1.0,
    }


@pytest.mark.parametrize(
    "change", ["missing", "extra", "path_overlap", "family_overlap"]
)
def test_incomplete_or_overlapping_cache_blocks_training(cached_inventory, change):
    manifest, cache = cached_inventory
    if change == "missing":
        cache.path(manifest["splits"]["train"][0]).unlink()
    elif change == "extra":
        (cache.directory / "unexpected.pt").write_bytes(b"unexpected")
    elif change == "path_overlap":
        manifest["splits"]["val"][0] = manifest["splits"]["train"][0]
    else:
        manifest["splits"]["val"][0]["family"] = manifest["splits"]["train"][0][
            "family"
        ]
    with pytest.raises(ValueError, match="eksik|çakışması"):
        audit_feature_cache(manifest, cache)


@pytest.mark.parametrize(
    "change",
    ["too_short", "too_long", "nonfinite", "order", "failure_index", "rejected_count"],
)
def test_invalid_frame_cache_rejected(cached_inventory, change):
    manifest, cache = cached_inventory
    record = manifest["splits"]["train"][-1 if change == "rejected_count" else 0]
    path = cache.path(record)
    saved = torch.load(path, weights_only=True)
    if change in {"too_short", "too_long"}:
        length = 2 if change == "too_short" else 7
        saved["features"] = torch.zeros(length, 1280)
        saved["report"]["valid_frames"] = length
    elif change == "nonfinite":
        saved["features"][0, 0] = float("nan")
    elif change == "order":
        saved["report"]["valid_indices"].reverse()
    elif change == "failure_index":
        saved["report"]["failed_frames"][0]["index"] = 99
    else:
        saved["report"]["valid_frames"] = 3
    torch.save(saved, path)
    with pytest.raises(ValueError, match="Bozuk|sırası|ret kare"):
        audit_feature_cache(manifest, cache)


def test_accepted_train_weights_match_completed_cache_counts():
    balance = training_class_balance([dict(label=0)] * 420 + [dict(label=1)] * 1833)
    assert balance["counts"] == dict(REAL=420, FAKE=1833)
    assert balance["class_weights"]["REAL"] == pytest.approx(2.6821428571)
    assert balance["class_weights"]["FAKE"] == pytest.approx(2253 / 3666)
    assert balance["source"] == "accepted_train_cache_only"


def test_imbalanced_metrics_measure_both_classes():
    result = calculate_video_metrics([0, 1, 1, 1, 1], [1] * 5, [0.8] * 5)
    assert result["accuracy"] == pytest.approx(0.8)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["per_class"]["REAL"] == dict(precision=0, recall=0, f1=0, support=1)
    assert result["per_class"]["FAKE"]["precision"] == pytest.approx(0.8)
    assert result["per_class"]["FAKE"]["recall"] == 1
    assert result["confusion_matrix"] == [[0, 1], [0, 4]]


def test_mixed_lengths_ignore_padding_and_padding_gradients(cached_inventory):
    torch.set_num_threads(2)
    manifest, cache = cached_inventory
    items = [
        cache.read(record, cache.directory.parent.parent)
        for record in manifest["splits"]["train"][:4]
    ]
    model = VideoClassifier(pretrained=False, hidden_size=8).eval()
    batch = next(iter(make_video_loader(items, batch_size=2)))
    padded, lengths = feature_batch(model, batch, torch.device("cpu"), True, False)
    assert lengths.tolist() == [3, 4] and padded.shape == (2, 4, 1280)
    # Exercise every valid length in one packed batch, including poisoned padding.
    padded = torch.nn.utils.rnn.pad_sequence(
        [item["values"] for item in items], batch_first=True
    )
    lengths = torch.tensor([3, 4, 5, 6])
    for index, length in enumerate(lengths):
        padded[index, length:] = float("nan")
    padded.requires_grad_()
    logits = model(padded, lengths)
    for index, item in enumerate(items):
        individual = model(item["values"].unsqueeze(0), lengths[index : index + 1])
        torch.testing.assert_close(logits[index : index + 1], individual)
    logits.sum().backward()
    for index, length in enumerate(lengths):
        assert torch.isfinite(padded.grad[index]).all()
        assert torch.all(padded.grad[index, :length].abs().sum(dim=1) > 0)
        assert torch.count_nonzero(padded.grad[index, length:]) == 0


def test_validation_macro_f1_controls_best_early_stop_and_export(
    cached_inventory, monkeypatch, tmp_path
):
    import src.video_train as training

    manifest, cache = cached_inventory
    real_audit = audit_feature_cache(manifest, cache)
    monkeypatch.setattr(training, "load_manifest", lambda path: manifest)
    monkeypatch.setattr(training, "audit_feature_cache", lambda *args: real_audit)
    monkeypatch.setattr(
        training,
        "VideoClassifier",
        lambda **kwargs: VideoClassifier(pretrained=False, hidden_size=8),
    )
    items = {
        name: [cache.read(record, tmp_path) for record in subset["accepted_records"]]
        for name, subset in real_audit["splits"].items()
    }
    monkeypatch.setattr(
        training,
        "FeatureDataset",
        lambda records, *args: items[
            "train"
            if records == real_audit["splits"]["train"]["accepted_records"]
            else "val"
        ],
    )
    metrics = calculate_video_metrics([0, 1], [0, 1], [0.1, 0.9])
    evaluations = iter([(0.6, 0.9), (0.5, 0.95)])

    def validation(*args):
        macro, fake = next(evaluations)
        result = copy.deepcopy(metrics)
        result.update(macro_f1=macro, f1=fake)
        return dict(
            metrics=result,
            video_loss=0.1,
            coverage=coverage([i["report"] for i in items["val"]]),
        )

    # No optimizer is run; high train/Fake-F1 must never choose the best epoch.
    monkeypatch.setattr(
        training,
        "train_epoch",
        lambda *args: dict(
            video_loss=0.1,
            metrics=metrics,
            metrics_scope="online_training_batches",
            coverage=coverage([i["report"] for i in items["train"]]),
        ),
    )
    monkeypatch.setattr(training, "evaluate_video", validation)
    args = SimpleNamespace(
        epochs=3,
        hidden_size=8,
        patience=1,
        learning_rate=3e-4,
        manifest=tmp_path / "unused.json",
        image_size=160,
        device="cpu",
        run_dir=tmp_path / "run",
        resume=None,
        batch_size=2,
        cache_dir=tmp_path / "cache",
    )
    run_training(args)
    best = torch.load(args.run_dir / "checkpoints/best_video.pth", weights_only=True)
    last = torch.load(args.run_dir / "checkpoints/last_video.pth", weights_only=True)
    assert best["epoch"] == 1 and last["epoch"] == 2
    assert last["best_macro_f1"] == 0.6 and last["early_stopping"]["bad_epochs"] == 1
    assert last["config"]["training_class_balance"]["counts"] == dict(REAL=2, FAKE=2)
    assert (
        last["history"][1]["validation_metrics"]["f1"]
        > last["history"][0]["validation_metrics"]["f1"]
    )
    assert not (args.run_dir / "checkpoints/epoch_003.pth").exists()
    record = json.loads(
        (args.run_dir / "experiment_record.json").read_text(encoding="utf-8")
    )
    assert record["results"]["best_validation_metrics"]["macro_f1"] == 0.6
    assert record["counts"]["train"]["rejected"] == 1
    assert record["counts"]["official_test"] == "erişilmedi"
    document = tmp_path / "EXPERIMENTS.md"
    document.write_text("# Existing photo results\n", encoding="utf-8")
    assert append_experiment(document, record)
    assert not append_experiment(document, record)
    assert document.read_text(encoding="utf-8").startswith("# Existing photo results\n")
