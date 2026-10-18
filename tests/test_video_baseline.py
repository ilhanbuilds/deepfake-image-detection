"""No validation leakage, masked mean pooling and read-only epoch selection."""

import json

import numpy as np
import pytest
import torch

from src.video_analysis import family_distribution, inspect_epochs, stratify_predictions
from src.video_baseline import (
    baseline_predictions,
    export_baseline,
    fit_baseline,
    temporal_mean,
)
from src.video_evaluate import calculate_video_metrics


@pytest.mark.parametrize("length", [3, 4, 5, 6])
def test_temporal_mean_ignores_padding_and_frame_order(length):
    values = (
        torch.arange(length, dtype=torch.float32).reshape(-1, 1).expand(length, 1280)
    )
    padded = torch.cat([values, torch.full((6 - length, 1280), float("nan"))])
    expected = np.full(1280, (length - 1) / 2)
    np.testing.assert_allclose(temporal_mean(padded, length), expected)
    np.testing.assert_allclose(temporal_mean(values.flip(0)), expected)
    if length < 6:
        with pytest.raises(ValueError, match="NaN/Inf"):
            temporal_mean(padded)


@pytest.mark.parametrize("length", [0, 2, 7])
def test_invalid_video_lengths_rejected(length):
    with pytest.raises(ValueError):
        temporal_mean(torch.zeros(length, 1280))


def test_baseline_train_only_scaling_weights_and_export(tmp_path):
    rng = np.random.default_rng(42)
    labels = np.array([0] * 8 + [1] * 24)
    features = rng.normal(size=(32, 1280))
    features[:, 0] += labels * 4
    baseline = fit_baseline(features, labels, dict(REAL=2, FAKE=2 / 3))
    scaler = baseline.named_steps["standardscaler"]
    original_mean = scaler.mean_.copy()
    np.testing.assert_allclose(original_mean, features.mean(axis=0))
    validation = features[:2] + 1000
    items = [dict(record=dict(path=f"val{i}", label=i)) for i in range(2)]
    result = baseline_predictions(baseline, validation, items)
    assert len(result["predictions"]) == 2
    np.testing.assert_array_equal(scaler.mean_, original_mean)
    assert baseline.named_steps["logisticregression"].class_weight == {0: 2, 1: 2 / 3}
    output = tmp_path / "parameters.npz"
    export_baseline(baseline, output)
    with np.load(output) as saved:
        normalized = (features - saved["mean"]) / saved["scale"]
        logits = normalized @ saved["coefficient"].T + saved["intercept"]
        probabilities = torch.from_numpy(logits).sigmoid().numpy()[:, 0]
    np.testing.assert_allclose(probabilities, baseline.predict_proba(features)[:, 1])
    with pytest.raises(FileExistsError):
        export_baseline(baseline, output)


def test_family_composition_and_length_metrics_keep_video_labels():
    records = [
        dict(path="Celeb-real/a", label=0, family="a"),
        dict(path="Celeb-synthesis/a1", label=1, family="a"),
        dict(path="Celeb-synthesis/a2", label=1, family="a"),
        dict(path="YouTube-real/b", label=0, family="b"),
    ]
    summary = family_distribution(records)
    assert summary["unique_families"] == 2 and summary["mixed_families"] == 1
    assert summary["class_families"] == dict(REAL=2, FAKE=1)
    assert summary["family_size_histogram"] == {1: 1, 3: 1}
    rows = [
        dict(
            path=r["path"],
            target=r["label"],
            prediction=r["label"],
            fake_probability=0.9 if r["label"] else 0.1,
        )
        for r in records
    ]
    items = [
        dict(record=r, values=torch.zeros(3 if i < 3 else 6, 1280))
        for i, r in enumerate(records)
    ]
    result = stratify_predictions(rows, items)
    assert result["by_valid_length"][3]["n"] == 3
    assert result["by_valid_length"][6]["n"] == 1
    assert result["by_valid_length"][6]["metrics"] is None
    assert result["by_valid_length"][4]["metrics"] is None
    assert result["by_folder"]["Celeb-real"]["class_recall"] == 1


def test_best_checkpoint_selection_is_validated_against_all_epoch_logs(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    history = []
    predictions = [
        dict(path="real", target=0, prediction=0, fake_probability=0.1),
        dict(path="fake", target=1, prediction=1, fake_probability=0.9),
    ]
    metrics = calculate_video_metrics([0, 1], [0, 1], [0.1, 0.9])
    for epoch in (1, 2):
        history.append(
            dict(epoch=epoch, validation_metrics=metrics, val_video_loss=0.1)
        )
        torch.save(
            dict(epoch=epoch, history=history.copy(), manifest_fingerprint="test"),
            checkpoint_dir / f"epoch_{epoch:03d}.pth",
        )
        (tmp_path / f"epoch_{epoch:03d}_coverage.json").write_text(
            json.dumps(
                dict(
                    train=dict(
                        evaluated_videos=2, attempted_videos=2, rejected_videos=[]
                    ),
                    validation=dict(
                        predictions=predictions,
                        video_loss=0.1,
                        coverage=dict(
                            evaluated_videos=2, attempted_videos=2, rejected_videos=[]
                        ),
                    ),
                )
            ),
            encoding="utf-8",
        )
    (tmp_path / "history.json").write_text(json.dumps(history), encoding="utf-8")
    (tmp_path / "cache_audit.json").write_text(
        json.dumps(
            dict(
                splits=dict(
                    val=dict(
                        accepted_records=[dict(path="real"), dict(path="fake")],
                        accepted=2,
                        total=2,
                        rejected=0,
                    ),
                    train=dict(accepted=2, total=2, rejected=0),
                )
            )
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "best_video.pth").write_bytes(
        (checkpoint_dir / "epoch_001.pth").read_bytes()
    )
    (checkpoint_dir / "last_video.pth").write_bytes(
        (checkpoint_dir / "epoch_002.pth").read_bytes()
    )
    assert inspect_epochs(tmp_path, dict(fingerprint="test"))[1] == 1
    (checkpoint_dir / "best_video.pth").write_bytes(
        (checkpoint_dir / "epoch_002.pth").read_bytes()
    )
    with pytest.raises(ValueError, match="Best checkpoint"):
        inspect_epochs(tmp_path, dict(fingerprint="test"))
