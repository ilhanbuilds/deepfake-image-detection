"""Sunuma hatalı metrik taşınmasını ve mevcut çıktıların ezilmesini engeller."""

import copy

import pytest

from scripts.build_final_report_pack import build_pack, validate_metrics


@pytest.fixture
def metrics():
    return {
        "class_names": ["REAL", "FAKE"],
        "positive_class": "FAKE",
        "confusion_matrix": [[3, 1], [1, 5]],
        "accuracy": 0.8,
        "precision": 5 / 6,
        "recall": 5 / 6,
        "f1": 5 / 6,
        "roc_auc": 0.91,
        "test_image_count": 10,
        "correct_predictions": 8,
    }


def test_valid_metrics_remain_unchanged(metrics):
    before = copy.deepcopy(metrics)
    assert validate_metrics(metrics) == 10
    assert metrics == before


@pytest.mark.parametrize(
    "key,value",
    [
        ("class_names", ["FAKE", "REAL"]),
        ("accuracy", 0.99),
        ("recall", 0.99),
        ("test_image_count", 20),
        ("correct_predictions", 9),
        ("roc_auc", float("nan")),
        ("confusion_matrix", [[3, -1], [1, 5]]),
    ],
)
def test_inconsistent_report_is_rejected(metrics, key, value):
    metrics[key] = value
    with pytest.raises(ValueError):
        validate_metrics(metrics)


def test_existing_final_files_are_not_overwritten(tmp_path):
    folder = tmp_path / "outputs/figures/final"
    folder.mkdir(parents=True)
    protected = folder / "existing.png"
    protected.write_bytes(b"keep original")
    with pytest.raises(FileExistsError, match="üzerine yazılmaz"):
        build_pack(tmp_path)
    assert protected.read_bytes() == b"keep original"
