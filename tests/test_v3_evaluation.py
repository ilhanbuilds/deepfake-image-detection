"""Sentetik fotoğraflarla final/robustness sırası, metrik ve koruma kontrolleri."""

import json

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

import src.evaluate_v3 as evaluation
from src.evaluate import calculate_metrics
from src.robustness import CONDITIONS, NORMALIZATION, RobustnessDataset, digest


class FrozenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sentinel", torch.tensor(1.0))

    def forward(self, images):
        assert not torch.is_grad_enabled() and not self.training
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack([mean, -mean], dim=1)


@pytest.fixture
def run_args(tmp_path, monkeypatch):
    data = tmp_path / "data"
    for name, color in (("real", "white"), ("fake", "black")):
        (data / "test" / name).mkdir(parents=True)
        for index in range(2):
            Image.new("RGB", (256, 256), color).save(
                data / "test" / name / f"{index}.png"
            )
    fingerprint = RobustnessDataset(data).fingerprint()
    checkpoint = tmp_path / "v3.pth"
    checkpoint.write_bytes(b"unchanged V3")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "test_fingerprint": fingerprint,
                "checkpoint_sha256": "V2-sha",
                "test_class_counts": {"REAL": 2, "FAKE": 2},
                "training_validation_fingerprint": "dev-sha",
            }
        ),
        encoding="utf-8",
    )
    baseline = tmp_path / "robustness_v2.json"
    baseline.write_text(
        json.dumps(
            {
                "seed": 42,
                "test_fingerprint": fingerprint,
                "checkpoint_sha256": "V2-sha",
                "label_mapping": {"REAL": 0, "FAKE": 1},
                "evaluation_precision": "float32",
                "settings": {"batch_size": 2, "conditions": list(CONDITIONS)},
                "normalization": {
                    key: list(value) for key, value in NORMALIZATION.items()
                },
                "test_image_count": 4,
                "class_counts": {"REAL": 2, "FAKE": 2},
                "results": {
                    c: calculate_metrics([0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1])
                    for c in CONDITIONS
                },
            }
        ),
        encoding="utf-8",
    )
    model = FrozenModel()
    monkeypatch.setattr(
        evaluation,
        "load_checkpoint",
        lambda *_: (
            model,
            {
                "pipeline": "presplit_image_v3_robust",
                "epoch": 5,
                "best_epoch": 5,
                "dataset_fingerprint": "dev-sha",
            },
        ),
    )
    return {
        "dataset_root": data,
        "checkpoint_path": checkpoint,
        "v2_report_path": baseline,
        "reference_path": reference,
        "output": tmp_path / "outputs",
        "log_path": tmp_path / "EXPERIMENTS.md",
        "device": "cpu",
        "batch_size": 2,
        "expected_count": 4,
    }


def test_clean_before_eight_conditions_outputs_and_protected_files(
    run_args, monkeypatch
):
    protected = {
        path: digest(path)
        for path in (
            run_args["checkpoint_path"],
            run_args["v2_report_path"],
            run_args["reference_path"],
            *run_args["dataset_root"].rglob("*.png"),
        )
    }
    original = evaluation.evaluate_conditions
    stages = []

    def ordered(model, loader, device, conditions, seed, stage):
        stages.append(tuple(conditions))
        if len(conditions) == 8:
            assert (run_args["output"] / "metrics/final_test_metrics_v3.json").is_file()
        return original(model, loader, device, conditions, seed, stage)

    monkeypatch.setattr(evaluation, "evaluate_conditions", ordered)
    report = evaluation.run_evaluation(**run_args)
    assert stages == [("CLEAN",), CONDITIONS]
    assert tuple(report["results"]) == CONDITIONS and report["clean_matches_final_test"]
    clean = json.loads(
        (run_args["output"] / "metrics/final_test_metrics_v3.json").read_text()
    )
    assert clean["confusion_matrix"] == [[2, 0], [0, 2]]
    assert (
        clean["best_epoch"],
        clean["correct_predictions"],
        clean["incorrect_predictions"],
    ) == (5, 4, 0)
    assert (
        clean["roc_auc"] == 1
        and clean["test_previously_inspected_for_v3_augmentation_design"]
    )
    assert all(digest(path) == sha for path, sha in protected.items())
    assert len(pd.read_csv(run_args["output"] / "metrics/robustness_v3.csv")) == 8
    with Image.open(run_args["output"] / "figures/robustness_v2_vs_v3.png") as image:
        image.verify()
    assert "completed" in run_args["log_path"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        evaluation.run_evaluation(**run_args)


def test_changed_photo_rejected_without_final_metrics(run_args):
    Image.new("RGB", (256, 256), "blue").save(
        run_args["dataset_root"] / "test/real/0.png"
    )
    with pytest.raises(ValueError, match="fotoğrafları"):
        evaluation.run_evaluation(**run_args)
    assert not (run_args["output"] / "metrics/final_test_metrics_v3.json").exists()
    assert "failed" in run_args["log_path"].read_text(encoding="utf-8")


def test_improvement_uses_percentage_points_and_preserves_negative_results():
    v2 = {c: {"accuracy": 0.8, "f1": 0.7} for c in CONDITIONS}
    v3 = {c: {"accuracy": 0.9, "f1": 0.65} for c in CONDITIONS}
    comparison = evaluation.compare_models(v2, v3)
    assert comparison["BLUR_SIGMA1"][
        "accuracy_improvement_percentage_points"
    ] == pytest.approx(10)
    assert comparison["RESIZE_112"][
        "f1_improvement_percentage_points"
    ] == pytest.approx(-5)
