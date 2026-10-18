"""Yalnızca sentetik fotoğraflar/toy model; gerçek robustness çalıştırılmaz."""

import copy
import io
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF

import src.robustness as robustness
from src.dataset import image_transform, read_image
from src.experiment_log import append_experiment
from src.robustness import (
    CONDITIONS,
    RobustnessDataset,
    compare_to_clean,
    corrupt_batch,
    digest,
)


def test_clean_exactly_matches_final_test_preprocessing(tmp_path):
    for folder, color in (("real", "white"), ("fake", "black")):
        (tmp_path / "test" / folder).mkdir(parents=True)
        Image.new("RGB", (256, 256), color).save(tmp_path / "test" / folder / "a.png")
    dataset = RobustnessDataset(tmp_path)
    assert {label for _, label in dataset.samples} == {0, 1}
    for i, (relative, label) in enumerate(dataset.samples):
        tensor, actual, path = dataset[i]
        assert actual == label == {"real": 0, "fake": 1}[Path(relative).parts[0]]
        expected = image_transform(False)(read_image(dataset.root / path))
        torch.testing.assert_close(
            TF.normalize(tensor, **robustness.NORMALIZATION), expected, rtol=0, atol=0
        )
    assert not (tmp_path / "train").exists() and not (tmp_path / "valid").exists()


@pytest.mark.parametrize("condition", CONDITIONS)
def test_corruptions_shape_range_finite_and_no_inplace(condition):
    tensor = torch.rand((1, 3, 224, 224), generator=torch.Generator().manual_seed(10))
    before = tensor.clone()
    result = corrupt_batch(tensor, condition, ["real/a.png"])
    assert result.shape == tensor.shape and torch.isfinite(result).all()
    assert result.min() >= 0 and result.max() <= 1
    torch.testing.assert_close(tensor, before, rtol=0, atol=0)


@pytest.mark.parametrize("quality", [70, 50])
def test_jpeg_exact_quality_roundtrip(quality):
    tensor = torch.rand((1, 3, 224, 224), generator=torch.Generator().manual_seed(5))
    with io.BytesIO() as buffer:
        TF.to_pil_image(tensor[0]).save(
            buffer, format="JPEG", quality=quality, subsampling=2
        )
        buffer.seek(0)
        with Image.open(buffer) as image:
            expected = TF.to_tensor(image.convert("RGB"))
    actual = corrupt_batch(tensor, f"JPEG_Q{quality}", ["a"])
    torch.testing.assert_close(actual[0], expected, rtol=0, atol=0)
    assert not torch.equal(actual, tensor)


def test_gaussian_sigma_and_resize_have_expected_effect():
    impulse = torch.zeros(1, 3, 224, 224)
    impulse[:, :, 112, 112] = 1
    blur1 = corrupt_batch(impulse, "BLUR_SIGMA1", ["a"])
    blur2 = corrupt_batch(impulse, "BLUR_SIGMA2", ["a"])
    assert 0 < blur2[0, 0, 112, 112] < blur1[0, 0, 112, 112] < 1
    assert blur1[0, 0].sum().item() == pytest.approx(1, abs=1e-5)
    pattern = (torch.arange(224)[None, :] + torch.arange(224)[:, None]) % 2
    checkerboard = pattern.float().repeat(1, 3, 1, 1)
    reduced = corrupt_batch(checkerboard, "RESIZE_112", ["a"])
    assert reduced.std() < checkerboard.std() / 10


def test_noise_std_shared_draws_path_seed_and_batch_independence():
    clean = torch.full((2, 3, 224, 224), 0.5)
    state = torch.get_rng_state().clone()
    a = corrupt_batch(clean, "NOISE_STD0.01", ["a", "b"])
    b = corrupt_batch(clean, "NOISE_STD0.03", ["a", "b"])
    torch.testing.assert_close(a - clean, (b - clean) / 3, rtol=0, atol=1e-7)
    assert (a - clean).std().item() == pytest.approx(0.01, rel=0.01)
    assert (b - clean).std().item() == pytest.approx(0.03, rel=0.01)
    reordered = corrupt_batch(clean.flip(0), "NOISE_STD0.01", ["b", "a"])
    torch.testing.assert_close(a, reordered.flip(0), rtol=0, atol=0)
    torch.testing.assert_close(
        a[:1], corrupt_batch(clean[:1], "NOISE_STD0.01", ["a"]), rtol=0, atol=0
    )
    assert not torch.equal(
        a, corrupt_batch(clean, "NOISE_STD0.01", ["a", "b"], seed=43)
    )
    assert torch.equal(state, torch.get_rng_state())


def test_drop_units_and_negative_improvements():
    report = compare_to_clean(
        {
            "CLEAN": {"accuracy": 0.9, "f1": 0.8, "roc_auc": 0.95},
            "JPEG_Q70": {"accuracy": 0.85, "f1": 0.82, "roc_auc": 0.91},
        }
    )
    assert report["CLEAN"]["accuracy_drop"] == 0
    assert report["JPEG_Q70"]["accuracy_drop_percentage_points"] == pytest.approx(5)
    assert report["JPEG_Q70"]["f1_drop_percentage_points"] == pytest.approx(-2)
    assert report["JPEG_Q70"]["roc_auc_drop"] == pytest.approx(0.04)


def log_record():
    return {
        "id": "synthetic-photo-1",
        "name": "Sentetik fotoğraf testi",
        "date": "2026-10-09",
        "model": "toy",
        "checkpoint": "fixture",
        "dataset": "fixture",
        "counts": {"test": 4},
        "objective": "Test",
        "settings": {},
        "results": {},
        "duration_seconds": 1,
        "observations": "Test",
        "conclusion": "Test",
        "files": [],
        "status": "completed",
    }


def test_log_appends_preserves_history_and_is_idempotent(tmp_path):
    path = tmp_path / "EXPERIMENTS.md"
    path.write_text("# Önceki fotoğraf deneyleri\n", encoding="utf-8")
    assert append_experiment(path, log_record())
    before = path.read_bytes()
    assert not append_experiment(path, log_record())
    assert path.read_bytes() == before
    failed = {
        **log_record(),
        "id": "failed-photo-2",
        "status": "failed",
        "observations": "Hata",
    }
    assert append_experiment(path, failed)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Önceki fotoğraf deneyleri") and "failed" in text


def test_log_atomic_failure_keeps_existing_history(tmp_path, monkeypatch):
    import src.experiment_log as logger

    path = tmp_path / "EXPERIMENTS.md"
    path.write_text("unchanged", encoding="utf-8")

    def fail(*args):
        raise OSError("Disk write failure")

    monkeypatch.setattr(logger.os, "replace", fail)
    with pytest.raises(OSError):
        append_experiment(path, log_record())
    assert path.read_text() == "unchanged"
    assert not path.with_suffix(".lock").exists()
    assert not list(tmp_path.glob("*.tmp"))


class FrozenToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("sentinel", torch.tensor(1.0))

    def forward(self, images):
        assert not torch.is_grad_enabled() and not self.training
        mean = images.mean(dim=(1, 2, 3))
        return torch.stack([mean, -mean], dim=1)


@pytest.fixture
def synthetic_run(tmp_path, monkeypatch):
    for folder, color in (("real", "white"), ("fake", "black")):
        (tmp_path / "data/test" / folder).mkdir(parents=True)
        for i in range(2):
            Image.new("RGB", (256, 256), color).save(
                tmp_path / "data/test" / folder / f"{i}.png"
            )
    checkpoint = tmp_path / "toy.pth"
    checkpoint.write_bytes(b"unchanged toy checkpoint")
    dataset = RobustnessDataset(tmp_path / "data")
    reference = tmp_path / "reference.json"
    reference.write_text(
        json.dumps(
            {
                "checkpoint_sha256": digest(checkpoint),
                "label_mapping": {"REAL": 0, "FAKE": 1},
                "test_image_count": 4,
                "test_class_counts": {"REAL": 2, "FAKE": 2},
                "test_fingerprint": dataset.fingerprint(),
                "best_epoch": 3,
                "confusion_matrix": [[2, 0], [0, 2]],
                "accuracy": 1,
                "f1": 1,
                "roc_auc": 1,
            }
        ),
        encoding="utf-8",
    )
    model = FrozenToyModel()
    monkeypatch.setattr(
        robustness,
        "load_checkpoint",
        lambda *args: (model, {"pipeline": "presplit_image_v2", "epoch": 3}),
    )
    return {
        "dataset_root": tmp_path / "data",
        "checkpoint_path": checkpoint,
        "reference_path": reference,
        "output": tmp_path / "output",
        "log_path": tmp_path / "docs/EXPERIMENTS.md",
        "device": "cpu",
        "batch_size": 2,
        "expected_count": 4,
    }, model


def test_synthetic_eight_condition_run_outputs_and_immutability(synthetic_run):
    kwargs, model = synthetic_run
    originals = {path: digest(path) for path in kwargs["dataset_root"].rglob("*.png")}
    originals.update(
        {
            kwargs["checkpoint_path"]: digest(kwargs["checkpoint_path"]),
            kwargs["reference_path"]: digest(kwargs["reference_path"]),
        }
    )
    state = copy.deepcopy(model.state_dict())
    report = robustness.run_robustness(**kwargs)
    assert tuple(report["results"]) == CONDITIONS
    assert report["reference_clean_confusion_matrix_matches"]
    assert all(
        result["confusion_matrix"] == [[2, 0], [0, 2]]
        for result in report["results"].values()
    )
    frame = pd.read_csv(kwargs["output"] / "metrics/robustness_v2.csv")
    assert len(frame) == 8 and tuple(frame.condition) == CONDITIONS
    with Image.open(kwargs["output"] / "figures/robustness_v2.png") as image:
        image.verify()
    assert all(digest(path) == sha for path, sha in originals.items())
    torch.testing.assert_close(
        state["sentinel"], model.state_dict()["sentinel"], rtol=0, atol=0
    )
    assert "completed" in kwargs["log_path"].read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        robustness.run_robustness(**kwargs)


def test_changed_test_file_rejected_and_failure_logged(synthetic_run):
    kwargs, _ = synthetic_run
    Image.new("RGB", (256, 256), "blue").save(
        kwargs["dataset_root"] / "test/real/0.png"
    )
    with pytest.raises(ValueError, match="içeriği"):
        robustness.run_robustness(**kwargs)
    assert "failed" in kwargs["log_path"].read_text(encoding="utf-8")
    assert not (kwargs["output"] / "metrics/robustness_v2.json").exists()


def test_unknown_condition_rejected():
    with pytest.raises(ValueError):
        corrupt_batch(torch.zeros(1, 3, 224, 224), "unknown", ["a"])
