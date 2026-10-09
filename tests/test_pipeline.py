"""Gerçek dataset/ön-eğitim indirmeden altyapı testleri."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from src.dataset import (
    CLASS_NAMES,
    FaceDataset,
    discover_samples,
    image_transform,
    make_loader,
    read_image,
    seed_everything,
    split_samples,
)
from src.evaluate import calculate_metrics, save_metrics
from src.model import build_model, load_checkpoint, resolve_device
from src.predict import predict_image


@pytest.fixture
def sample_root(tmp_path: Path) -> Path:
    for folder_index, folder in enumerate(("Real", "Fake")):
        (tmp_path / folder).mkdir()
        for index in range(20):
            Image.new(
                "RGB", (32, 48), color=(index, 100 + folder_index * 50, 200)
            ).save(tmp_path / folder / f"{index:02}.png")
    return tmp_path


@pytest.fixture(scope="module")
def model():
    torch.set_num_threads(2)
    seed_everything()
    return build_model(pretrained=False).eval()


def test_split_ratios_reproducibility_and_no_overlap(sample_root):
    samples = discover_samples(sample_root)
    splits = split_samples(samples)
    assert splits == split_samples(samples)
    assert splits != split_samples(samples, seed=43)
    assert [len(splits[name]) for name in ("train", "val", "test")] == [28, 6, 6]
    sets = [set(subset) for subset in splits.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert set.union(*sets) == set(samples)
    for subset in splits.values():
        assert sum(label == 0 for _, label in subset) == len(subset) // 2


def test_dataset_tensor_and_default_batch(sample_root):
    samples = discover_samples(sample_root)
    dataset = FaceDataset(sample_root, samples)
    tensor, label = dataset[0]
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32 and torch.isfinite(tensor).all()
    assert label == 0
    images, labels = next(iter(make_loader(dataset)))
    assert images.shape == (8, 3, 224, 224)
    assert labels.shape == (8,)


@pytest.mark.parametrize("mode", ["L", "RGBA"])
def test_non_rgb_image(mode, tmp_path):
    path = tmp_path / "image.png"
    Image.new(mode, (17, 33)).save(path)
    assert image_transform()(read_image(path)).shape == (3, 224, 224)


def test_missing_class_is_actionable(tmp_path):
    with pytest.raises(ValueError, match="Sınıf klasörü"):
        discover_samples(tmp_path)


def test_insufficient_dataset_rejected():
    with pytest.raises(ValueError, match="yeterli görüntü"):
        split_samples([("Real/a.png", 0), ("Fake/b.png", 1)])


def test_corrupt_image_rejected(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"invalid image")
    with pytest.raises(ValueError, match="Görüntü okunamadı"):
        read_image(path)


def test_model_two_logits(model):
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 224, 224))
    assert logits.shape == (1, 2)
    assert torch.isfinite(logits).all()


def test_checkpoint_roundtrip_and_prediction(model, tmp_path):
    checkpoint_path = tmp_path / "test.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "image_size": 224,
        },
        checkpoint_path,
    )
    restored, _ = load_checkpoint(checkpoint_path, torch.device("cpu"))
    path = tmp_path / "face.png"
    Image.new("RGB", (50, 40)).save(path)
    prediction = predict_image(restored, path, torch.device("cpu"))
    assert prediction == predict_image(model, path, torch.device("cpu"))
    assert prediction["label"] in CLASS_NAMES
    assert sum(prediction["probabilities"].values()) == pytest.approx(1.0)


def test_checkpoint_class_order_rejected(tmp_path):
    path = tmp_path / "wrong.pt"
    torch.save({"class_names": ["FAKE", "REAL"], "image_size": 224}, path)
    with pytest.raises(ValueError, match="uyumlu değil"):
        load_checkpoint(path, torch.device("cpu"))


def test_metrics_fake_positive_class_and_output(tmp_path):
    metrics = calculate_metrics([0, 0, 0, 1, 1], [0, 1, 1, 1, 1])
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["confusion_matrix"] == [[1, 2], [0, 2]]
    save_metrics(metrics, tmp_path)
    assert json.loads((tmp_path / "metrics/test.json").read_text()) == metrics
    assert (tmp_path / "figures/test_confusion_matrix.png").stat().st_size > 0


def test_no_fake_predictions_has_zero_precision():
    metrics = calculate_metrics([0, 1], [0, 0])
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert metrics["confusion_matrix"] == [[1, 0], [1, 0]]


def test_empty_metrics_rejected():
    with pytest.raises(ValueError):
        calculate_metrics([], [])


def test_seed_reproducibility():
    seed_everything()
    first = torch.rand(4)
    seed_everything()
    assert torch.equal(first, torch.rand(4))


def test_invalid_device_rejected():
    with pytest.raises(ValueError):
        resolve_device("invalid")


def test_train_requires_explicit_mode():
    result = subprocess.run(
        [sys.executable, "-m", "src.train"], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "--run-training" in result.stderr


def test_check_data_does_not_train(sample_root, tmp_path):
    output_dir = tmp_path / "outputs"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train",
            "--check-data",
            "--data-dir",
            str(sample_root),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert not output_dir.exists()
