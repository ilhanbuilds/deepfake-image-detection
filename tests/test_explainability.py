"""Eğitim yapmadan Grad-CAM, durum koruma ve bağımsız CLI testleri."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from src.dataset import CLASS_NAMES
from src.explainability import generate_gradcam, save_gradcam, select_target_layer
from src.model import build_model
from src.predict import predict_image


@pytest.fixture
def image_path(tmp_path):
    pixels = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    path = tmp_path / "face.png"
    Image.fromarray(pixels).save(path)
    return path


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    return build_model(pretrained=False).eval()


def test_last_convolution_is_selected_automatically(model):
    name, layer = select_target_layer(model)
    convolutions = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Conv2d)
    ]
    assert (name, layer) == convolutions[-1]
    assert layer.out_channels == model.classifier[1].in_features


def test_unsupported_model_rejected():
    with pytest.raises(ValueError, match="EfficientNet"):
        select_target_layer(nn.Linear(3, 2))


def test_gradcam_prediction_matches_predict_and_preserves_image_size(model, image_path):
    expected = predict_image(model, image_path, torch.device("cpu"))
    result = generate_gradcam(model, image_path, torch.device("cpu"))
    assert result.label == expected["label"]
    assert result.confidence_percent == pytest.approx(
        expected["probabilities"][result.label] * 100
    )
    assert result.probabilities == pytest.approx(expected["probabilities"])
    assert result.original.size == (96, 64)
    assert result.heatmap.shape == (64, 96)
    assert result.heatmap.dtype == np.float32
    assert np.isfinite(result.heatmap).all()
    assert 0 <= result.heatmap.min() <= result.heatmap.max() <= 1


def test_gradcam_preserves_weights_gradients_modes_and_removes_hooks(model, image_path):
    model.train()
    model.features[0].eval()
    modes = [module.training for module in model.modules()]
    state = {name: value.clone() for name, value in model.state_dict().items()}
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    saved_grad = parameter.grad.clone()
    _, layer = select_target_layer(model)
    hooks = len(layer._forward_hooks)
    generate_gradcam(model, image_path, torch.device("cpu"))
    assert [module.training for module in model.modules()] == modes
    assert all(
        torch.equal(state[name], value) for name, value in model.state_dict().items()
    )
    assert torch.equal(parameter.grad, saved_grad)
    assert len(layer._forward_hooks) == hooks
    assert all(p.grad is None for p in list(model.parameters())[1:])


def test_frozen_model_and_inference_mode_supported(model, image_path):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with torch.inference_mode():
        result = generate_gradcam(model, image_path, torch.device("cpu"))
    assert result.label in CLASS_NAMES
    assert 0 <= result.confidence_percent <= 100
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in model.parameters()
    )


def test_hook_and_modes_restored_after_invalid_output(model, image_path):
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.train()
    _, layer = select_target_layer(model)
    hooks = len(layer._forward_hooks)
    with pytest.raises(ValueError, match="logitleri"):
        generate_gradcam(model, image_path, torch.device("cpu"))
    assert model.training
    assert len(layer._forward_hooks) == hooks


def test_zero_gradients_produce_empty_finite_heatmap(model, image_path):
    with torch.no_grad():
        model.classifier[1].weight.zero_()
        model.classifier[1].bias.zero_()
    result = generate_gradcam(model, image_path, torch.device("cpu"))
    assert not np.any(result.heatmap)
    assert np.isfinite(result.heatmap).all()
    assert result.confidence_percent == pytest.approx(50.0)


def test_save_outputs_without_overwriting_prior_explanation(
    model, image_path, tmp_path
):
    result = generate_gradcam(model, image_path, torch.device("cpu"))
    output_dir = tmp_path / "outputs"
    first = save_gradcam(result, output_dir, image_path, tmp_path / "best_model.pth", 5)
    second = save_gradcam(
        result, output_dir, image_path, tmp_path / "best_model.pth", 5
    )
    assert first["metadata_file"] != second["metadata_file"]
    assert (
        json.loads(Path(first["metadata_file"]).read_text(encoding="utf-8"))[
            "best_epoch"
        ]
        == 5
    )
    for name in ("original", "heatmap", "overlay", "comparison"):
        with Image.open(first["files"][name]) as image:
            image.load()
            assert image.width > 0 and image.height > 0
            if name != "comparison":
                assert image.size == (96, 64)
    with Image.open(first["files"]["original"]) as saved:
        assert np.array_equal(np.asarray(saved), np.asarray(result.original))


def test_cli_from_another_working_directory(model, image_path, tmp_path):
    checkpoint_path = tmp_path / "best_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": list(CLASS_NAMES),
            "image_size": 224,
            "epoch": 5,
        },
        checkpoint_path,
    )
    script = Path(__file__).resolve().parents[1] / "scripts/generate_gradcam.py"
    output_dir = tmp_path / "cli_outputs"
    run = subprocess.run(
        [
            sys.executable,
            str(script),
            "--image",
            str(image_path),
            "--checkpoint",
            str(checkpoint_path),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd=tmp_path,
        capture_output=True,
    )
    assert run.returncode == 0, run.stderr.decode(errors="replace")
    metadata_files = list(output_dir.glob("*/prediction.json"))
    assert len(metadata_files) == 1
    metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert metadata["label"] in CLASS_NAMES
    assert metadata["target_layer"].startswith("features.")
