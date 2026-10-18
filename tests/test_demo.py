"""Lokal demo: gerçek model/Grad-CAM, upload sınırları ve CLEAN eşdeğerliği."""

import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from streamlit.testing.v1 import AppTest

import app
from scripts.generate_gradcam_v3 import select_correct_examples
from src.explainability import generate_gradcam, generate_gradcam_from_image
from src.model import build_model
from src.predict import predict_image


@pytest.fixture
def photograph():
    image = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    )
    with io.BytesIO() as buffer:
        image.save(buffer, format="PNG")
        return buffer.getvalue()


@pytest.fixture
def demo_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = build_model(pretrained=False).eval()
    checkpoint = tmp_path / "demo.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": ["REAL", "FAKE"],
            "image_size": 224,
            "epoch": 5,
        },
        checkpoint,
    )
    monkeypatch.setattr(app, "CHECKPOINT", checkpoint)
    app.load_demo_model.clear()
    yield model
    app.load_demo_model.clear()


def test_idle_demo_does_not_load_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CHECKPOINT", tmp_path / "absent.pth")
    demo = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()
    assert not demo.exception
    assert demo.title[0].value == "Deepfake Yüz Görüntüsü Tespiti"
    assert len(demo.file_uploader) == 1
    assert len(demo.info) == 1
    assert not demo.error


def test_upload_prediction_gradcam_and_clean_preprocessing(
    demo_checkpoint, photograph, monkeypatch
):
    originals = []
    generate = app.generate_gradcam_from_image

    def capture(model, image, device):
        originals.append(image.copy())
        return generate(model, image, device)

    monkeypatch.setattr(app, "generate_gradcam_from_image", capture)
    demo = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()
    demo.file_uploader[0].set_value(("face.png", photograph, "image/png")).run()
    demo.button[0].click().run()
    assert not demo.exception
    assert not demo.error
    assert demo.subheader[0].value in {"Tahmin: REAL", "Tahmin: FAKE"}
    assert 50 <= float(demo.metric[0].value.lstrip("%")) <= 100
    assert len(demo.image) == 2
    assert [image.captions for image in demo.image] == [
        ["Orijinal görüntü"],
        ["Grad-CAM görüntüsü"],
    ]
    assert len(originals) == 1
    assert originals[0].size == (96, 64)
    np.testing.assert_array_equal(
        np.asarray(originals[0]), np.asarray(app.decode_upload(photograph))
    )
    assert demo.markdown[0].value == (
        "Model, eğitim dağılımına benzer yüz ağırlıklı görüntülerde "
        "deepfake/sentetik yüz sınıflandırması yapar. "
        "Genel amaçlı AI görüntü tespiti değildir."
    )
    assert "Best epoch: 5" in demo.caption[-1].value


def test_broken_upload_is_rejected_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "CHECKPOINT", tmp_path / "absent.pth")
    demo = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()
    demo.file_uploader[0].set_value(("broken.jpg", b"not an image", "image/jpeg")).run()
    demo.button[0].click().run()
    assert not demo.exception
    assert "Fotoğraf açılamadı" in demo.error[0].value


def test_missing_checkpoint_has_clear_error(tmp_path, monkeypatch, photograph):
    monkeypatch.setattr(app, "CHECKPOINT", tmp_path / "absent.pth")
    app.load_demo_model.clear()
    demo = AppTest.from_string("from app import main\nmain()", default_timeout=30).run()
    demo.file_uploader[0].set_value(("face.png", photograph, "image/png")).run()
    demo.button[0].click().run()
    assert not demo.exception
    assert "V3 checkpoint bulunamadı" in demo.error[0].value
    app.load_demo_model.clear()


def test_in_memory_gradcam_equals_file_prediction(
    demo_checkpoint, photograph, tmp_path
):
    path = tmp_path / "face.png"
    path.write_bytes(photograph)
    device = torch.device("cpu")
    in_memory = generate_gradcam_from_image(
        demo_checkpoint, app.decode_upload(photograph), device
    )
    from_file = generate_gradcam(demo_checkpoint, path, device)
    prediction = predict_image(demo_checkpoint, path, device)
    assert in_memory.label == from_file.label == prediction["label"]
    assert (
        in_memory.probabilities
        == from_file.probabilities
        == prediction["probabilities"]
    )
    np.testing.assert_array_equal(in_memory.heatmap, from_file.heatmap)


@pytest.mark.parametrize("format_name", ["JPEG", "PNG"])
def test_supported_rgb_and_grayscale_upload(format_name):
    with io.BytesIO() as buffer:
        Image.new("L", (32, 48), 120).save(buffer, format=format_name)
        assert app.decode_upload(buffer.getvalue()).size == (32, 48)


def test_unsupported_format_and_upload_limits(photograph, monkeypatch):
    with io.BytesIO() as buffer:
        Image.new("RGB", (10, 10)).save(buffer, format="GIF")
        with pytest.raises(ValueError, match="JPG/JPEG/PNG"):
            app.decode_upload(buffer.getvalue())
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 1)
    with pytest.raises(ValueError, match="10 MB"):
        app.decode_upload(photograph)
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", len(photograph))
    monkeypatch.setattr(app, "MAX_IMAGE_PIXELS", 1)
    with pytest.raises(ValueError, match="megapiksel"):
        app.decode_upload(photograph)


def test_selection_keeps_only_correct_labels_and_is_repeatable(monkeypatch):
    samples = [
        (f"{label}/{index}.png", label) for label in (0, 1) for index in range(9)
    ]
    dataset = SimpleNamespace(root=Path("test"), samples=samples)

    def prediction(model, path, device):
        label = int(path.parent.name)
        if path.stem == "0":
            label = 1 - label
        return {"label": ["REAL", "FAKE"][label]}

    monkeypatch.setattr("src.predict.predict_image", prediction)
    first, examined = select_correct_examples(None, dataset, None)
    assert first == select_correct_examples(None, dataset, None)[0]
    assert len(first) == len(set(path for path, _ in first)) == 10
    assert [label for _, label in first] == [0] * 5 + [1] * 5
    assert all(path.stem != "0" for path, _ in first)
    assert all(value >= 5 for value in examined.values())


def test_selection_fails_when_not_enough_correct_examples(monkeypatch):
    dataset = SimpleNamespace(root=Path("test"), samples=[("real/1.png", 0)])
    monkeypatch.setattr("src.predict.predict_image", lambda *args: {"label": "FAKE"})
    with pytest.raises(ValueError, match="doğru örnek"):
        select_correct_examples(None, dataset, None)
