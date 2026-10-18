"""Synthetic adapter/UI checks; no official video data, training or real model load."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

import app
import app_unified as ui
from scripts import demo_worker as worker
from src import demo_media as media
from src import demo_services as services
from src import demo_preview as preview


@pytest.fixture(autouse=True)
def synthetic_preview(monkeypatch):
    monkeypatch.setattr(
        ui,
        "prepare_video_preview",
        lambda data, name: (
            data,
            dict(duration_seconds=5.0, width=640, height=480),
            [],
        ),
    )


@pytest.fixture
def photo():
    stream = io.BytesIO()
    Image.new("RGB", (80, 60), (150, 90, 70)).save(stream, format="PNG")
    return stream.getvalue()


def photo_analysis(photo):
    return services.Analysis(
        dict(
            mode="photo",
            status="accepted",
            label="REAL",
            fake_score=0.15,
            heatmap_present=True,
            wall_seconds=1.2,
        ),
        photo,
        photo,
    )


def video_analysis():
    return services.Analysis(
        dict(
            mode="video",
            status="accepted",
            label="FAKE",
            fake_score=0.72,
            p6=0.55,
            expanded=True,
            processed_frames=12,
            selected_frames=12,
            valid_frames=11,
            wall_seconds=8.0,
            inference_seconds=4.0,
            model_load_seconds=2.0,
            warmup_seconds=1.0,
            warnings=["Bir karede yüz bulunamadı."],
        )
    )


def test_photo_decoder_exactly_reuses_original(photo):
    assert (
        media.photo_image(photo).tobytes()
        == app.decode_upload(photo).convert("RGB").tobytes()
    )


@pytest.mark.parametrize(
    "name,data,mode",
    [
        ("photo.gif", b"GIF89a", "photo"),
        ("video.exe", b"x", "video"),
        ("video.mp4", b"not video", "video"),
        ("video.webm", b"not webm", "video"),
        ("video.mp4", b"", "video"),
        ("photo.png", b"", "photo"),
        ("video.mp4", b"video", "other"),
    ],
)
def test_invalid_upload_types_and_content_rejected(name, data, mode):
    with pytest.raises(media.DemoError):
        media.validate_upload(data, name, mode)


def test_upload_size_limits_checked_before_subprocess(photo, monkeypatch):
    monkeypatch.setattr(media, "MAX_VIDEO_BYTES", 8)
    with pytest.raises(media.DemoError, match="100 MB"):
        media.validate_upload(b"0000ftypmore", "demo.mp4", "video")
    monkeypatch.setattr(media, "MAX_PHOTO_BYTES", 1)
    with pytest.raises(media.DemoError, match="10 MB"):
        media.validate_upload(photo, "demo.png", "photo")


class Capture:
    def __init__(self, values, opened=True):
        self.values, self.opened, self.closed = values, opened, False

    def isOpened(self):
        return self.opened

    def get(self, key):
        return self.values[key]

    def release(self):
        self.closed = True


def probe(values, opened=True):
    capture = Capture(values, opened)
    cv = SimpleNamespace(
        CAP_PROP_FRAME_COUNT=0,
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        VideoCapture=lambda _: capture,
    )
    return capture, cv


@pytest.mark.parametrize(
    "values",
    [
        (1801, 30, 640, 480),
        (6, 0, 640, 480),
        (30, 30, 3840, 2160),
        (5, 30, 640, 480),
        (30, float("nan"), 640, 480),
        (7300, 120, 640, 480),
    ],
)
def test_video_metadata_limits_and_capture_cleanup(values):
    capture, cv = probe(values)
    with pytest.raises(media.DemoError):
        media.probe_video(Path("synthetic.mp4"), cv)
    assert capture.closed


def test_valid_video_metadata_and_unsupported_codec():
    capture, cv = probe((300, 30, 640, 480))
    assert media.probe_video(Path("synthetic.mp4"), cv)["duration_seconds"] == 10
    assert capture.closed
    capture, cv = probe((300, 30, 640, 480), False)
    with pytest.raises(media.DemoError, match="codec"):
        media.probe_video(Path("synthetic.mp4"), cv)
    assert capture.closed


def test_temporary_input_and_artifacts_removed_on_success(photo, monkeypatch):
    directories = []

    def run(mode, directory, callback, timeout):
        directories.append(directory)
        request = json.loads((directory / "request.json").read_text())
        assert request["input_name"] == "input.png"
        assert (directory / request["input_name"]).read_bytes() == photo
        (directory / "result.json").write_text(json.dumps(photo_analysis(photo).data))
        (directory / "preview.png").write_bytes(photo)
        (directory / "gradcam.png").write_bytes(photo)
        callback(1, "Tamamlandı")

    monkeypatch.setattr(services, "_run_worker", run)
    events = []
    result = services.analyze_photo(
        photo, "unsafe;name.png", lambda *e: events.append(e)
    )
    assert result.overlay == photo and result.data["wall_seconds"] >= 0
    assert events and all(not directory.exists() for directory in directories)


def test_temporary_input_removed_after_failure_and_lock_released(photo, monkeypatch):
    paths = []

    def fail(mode, directory, *args):
        paths.append(directory)
        raise media.DemoError("Sentetik hata")

    monkeypatch.setattr(services, "_run_worker", fail)
    with pytest.raises(media.DemoError):
        services.analyze_photo(photo, "face.png")
    assert not paths[0].exists()
    with services.analysis_slot():
        with pytest.raises(media.DemoError, match="Başka bir analiz"):
            with services.analysis_slot():
                pytest.fail("Concurrent GPU analysis allowed")


def test_worker_command_is_fixed_no_shell_and_output_progress(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.write_bytes(b"synthetic executable marker")
    monkeypatch.setattr(services, "VIDEO_PYTHON", python)
    observed = {}

    class Process:
        stdout = io.StringIO(
            '{"event":"progress","message":"İnceleniyor","progress":0.5}\n'
        )
        returncode = 0

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

    def launch(command, **kwargs):
        observed.update(command=command, **kwargs)
        return Process()

    monkeypatch.setattr(services.subprocess, "Popen", launch)
    events = []
    services._run_worker("video", tmp_path, lambda *event: events.append(event), 5)
    assert observed["shell"] is False and observed["command"][0] == str(python)
    assert "scripts.demo_worker" in observed["command"] and events == [
        (0.5, "İnceleniyor")
    ]


def test_worker_timeout_terminates_owned_child(tmp_path, monkeypatch):
    python = tmp_path / "python.exe"
    python.write_bytes(b"synthetic")
    monkeypatch.setattr(services, "PHOTO_PYTHON", python)
    stopped = []

    class Process:
        stdout = io.StringIO("")
        returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            stopped.append(True)
            self.returncode = -1

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(services.subprocess, "Popen", lambda *a, **kw: Process())
    times = iter([0.0, 2.0])
    monkeypatch.setattr(services, "monotonic", lambda: next(times))
    with pytest.raises(media.DemoError, match="süre sınırını"):
        services._run_worker("photo", tmp_path, None, 1)
    assert stopped


def test_worker_rejects_non_private_job_and_path_escape(tmp_path, monkeypatch):
    with pytest.raises(media.DemoError, match="klasörü"):
        worker.request_input(tmp_path, "video")
    monkeypatch.setattr(worker.tempfile, "gettempdir", lambda: str(tmp_path))
    job = tmp_path / "deepfake-demo-synthetic"
    job.mkdir()
    (job / "request.json").write_text(
        json.dumps(dict(mode="video", input_name="../outside.mp4"))
    )
    with pytest.raises(media.DemoError, match="yolu"):
        worker.request_input(job, "video")


def test_missing_face_and_short_video_warnings_are_turkish():
    messages = media.video_warnings(
        dict(
            status="rejected",
            failures=[{}],
            budget_shortfall=2,
            expansion_reason="incomplete_initial_six",
        )
    )
    assert len(messages) == 4 and any("tahmini üretilmedi" in x for x in messages)


def test_preview_conversion_preserves_original_and_cleans_files(monkeypatch):
    monkeypatch.setattr(preview, "probe_video", lambda *a: dict(duration_seconds=1))
    monkeypatch.setattr(preview.shutil, "which", lambda _: "trusted-existing-ffmpeg")
    observed = []
    original = b"0000ftyporiginal"

    def convert(command, **kwargs):
        source = Path(command[command.index("-i") + 1])
        observed.append(source.parent)
        assert source.read_bytes() == original and kwargs["shell"] is False
        assert "libx264" in command
        Path(command[-1]).write_bytes(b"0000ftypbrowser")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(preview.subprocess, "run", convert)
    result, _, warnings = preview.prepare_video_preview(original, "demo.mp4")
    assert result == b"0000ftypbrowser" and not warnings
    assert not observed[0].exists()


def test_preview_missing_converter_has_explicit_fallback(monkeypatch):
    monkeypatch.setattr(preview, "probe_video", lambda *a: {})
    monkeypatch.setattr(preview.shutil, "which", lambda _: None)
    data = b"0000ftyporiginal"
    result, _, warnings = preview.prepare_video_preview(data, "demo.mp4")
    assert result == data and warnings


def test_idle_ui_has_two_tabs_and_never_loads_models(monkeypatch):
    monkeypatch.setattr(
        ui, "analyze_photo", lambda *a: pytest.fail("idle photo inference")
    )
    monkeypatch.setattr(
        ui, "analyze_video", lambda *a: pytest.fail("idle video inference")
    )
    app_test = AppTest.from_string(
        "from app_unified import main\nmain()", default_timeout=30
    ).run()
    assert not app_test.exception
    assert [tab.label for tab in app_test.tabs] == ["Fotoğraf analizi", "Video analizi"]
    assert (
        len(app_test.file_uploader) == 2
        and app_test.button[0].disabled
        and app_test.button[1].disabled
    )


def test_photo_ui_preview_result_gradcam_and_clear(photo, monkeypatch):
    calls = []

    def analyze(data, name, callback):
        calls.append(data)
        callback(0.5, "İnceleniyor")
        return photo_analysis(photo)

    monkeypatch.setattr(ui, "analyze_photo", analyze)
    test = AppTest.from_string(
        "from app_unified import main\nmain()", default_timeout=30
    ).run()
    test.file_uploader[0].set_value(("face.png", photo, "image/png")).run()
    test.button[0].click().run()
    assert not test.exception and calls == [photo]
    assert any(
        metric.label == "FAKE skoru" and metric.value == "0.1500"
        for metric in test.metric
    )
    assert any("Grad-CAM" in " ".join(image.captions) for image in test.image)
    test.button[-1].click().run()
    assert not test.exception and not test.image


def test_video_ui_preview_progress_and_expanded_result(monkeypatch):
    def analyze(data, name, callback):
        callback(0.6, "Ek kareler inceleniyor")
        return video_analysis()

    monkeypatch.setattr(ui, "analyze_video", analyze)
    test = AppTest.from_string(
        "from app_unified import main\nmain()", default_timeout=30
    ).run()
    test.file_uploader[1].set_value(
        ("demo.mp4", b"0000ftypisomsynthetic", "video/mp4")
    ).run()
    test.button[1].click().run()
    assert not test.exception
    assert any(
        metric.label == "İşlenen kare" and metric.value == "12"
        for metric in test.metric
    )
    assert any("genişletildi" in value.value for value in test.success)
    assert any("yüz bulunamadı" in warning.value for warning in test.warning)


def test_rejected_video_ui_has_no_fake_prediction(monkeypatch):
    monkeypatch.setattr(
        ui,
        "analyze_video",
        lambda *a: services.Analysis(
            dict(mode="video", status="rejected", warnings=["Yüz bulunamadı."])
        ),
    )
    test = AppTest.from_string(
        "from app_unified import main\nmain()", default_timeout=30
    ).run()
    test.file_uploader[1].set_value(
        ("demo.mp4", b"0000ftypisomsynthetic", "video/mp4")
    ).run()
    test.button[1].click().run()
    assert not test.exception and not test.metric
    assert any("Tahmin üretilmedi" in info.value for info in test.info)


def test_invalid_image_never_starts_worker(monkeypatch):
    monkeypatch.setattr(
        ui, "analyze_photo", lambda *a: pytest.fail("bad file reached model")
    )
    test = AppTest.from_string(
        "from app_unified import main\nmain()", default_timeout=30
    ).run()
    test.file_uploader[0].set_value(("broken.jpg", b"broken", "image/jpeg")).run()
    assert not test.exception and test.error and test.button[0].disabled
