"""Delivery portability and setup safety; no downloads or model training."""

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import setup_delivery as setup
from src.runtime_provenance import IMPLEMENTATIONS, verify_implementations
from src import studio_server
from tests.test_studio import photo


@pytest.mark.parametrize("machine", ["arm64", "x86_64"])
def test_mac_setup_requires_native_apple_silicon_before_install(
    tmp_path, monkeypatch, machine
):
    errors = io.StringIO()
    monkeypatch.setattr("sys.argv", ["setup_delivery.py", "--setup-only"])
    monkeypatch.setattr(
        setup,
        "sys",
        SimpleNamespace(version_info=(3, 12), platform="darwin", stderr=errors),
    )
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    installed = []

    def install(root, photo_only):
        installed.append(photo_only)
        return tmp_path / "python"

    monkeypatch.setattr(setup, "install", install)
    if machine == "arm64":
        setup.main()
        assert installed == [True]
    else:
        with pytest.raises(SystemExit) as stopped:
            setup.main()
        assert stopped.value.code == 1 and installed == []
        assert "ARM64 Python" in errors.getvalue()


def recorded_sources(root):
    source = root / "src"
    source.mkdir()
    recorded = {}
    for name in IMPLEMENTATIONS:
        contents = f"# {name}\r\n".encode()
        (source / name).write_bytes(contents)
        historical = (
            f"C:\\unavailable\\original-project\\src\\{name}"
            if name == "video_forensic_train.py"
            else f"src\\{name}"
        )
        recorded[historical] = hashlib.sha256(contents).hexdigest()
    return recorded


def test_relocated_provenance_verifies_current_root_not_historical_root(tmp_path):
    recorded = recorded_sources(tmp_path)
    verify_implementations(recorded, tmp_path)
    (tmp_path / "src/video_forensic_train.py").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_implementations(recorded, tmp_path)


@pytest.mark.parametrize(
    "bad", ["src/../video_train.py", "other/video_train.py", "unknown.py"]
)
def test_provenance_rejects_unknown_or_traversing_sources(tmp_path, bad):
    recorded = recorded_sources(tmp_path)
    recorded[bad] = "0" * 64
    with pytest.raises(ValueError, match="provenance path"):
        verify_implementations(recorded, tmp_path)


def test_provenance_cannot_drop_a_protected_source(tmp_path):
    recorded = recorded_sources(tmp_path)
    recorded.pop(next(iter(recorded)))
    with pytest.raises(ValueError, match="Incomplete"):
        verify_implementations(recorded, tmp_path)


def test_incompatible_environment_is_preserved_without_install(tmp_path, monkeypatch):
    environment = tmp_path / ".venv"
    environment.mkdir()
    marker = environment / "keep.txt"
    marker.write_text("existing environment")

    def unexpected_run(*args, **kwargs):
        pytest.fail("An incompatible existing environment must not be overwritten")

    monkeypatch.setattr(setup.subprocess, "run", unexpected_run)
    with pytest.raises(RuntimeError, match="Uyumsuz"):
        setup.ensure_environment(environment)
    assert marker.read_text() == "existing environment"


def test_missing_models_fail_before_any_environment_install(tmp_path, monkeypatch):
    def unexpected_environment(*args):
        pytest.fail("Missing assets must not trigger installation")

    monkeypatch.setattr(setup, "ensure_environment", unexpected_environment)
    with pytest.raises(RuntimeError, match="Paket eksik"):
        setup.install(tmp_path, photo_only=True)


def test_valid_models_are_hashed_not_just_checked_for_existence(tmp_path, monkeypatch):
    path = Path("model.pth")
    monkeypatch.setattr(
        setup, "ASSETS", {str(path): hashlib.sha256(b"original").hexdigest()}
    )
    (tmp_path / path).write_bytes(b"original")
    setup.verify_assets(tmp_path)
    (tmp_path / path).write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="bütünlüğü"):
        setup.verify_assets(tmp_path)


def test_photo_only_session_rejects_video_before_decoder_and_still_accepts_photo(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("DEEPFAKE_PHOTO_ONLY", "1")
    monkeypatch.setattr(studio_server, "detect_media", lambda contents, name: "video")
    studio = studio_server.Studio(tmp_path)
    with pytest.raises(studio_server.DemoError, match="yalnız fotoğraf"):
        studio.upload(b"0000ftypvideo", "video.mp4")
    assert studio.snapshot()["status"] == "empty"
    monkeypatch.setattr(studio_server, "detect_media", lambda contents, name: "photo")
    assert studio.upload(photo(), "photo.png")["status"] == "ready"
