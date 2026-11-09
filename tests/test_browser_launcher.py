"""Browser launcher portability; no models, training or server processes."""

from types import SimpleNamespace

import pytest

from scripts import start_deepfake_app as launcher
from src import demo_services as services
from src import studio_browser as browser
from src import studio_server as studio
from src.demo_media import DemoError


@pytest.mark.parametrize(
    "platform,executable", [("nt", "Scripts/python.exe"), ("posix", "bin/python")]
)
def test_launcher_uses_project_environment_and_opens_local_browser(
    tmp_path, monkeypatch, platform, executable
):
    root = tmp_path / "project with spaces"
    python = root / ".venv" / executable
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(launcher, "__file__", str(root / "scripts/launcher.py"))
    monkeypatch.setattr(launcher, "os", SimpleNamespace(name=platform))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(launcher.subprocess, "run", run)
    with pytest.raises(SystemExit) as stopped:
        launcher.main()
    assert stopped.value.code == 7
    command, options = calls[0]
    assert command == [
        str(python),
        "-B",
        "-m",
        "src.studio_browser",
    ]
    assert options == {"cwd": root, "check": False}


def test_missing_environment_does_not_start_server(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "__file__", str(tmp_path / "scripts/launcher.py"))

    def unexpected_start(*args, **kwargs):
        pytest.fail("A missing environment must not start a process")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected_start)
    with pytest.raises(SystemExit, match="Python ortamı bulunamadı"):
        launcher.main()


def test_mac_video_reports_cuda_requirement_before_starting_worker(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(services, "sys", SimpleNamespace(platform="darwin"))

    def unexpected_start(*args, **kwargs):
        pytest.fail("Mac video must not start a CUDA worker")

    monkeypatch.setattr(services.subprocess, "Popen", unexpected_start)
    with pytest.raises(DemoError, match="CUDA destekli NVIDIA GPU"):
        services._run_worker("video", tmp_path, None, 5)
    monkeypatch.setattr(studio, "sys", SimpleNamespace(platform="darwin"))
    with pytest.raises(DemoError, match="CUDA destekli NVIDIA GPU"):
        studio.Studio(tmp_path).run_worker("video", tmp_path, 5)


@pytest.mark.parametrize("browser_opens", [True, False])
def test_browser_uses_studio_server_and_cancels_analysis_on_exit(
    tmp_path, monkeypatch, browser_opens
):
    import json
    import threading

    cancelled = threading.Event()
    closed, joined, opened = [], [], []
    server = SimpleNamespace(
        server_port=8765,
        studio=SimpleNamespace(
            cancelled=cancelled,
            thread=SimpleNamespace(join=lambda **kwargs: joined.append(kwargs)),
        ),
        server_close=lambda: closed.append(True),
    )

    def serve_forever(**kwargs):
        raise KeyboardInterrupt

    def create(session, token, port):
        assert port == 8502 and len(token) == 48
        assert session.is_dir()
        return server

    def open_browser(url):
        opened.append(url)
        return browser_opens

    server.serve_forever = serve_forever
    monkeypatch.setattr(browser, "ROOT", tmp_path)
    monkeypatch.setattr(browser, "create_server", create)
    monkeypatch.setattr(browser.webbrowser, "open", open_browser)
    browser.run()
    record = next((tmp_path / "outputs/studio_browser_sessions").glob("*/server.json"))
    assert json.loads(record.read_text())["url"] == opened[0]
    assert opened[0].startswith("http://127.0.0.1:8765/")
    assert cancelled.is_set() and closed == [True] and joined == [{"timeout": 10}]


def test_browser_port_collision_never_opens_browser(tmp_path, monkeypatch):
    monkeypatch.setattr(browser, "ROOT", tmp_path)

    def occupied(*args):
        raise OSError("Address already in use")

    def unexpected_open(*args):
        pytest.fail("A failed server must not open a browser")

    monkeypatch.setattr(browser, "create_server", occupied)
    monkeypatch.setattr(browser.webbrowser, "open", unexpected_open)
    with pytest.raises(SystemExit, match="port 8502"):
        browser.run()


def test_posix_worker_cleanup_only_kills_its_dedicated_group(monkeypatch):
    calls = []
    monkeypatch.setattr(studio, "signal", SimpleNamespace(SIGKILL=9))
    monkeypatch.setattr(
        studio,
        "os",
        SimpleNamespace(
            getpgid=lambda pid: pid,
            killpg=lambda group, sig: calls.append((group, sig)),
        ),
    )
    job = studio._PosixWorkerJob()
    job.assign(SimpleNamespace(pid=12345))
    job.close()
    job.close()
    assert calls == [(12345, studio.signal.SIGKILL)]


def test_posix_worker_rejects_shared_process_group(monkeypatch):
    monkeypatch.setattr(studio, "os", SimpleNamespace(getpgid=lambda pid: 1))
    job = studio._PosixWorkerJob()
    with pytest.raises(OSError, match="bağımsız"):
        job.assign(SimpleNamespace(pid=12345))
    job.close()
