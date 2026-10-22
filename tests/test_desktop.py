"""Small lifecycle tests only: no model, dataset, historical cache or training."""

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from src import desktop_runtime as runtime


class Pipe:
    def __init__(self, process):
        self.process = process
        self.closed = False

    def write(self, data):
        assert self.process.assigned  # Gate cannot open before Job assignment.
        self.process.gate = data

    def flush(self):
        pass

    def close(self):
        self.closed = True


class Process:
    pid = 123

    def __init__(self, code=None):
        self.code, self.assigned = code, False
        self.stdin = Pipe(self)
        self.killed = False

    def poll(self):
        return self.code

    def wait(self, timeout):
        self.code = -1 if self.code is None else self.code
        return self.code

    def kill(self):
        self.killed, self.code = True, -1


class Job:
    def __init__(self):
        self.process, self.closed = None, False

    def assign(self, process):
        self.process = process
        process.assigned = True

    def close(self):
        self.closed = True
        if self.process:
            self.process.code = -1


@pytest.fixture
def factory(tmp_path, monkeypatch):
    (tmp_path / ".venv/Scripts").mkdir(parents=True)
    (tmp_path / ".venv/Scripts/python.exe").write_bytes(b"test")
    jobs, processes, invocations = [], [], []

    def job():
        item = Job()
        jobs.append(item)
        return item

    def popen(argv, **kwargs):
        process = Process()
        processes.append(process)
        invocations.append((argv, kwargs))
        port, base = argv[argv.index("--port") + 1], argv[argv.index("--base") + 1]
        kwargs["stdout"].write(f"URL: http://127.0.0.1:{port}/{base}".encode())
        kwargs["stdout"].flush()
        return process

    monkeypatch.setattr(runtime, "choose_port", lambda: 49152)
    monkeypatch.setattr(runtime, "healthy", lambda url: True)
    return SimpleNamespace(
        server=runtime.DesktopServer(tmp_path, job, popen),
        jobs=jobs,
        processes=processes,
        invocations=invocations,
    )


def test_separate_loopback_server_gated_before_start(factory):
    server = factory.server
    url = server.start()
    argv, options = factory.invocations[0]
    assert url.startswith("http://127.0.0.1:49152/desktop-")
    assert argv[0].endswith(".venv\\Scripts\\python.exe") or argv[0].endswith(
        ".venv/Scripts/python.exe"
    )
    assert argv[1:5] == ["-B", "-u", "-m", "src.desktop_server"]
    assert (
        options["shell"] is False and options["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
    )
    assert factory.processes[0].gate.strip().decode() == argv[argv.index("--base") + 1]
    assert json.loads((server.session / "server.json").read_text())["pid"] == 123
    server.stop()
    assert factory.jobs[0].closed and factory.processes[0].code == -1
    assert not factory.processes[0].killed


def test_stop_before_start_and_idempotent_stop(factory):
    factory.server.stop()
    assert factory.server.start() is None
    factory.server.stop()
    assert not factory.processes


def test_missing_model_environment_is_turkish(tmp_path):
    server = runtime.DesktopServer(tmp_path)
    with pytest.raises(RuntimeError, match="Ana Python ortamı"):
        server.start()
    server.stop()


def test_timeout_cleans_only_owned_job(factory, monkeypatch):
    monkeypatch.setattr(runtime, "healthy", lambda url: False)
    with pytest.raises(RuntimeError, match="başlatılamadı"):
        factory.server.start(timeout=0)
    assert factory.jobs[0].closed
    factory.server.stop()


def test_health_without_owned_banner_never_accepts_foreign_process(factory):
    original = factory.server.popen

    def blank(*args, **kwargs):
        process = original(*args, **kwargs)
        kwargs["stdout"].truncate(0)
        kwargs["stdout"].flush()
        return process

    factory.server.popen = blank
    with pytest.raises(RuntimeError, match="başlatılamadı"):
        factory.server.start(timeout=0.01)
    assert factory.jobs[0].closed


def test_collision_retries_and_preserves_unrelated_owner(factory):
    original = factory.server.popen

    def collision(*args, **kwargs):
        process = original(*args, **kwargs)
        if len(factory.processes) == 1:
            kwargs["stdout"].write(b"\nPort 49152 is already in use")
            kwargs["stdout"].flush()
            process.code = 1
        return process

    factory.server.popen = collision
    assert factory.server.start()
    assert len(factory.processes) == 2 and factory.jobs[0].closed
    assert not any(p.killed for p in factory.processes)
    factory.server.stop()


def test_assignment_failure_never_opens_gate(factory):
    class BadJob(Job):
        def assign(self, process):
            raise OSError("job assignment failure")

    factory.server.job_factory = BadJob
    with pytest.raises(OSError, match="assignment"):
        factory.server.start()
    assert factory.processes[0].killed
    assert not hasattr(factory.processes[0], "gate")
    factory.server.stop()


def test_health_uses_no_proxy_and_checks_body(monkeypatch):
    class Response(io.BytesIO):
        status = 200

    def build(handler):
        assert handler.proxies == {}
        return SimpleNamespace(open=lambda url, timeout: Response(b"ok"))

    monkeypatch.setattr(runtime.urllib.request, "build_opener", build)
    assert runtime.healthy("http://127.0.0.1:123/desktop-test")


def test_temp_cleanup_only_recorded_private_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    private = tmp_path / "deepfake-demo-fixture"
    private.mkdir()
    (private / "input.png").write_bytes(b"test")
    unrelated = tmp_path / "other"
    unrelated.mkdir()
    record = tmp_path / "owned.jsonl"
    record.write_text(json.dumps(str(private)) + "\n")
    runtime.cleanup_temps(record)
    assert not private.exists() and unrelated.exists()


def test_temp_cleanup_refuses_outside_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    target = tmp_path / "keep"
    target.mkdir()
    record = tmp_path / "owned.jsonl"
    record.write_text(json.dumps(str(target)) + "\n")
    with pytest.raises(ValueError, match="sınırın dışında"):
        runtime.cleanup_temps(record)
    assert target.exists()


def test_loading_and_error_text_is_escaped():
    html = runtime.page("<script>bad</script>", error=True)
    assert "&lt;script&gt;" in html and "<script>bad" not in html
    assert 'lang="tr"' in html and "Uygulama açılamadı" in html
    assert "hazırlanıyor" in runtime.page("başlatılıyor")


def test_child_rejects_wrong_gate_without_starting_server(tmp_path):
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "src.desktop_server",
            "--port",
            "49152",
            "--base",
            "expected-token",
            "--session",
            str(tmp_path),
        ],
        cwd=root,
        input="wrong-token\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=dict(os.environ, PYTHONUTF8="1"),
        timeout=10,
    )
    assert result.returncode != 0
    assert "doğrulanamadı" in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("fail_start", [False, True])
def test_window_close_and_turkish_startup_error(tmp_path, monkeypatch, fail_start):
    from scripts import desktop_app

    class Event:
        def __init__(self):
            self.callbacks = []

        def __iadd__(self, callback):
            self.callbacks.append(callback)
            return self

        def fire(self):
            for callback in self.callbacks:
                callback()

    class Server:
        def __init__(self):
            self.session = tmp_path
            self.cancelled = threading.Event()
            self.stopped = False

        def start(self):
            if fail_start:
                raise RuntimeError("Deneme başlatma hatası")
            return "http://127.0.0.1:49152/desktop-unit"

        def running(self):
            return not self.stopped

        def stop(self):
            self.stopped = True
            self.cancelled.set()

    server = Server()
    window = SimpleNamespace(
        events=SimpleNamespace(shown=Event(), loaded=Event(), closed=Event())
    )
    pages = []

    def load(url):
        assert url.startswith("http://127.0.0.1:")
        window.events.closed.fire()

    window.load_url = load
    window.load_html = pages.append
    window.get_current_url = lambda: None
    module = SimpleNamespace(settings={})

    def start(callback, **options):
        assert options["gui"] == "edgechromium" and options["private_mode"]
        assert options["debug"] is False
        window.events.shown.fire()
        callback()

    module.create_window = lambda *args, **kwargs: window
    module.start = start
    monkeypatch.setitem(sys.modules, "webview", module)
    monkeypatch.setattr(runtime, "DesktopServer", lambda: server)
    monkeypatch.setattr(
        desktop_app.ctypes,
        "windll",
        SimpleNamespace(
            shell32=SimpleNamespace(
                SetCurrentProcessExplicitAppUserModelID=lambda value: None
            )
        ),
        raising=False,
    )
    desktop_app.main()
    assert server.stopped
    assert module.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] is False
    assert module.settings["ALLOW_FILE_URLS"] is False
    if fail_start:
        assert pages and "Deneme başlatma hatası" in pages[0]
        assert (tmp_path / "error.log").is_file()
    else:
        assert not pages


def test_unwritable_output_folder_has_native_turkish_error(monkeypatch):
    from scripts import desktop_app

    dialogs = []

    def fail():
        raise PermissionError("test access failure")

    monkeypatch.setattr(runtime, "DesktopServer", fail)
    monkeypatch.setattr(
        desktop_app.ctypes,
        "windll",
        SimpleNamespace(
            user32=SimpleNamespace(MessageBoxW=lambda *args: dialogs.append(args))
        ),
        raising=False,
    )
    desktop_app.main()
    assert len(dialogs) == 1 and "yazma iznini" in dialogs[0][1]
