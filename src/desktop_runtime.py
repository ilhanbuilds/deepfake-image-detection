"""Separate loopback server lifecycle; leaves existing browser/model runtimes intact."""

import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import tempfile
import threading
from time import monotonic
import urllib.error
import urllib.request

from src.desktop_job import WindowsJob

ROOT = Path(__file__).resolve().parents[1]


def choose_port():
    with socket.socket() as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def healthy(url):
    # Ignore system HTTP proxies even for local connections.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url + "/_stcore/health", timeout=0.6) as response:
            return response.status == 200 and response.read(16).strip() == b"ok"
    except (OSError, urllib.error.URLError):
        return False


def cleanup_temps(record):
    """Only private job directories recorded by this child; never follow links."""
    if not record.exists():
        return
    temp_root = Path(tempfile.gettempdir()).resolve()
    for line in record.read_text(encoding="utf-8").splitlines():
        path = Path(json.loads(line))
        if path.parent != temp_root or not path.name.startswith(
            ("deepfake-demo-", "deepfake-preview-")
        ):
            raise ValueError("Geçici dosya kaydı güvenli sınırın dışında.")
        if not path.exists():
            continue
        if path.is_symlink() or path.is_junction():
            raise ValueError("Geçici klasörde beklenmeyen bağlantı bulundu.")
        for folder, directories, files in os.walk(path, followlinks=False):
            for name in directories + files:
                item = Path(folder) / name
                if item.is_symlink() or item.is_junction():
                    raise ValueError("Geçici klasörde beklenmeyen bağlantı bulundu.")
        shutil.rmtree(path)


class DesktopServer:
    def __init__(self, root=ROOT, job_factory=WindowsJob, popen=subprocess.Popen):
        self.root = Path(root)
        self.job_factory, self.popen = job_factory, popen
        self.cancelled = threading.Event()
        self.lock = threading.RLock()
        self.process = self.job = self.log = None
        self.url = None
        self.session = self.root / "outputs/desktop_runtime" / secrets.token_hex(12)
        self.session.mkdir(parents=True, exist_ok=False)

    def _release(self):
        if self.job is not None:
            self.job.close()
            self.job = None
        if self.process is not None:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Only our own gated child, never an unrelated port owner.
                self.process.kill()
                self.process.wait(timeout=5)
            self.process = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def start(self, timeout=45, attempts=4):
        python = self.root / ".venv/Scripts/python.exe"
        if not python.is_file():
            raise RuntimeError(
                "Ana Python ortamı bulunamadı. .venv klasörünü kontrol edin."
            )
        for attempt in range(attempts):
            if self.cancelled.is_set():
                return None
            base = "desktop-" + secrets.token_hex(16)
            port = choose_port()
            url = f"http://127.0.0.1:{port}/{base}"
            log_path = self.session / f"server-{attempt}.log"
            with self.lock:
                if self.cancelled.is_set():
                    return None
                self.job = self.job_factory()
                self.log = log_path.open("wb")
                try:
                    self.process = self.popen(
                        [
                            str(python),
                            "-B",
                            "-u",
                            "-m",
                            "src.desktop_server",
                            "--port",
                            str(port),
                            "--base",
                            base,
                            "--session",
                            str(self.session),
                        ],
                        cwd=self.root,
                        env=dict(
                            os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1"
                        ),
                        stdin=subprocess.PIPE,
                        stdout=self.log,
                        stderr=subprocess.STDOUT,
                        shell=False,
                        creationflags=subprocess.CREATE_NO_WINDOW
                        if os.name == "nt"
                        else 0,
                    )
                    self.job.assign(self.process)
                    self.process.stdin.write((base + "\n").encode())
                    self.process.stdin.flush()
                    self.process.stdin.close()
                except Exception:
                    if self.process is not None and self.process.poll() is None:
                        self.process.kill()  # Gate is still closed; no descendants exist.
                    self._release()
                    raise
            deadline = monotonic() + timeout
            while not self.cancelled.is_set() and monotonic() < deadline:
                with self.lock:
                    if self.process is None:
                        return None
                    code = self.process.poll()
                if code is not None:
                    break
                # Readiness must come from OUR child's post-bind URL banner as
                # well as its unique base-path health endpoint (port-probe race).
                banner = log_path.read_text(encoding="utf-8", errors="replace")
                if url in banner and healthy(url):
                    with self.lock:
                        if self.process is not None and self.process.poll() is None:
                            self.url = url
                            (self.session / "server.json").write_text(
                                json.dumps(
                                    dict(
                                        pid=self.process.pid,
                                        owner_pid=os.getpid(),
                                        url=url,
                                    ),
                                    indent=2,
                                ),
                                encoding="utf-8",
                            )
                            return url
                self.cancelled.wait(0.15)
            with self.lock:
                self._release()
            if self.cancelled.is_set():
                return None
            log = log_path.read_text(encoding="utf-8", errors="replace")
            if (
                "is already in use" not in log
                and f"Port {port} is not available" not in log
            ):
                raise RuntimeError(
                    "Yerel arayüz başlatılamadı veya zamanında yanıt vermedi."
                )
        raise RuntimeError(
            "Boş bir yerel port ayrılamadı. Diğer uygulamalar kapatılmadı."
        )

    def running(self):
        with self.lock:
            return self.process is not None and self.process.poll() is None

    def stop(self):
        self.cancelled.set()
        with self.lock:
            self._release()
            cleanup_temps(self.session / "owned_temps.jsonl")


def page(message, error=False):
    from html import escape

    title = "Uygulama açılamadı" if error else "Analiz alanınız hazırlanıyor"
    return f"""<!doctype html><html lang="tr"><meta charset="utf-8">
    <style>body{{margin:0;background:#f6f9ff;color:#172644;font:17px 'Segoe UI',sans-serif;
    display:grid;place-items:center;height:100vh}}main{{max-width:580px;padding:48px}}
    small{{color:#2563eb;font-weight:700;letter-spacing:3px}}h1{{font-size:34px}}
    p{{color:#61718b;line-height:1.8}}.loader{{width:32px;height:32px;border:3px solid #dce6fa;
    border-top-color:#2563eb;border-radius:50%;animation:spin 1s linear infinite}}
    @keyframes spin{{to{{transform:rotate(360deg)}}}}
    @media(prefers-reduced-motion:reduce){{.loader{{animation:none}}}}</style>
    <main><small>DEEPFAKE ANALİZ · MASAÜSTÜ</small><h1>{title}</h1>
    {"" if error else '<div class="loader"></div>'}<p>{escape(message)}</p>
    {"<p>Pencereyi kapatıp yeniden başlatabilirsiniz.</p>" if error else ""}</main></html>"""
