"""Session-scoped loopback UI server. Model and video algorithms are unchanged."""

import argparse
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import queue
import secrets
import subprocess
import sys
import tempfile
import threading
from time import monotonic
from urllib.parse import parse_qs, urlsplit

from PIL import Image

from src.demo_media import DemoError, MAX_VIDEO_BYTES, validate_upload
from src.demo_preview import prepare_video_preview
from src.demo_services import analysis_slot, PHOTO_PYTHON, VIDEO_PYTHON
from src.desktop_job import WindowsJob

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets/studio"


def detect_media(contents, name):
    suffix = Path(name).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png"}:
        validate_upload(contents, name, "photo")
        try:
            with Image.open(io.BytesIO(contents)) as image:
                expected = "PNG" if suffix == ".png" else "JPEG"
                if image.format != expected:
                    raise DemoError(
                        "Dosyanın uzantısı gerçek görüntü biçimiyle uyuşmuyor."
                    )
                image.verify()
        except DemoError:
            raise
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            raise DemoError(
                "Fotoğraf açılamadı. Geçerli bir JPG veya PNG seçin."
            ) from error
        return "photo"
    validate_upload(contents, name, "video")
    return "video"


class Studio:
    def __init__(self, session):
        self.session = Path(session)
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        self.thread = None
        self.state = dict(
            status="empty", progress=None, message="", file=None, result=None
        )
        self.media = {}
        self.contents = None
        self.file_id = None

    def snapshot(self):
        with self.lock:
            return json.loads(json.dumps(self.state))

    def upload(self, contents, name):
        with self.lock:
            if self.state["status"] in {"running", "uploading"}:
                raise DemoError("Önce devam eden işlemi tamamlayın veya iptal edin.")
            self.state = dict(
                status="uploading",
                progress=None,
                message="Dosya kontrol ediliyor…",
                file=None,
                result=None,
            )
            self.media, self.contents = {}, None
        try:
            mode = detect_media(contents, name)
            if mode == "photo":
                from src.demo_media import photo_image

                image = photo_image(contents)
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                preview, mime = buf.getvalue(), "image/png"
                metadata, warnings = dict(width=image.width, height=image.height), []
            else:
                preview, metadata, warnings = prepare_video_preview(contents, name)
                mime = "video/mp4" if preview[4:8] == b"ftyp" else "video/webm"
            with self.lock:
                self.file_id = secrets.token_hex(12)
                self.contents = contents
                self.media = {"preview": (preview, mime)}
                self.state = dict(
                    status="ready",
                    progress=None,
                    message="",
                    result=None,
                    file=dict(
                        id=self.file_id,
                        name=Path(name).name,
                        bytes=len(contents),
                        mode=mode,
                        metadata=metadata,
                        warnings=warnings,
                    ),
                )
            return self.snapshot()
        except Exception:
            self.clear()
            raise

    def clear(self):
        with self.lock:
            if self.state["status"] == "running":
                raise DemoError("Önce analizi iptal edin.")
            self.media, self.contents, self.file_id = {}, None, None
            self.state = dict(
                status="empty", progress=None, message="", file=None, result=None
            )

    def start(self):
        with self.lock:
            if not self.contents or self.state["status"] in {"running", "uploading"}:
                raise DemoError(
                    "Analiz için bir dosya seçin; aynı anda yalnız bir analiz yapılabilir."
                )
            self.cancelled.clear()
            self.state.update(
                status="running",
                progress=None,
                message="Analiz başlatılıyor…",
                result=None,
            )
            self.media.pop("gradcam", None)
            self.thread = threading.Thread(target=self.analyze, daemon=True)
            self.thread.start()

    def analyze(self):
        began = monotonic()
        mode = self.state["file"]["mode"]
        timeout = 120 if mode == "photo" else 300
        try:
            with (
                analysis_slot(),
                tempfile.TemporaryDirectory(prefix="deepfake-demo-") as temporary,
            ):
                directory = Path(temporary)
                suffix = validate_upload(
                    self.contents, self.state["file"]["name"], mode
                )
                (directory / ("input" + suffix)).write_bytes(self.contents)
                (directory / "request.json").write_text(
                    json.dumps(
                        dict(
                            mode=mode,
                            input_name="input" + suffix,
                            sha256=hashlib.sha256(self.contents).hexdigest(),
                        )
                    ),
                    encoding="utf-8",
                )
                self.run_worker(mode, directory, timeout)
                if self.cancelled.is_set():
                    raise DemoError("Analiz iptal edildi.")
                result = json.loads(
                    (directory / "result.json").read_text(encoding="utf-8")
                )
                score = result.get("fake_score")
                if result.get("mode") != mode or result.get("status") not in {
                    "accepted",
                    "rejected",
                }:
                    raise DemoError("Analiz sonucu doğrulanamadı.")
                if result["status"] == "accepted" and (
                    not isinstance(score, (int, float))
                    or not math.isfinite(score)
                    or not 0 <= score <= 1
                    or result.get("label") not in {"REAL", "FAKE"}
                ):
                    raise DemoError("Model geçerli bir skor üretmedi.")
                result["wall_seconds"] = monotonic() - began
                with self.lock:
                    if mode == "photo":
                        self.media["gradcam"] = (
                            (directory / "gradcam.png").read_bytes(),
                            "image/png",
                        )
                    self.state.update(
                        status="done", progress=1, message="", result=result
                    )
        except Exception as error:
            with (self.session / "analysis_errors.log").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(repr(error) + "\n")
            message = (
                str(error)
                if isinstance(error, DemoError)
                else "Analiz tamamlanamadı. Dosyayı ve model ortamını kontrol edin."
            )
            with self.lock:
                self.state.update(
                    status="ready" if self.cancelled.is_set() else "error",
                    progress=None,
                    message=message,
                    result=None,
                )

    def run_worker(self, mode, directory, timeout):
        executable = PHOTO_PYTHON if mode == "photo" else VIDEO_PYTHON
        if not executable.is_file():
            raise DemoError("Modelin Python ortamı bulunamadı.")
        messages = queue.Queue()
        process = None
        with _worker_job() as job:
            with (self.session / "worker.log").open("ab") as log:
                process = subprocess.Popen(
                    [
                        str(executable),
                        "-B",
                        "-u",
                        "-m",
                        "src.studio_worker_gate",
                        "--mode",
                        mode,
                        "--job-dir",
                        str(directory),
                    ],
                    cwd=ROOT,
                    env=dict(
                        os.environ,
                        PYTHONUTF8="1",
                        PYTHONDONTWRITEBYTECODE="1",
                        NO_ALBUMENTATIONS_UPDATE="1",
                        OMP_NUM_THREADS="4",
                        MKL_NUM_THREADS="4",
                        CUBLAS_WORKSPACE_CONFIG=":4096:8",
                    ),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=log,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                with (self.session / "owned_workers.jsonl").open(
                    "a", encoding="utf-8"
                ) as record:
                    record.write(json.dumps(dict(pid=process.pid, mode=mode)) + "\n")
                try:
                    job.assign(process)
                    process.stdin.write("start\n")
                    process.stdin.close()

                    def read():
                        for line in process.stdout:
                            messages.put(line)
                        messages.put(None)

                    reader = threading.Thread(target=read, daemon=True)
                    reader.start()
                    deadline = monotonic() + timeout
                    failure = None
                    while True:
                        if self.cancelled.is_set():
                            raise DemoError("Analiz iptal edildi.")
                        if monotonic() > deadline:
                            raise DemoError(
                                "Analiz süre sınırını aştı. Daha kısa bir dosya deneyin."
                            )
                        try:
                            line = messages.get(timeout=0.15)
                        except queue.Empty:
                            continue
                        if line is None:
                            break
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("event") == "error":
                            failure = event.get("message")
                        elif event.get("event") == "progress":
                            progress = event.get("progress")
                            if isinstance(progress, (int, float)) and math.isfinite(
                                progress
                            ):
                                with self.lock:
                                    self.state.update(
                                        progress=max(0, min(1, progress)),
                                        message=str(
                                            event.get("message", "Analiz sürüyor…")
                                        ),
                                    )
                    process.wait(timeout=5)
                    if process.returncode:
                        raise DemoError(failure or "Analiz süreci tamamlanamadı.")
                finally:
                    job.close()
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)
                    process.stdout.close()


@contextmanager
def _worker_job():
    job = WindowsJob()
    try:
        yield job
    finally:
        job.close()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send(self, status, raw, mime="application/json", extra=None):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
        )
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def valid_path(self):
        host = f"127.0.0.1:{self.server.server_port}"
        if self.headers.get("Host") != host:
            return None
        parsed = urlsplit(self.path)
        prefix = "/" + self.server.token + "/"
        if not parsed.path.startswith(prefix):
            return None
        return parsed.path[len(prefix) :]

    def do_GET(self):
        route = self.valid_path()
        if route is None:
            return self.send(404, b"{}")
        if route == "state":
            return self.send(
                200,
                json.dumps(self.server.studio.snapshot(), ensure_ascii=False).encode(),
            )
        assets = {
            "": ("index.html", "text/html; charset=utf-8"),
            "app.js": ("app.js", "text/javascript; charset=utf-8"),
            "app.css": ("app.css", "text/css; charset=utf-8"),
        }
        if route in assets:
            name, mime = assets[route]
            return self.send(200, (ASSETS / name).read_bytes(), mime)
        if route.startswith("media/"):
            parts = route.split("/")
            studio = self.server.studio
            with studio.lock:
                if (
                    len(parts) != 3
                    or parts[1] != studio.file_id
                    or parts[2] not in studio.media
                ):
                    return self.send(404, b"{}")
                raw, mime = studio.media[parts[2]]
            start, end = 0, len(raw) - 1
            request = self.headers.get("Range")
            if request:
                import re

                match = re.fullmatch(r"bytes=(\d+)-(\d*)", request)
                if not match:
                    return self.send(
                        416, b"", extra={"Content-Range": f"bytes */{len(raw)}"}
                    )
                start = int(match[1])
                end = min(int(match[2]), end) if match[2] else end
                if start > end:
                    return self.send(
                        416, b"", extra={"Content-Range": f"bytes */{len(raw)}"}
                    )
            headers = {"Accept-Ranges": "bytes"}
            if request:
                headers["Content-Range"] = f"bytes {start}-{end}/{len(raw)}"
            return self.send(
                206 if request else 200, raw[start : end + 1], mime, headers
            )
        self.send(404, b"{}")

    def do_POST(self):
        route = self.valid_path()
        origin = f"http://127.0.0.1:{self.server.server_port}"
        if (
            route is None
            or self.headers.get("X-Studio-Token") != self.server.token
            or self.headers.get("Origin") != origin
        ):
            self.close_connection = True
            return self.send(403, b"{}")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 0 or size > MAX_VIDEO_BYTES:
                self.close_connection = True
                raise DemoError("Dosya en fazla 100 MB olmalı; fotoğraf sınırı 10 MB.")
            if self.headers.get("Transfer-Encoding"):
                self.close_connection = True
                raise DemoError("Dosya aktarım biçimi desteklenmiyor.")
            contents = self.rfile.read(size)
            if len(contents) != size:
                raise DemoError("Dosya aktarımı tamamlanamadı.")
            if route == "file":
                name = parse_qs(urlsplit(self.path).query).get("name", [""])[0]
                result = self.server.studio.upload(contents, name)
            elif route == "analyze":
                self.server.studio.start()
                result = self.server.studio.snapshot()
            elif route == "cancel":
                self.server.studio.cancelled.set()
                result = self.server.studio.snapshot()
            elif route == "clear":
                self.server.studio.clear()
                result = self.server.studio.snapshot()
            else:
                return self.send(404, b"{}")
            self.send(200, json.dumps(result, ensure_ascii=False).encode())
        except (DemoError, ValueError, OSError) as error:
            message = (
                str(error)
                if isinstance(error, DemoError)
                else "Dosya veya işlem tamamlanamadı. Lütfen yeniden deneyin."
            )
            self.send(400, json.dumps(dict(error=message), ensure_ascii=False).encode())


def create_server(session, token, port=0):
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.token = token
    server.studio = Studio(session)
    return server


def record_owned_temps(session):
    lock = threading.Lock()
    temp_root = Path(tempfile.gettempdir()).resolve()

    def audit(event, values):
        if event == "tempfile.mkdtemp":
            path = Path(values[0])
            if path.parent.resolve() == temp_root and path.name.startswith(
                ("deepfake-demo-", "deepfake-preview-")
            ):
                with (
                    lock,
                    (Path(session) / "owned_temps.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle,
                ):
                    handle.write(json.dumps(str(path)) + "\n")

    sys.addaudithook(audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    token = sys.stdin.readline().strip()
    if len(token) != 48 or any(c not in "0123456789abcdef" for c in token):
        raise SystemExit("Ownership gate not opened")
    record_owned_temps(args.session)
    server = create_server(args.session, token)
    (args.session / "server.json").write_text(
        json.dumps(
            dict(url=f"http://127.0.0.1:{server.server_port}/{token}/", pid=os.getpid())
        ),
        encoding="utf-8",
    )
    server.serve_forever()
