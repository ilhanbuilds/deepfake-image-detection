"""Short-lived isolated model workers; no model stays resident in the UI process."""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
from time import monotonic

from src.demo_media import DemoError, validate_upload

ROOT = Path(__file__).resolve().parents[1]
_PYTHON_PATH = "Scripts/python.exe" if os.name == "nt" else "bin/python"
PHOTO_PYTHON = ROOT / ".venv" / _PYTHON_PATH
VIDEO_PYTHON = ROOT / "outputs/video_research_v1/env" / _PYTHON_PATH
_LOCK = threading.Lock()


@dataclass
class Analysis:
    data: dict
    preview: bytes | None = None
    overlay: bytes | None = None


@contextmanager
def analysis_slot():
    if not _LOCK.acquire(blocking=False):
        raise DemoError(
            "Başka bir analiz sürüyor. Tamamlanmasını bekleyip yeniden deneyin."
        )
    handle = None
    locked = False
    try:
        handle = (Path(tempfile.gettempdir()) / "deepfake_unified_gpu.lock").open("a+b")
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise DemoError(
                "Başka bir uygulama oturumu analiz yapıyor. Lütfen bekleyin."
            ) from error
        yield
    finally:
        if handle is not None:
            if locked:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
        _LOCK.release()


def _stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def _run_worker(mode, directory, on_event, timeout):
    if mode == "video" and sys.platform == "darwin":
        raise DemoError(
            "Mac'te mevcut video modeli çalıştırılamıyor; video analizi CUDA destekli NVIDIA GPU gerektiriyor. Fotoğraf analizini kullanabilirsiniz."
        )
    executable = PHOTO_PYTHON if mode == "photo" else VIDEO_PYTHON
    if not executable.is_file():
        raise DemoError(
            "Modelin Python ortamı bulunamadı. Projenin mevcut yerel ortamlarını kontrol edin."
        )
    env = dict(
        os.environ,
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    messages = queue.Queue()
    with (directory / "worker_error.log").open("wb") as errors:
        process = subprocess.Popen(
            [
                str(executable),
                "-B",
                "-u",
                "-m",
                "scripts.demo_worker",
                "--mode",
                mode,
                "--job-dir",
                str(directory),
            ],
            cwd=ROOT,
            env=env,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=errors,
            text=True,
            encoding="utf-8",
            errors="replace",
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

        def read_output():
            try:
                for line in process.stdout:
                    messages.put(line)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        deadline, failure, finished = monotonic() + timeout, None, False
        try:
            while not finished:
                if monotonic() >= deadline:
                    raise DemoError(
                        "Analiz süre sınırını aştı. Daha kısa veya daha düşük çözünürlüklü bir dosya deneyin."
                    )
                try:
                    line = messages.get(timeout=0.2)
                except queue.Empty:
                    continue
                if line is None:
                    finished = True
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("event") == "error":
                    failure = event.get("message", "Analiz tamamlanamadı.")
                elif event.get("event") == "progress" and on_event:
                    on_event(
                        float(event.get("progress", 0)),
                        str(event.get("message", "Analiz sürüyor…")),
                    )
            try:
                process.wait(timeout=max(1, deadline - monotonic()))
            except subprocess.TimeoutExpired as error:
                raise DemoError(
                    "Analiz süreci zamanında tamamlanamadı. Lütfen daha kısa bir dosya deneyin."
                ) from error
            if process.returncode != 0:
                raise DemoError(
                    failure
                    or "Analiz süreci beklenmedik biçimde durdu. GPU belleğini ve dosya biçimini kontrol edin."
                )
        finally:
            _stop(process)
            process.stdout.close()
            reader.join(timeout=2)


def analyze(contents, name, mode, on_event=None, timeout=None):
    began = monotonic()
    suffix = validate_upload(contents, name, mode)
    with (
        analysis_slot(),
        tempfile.TemporaryDirectory(prefix="deepfake-demo-") as temporary,
    ):
        directory = Path(temporary)
        (directory / ("input" + suffix)).write_bytes(contents)
        request = dict(
            mode=mode,
            input_name="input" + suffix,
            sha256=hashlib.sha256(contents).hexdigest(),
        )
        (directory / "request.json").write_text(json.dumps(request), encoding="utf-8")
        _run_worker(
            mode, directory, on_event, timeout or (120 if mode == "photo" else 300)
        )
        path = directory / "result.json"
        if not path.is_file():
            raise DemoError("Analiz sonucu alınamadı. Lütfen yeniden deneyin.")
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("mode") != mode or data.get("status") not in {
            "accepted",
            "rejected",
        }:
            raise DemoError("Analiz sonucu istekle uyuşmuyor.")
        if data["status"] == "accepted":
            score = data.get("fake_score")
            if (
                not isinstance(score, (float, int))
                or not math.isfinite(score)
                or not 0 <= score <= 1
                or data.get("label") not in {"REAL", "FAKE"}
            ):
                raise DemoError("Model geçerli bir sınıflandırma skoru üretmedi.")
        data["wall_seconds"] = monotonic() - began
        return Analysis(
            data,
            (directory / "preview.png").read_bytes() if mode == "photo" else None,
            (directory / "gradcam.png").read_bytes() if mode == "photo" else None,
        )


def analyze_photo(contents, name, on_event=None):
    return analyze(contents, name, "photo", on_event)


def analyze_video(contents, name, on_event=None):
    return analyze(contents, name, "video", on_event)
