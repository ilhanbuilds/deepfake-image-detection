"""New offline desktop UI; old desktop_app.py and both Streamlit apps stay intact."""

import ctypes
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StudioProcess:
    def __init__(self):
        from src.desktop_job import WindowsJob

        self.session = ROOT / "outputs/desktop_studio_sessions" / secrets.token_hex(12)
        self.session.mkdir(parents=True, exist_ok=False)
        self.job = WindowsJob()
        self.process = None
        self.log = None
        self.stopped = threading.Event()
        self.lock = threading.RLock()

    def start(self):
        token = secrets.token_hex(24)
        with self.lock:
            if self.stopped.is_set():
                return None
            self.log = (self.session / "server.log").open("wb")
            self.process = subprocess.Popen(
                [
                    str(ROOT / ".venv/Scripts/python.exe"),
                    "-B",
                    "-u",
                    "-m",
                    "src.studio_server",
                    "--session",
                    str(self.session),
                ],
                cwd=ROOT,
                env=dict(
                    os.environ,
                    PYTHONUTF8="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    NO_ALBUMENTATIONS_UPDATE="1",
                ),
                stdin=subprocess.PIPE,
                stdout=self.log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            try:
                self.job.assign(self.process)
                self.process.stdin.write((token + "\n").encode())
                self.process.stdin.close()
            except Exception:
                self.process.kill()
                raise
        deadline = monotonic() + 35
        while not self.stopped.wait(0.1):
            if self.process.poll() is not None:
                raise RuntimeError("Yerel arayüz sunucusu başlatılamadı.")
            path = self.session / "server.json"
            if path.exists():
                state = json.loads(path.read_text(encoding="utf-8"))
                from urllib.parse import urlsplit
                import urllib.request

                parsed = urlsplit(state["url"])
                if parsed.hostname == "127.0.0.1" and parsed.path == "/" + token + "/":
                    opener = urllib.request.build_opener(
                        urllib.request.ProxyHandler({})
                    )
                    with opener.open(state["url"] + "state", timeout=2) as response:
                        if response.status == 200:
                            return state["url"]
            if monotonic() > deadline:
                raise RuntimeError("Yerel arayüzün başlatılması zaman aşımına uğradı.")
        return None

    def stop(self):
        from src.desktop_runtime import cleanup_temps

        with self.lock:
            self.stopped.set()
            self.job.close()
            if self.process:
                self.process.wait(timeout=8)
            if self.log and not self.log.closed:
                self.log.close()
            cleanup_temps(self.session / "owned_temps.jsonl")


def main():
    import webview

    runtime = StudioProcess()
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
        "Deepfake.Analiz.Studio"
    )
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
    webview.settings["ALLOW_FILE_URLS"] = False
    window = webview.create_window(
        "Deepfake Analiz",
        html='<html lang="tr"><body style="margin:24px;background:#11151c;color:#e9edf5;font:16px Segoe UI">Deepfake Analiz</body></html>',
        width=1180,
        height=760,
        min_size=(640, 460),
        background_color="#11151c",
        text_select=True,
        zoomable=False,
    )
    window.events.closed += runtime.stop

    def run():
        try:
            url = runtime.start()
            if url and not runtime.stopped.is_set():
                window.load_url(url)
        except Exception as error:
            (runtime.session / "startup_error.txt").write_text(
                repr(error), encoding="utf-8"
            )
            if not runtime.stopped.is_set():
                ctypes.windll.user32.MessageBoxW(
                    None,
                    "Arayüz başlatılamadı. Kayıt: " + str(runtime.session),
                    "Deepfake Analiz",
                    0x10,
                )
            runtime.stop()

    try:
        webview.start(
            run,
            gui="edgechromium",
            private_mode=True,
            icon=str(ROOT / "assets/desktop/deepfake.ico"),
            debug=False,
        )
    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
