"""Windows WebView2 shell; runs Streamlit and models in their existing environments."""

import ctypes
import json
from pathlib import Path
import sys
import threading
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from src.desktop_runtime import DesktopServer, page

    try:
        server = DesktopServer()
    except OSError:
        ctypes.windll.user32.MessageBoxW(
            None,
            "Yerel çalışma klasörü oluşturulamadı. Projenin outputs klasörüne yazma iznini ve boş disk alanını kontrol edin.",
            "Deepfake Analiz",
            0x10,
        )
        return
    window = None
    event_lock = threading.Lock()

    def record(name, **details):
        with (
            event_lock,
            (server.session / "window.jsonl").open("a", encoding="utf-8") as handle,
        ):
            handle.write(
                json.dumps(dict(event=name, **details), ensure_ascii=False) + "\n"
            )

    try:
        import webview

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "Deepfake.Analiz.Desktop"
        )
        webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False
        webview.settings["ALLOW_FILE_URLS"] = False
        window = webview.create_window(
            "Deepfake Analiz — Fotoğraf ve Video",
            html=page(
                "Yerel sunucu başlatılıyor. Model dosyalarınız ve mevcut analiz kuralları korunuyor."
            ),
            width=1180,
            height=820,
            min_size=(780, 600),
            background_color="#F6F9FF",
            text_select=True,
            zoomable=True,
        )
        window.events.shown += lambda: record("shown")
        window.events.loaded += lambda: record(
            "document_loaded", url=window.get_current_url()
        )

        def close():
            try:
                record("closed")
            finally:
                server.stop()

        def run():
            try:
                url = server.start()
                if not url or server.cancelled.is_set():
                    return
                record("server_ready", url=url)
                window.load_url(url)
                while not server.cancelled.wait(0.5):
                    if not server.running():
                        raise RuntimeError(
                            "Yerel analiz sunucusu beklenmedik biçimde kapandı."
                        )
            except Exception as error:
                (server.session / "error.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
                if not server.cancelled.is_set():
                    window.load_html(
                        page(f"{error} Ayrıntı kaydı: {server.session}", error=True)
                    )
                server.stop()

        window.events.closed += close
        webview.start(
            run,
            gui="edgechromium",
            private_mode=True,
            icon=str(ROOT / "assets/desktop/deepfake.ico"),
            debug=False,
        )
    except Exception:
        (server.session / "error.log").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        ctypes.windll.user32.MessageBoxW(
            None,
            "Masaüstü penceresi açılamadı. .desktop-venv ortamı ve Microsoft Edge WebView2 Runtime kurulumunu kontrol edin.\n\n"
            f"Ayrıntı kaydı: {server.session}",
            "Deepfake Analiz",
            0x10,
        )
    finally:
        server.stop()


if __name__ == "__main__":
    main()
