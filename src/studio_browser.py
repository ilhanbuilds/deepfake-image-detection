"""Run the same Studio UI and analysis server in a local browser."""

import argparse
import json
from pathlib import Path
import secrets
import webbrowser

from src.studio_server import create_server

ROOT = Path(__file__).resolve().parents[1]


def run(port=8502, open_browser=True):
    session = ROOT / "outputs/studio_browser_sessions" / secrets.token_hex(12)
    session.mkdir(parents=True, exist_ok=False)
    token = secrets.token_hex(24)
    try:
        server = create_server(session, token, port)
    except OSError as error:
        raise SystemExit(
            f"Yerel Studio sunucusu başlatılamadı (port {port}). Önceki web oturumunu kapatın veya --port 0 ile boş port kullanın."
        ) from error
    try:
        url = f"http://127.0.0.1:{server.server_port}/{token}/"
        (session / "server.json").write_text(
            json.dumps(dict(url=url)), encoding="utf-8"
        )
        print(f"Deepfake Analiz Studio: {url}", flush=True)
        print("Durdurmak için bu Terminal'de Control+C kullanın.", flush=True)
        if open_browser:
            try:
                opened = webbrowser.open(url)
            except (OSError, webbrowser.Error):
                opened = False
            if not opened:
                print("Tarayıcı açılamadı; yukarıdaki adresi elle açın.", flush=True)
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            pass
    finally:
        server.studio.cancelled.set()
        server.server_close()
        if server.studio.thread is not None:
            server.studio.thread.join(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.port <= 65535:
        parser.error("Port 0–65535 arasında olmalı.")
    run(args.port, not args.no_browser)


if __name__ == "__main__":
    main()
