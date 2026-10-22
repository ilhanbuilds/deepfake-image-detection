"""Gated Streamlit child: no server/worker starts before Windows Job assignment."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import threading


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    if sys.stdin.readline().strip() != args.base:
        raise SystemExit("Masaüstü oturumu doğrulanamadı.")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    lock = threading.Lock()
    temp_root = Path(tempfile.gettempdir()).resolve()

    def audit(event, values):
        if event != "tempfile.mkdtemp":
            return
        path = Path(values[0])
        if path.parent.resolve() == temp_root and path.name.startswith(
            ("deepfake-demo-", "deepfake-preview-")
        ):
            with (
                lock,
                (args.session / "owned_temps.jsonl").open(
                    "a", encoding="utf-8"
                ) as handle,
            ):
                handle.write(json.dumps(str(path)) + "\n")

    sys.addaudithook(audit)
    from streamlit.web import cli

    sys.argv = [
        "streamlit",
        "run",
        str(root / "app_unified.py"),
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(args.port),
        "--server.baseUrlPath",
        args.base,
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--server.maxUploadSize",
        "100",
        "--browser.gatherUsageStats",
        "false",
    ]
    cli.main()


if __name__ == "__main__":
    main()
