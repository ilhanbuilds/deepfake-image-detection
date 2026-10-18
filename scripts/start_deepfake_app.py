"""Launch the separate Turkish UI without changing existing Streamlit settings."""

from pathlib import Path
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    python = root / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise SystemExit("Ana Python ortamı bulunamadı: .venv")
    try:
        completed = subprocess.run(
            [
                str(python),
                "-B",
                "-m",
                "streamlit",
                "run",
                str(root / "app_unified.py"),
                "--server.port",
                "8502",
                "--server.address",
                "127.0.0.1",
                "--server.maxUploadSize",
                "100",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=root,
            check=False,
        )
    except KeyboardInterrupt:
        return
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
