"""Open the desktop Studio interface in the default browser."""

from pathlib import Path
import os
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    python = (
        root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    )
    if not python.is_file():
        raise SystemExit("Ana Python ortamı bulunamadı: .venv")
    try:
        completed = subprocess.run(
            [
                str(python),
                "-B",
                "-m",
                "src.studio_browser",
            ],
            cwd=root,
            check=False,
        )
    except KeyboardInterrupt:
        return
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
