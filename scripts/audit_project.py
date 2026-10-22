"""Read-only inventory and protected hashes; never follows directory links."""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/project_cleanup_20261022"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def category(path):
    parts = path.parts
    if "__pycache__" in parts or parts[0] in {".pytest_cache", ".ruff_cache"}:
        return "C"
    if parts[0] in {"data", "outputs"}:
        return "B"
    if parts[0] in {"src", "scripts", "assets", "tests", ".venv", ".desktop-venv"}:
        return "A"
    return "D"


def inventory(phase):
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / f"{phase}_inventory.json").exists():
        raise FileExistsError(
            "Existing audit snapshot is immutable; use a new audit directory."
        )
    folders, files, links, total, count = {}, [], [], 0, 0
    with gzip.open(OUT / f"{phase}_files.jsonl.gz", "wt", encoding="utf-8") as stream:
        pending = [ROOT]
        while pending:
            folder = pending.pop()
            with os.scandir(folder) as entries:
                for item in entries:
                    path = Path(item.path)
                    rel = path.relative_to(ROOT)
                    stat = item.stat(follow_symlinks=False)
                    if getattr(stat, "st_file_attributes", 0) & 0x400:
                        links.append(str(rel))
                        continue
                    if item.is_dir(follow_symlinks=False):
                        pending.append(path)
                    elif item.is_file(follow_symlinks=False):
                        row = dict(
                            path=rel.as_posix(),
                            bytes=stat.st_size,
                            category=category(rel),
                        )
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total += stat.st_size
                        count += 1
                        key = rel.parts[0] if len(rel.parts) > 1 else "[root files]"
                        folders[key] = folders.get(key, 0) + stat.st_size
                        files.append(row)
    result = dict(
        logical_bytes=total,
        file_count=count,
        folders=folders,
        largest_50=sorted(files, key=lambda r: r["bytes"], reverse=True)[:50],
        skipped_reparse_points=links,
        measure="Logical file sizes, includes .git and audit outputs; not allocated clusters",
    )
    (OUT / f"{phase}_inventory.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    protected = {}
    for row in files:
        path = ROOT / row["path"]
        rel = Path(row["path"])
        if rel.parts[0] == "data" or rel.parts[:2] == (
            "outputs",
            "project_cleanup_20261022",
        ):
            continue
        if (
            rel.parts[0] == "outputs"
            and "env" not in rel.parts
            and (
                path.suffix in {".pth", ".pt", ".ckpt"}
                or (
                    path.suffix in {".json", ".csv", ".jsonl", ".md", ".py"}
                    and path.stat().st_size < 10_000_000
                )
            )
        ):
            with path.open("rb") as source:
                protected[row["path"]] = hashlib.file_digest(
                    source, "sha256"
                ).hexdigest()
        elif rel.parts[0] in {"src", "assets"} or row["path"] in {
            "app.py",
            "app_unified.py",
            "scripts/demo_worker.py",
            "scripts/studio_app.py",
        }:
            if "__pycache__" not in rel.parts:
                with path.open("rb") as source:
                    protected[row["path"]] = hashlib.file_digest(
                        source, "sha256"
                    ).hexdigest()
    (OUT / f"{phase}_protected.json").write_text(
        json.dumps(protected, indent=2), encoding="utf-8"
    )
    if phase == "before":
        states = {
            key: git(*args)
            for key, args in {
                "status": ("status", "--porcelain=v1", "-uall"),
                "tracked": ("ls-files",),
                "untracked": ("ls-files", "--others", "--exclude-standard"),
                "ignored": (
                    "ls-files",
                    "--others",
                    "--ignored",
                    "--exclude-standard",
                    "--directory",
                ),
                "tracked_ignored": ("ls-files", "-ci", "--exclude-standard"),
            }.items()
        }
        (OUT / "git_before.json").write_text(
            json.dumps(states, indent=2), encoding="utf-8"
        )
        candidates = [
            ROOT / line
            for line in git("ls-files", "-co", "--exclude-standard", "-z").split("\0")
            if line
        ]
        candidates += list((ROOT / "docs").glob("*.pdf"))
        with zipfile.ZipFile(
            OUT / "source_before.zip", "w", zipfile.ZIP_DEFLATED
        ) as backup:
            for path in sorted(set(candidates)):
                if path.is_file():
                    backup.write(path, path.relative_to(ROOT))
    print(
        json.dumps(
            dict(
                phase=phase,
                bytes=total,
                files=count,
                folders=folders,
                protected=len(protected),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["before", "after"])
    inventory(parser.parse_args().phase)
