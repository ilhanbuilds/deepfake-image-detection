"""Fotoğraf deneylerini kronolojik günlüğe atomik ve idempotent ekleme."""

import json
import os
import re
import tempfile
from pathlib import Path


def append_experiment(path: Path, record: dict) -> bool:
    required = {
        "id",
        "name",
        "date",
        "model",
        "checkpoint",
        "dataset",
        "counts",
        "objective",
        "settings",
        "results",
        "duration_seconds",
        "observations",
        "conclusion",
        "files",
        "status",
    }
    if not required.issubset(record) or record["status"] not in {"completed", "failed"}:
        raise ValueError("Deney kaydı eksik veya durum geçersiz.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", record["id"]):
        raise ValueError("Deney id geçersiz.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(descriptor)
    temporary = None
    try:
        content = (
            path.read_text(encoding="utf-8")
            if path.exists()
            else "# Fotoğraf Deney Günlüğü\n"
        )
        marker = f"<!-- experiment:{record['id']} -->"
        if marker in content:
            return False
        fields = {
            "date": "Tarih",
            "status": "Durum",
            "model": "Model",
            "checkpoint": "Checkpoint",
            "dataset": "Dataset",
            "counts": "Train/Val/Test sayıları",
            "objective": "Amaç",
            "settings": "Ayarlar",
            "results": "Sonuç",
            "duration_seconds": "Süre (saniye)",
            "observations": "Önemli gözlem",
            "conclusion": "Sonuç / yorum",
            "files": "İlgili dosyalar",
        }
        entry = f"\n\n{marker}\n## {record['name']}\n\n"
        for key, title in fields.items():
            value = record[key]
            if isinstance(value, (dict, list)):
                entry += f"**{title}:**\n\n```json\n{json.dumps(value, indent=2, ensure_ascii=False)}\n```\n\n"
            else:
                entry += f"**{title}:** {value}\n\n"
        descriptor, temporary = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp", prefix=path.name
        )
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content.rstrip() + entry.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return True
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
        lock.unlink(missing_ok=True)
