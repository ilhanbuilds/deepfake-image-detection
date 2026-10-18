"""Upload limits and bounded video metadata checks for the separate demo."""

import math
from pathlib import Path

MAX_PHOTO_BYTES = 10 * 1024 * 1024
MAX_VIDEO_BYTES = 100 * 1024 * 1024
MAX_VIDEO_SECONDS = 60
MAX_VIDEO_FRAMES = 7200
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm"}


class DemoError(ValueError):
    """A Turkish message safe to display to the person using the app."""


def photo_image(contents):
    # Reuse the exact original decoder and EXIF/RGB behavior; never load its model.
    from app import decode_upload
    from PIL import ImageOps

    return ImageOps.exif_transpose(decode_upload(contents)).convert("RGB")


def validate_upload(contents, name, mode):
    if mode == "photo":
        if Path(name).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            raise DemoError("Fotoğraf için JPG, JPEG veya PNG seçin.")
        if not contents or len(contents) > MAX_PHOTO_BYTES:
            raise DemoError("Fotoğraf boş olmamalı ve en fazla 10 MB olmalı.")
        return ".png" if contents.startswith(b"\x89PNG\r\n\x1a\n") else ".jpg"
    if mode != "video":
        raise DemoError("Analiz türü geçersiz.")
    suffix = Path(name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS:
        raise DemoError("Video için MP4, M4V, MOV veya WebM seçin.")
    if not contents or len(contents) > MAX_VIDEO_BYTES:
        raise DemoError("Video boş olmamalı ve en fazla 100 MB olmalı.")
    valid_header = (
        contents[:4] == b"\x1aE\xdf\xa3"
        if suffix == ".webm"
        else contents[4:8] == b"ftyp"
    )
    if not valid_header:
        raise DemoError(
            "Dosya içeriği seçilen video biçimiyle uyuşmuyor. Standart bir MP4 veya WebM kullanın."
        )
    return suffix


def probe_video(path, cv):
    cap = cv.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise DemoError(
                "Video açılamadı. Dosya bozuk olabilir veya codec desteklenmiyor olabilir."
            )
        values = [
            cap.get(key)
            for key in (
                cv.CAP_PROP_FRAME_COUNT,
                cv.CAP_PROP_FPS,
                cv.CAP_PROP_FRAME_WIDTH,
                cv.CAP_PROP_FRAME_HEIGHT,
            )
        ]
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise DemoError("Videonun süre veya çözünürlük bilgisi okunamadı.")
        count, fps, width, height = values
        duration = count / fps
        if duration > MAX_VIDEO_SECONDS or count > MAX_VIDEO_FRAMES or fps > 120:
            raise DemoError("Video en fazla 60 saniye, 120 FPS ve 7.200 kare olmalı.")
        if width * height > 1920 * 1080 or max(width, height) > 1920:
            raise DemoError(
                "Video çözünürlüğü en fazla Full HD olmalı (1920×1080 veya dikey karşılığı)."
            )
        if count < 6:
            raise DemoError("Analiz için en az 6 farklı video karesi gerekiyor.")
        return dict(
            frame_count=int(count),
            fps=fps,
            duration_seconds=duration,
            width=int(width),
            height=int(height),
        )
    finally:
        cap.release()


def video_warnings(result):
    messages = []
    if result.get("budget_shortfall", 0):
        messages.append(
            "Videoda 12 benzersiz kare yok; yalnız mevcut kareler kullanıldı."
        )
    if result.get("failures"):
        messages.append(
            f"{len(result['failures'])} karede yüz veya görüntü okunamadı. Eksik karelere skor atanmadı."
        )
    if result.get("expansion_reason") == "incomplete_initial_six":
        messages.append(
            "İlk altı karede eksik görüntü/yüz olduğu için ek kareler incelendi."
        )
    if result.get("status") != "accepted":
        messages.append(
            "En az üç geçerli yüz karesi elde edilemedi; REAL/FAKE tahmini üretilmedi."
        )
    return messages
