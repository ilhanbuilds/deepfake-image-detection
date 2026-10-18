"""Browser-only H.264 preview; model inference always receives original upload bytes."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile

from src.demo_media import DemoError, probe_video, validate_upload


def prepare_video_preview(contents, name):
    import cv2

    suffix = validate_upload(contents, name, "video")
    with tempfile.TemporaryDirectory(prefix="deepfake-preview-") as temporary:
        directory = Path(temporary)
        source = directory / ("source" + suffix)
        source.write_bytes(contents)
        metadata = probe_video(source, cv2)
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            return (
                contents,
                metadata,
                [
                    "Önizleme dönüştürücüsü bulunamadı. Tarayıcı videoyu açamıyorsa H.264 kodlu MP4 kullanın; analiz orijinal dosyada yapılır."
                ],
            )
        output = directory / "preview.mp4"
        try:
            conversion = subprocess.run(
                [
                    ffmpeg,
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0?",
                    "-vf",
                    "scale=w='min(960,iw)':h=-2",
                    "-c:v",
                    "libx264",
                    "-threads",
                    "2",
                    "-preset",
                    "ultrafast",
                    "-crf",
                    "25",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    "-movflags",
                    "+faststart",
                    str(output),
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=45,
                check=False,
                shell=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        except subprocess.TimeoutExpired as error:
            raise DemoError(
                "Video önizlemesi zamanında hazırlanamadı. Daha kısa veya düşük çözünürlüklü bir video deneyin."
            ) from error
        except OSError:
            return (
                contents,
                metadata,
                [
                    "Önizleme dönüştürücüsü başlatılamadı. Analiz yapılabilir; görüntüleme için H.264 MP4 kullanın."
                ],
            )
        if conversion.returncode != 0 or not output.is_file():
            return (
                contents,
                metadata,
                [
                    "Tarayıcı uyumlu önizleme üretilemedi. Video analizi orijinal dosyada yapılabilir; görüntüleme için H.264 MP4 kullanın."
                ],
            )
        if output.stat().st_size > 100 * 1024 * 1024:
            raise DemoError(
                "Hazırlanan önizleme boyut sınırını aşıyor. Daha kısa bir video seçin."
            )
        return output.read_bytes(), metadata, []
