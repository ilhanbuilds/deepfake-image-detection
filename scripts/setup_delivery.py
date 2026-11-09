"""Install private local runtimes and open the supplied model-backed Studio."""

import argparse
import hashlib
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "outputs/checkpoints/best_model_v3_robust.pth": "c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c",
    "outputs/video_research_v1/finetune12/main_20261014/checkpoints/best_video.pth": "a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c",
    "outputs/video_research_v1/retinaface.pth": "b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2",
    "outputs/video_research_v1/upstream/src_inference_preprocess.py": "cd9d03e4e85fa39fb83e8bf6fe0756ed3a1a92c32fe252918225490cb018341e",
}


def verify_assets(root):
    for relative, expected in ASSETS.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Paket eksik: {relative}. ZIP'in tamamını çıkarın.")
        with path.open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise RuntimeError(f"Model/dosya bütünlüğü hatası: {relative}")


def environment_python(environment):
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def ensure_environment(environment):
    python = environment_python(environment)
    if python.is_file():
        return python
    if environment.exists():
        raise RuntimeError(
            f"Uyumsuz veya eksik ortam: {environment}. ZIP'i yeni bir klasöre çıkarın; başka bilgisayarın ortamını kopyalamayın."
        )
    subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    return python


def probe(python, code, root):
    return (
        subprocess.run(
            [str(python), "-B", "-c", code],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode
        == 0
    )


def install(root, photo_only):
    verify_assets(root)
    python = ensure_environment(root / ".venv")
    runtime_probe = "import streamlit, torch, torchvision, cv2, PIL, sklearn, pandas, matplotlib; from src.studio_server import create_server"
    cuda_probe = "import torch; assert torch.cuda.is_available()"
    needs_cuda = not photo_only and not probe(python, cuda_probe, root)
    if not probe(python, runtime_probe, root) or needs_cuda:
        print(
            "İlk kurulum: bağımlılıklar internetten indiriliyor. Bu işlem uzun sürebilir.",
            flush=True,
        )
        if not photo_only:
            subprocess.run(
                [
                    str(python),
                    "-m",
                    "pip",
                    "install",
                    "--upgrade",
                    "--force-reinstall",
                    "--no-deps",
                    "torch==2.11.0",
                    "torchvision==0.26.0",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                ],
                cwd=root,
                check=True,
            )
        subprocess.run(
            [str(python), "-m", "pip", "install", "-r", "requirements-video.txt"],
            cwd=root,
            check=True,
        )
        if not probe(python, runtime_probe, root):
            raise RuntimeError(
                "Ana Python ortamı doğrulanamadı. Yukarıdaki kurulum hatalarını kontrol edin."
            )
    if not photo_only:
        if not probe(python, cuda_probe, root):
            raise RuntimeError(
                "Video için NVIDIA CUDA GPU ve uygun sürücü gerekiyor. Fotoğraf için Windows Fotograf CPU.cmd dosyasını kullanın."
            )
        video = root / "outputs/video_research_v1/env"
        video_python = ensure_environment(video)
        packages = subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            cwd=root,
            text=True,
        ).strip()
        video_packages = subprocess.check_output(
            [
                str(video_python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            cwd=root,
            text=True,
        ).strip()
        (Path(video_packages) / "project_runtime.pth").write_text(
            packages + "\n", encoding="utf-8"
        )
        video_probe = "import torch, efficientnet_pytorch, albumentations, cv2; from retinaface.predict_single import Model; assert torch.cuda.is_available()"
        if not probe(video_python, video_probe, root):
            subprocess.run(
                [
                    str(video_python),
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "-r",
                    "requirements-video-forensic.txt",
                ],
                cwd=root,
                check=True,
            )
        if not probe(video_python, video_probe, root):
            raise RuntimeError(
                "SBI video ortamı doğrulanamadı. Yukarıdaki kurulum çıktısını kontrol edin."
            )
    return python


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo-only", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    try:
        if sys.version_info[:2] != (3, 12):
            raise RuntimeError(
                "Bu paket Python 3.12 ile hazırlanmıştır. Python 3.12 kurup tekrar başlatın."
            )
        if sys.platform == "darwin" and platform.machine().lower() not in {
            "arm64",
            "aarch64",
        }:
            raise RuntimeError(
                "Mac başlatıcısı Apple Silicon ve yerel ARM64 Python 3.12 gerektirir. Rosetta/x86 Python yerine ARM64 Python kurun."
            )
        photo_only = args.photo_only or sys.platform == "darwin"
        if os.name != "nt" and not photo_only:
            raise RuntimeError(
                "Bu video kurulum başlatıcısı Windows içindir. Fotoğraf için --photo-only kullanın."
            )
        python = install(ROOT, photo_only)
        if args.setup_only:
            print("Kurulum ve model kontrolleri tamamlandı.", flush=True)
            return
        print(
            "Studio açılıyor. Bu pencereyi açık tutun; durdurmak için Control+C kullanın.",
            flush=True,
        )
        completed = subprocess.run(
            [str(python), "-B", "-m", "src.studio_browser", "--port", "0"],
            cwd=ROOT,
            env=dict(os.environ, DEEPFAKE_PHOTO_ONLY="1" if photo_only else "0"),
            check=False,
        )
        raise SystemExit(completed.returncode)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Başlatılamadı: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1) from error
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
