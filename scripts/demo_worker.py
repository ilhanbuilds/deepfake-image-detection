"""Single private upload job, then exit to release all CUDA allocations."""

import argparse
from contextlib import redirect_stdout
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter

from src.demo_media import (
    DemoError,
    photo_image,
    probe_video,
    validate_upload,
    video_warnings,
)

ROOT = Path(__file__).resolve().parents[1]
PHOTO_CHECKPOINT = ROOT / "outputs/checkpoints/best_model_v3_robust.pth"
PHOTO_SHA = "c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c"
EVENTS = sys.stdout


def emit(message, progress):
    EVENTS.write(
        json.dumps(
            dict(event="progress", message=message, progress=progress),
            ensure_ascii=False,
        )
        + "\n"
    )
    EVENTS.flush()


def request_input(directory, mode):
    directory = directory.resolve()
    if directory.parent != Path(
        tempfile.gettempdir()
    ).resolve() or not directory.name.startswith("deepfake-demo-"):
        raise DemoError("Geçici analiz klasörü geçersiz.")
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if request.get("mode") != mode:
        raise DemoError("Analiz isteği geçersiz.")
    source = directory / request["input_name"]
    if (
        source.is_symlink()
        or source.resolve().parent != directory
        or not source.name.startswith("input.")
    ):
        raise DemoError("Yüklenen dosyanın yolu geçersiz.")
    limit = 10 if mode == "photo" else 100
    if source.stat().st_size > limit * 1024 * 1024:
        raise DemoError("Yüklenen dosya boyut sınırını aşıyor.")
    contents = source.read_bytes()
    validate_upload(contents, source.name, mode)
    if hashlib.sha256(contents).hexdigest() != request["sha256"]:
        raise DemoError("Dosya aktarımı doğrulanamadı.")
    return source, contents


def photo_job(contents, directory):
    from src.explainability import generate_gradcam_from_image, gradcam_images
    from src.model import load_checkpoint, resolve_device
    from src.video_pipeline import file_sha256

    image = photo_image(contents)
    emit("Fotoğraf doğrulandı; V3 modeli yükleniyor…", 0.18)
    if file_sha256(PHOTO_CHECKPOINT) != PHOTO_SHA:
        raise DemoError("Görsel model dosyasının bütünlüğü doğrulanamadı.")
    device = resolve_device("auto")
    model, checkpoint = load_checkpoint(PHOTO_CHECKPOINT, device)
    emit("Fotoğraf ve Grad-CAM açıklaması hazırlanıyor…", 0.5)
    result = generate_gradcam_from_image(model, image, device)
    _, overlay = gradcam_images(result)
    result.original.save(directory / "preview.png")
    overlay.save(directory / "gradcam.png")
    return dict(
        mode="photo",
        status="accepted",
        label=result.label,
        fake_score=result.probabilities["FAKE"],
        probabilities=result.probabilities,
        epoch=checkpoint["epoch"],
        device=device.type,
        heatmap_present=bool(result.heatmap.any()),
        warnings=[],
        checkpoint_sha256=PHOTO_SHA,
    )


def video_job(source):
    import cv2
    import torch
    from scripts.benchmark_video_adaptive import load_engine
    from src.dataset import seed_everything
    from src.video_adaptive import AdaptiveVideo, CHECKPOINT_SHA, MARGIN

    metadata = probe_video(source, cv2)
    emit("Video doğrulandı; SBI modeli ve yüz algılayıcı yükleniyor…", 0.12)
    if not torch.cuda.is_available():
        raise DemoError(
            "Video analizi için CUDA destekli GPU gerekiyor. Mevcut video ortamında CUDA kullanılamıyor."
        )
    if (
        CHECKPOINT_SHA
        != "a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c"
        or MARGIN != 0.10
    ):
        raise DemoError("Sabit video modeli veya adaptif karar kuralı değişmiş.")
    seed_everything(42)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    engine, environment = load_engine()
    calls = 0
    tick = perf_counter()

    def processor(frame, index):
        nonlocal calls
        measured = engine.processor(frame, index)
        calls += 1
        phase = (
            "İlk 6 kare inceleniyor"
            if calls <= 6
            else "Analiz ek karelerle genişletiliyor"
        )
        emit(f"{phase} · {calls} kare işlendi", 0.2 + 0.7 * min(calls, 12) / 12)
        return measured

    emit("İlk 6 kare için adaptif analiz başlıyor…", 0.2)
    result = AdaptiveVideo(engine.cv, processor).predict(
        source, "adaptive", metadata["frame_count"]
    )
    torch.cuda.synchronize()
    inference_seconds = perf_counter() - tick
    return dict(
        mode="video",
        status=result["status"],
        label=("FAKE" if result["prediction"] else "REAL")
        if result["prediction"] is not None
        else None,
        fake_score=result["fake_probability"],
        p6=result["p6"],
        processed_frames=result["processed_frames"],
        valid_frames=result["valid_face_frames"],
        expanded=result["expanded"],
        selected_frames=len(result["selected_indices"]),
        warnings=video_warnings(result),
        rejection_reason=result["rejection_reason"],
        inference_seconds=inference_seconds,
        model_load_seconds=environment["model_and_detector_load_seconds"],
        warmup_seconds=environment["synthetic_warmup_seconds"],
        peak_allocated_mib=result["peak_allocated_mib"],
        metadata=metadata,
        checkpoint_sha256=CHECKPOINT_SHA,
        threshold=0.5,
        margin=0.10,
        device="cuda",
    )


def run(directory, mode):
    began = perf_counter()
    source, contents = request_input(directory, mode)
    emit("Dosya kontrol ediliyor…", 0.05)
    with redirect_stdout(sys.stderr):
        result = (
            photo_job(contents, directory) if mode == "photo" else video_job(source)
        )
    result["analysis_seconds"] = perf_counter() - began
    (directory / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    emit("Analiz tamamlandı.", 1.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("photo", "video"), required=True)
    parser.add_argument("--job-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args.job_dir, args.mode)
    except BaseException as error:
        if isinstance(error, DemoError):
            message = str(error)
        elif "out of memory" in str(error).lower():
            message = "GPU belleği yetersiz. Diğer GPU uygulamalarını kapatıp yeniden deneyin."
        elif isinstance(error, (ValueError, FileNotFoundError)):
            message = "Dosya veya model doğrulaması tamamlanamadı. Geçerli bir dosya ve projenin sabit modellerini kullanın."
        else:
            message = "Analiz tamamlanamadı. Dosya biçimini, CUDA ortamını ve model dosyalarını kontrol edin."
        EVENTS.write(
            json.dumps(dict(event="error", message=message), ensure_ascii=False) + "\n"
        )
        EVENTS.flush()
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
