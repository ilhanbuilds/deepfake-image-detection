"""Sabit V3 checkpoint ile tamamen lokal fotoğraf sınıflandırma demosu."""

import io
import threading
import warnings
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src.explainability import generate_gradcam_from_image, gradcam_images
from src.model import load_checkpoint, resolve_device

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT = PROJECT_ROOT / "outputs/checkpoints/best_model_v3_robust.pth"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 16_000_000


def decode_upload(contents: bytes) -> Image.Image:
    """Uzantıya güvenmeden biçimi ve boyutu doğrular; fotoğrafı diske yazmaz."""
    if not contents or len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Lütfen en fazla 10 MB büyüklüğünde bir fotoğraf seçin.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(contents)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Yalnızca JPG/JPEG/PNG fotoğraflar desteklenir.")
                if image.width * image.height > MAX_IMAGE_PIXELS:
                    raise ValueError("Fotoğraf en fazla 16 megapiksel olmalı.")
                image.load()
                return image.copy()
    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as error:
        raise ValueError(
            "Fotoğraf açılamadı; geçerli bir JPG/JPEG/PNG seçin."
        ) from error


@st.cache_resource(show_spinner=False)
def load_demo_model():
    device = resolve_device("auto")
    model, checkpoint = load_checkpoint(CHECKPOINT, device)
    # Cache tüm oturumlarca paylaşılır; Grad-CAM hook'ları aynı anda çalışmamalı.
    return model, checkpoint, device, threading.RLock()


def main() -> None:
    st.set_page_config(page_title="Deepfake Yüz Görüntüsü Tespiti", page_icon="🔎")
    st.title("Deepfake Yüz Görüntüsü Tespiti")
    st.write(
        "Model, eğitim dağılımına benzer yüz ağırlıklı görüntülerde "
        "deepfake/sentetik yüz sınıflandırması yapar. "
        "Genel amaçlı AI görüntü tespiti değildir."
    )
    st.caption("Tamamen lokal analiz • Fotoğraf diske kaydedilmez • REAL=0, FAKE=1")
    uploaded = st.file_uploader("Bir yüz fotoğrafı seçin", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("Sunum için bir JPG/JPEG/PNG yüz fotoğrafı yükleyin.")
        return
    if not st.button("Fotoğrafı analiz et", type="primary"):
        return
    try:
        original = ImageOps.exif_transpose(decode_upload(uploaded.getvalue())).convert(
            "RGB"
        )
        with st.spinner("Fotoğraf analiz ediliyor…"):
            model, checkpoint, device, lock = load_demo_model()
            with lock:
                result = generate_gradcam_from_image(model, original, device)
            _, overlay = gradcam_images(result)
    except FileNotFoundError as error:
        st.error(f"V3 checkpoint bulunamadı: {error}")
        return
    except (ValueError, RuntimeError, OSError) as error:
        st.error(f"Analiz tamamlanamadı: {error}")
        return
    st.subheader(f"Tahmin: {result.label}")
    st.metric("Güven", f"%{result.confidence_percent:.2f}")
    left, right = st.columns(2)
    left.image(result.original, caption="Orijinal görüntü", width="stretch")
    right.image(overlay, caption="Grad-CAM görüntüsü", width="stretch")
    if not result.heatmap.any():
        st.info("Pozitif uzamsal Grad-CAM sinyali oluşmadı.")
    st.caption(
        "Kırmızı bölgeler tahmin edilen sınıf için daha güçlü Grad-CAM sinyalini "
        "gösterir. Güven, modelin softmax skorudur; doğruluk garantisi değildir."
    )
    st.caption(
        f"Ana model: V3 Robust EfficientNet-B0 • Best epoch: {checkpoint['epoch']}"
    )


if __name__ == "__main__":
    main()
