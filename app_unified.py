"""Separate presentation app; original app.py and all experiment assets stay intact."""

import hashlib

import streamlit as st

from src.demo_media import DemoError, photo_image, validate_upload
from src.demo_preview import prepare_video_preview
from src.demo_services import analyze_photo, analyze_video

STYLE = """
<style>
html, body, [data-testid="stApp"] {font-family: 'Segoe UI', sans-serif;}
[data-testid="stAppViewContainer"] {background: radial-gradient(ellipse at 90% 0%, #e9f2ff 0%, transparent 42%), #f7f9fc;}
.block-container {max-width: 1140px; padding-top: 2.4rem; padding-bottom: 3rem;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stFileUploaderDropzone"] button[kind="secondary"] > div {display: none;}
[data-testid="stFileUploaderDropzone"] button[kind="secondary"]::after {content: "Dosya seç"; font-size: .9rem;}
[data-testid="stFileUploaderDropzoneInstructions"] {display: none;}
.brand {font-size: .78rem; font-weight: 800; color: #2563eb; letter-spacing: .17em; margin-bottom: 1.1rem;}
.hero-title {color: #14213d; font-size: clamp(2rem, 4vw, 3.3rem); font-weight: 800; line-height: 1.14; letter-spacing: -.045em; margin: 0 0 .9rem;}
.hero-copy {color: #61718b; font-size: 1rem; line-height: 1.8; max-width: 730px; margin-bottom: 1.5rem;}
.tags {display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: 1.6rem;}
.tag {padding: .38rem .75rem; background: #edf3ff; color: #345c9d; border: 1px solid #dce8fc; border-radius: 30px; font-size: .76rem; font-weight: 600;}
[data-testid="stTabs"] button[role="tab"] {font-size: 1rem; padding: .8rem 1.4rem;}
[data-testid="stTabs"] [role="tablist"] {gap: 1rem; margin-bottom: 1rem;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 18px;}
[data-testid="stFileUploader"] {border-radius: 14px;}
[data-testid="stMetric"] {background: #f4f7fc; border: 1px solid #e6edf6; border-radius: 12px; padding: .85rem 1rem;}
[data-testid="stMetricValue"] {font-size: 1.65rem; color: #192d51;}
[data-testid="stButton"] button[kind="primary"] {border-radius: 10px; min-height: 3rem; background: #2563eb; border-color: #2563eb; font-weight: 700;}
.verdict {padding: 1.1rem 1.25rem; border-radius: 14px; margin: .5rem 0 1rem; border: 1px solid;}
.real {background: #eefaf5; color: #156b4b; border-color: #ccebdd;}
.fake {background: #fff2f0; color: #a33b35; border-color: #f3d5d1;}
.verdict small {display: block; opacity: .8; font-size: .75rem; margin-bottom: .25rem;}
.verdict strong {font-size: 1.4rem;}
.section-kicker {color: #2563eb; font-size: .75rem; font-weight: 800; letter-spacing: .1em; margin-top: .4rem;}
.footer {border-top: 1px solid #e0e7f0; padding-top: 1.2rem; margin-top: 2rem; color: #7d8ba2; font-size: .78rem; line-height: 1.8;}
@media (max-width: 650px) {.block-container {padding: 1.3rem 1rem;} .hero-title {font-size: 2.1rem;} [data-testid="stMetricValue"] {font-size: 1.25rem;}}
</style>
"""


def result_panel(analysis, mode):
    data = analysis.data
    for warning in data.get("warnings", []):
        st.warning(warning)
    if data["status"] != "accepted":
        st.info("Bu dosyada güvenilir bir yüz örneklemi oluşmadı. Tahmin üretilmedi.")
        return
    label = data["label"]
    title = "REAL · Gerçek sınıfı" if label == "REAL" else "FAKE · Sahte sınıfı"
    st.markdown(
        f'<div class="verdict {"real" if label == "REAL" else "fake"}"><small>MODELİN SINIFLANDIRMASI</small><strong>{title}</strong></div>',
        unsafe_allow_html=True,
    )
    if mode == "photo":
        a, b = st.columns(2)
        a.metric("FAKE skoru", f"{data['fake_score']:.4f}")
        b.metric("Analiz süresi", f"{data['wall_seconds']:.1f} sn")
        st.image(
            analysis.overlay,
            caption="Grad-CAM · Tahmin edilen sınıfa katkı veren bölgeler",
            width="stretch",
        )
        if not data.get("heatmap_present"):
            st.info("Bu görüntüde pozitif uzamsal Grad-CAM sinyali oluşmadı.")
        st.caption(
            "Grad-CAM bir açıklama haritasıdır; manipülasyonun konumunu kesin olarak göstermez."
        )
    else:
        a, b = st.columns(2)
        a.metric("FAKE skoru", f"{data['fake_score']:.4f}")
        b.metric("İşlenen kare", str(data["processed_frames"]))
        st.metric("Toplam analiz süresi", f"{data['wall_seconds']:.1f} sn")
        route = (
            "Ek karelere genişletildi" if data["expanded"] else "6 karede tamamlandı"
        )
        st.success(route, icon="✅")
        st.caption(
            f"Yüz bulunan kare: {data['valid_frames']} / {data['selected_frames']} · Karar eşiği: 0,5"
        )
        with st.expander("Analizin ayrıntıları"):
            st.write(
                f"İlk 6 karenin FAKE skoru: {data['p6']:.4f}"
                if data.get("p6") is not None
                else "İlk aşamada geçerli yüz skoru üretilemedi."
            )
            st.write(
                f"Video analizi: {data['inference_seconds']:.2f} sn · Model yükleme: {data['model_load_seconds']:.2f} sn · Isınma: {data['warmup_seconds']:.2f} sn"
            )
            st.caption(
                "Toplam süre, ayrı analiz sürecinin açılışını ve sonuç aktarımını da içerir. Ek karelerde aynı kare tekrar analiz edilmez."
            )
    st.caption(
        "Skor 0–1 aralığında model çıktısıdır; kalibre edilmiş kesinlik veya doğruluk olasılığı değildir."
    )


def upload_panel(mode):
    photo = mode == "photo"
    title = "Fotoğraf" if photo else "Video"
    key = mode + "_analysis"
    contents, start, valid = None, False, False
    left, right = st.columns([1, 1.08], gap="large")
    with left, st.container(border=True):
        st.markdown(
            f'<div class="section-kicker">01 · {"FOTOĞRAF" if photo else "VİDEO"} SEÇİN</div>',
            unsafe_allow_html=True,
        )
        st.subheader("İçeriğinizi yükleyin")
        types = ["jpg", "jpeg", "png"] if photo else ["mp4", "m4v", "mov", "webm"]
        uploaded = st.file_uploader(
            f"{title} dosyası",
            type=types,
            max_upload_size=10 if photo else 100,
            key=f"{mode}_upload_{st.session_state.get('upload_version', 0)}",
            help="Dosyalar yalnız bu bilgisayarda işlenir; analiz için kullanılan geçici kopyalar silinir.",
        )
        st.caption(
            "JPG / PNG · En fazla 10 MB · 16 megapiksel"
            if photo
            else "MP4 / MOV / WebM · En fazla 100 MB · 60 saniye · Full HD"
        )
        if uploaded is None:
            st.info(
                "Başlamak için bir yüz fotoğrafı seçin."
                if photo
                else "Başlamak için yüz içeren kısa bir video seçin."
            )
            st.session_state.pop(key, None)
        else:
            contents = uploaded.getvalue()
            fingerprint = hashlib.sha256(contents).hexdigest()
            if st.session_state.get(mode + "_fingerprint") != fingerprint:
                st.session_state.pop(key, None)
                st.session_state.pop(mode + "_preview", None)
                st.session_state[mode + "_fingerprint"] = fingerprint
            try:
                validate_upload(contents, uploaded.name, mode)
                if photo:
                    st.image(
                        photo_image(contents),
                        caption="Yüklenen fotoğraf",
                        width="stretch",
                    )
                else:
                    if mode + "_preview" not in st.session_state:
                        with st.spinner(
                            "Video önizlemesi ve dosya bilgileri hazırlanıyor…"
                        ):
                            st.session_state[mode + "_preview"] = prepare_video_preview(
                                contents, uploaded.name
                            )
                    preview, metadata, preview_warnings = st.session_state[
                        mode + "_preview"
                    ]
                    st.video(
                        preview,
                        format="video/mp4" if preview[4:8] == b"ftyp" else "video/webm",
                    )
                    st.caption(
                        f"{metadata['duration_seconds']:.1f} saniye · {metadata['width']}×{metadata['height']} · Analiz orijinal dosyada yapılır."
                    )
                    for message in preview_warnings:
                        st.warning(message)
                valid = True
            except (ValueError, OSError) as error:
                st.error(str(error))
        start = st.button(
            f"{title}ı analiz et" if photo else "Videoyu analiz et",
            key=mode + "_start",
            type="primary",
            width="stretch",
            disabled=not valid,
        )
    with right, st.container(border=True):
        st.markdown(
            '<div class="section-kicker">02 · ANALİZ SONUCU</div>',
            unsafe_allow_html=True,
        )
        st.subheader("İçeriğe daha yakından bakın")
        if start:
            st.session_state.pop(key, None)
            progress = st.progress(0.0, text="Analiz hazırlanıyor…")

            def update(value, message):
                progress.progress(min(1.0, max(0.0, value)), text=message)

            try:
                with st.spinner(
                    "Fotoğraf inceleniyor…"
                    if photo
                    else "Yüzler ve adaptif kareler inceleniyor…"
                ):
                    service = analyze_photo if photo else analyze_video
                    st.session_state[key] = service(contents, uploaded.name, update)
                progress.empty()
            except (DemoError, ValueError, OSError) as error:
                progress.empty()
                st.error(str(error))
        if key in st.session_state:
            result_panel(st.session_state[key], mode)
        else:
            st.write(
                "Dosyanızı seçip analiz düğmesine bastığınızda sonuç burada görünecek."
            )
            st.caption(
                "Sonuç; model tahmini, FAKE skoru ve analiz bilgileriyle birlikte sunulur."
            )


def main():
    st.set_page_config(
        page_title="Deepfake Analiz · Görüntü & Video", page_icon="🔎", layout="wide"
    )
    st.markdown(STYLE, unsafe_allow_html=True)
    st.markdown(
        '<div class="brand">◈ DEEPFAKE ANALİZ LABORATUVARI</div><h1 class="hero-title">Dijital içeriğe<br>ikinci bir bakış.</h1><div class="hero-copy">Yüz fotoğraflarını ve kısa videoları iki uzman modelle inceleyin. Açıklanabilir görüntü analizi ve ihtiyaca göre ek kare kullanan video tespiti, tek bir yerel uygulamada.</div><div class="tags"><span class="tag">Tamamen yerel işlem</span><span class="tag">Görüntü · V3 Robust</span><span class="tag">Video · Adaptif SBI</span></div>',
        unsafe_allow_html=True,
    )
    photo, video = st.tabs(["Fotoğraf analizi", "Video analizi"])
    with photo:
        upload_panel("photo")
    with video:
        upload_panel("video")
    with st.expander("Modeller ve sonuçların sınırları"):
        a, b = st.columns(2)
        a.markdown(
            "**Görüntü · EfficientNet-B0 V3 Robust**\n\nGörsel test doğruluğu: **%99,635**. Yüz ağırlıklı görüntüler için eğitildi; genel amaçlı yapay zekâ görsel dedektörü değildir."
        )
        b.markdown(
            "**Video · SBI EfficientNet-B4, epoch 6**\n\nVIDEO-HOLDOUT doğruluğu: **%97,4903**. Önce 6 kare; ilk skor 0,4–0,6 arasındaysa 12 kareye tamamlanır. Eksik ilk kareler de genişlemeyi tetikler."
        )
        st.write(
            "Bu başarılar farklı değerlendirme kümelerine aittir; yüklediğiniz içerikte doğruluk garantisi vermez. Video değerlendirmesi kişi-bağımsız veya farklı veri kümesine genelleme kanıtı değildir. Görüntü işleme, yüz görünürlüğü ve sıkıştırma sonucu etkileyebilir."
        )
        st.caption(
            "Dosyalar harici bir servise gönderilmez. Analiz sırasında geçici kopya kullanılır ve işlem bitince silinir. Önizleme ve sonuçlar oturum belleğinde kalır; 'Oturumu temizle' ile kaldırabilirsiniz. Ani sistem kapanışında geçici bir klasör kalabilir."
        )
    if st.button("Oturumu temizle", key="clear_session"):
        for key in list(st.session_state):
            if key.endswith(("_analysis", "_fingerprint", "_preview")):
                del st.session_state[key]
        st.session_state["upload_version"] = (
            st.session_state.get("upload_version", 0) + 1
        )
        st.rerun()
    st.markdown(
        '<div class="footer">Bitirme projesi · Yerel araştırma prototipi<br>REAL / FAKE bir model sınıflandırmasıdır. Kritik kararlar için tek başına kullanılmamalıdır.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
