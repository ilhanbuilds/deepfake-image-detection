# Deepfake Görüntü ve Video Tespiti

Türkçe birleşik uygulama **`app_unified.py`** üzerinden fotoğrafta V3 Robust
EfficientNet-B0 ve videoda SBI EfficientNet-B4 epoch 6 kullanır. Video önce
6 kareyi işler; `|p6 - 0.5| < 0.10` olduğunda 12 kareye genişler. Karar
eşiği 0.5'tir. Eski `app.py` fotoğraf uygulaması ayrıca korunmuştur.

Yerel ortam ve harici model dosyaları hazırken repository kökünden:

```powershell
.\.venv\Scripts\python.exe scripts/start_deepfake_app.py
```

Adres: **http://127.0.0.1:8502**. Fotoğraf önizlemesi/Grad-CAM, video
önizlemesi, ilerleme, kare sayısı ve süre gösterilir. Skorlar kalibre edilmiş
kesinlik yüzdeleri değildir. İki model ayrı süreçlerde, sırayla çalışır.

- [Birleşik uygulama kullanımı ve sınırları](docs/UNIFIED_APP_TR.md)
- [GitHub'dan yeniden kurulum ve harici dosyalar](docs/GITHUB_RESTORE_TR.md)
- [Tamamlanmış VIDEO-HOLDOUT sonuç özeti](docs/VIDEO_HOLDOUT_RESULTS.md)

**Kod deposu dataset veya model yedeği değildir.** V3/SBI/RetinaFace
ağırlıkları, video provenance dosyaları, upstream preprocessing ve ayrı video
ortamı haricen gerekir. Video için NVIDIA CUDA kullanılır. Yeni checkout
yalnız `pip install` ile tam video uygulamasına dönüşmez; ayrıntılı envanter
ve taşınabilirlik sınırlamaları kurulum belgesindedir. Model geliştirme ve
resmî test tamamlanmıştır; eski deney komutları otomatik çalıştırılmamalıdır.

## Görüntü modeli ve tarihsel deneyler

Üniversite Derin Öğrenme dersi için **fotoğraf tabanlı** REAL / FAKE yüz
sınıflandırma projesi. Ana model **V3 Robust EfficientNet-B0**;
`0 = REAL`, `1 = FAKE`. Eğitim ve final değerlendirmeler tamamlandı.
V1/V2 modelleri, eğitim geçmişleri ve sonuçları korunur; yeni eğitim veya
test setiyle tuning yapılmaz.

Model deepfake / sentetik yüz fotoğraflarını sınıflandırır. Fotoğraftaki
her türlü AI düzenlemesini tespit ettiği iddia edilmez. Confidence, softmax
skorudur; kalibre edilmiş doğruluk garantisi değildir.

## Kurulum — Windows / PowerShell

Python 3.12, PyTorch 2.11.0, torchvision 0.26.0; GTX 1650 4 GB için CUDA 12.8
paketleri kullanılır. İlk paket kurulumu internet gerektirir.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

CPU kurulumu için PyTorch index URL'sinde `cu128` yerine `cpu` kullanın.
Ortam kontrolü: `nvidia-smi` ve `.\.venv\Scripts\python.exe -m src.model`.
Demo yerel checkpoint'i yükler; ImageNet ağırlığı indirmez.

## Ayrı video modeli V1

EfficientNet-B0 frozen + zamansal GRU video classifier altyapısı hazırdır.
Fotoğraf V3 ve Streamlit demo korunur. Küçük video smoke/cache ölçümleri
yapıldı; tam preprocessing ve GRU eğitimi başlatılmadı. Train-only ağırlıklı
video loss ve frozen backbone feature cache desteği vardır. Celeb-DF-v2
resmi test ve dosya adı kaynak aileleriyle ayrılan
train/validation manifesti, smoke/cache/eğitim/evaluation/inference komutları
ve kaynak/identity sınırları: [Video V1 kılavuzu](docs/VIDEO_V1.md).
Video bağımlılığı ayrı `requirements-video.txt` içindedir.

## Lokal Streamlit demo

Repository kökünde çalıştırın:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Tarayıcı adresi: `http://127.0.0.1:8501`. JPG/JPEG/PNG yüz fotoğrafını yükleyip
**Fotoğrafı analiz et** düğmesine basın. REAL/FAKE tahmini, confidence,
orijinal fotoğraf ve Grad-CAM overlay gösterilir. En fazla 10 MB / 16 megapiksel
kabul edilir. Fotoğraf bellekte işlenir, diske kaydedilmez. Uygulama yerel
arayüze bağlanır, kullanım telemetrisi kapalıdır; çalışma sırasında internet
gerekmez. CUDA varsa kullanılır, aksi durumda CPU seçilir.

Model, eğitim dağılımına benzer yüz ağırlıklı görüntülerde deepfake/sentetik yüz
sınıflandırması yapar. Genel amaçlı AI görüntü tespiti değildir.

Gerekli checkpoint: `outputs/checkpoints/best_model_v3_robust.pth`.
Dataset demo için gerekli değildir. Checkpoint Git dışında tutulur; yeni bir
checkout'ta yerel dosyayı bu yola ayrıca kopyalayın.

## Ana model ve CLEAN preprocessing

EfficientNet-B0, ImageNet `IMAGENET1K_V1` başlangıcı, iki sınıf çıkışı.
CLEAN preprocessing hem demo hem final testte aynı fonksiyondur:
EXIF yön düzeltme → RGB → bilinear/antialias 224×224 → tensor → ImageNet
normalizasyonu (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`).
Yüklenen görüntünün tamamı bu dönüşümden geçirilir; eğitim dağılımına benzer,
yüz ağırlıklı fotoğraflar hedeflenir.

Tamamlanmış V3 eğitim ayarları: batch 8, AdamW, LR `3e-4`, mixed precision,
seed 42, workers 0, maksimum 6 epoch, early stopping patience 2.
Altı epoch tamamlandı; **best epoch 5** CLEAN validation FAKE F1 ile seçildi.
Son epoch'ta patience sınırına ulaşılmadığı için duruş early stopping değildir.

Rastgele augmentation **yalnızca train** fotoğraflarında uygulandı:

| Augmentation | Olasılık | Ayar |
|---|---:|---|
| JPEG compression | %50 | Quality 50–95 |
| Gaussian blur | %15 | Sigma 0,1–1,5 |
| Gaussian noise | %15 | RGB [0,1], std 0–0,02 |
| Downscale/upscale | %30 | 224→rastgele 112–224→224 |
| Horizontal flip | %50 | Yatay çevirme |

Validation, final test ve demo CLEAN kalır; rastgele augmentation uygulanmaz.

## Final sonuçlar

V3, **140k Real and Fake Faces test setindeki** hazır ayrılmış 20.000
fotoğrafta ölçüldü (10.000 REAL / 10.000 FAKE).
Precision, recall, F1 ve ROC-AUC için pozitif sınıf FAKE'tir.

| Ölçü | V3 |
|---|---:|
| Best epoch | 5 |
| Accuracy | %99,635 |
| Precision FAKE | %99,7893 |
| Recall FAKE | %99,480 |
| F1 FAKE | %99,6344 |
| ROC-AUC | 0,99992647 |
| Doğru / yanlış | 19.927 / 73 |

Confusion matrix (satır gerçek / sütun tahmin; REAL, FAKE):
`[[9979, 21], [52, 9948]]`.
Kaynak: `outputs/metrics/final_test_metrics_v3.json`.

## Robustness: V2 → V3

Aynı test fotoğrafları ve aynı sekiz koşul, sabit checkpoint'lerle ölçüldü.

| Koşul | V2 accuracy % | V3 accuracy % | Artış (yüzde puan) |
|---|---:|---:|---:|
| CLEAN | 99,410 | 99,635 | +0,225 |
| JPEG Q70 | 85,690 | 99,220 | +13,530 |
| JPEG Q50 | 76,340 | 98,525 | +22,185 |
| Blur sigma 1 | 50,955 | 98,885 | +47,930 |
| Blur sigma 2 | 50,005 | 91,030 | +41,025 |
| Noise std 0,01 | 94,895 | 99,430 | +4,535 |
| Noise std 0,03 | 75,455 | 97,850 | +22,395 |
| Resize 224→112→224 | 50,185 | 98,230 | +48,045 |

En büyük accuracy artışı resize koşulundadır. Blur ve resize altında FAKE→REAL
hataları azaldı: sigma 1 **9809→198**, sigma 2 **9999→1723**, resize **9962→319**.
V2'nin yüksek frekanslı görsel ipuçlarına bağımlılığı olası bir açıklamadır;
bu gözlem tek başına nedensel kanıt değildir. Sigma 2 hâlâ en zor koşuldur.
V2 test sonuçları V3 augmentation tasarımına yön verdi; bu karşılaştırma
geliştirme boyunca hiç görülmemiş bağımsız test iddiasıyla sunulmaz.
Test eğitim veya checkpoint seçiminde kullanılmadı.

Tam precision/recall/F1/ROC-AUC/confusion matrix:
`outputs/metrics/robustness_v3.json` ve `robustness_v3.csv`.
Grafik: `outputs/figures/robustness_v2_vs_v3.png`.
V1/V2/V3 eğitim ve tüm deney ayrıntıları [deney günlüğünde](docs/EXPERIMENTS.md).

## Final rapor ve sunum paketi

Kısa akademik özet: [FINAL_SUMMARY.md](docs/FINAL_SUMMARY.md).
`outputs/figures/final/` klasöründe V3 confusion matrix, V2–V3 robustness,
V1–V2–V3 sonuç karşılaştırması, üç eğitim/validation grafiği ve iki REAL +
iki FAKE Grad-CAM örneği bulunur. Grafikler 300 DPI PNG ve SVG; Grad-CAM'ler
önceki karşılaştırma PNG'lerinin kopyalarıdır. `package_manifest.json` kaynak
ve çıktı SHA-256 kayıtlarını içerir. Yeni eğitim/inference yapılmaz.

Mevcut metrik/log/Grad-CAM dosyalarından boş bir final klasörüne üretmek için:

```powershell
.\.venv\Scripts\python.exe scripts/build_final_report_pack.py
```

Script dolu bir final klasörünün üzerine yazmaz. Datasetler/checkpointler ve
üretilmiş görseller Git dışında tutulur; teslimde final klasörünü ayrıca
paylaşın. V1 farklı test datasetinde ölçüldüğünden karşılaştırma ortak bir
benchmark değildir. V2 gözlemlerinin V3 tasarımına etkisi özette açıklanır.

## V3 Grad-CAM

Tahmin edilen sınıfın logiti açıklanır. EfficientNet'in son convolution katmanı
yapıdan otomatik seçilip classification girdisiyle doğrulanır (`features.8.0`).
Gradyanlar açıklama içindir; optimizer adımı ve ağırlık güncellemesi yapılmaz.
Isı haritası yapay yüz bölgesinin ground-truth segmentasyonu değildir.

Tek fotoğraf:

```powershell
.\.venv\Scripts\python.exe scripts/generate_gradcam.py --image "C:\fotograflar\yuz.jpg"
```

Varsayılan V3 checkpoint ve `outputs/figures/gradcam_v3/` kullanılır.
Her çalıştırma yeni klasör oluşturur: `original.png`, `heatmap.png`,
`overlay.png`, `comparison.png`, `prediction.json`.
Eski Grad-CAM sonuçları `outputs/figures/gradcam/` altında korunur.
V1/V2 için açıkça `--checkpoint` ve `--output-dir` verilebilir.

Testten otomatik 5 doğru REAL + 5 doğru FAKE açıklaması:

```powershell
.\.venv\Scripts\python.exe scripts/generate_gradcam_v3.py --device cuda
```

Seed 42 ile her sınıf rastgele sıralanır, ilk beş doğru tahmin seçilir;
confidence'a göre sıralama yapılmaz. Checkpoint hash'i ve test fingerprint'i
final sonuçlarla doğrulanır. Örnekler açıklayıcıdır; genel başarı ölçümü veya
tuning değildir. Her çalıştırmanın seçim kaydı `selection_summary.json` ile
üretilir, önceki çıktılar korunur.

## Dataset ve kod yapısı

- `data/raw/Real`, `Fake`, `dataset.csv`: V1 fotoğraf dataseti, korunur.
- `data/real_vs_fake/{train,valid,test}/{real,fake}`: hazır 100k/20k/20k ayrımları;
  yeniden split edilmez. Bu bilgisayarda mevcut klasör adı `real-vs-fake`;
  loader aynı dataset için bu adı da kabul eder.
- `src/dataset.py`, `model.py`, `train.py`, `evaluate.py`, `predict.py`: V1 pipeline.
- `src/presplit_dataset.py`, `train_v2.py`, `train_v3.py`, `robust_augmentation.py`:
  tamamlanmış V2/V3 eğitimlerinin tekrar üretilebilir kodu, korunur.
- `src/robustness.py`, `evaluate_v3.py`, `experiment_log.py`: değerlendirme/günlük.
- `src/explainability.py`, `scripts/generate_gradcam*.py`: Grad-CAM.
- `app.py`, `.streamlit/config.toml`: lokal demo.
- `tests/`: dataset, eğitim güvenliği, metrik, Grad-CAM ve demo kontrolleri.

ImageFolder'ın alfabetik varsayılanı kullanılmaz: proje genelinde REAL=0,
FAKE=1 zorunludur. Dataset, checkpoint ve üretilen çıktıların tamamı Git dışında
tutulur. V1 `best_model.pth`, V2 `best_model_v2.pth` ve V3
`best_model_v3_robust.pth` silinmez veya değiştirilmez.

## Kontroller

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q app.py src scripts tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

GitHub'a kod, Türkçe dokümantasyon ve testler gönderilir; dataset/checkpoint/
çıktılar ayrıca yerel olarak korunmalıdır. Demo bağımlılıkları kurulduktan
ve V3 checkpoint yerleştirildikten sonra internet olmadan çalışır.
