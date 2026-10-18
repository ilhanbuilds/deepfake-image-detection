# GitHub'dan yeniden kurulum ve koruma

Bu depo proje kaynaklarını, testleri ve küçük raporları içerir. Dataset,
checkpoint, üretilmiş çıktılar ve Python ortamları GitHub'a yüklenmez.
GitHub push bunların yedeği değildir. Büyük ağırlıklar için LFS kurulmadı;
upstream kod/ağırlıkların yeniden dağıtım izni varsayılmamalıdır.

## Ortam kurulumu

Windows, Python 3.12 ve desteklenen NVIDIA sürücüsü kullanılır. Repository
kökünde yeni `.venv` için README'deki CUDA torch/torchvision kurulumunu
uygulayın; ardından video decoder dahil ana bağımlılıkları kurun:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt
```

Video ortamı ana ortamın paketlerine `project_runtime.pth` ile erişir;
legacy RetinaFace paket metadata'sı eski torch istediğinden video ek paketleri
yalnız ayrı ortamda `--no-deps` ile kurulmalıdır. Çalışan ortamlarda paket
yükseltmesi yapmayın. Yeni bilgisayarda ana ortam hazırlandıktan sonra:

```powershell
.\.venv\Scripts\python.exe -m venv outputs/video_research_v1/env
$runtimePackages = (Resolve-Path .venv/Lib/site-packages).Path
$bridgePath = 'outputs/video_research_v1/env/Lib/site-packages/project_runtime.pth'
[IO.File]::WriteAllText((Join-Path (Get-Location) $bridgePath), $runtimePackages + [Environment]::NewLine)
.\outputs\video_research_v1\env\Scripts\python.exe -m pip install --no-deps -r requirements-video-forensic.txt
```

Bağımlılıklar mevcut sürüm aralıklarını sabitler; tam wheel/OS arşivi değildir.
Tarihsel runtime sürümleri korunan deney config/protocol dosyalarında bulunur.
FFmpeg isteğe bağlıdır; tarayıcı uyumlu önizleme üretir, model orijinali işler.

## Uygulamanın haricen gereken dosyaları

| Göreli yol | Amaç |
|---|---|
| `outputs/checkpoints/best_model_v3_robust.pth` | Fotoğraf V3 ağırlığı |
| `outputs/video_research_v1/finetune12/main_20261014/checkpoints/best_video.pth` | SBI epoch 6 ağırlığı |
| `outputs/video_research_v1/finetune12/main_20261014/config.json` | Model/preprocessing provenance doğrulaması |
| `outputs/video_research_v1/retinaface.pth` | Yüz detector ağırlığı |
| `outputs/video_research_v1/upstream/src_inference_preprocess.py` | Hash doğrulanan resmî SBI crop kodu |
| `outputs/video_frames_v1/full.json` | Yükleyicinin karşılaştırdığı tarihsel preprocessing imzası |
| `.venv/` ve `outputs/video_research_v1/env/` | Çalışan iki Python ortamı; taşımak yerine aynı paketlerle kurun |

V3 SHA-256:
`c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c`

SBI epoch 6 SHA-256:
`a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c`

RetinaFace SHA-256:
`b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2`

Uygulama upload analizi için eğitim datasetini, train crop cache'ini veya
resmî test videolarını kullanmaz. Fakat deneylerin tekrar üretimi için ham
veri, manifestler, auditler, skorlar ve tüm mühürler ayrı yedeklenmelidir.

## Taşınabilirlik sınırı

Korunan eğitim `config.json` dosyasının `implementation_sha256` alanında
`C:\Users\kilic\Desktop\deepfake-image-detection\src\video_forensic_train.py`
mutlak yolu vardır. Mevcut video yükleyicisi bu yolu ve kaynak bayt hash'lerini
aynen doğrular. Bu yüzden video uygulaması başka bir köke taşındığında güvenlik
kontrolü durabilir. Burada config veya kontrol değiştirilmedi. Aynı yerel
klasörde geri yükleme doğrulanmış akıştır; farklı kök/işletim sistemi için
ayrı, denetlenen taşınabilirlik çalışması gerekir. Kaynak satır sonları da
hash'leri etkileyebilir; restore sonrası provenance doğrulaması şarttır.

Upstream dosyalar `scripts/setup_video_forensic.py` içinde revizyon ve hash ile
tanımlıdır; script açık yetkiyle indirir ve paket kurar. Uygulamayı başlatmak
bu script'i, eğitimi veya nihai testi otomatik başlatmaz. Önce izin/lisans
koşullarını ve gerekli yedekleri kontrol edin; hiçbir modeli otomatik paylaşmayın.

## Başlatma ve kontroller

```powershell
.\.venv\Scripts\python.exe scripts/start_deepfake_app.py
```

Sentetik UI/adaptör testleri, model inference başlatmadan:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_unified_demo.py tests/test_demo.py tests/test_video_adaptive.py tests/test_video_holdout.py -q
```

Resmî test bir kez karar değerlendirmesi için kullanılmıştır. Yeniden kurulum
kontrolü olarak resmî test çalıştırıcısını kullanmayın. `.env`, özel anahtarlar
ve `.streamlit/secrets.toml` Git dışında tutulur; gizli bilgileri kaynak veya
belgeye yazmayın.
