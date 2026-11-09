# Geliştirici kılavuzu

Hazır uygulama için [ana sayfadaki kurulumu](../README.md) izleyin. Bu belge kod incelemesi, manuel ortam kurulumu ve doğrulama içindir.

## Uygulama akışı

`src/studio_browser.py` Studio sunucusunu tarayıcıda; `scripts/studio_app.py` aynı sunucuyu Windows masaüstü penceresinde açar. İki giriş noktası da `assets/studio/` dosyalarını ve `src/studio_server.py` analiz akışını kullanır.

Sunucu yüklemeyi doğrular, önizlemeyi hazırlar ve ayrı model sürecini başlatır. `scripts/demo_worker.py` V3/Grad-CAM veya adaptif SBI analizini yapar. İş bitince model süreci kapanır; GPU kilidi eşzamanlı model işlerini önler. Oturuma özel yerel URL ve Origin/token kontrolü kullanılır.

| Girdi | Sınır |
|---|---|
| Fotoğraf | JPG/JPEG/PNG, 10 MiB, 16 megapiksel |
| Video | MP4/M4V/MOV/WebM, 100 MiB, 60 saniye, Full HD; en az 6 kare |
| Analiz zaman aşımı | Fotoğraf 120 saniye, video 300 saniye |

FFmpeg varsa önizleme için tarayıcı uyumlu kopya üretilir; model orijinal videoyu analiz eder. Dosyalar geçici klasörlerde işlenir. Normal bitişte temizlenir; ani sistem kapanışı artık bırakabilir.

## Kaynak koddan tarayıcıyı açma

Ana ortam ve model dosyaları hazırsa proje kökünde:

```powershell
.\.venv\Scripts\python.exe -B scripts/start_deepfake_app.py
```

Başlatıcı varsayılan olarak 8502 portunu kullanır. Boş port seçmek için:

```powershell
.\.venv\Scripts\python.exe -B -m src.studio_browser --port 0
```

## Windows masaüstü penceresi

Masaüstü ortamı ve WebView2 hazırsa:

```powershell
.\.desktop-venv\Scripts\pythonw.exe -B scripts/studio_app.py
```

`Deepfake Analiz.vbs` aynı pencereyi açar. `scripts/create_desktop_shortcut.ps1`, mevcut kısayolu ezmeden bu bilgisayara uygun Windows kısayolu oluşturur.

## Eski arayüzler ve deney kodları

`app.py`, `app_unified.py` ve eski masaüstü yardımcıları kaynak geçmişinin parçasıdır. `app.py` görüntü decoder'ı güncel uygulamada da kullanılır; eski servisler test ve süreç yönetimi bağımlılıkları nedeniyle korunur. Güncel giriş noktaları ana sayfada belirtilmiştir. `src/train*.py` ve `src/video_*.py` eğitim/değerlendirme çalışmalarını içerir; uygulamayı başlatmak eğitim çalıştırmaz.

## Manuel kurulum ve geri yükleme

Python3.12, Windows, NVIDIA sürücüsü, PyTorch2.11.0/torchvision0.26.0 CUDA12.8 ve GTX1650 4GB kullanılan ortamdır. Aşağıdaki komutlar **yeni kurulum** içindir; çalışan ortamları otomatik yükseltmeyin.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt
.\.venv\Scripts\python.exe -m venv .desktop-venv
.\.desktop-venv\Scripts\python.exe -m pip install -r requirements-desktop.txt
```

Yalnız fotoğraf CPU kurulumu için PyTorch URL'sindeki `cu128` yerine `cpu` seçilebilir; video CUDA ister. WebView2 Runtime ve pywebview'in Windows/.NET gereksinimleri ayrıca sağlanmalıdır; uygulama bunları otomatik indirmez. FFmpeg isteğe bağlı önizleme bağımlılığıdır.

SBI ortamı ana ortamın paketlerini `.pth` köprüsüyle görür. Eski RetinaFace metadata pinleri yeni Torch ile uyuşmadığından yalnız ayrı ortamda `--no-deps` kullanılmıştır:

```powershell
.\.venv\Scripts\python.exe -m venv outputs/video_research_v1/env
$runtimePackages = (Resolve-Path .venv/Lib/site-packages).Path
$bridgePath = 'outputs/video_research_v1/env/Lib/site-packages/project_runtime.pth'
[IO.File]::WriteAllText((Join-Path (Get-Location) $bridgePath), $runtimePackages + [Environment]::NewLine)
.\outputs\video_research_v1\env\Scripts\python.exe -m pip install --no-deps -r requirements-video-forensic.txt
```

`requirements.txt` ana ML/Streamlit/test araçları, `requirements-video.txt` OpenCV ekini, `requirements-video-forensic.txt` izole SBI eklerini, `requirements-desktop.txt` yalnız kabuğu tanımlar. Aralıklar tam wheel/OS kilidi değildir. SBI ortamında `pip check` eski RetinaFace torch/torchvision/albumentations pinleri nedeniyle hata verebilir; bu bilinen metadata uyumsuzluğu sessizce “geçti” sayılmaz. Çalışma uyumluluğu mevcut gerçek inference/backward kayıtlarıyla gösterilmiştir; eski runtime ile bit düzeyinde eşdeğerlik iddiası değildir.

### Git dışında gereken varlıklar

| Yol | Temin / geri yükleme |
|---|---|
| `outputs/checkpoints/best_model_v3_robust.pth` | Projenin özgün yerel/yetkili yedeği; genel ImageNet ağırlığı yerine konamaz |
| `outputs/video_research_v1/finetune12/main_20261014/checkpoints/best_video.pth` | Yerel fine-tune epoch6 yedeği; upstream FF-raw aynı model değildir |
| `outputs/video_research_v1/finetune12/main_20261014/config.json` | Aynı run'dan provenance/config; yeniden yazıp kontrolleri atlamayın |
| `outputs/video_research_v1/retinaface.pth` | Özgün doğrulanmış yedek veya izinli upstream kaynağı |
| `outputs/video_research_v1/upstream/src_inference_preprocess.py` | Aynı sabitlenmiş SBI revizyonu ve hash; üçüncü taraf kodu yeniden dağıtılmaz |
| `outputs/video_frames_v1/full.json` | Yerel bilimsel yedek; yükleyici tarihsel preprocessing imzasını karşılaştırır |
| `data/`, manifestler, cache'ler, mühürler ve skorlar | Eğitim/deney yeniden üretimi için ayrı bilimsel yedek; kullanıcı yüklemesini analiz etmek için dataset gerekmez |

| Varlık | Sabit SHA-256 |
|---|---|
| V3 Robust | `c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c` |
| SBI epoch6 | `a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c` |
| RetinaFace | `b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2` |
| SBI FF-raw başlangıcı | `e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2` |

Klon sonrası gerekli varlıklar yoksa analiz çalışmaz. Kullanılan V3, SBI ve RetinaFace dosyaları modeller dahil indirme paketindedir. Tarihsel başlangıç checkpointleri eğitim arşivine aittir. Upstream kaynak/revizyon/hash listesi `scripts/setup_video_forensic.py` içindedir; script kurulum/indirmeyi yalnız açık bayrakla yapar. Günlük başlatma eğitim, veri indirme veya resmî test başlatmaz.

**Taşınabilirlik doğrulaması:** Eğitim config'inin `implementation_sha256` alanındaki tarihsel Windows yolları, `src/runtime_provenance.py` tarafından mevcut proje kökündeki dokuz bilinen kaynak kimliğine eşlenir. Özgün SHA-256 değerlerinin tamamı doğrulanır; config, model ve preprocessing değiştirilmez. İlgili kaynakların satır sonları `.gitattributes` ile korunur. Farklı klasörde gerçek fotoğraf ve CUDA video smoke kontrolü aynı skorları üretti. Bu, gerçek Mac veya farklı donanımda kabul testi değildir. Git yedeği büyük bilimsel dosyaların fiziksel bağımsız yedeği değildir.

## Doğrulama

Eğitim veya resmi final testi çalıştırmayan dar kontroller:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_delivery_runtime.py tests/test_browser_launcher.py tests/test_studio.py tests/test_unified_demo.py tests/test_desktop.py tests/test_desktop_windows.py tests/test_video_adaptive.py tests/test_explainability.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src scripts tests app.py app_unified.py
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests app.py app_unified.py
node --check assets/studio/app.js
git diff --check
```

Bu dilimde 124 test geçti. Farklı yerel kökte fotoğraf, CPU Grad-CAM ve CUDA video analizi doğrulandı; GPU skorları önceki demo kaydıyla aynıydı. Gerçek Mac, farklı donanım ve internetten sıfır kurulum henüz doğrulanmadı.

Tüm tarihsel testleri topluca çalıştırmak yerel deney arşivi gerektirebilir veya büyük fixture dosyaları oluşturabilir. Resmi 20K fotoğraf/518 video final testi kurulum kontrolü değildir. Proje Python ve statik JavaScript kullanır; TypeScript veya ayrı bir frontend build adımı yoktur.
