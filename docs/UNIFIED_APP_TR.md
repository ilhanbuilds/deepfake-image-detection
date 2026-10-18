# Deepfake Analiz — birleşik sunum uygulaması

Yeni giriş noktası `app_unified.py`; eski `app.py` değiştirilmedi.

Herhangi bir PowerShell klasöründen tek komut:

```powershell
& "C:\Users\kilic\Desktop\deepfake-image-detection\.venv\Scripts\python.exe" "C:\Users\kilic\Desktop\deepfake-image-detection\scripts\start_deepfake_app.py"
```

Adres: **http://127.0.0.1:8502**. Terminali açık tutun; kapatmak için Ctrl+C kullanın. 8502 portunda uygulama zaten açıksa mevcut adresi kullanın. Başlatıcı yalnız yeni süreç için 100 MB Streamlit yükleme sınırı uygular; mevcut `.streamlit/config.toml` dosyasını değiştirmez. Dış ağa açılmaz.

## Kullanım

- **Fotoğraf analizi:** JPG/JPEG/PNG seçin; önizlemeyi kontrol edip “Fotoğrafı analiz et” düğmesine basın. REAL/FAKE sınıfı, 0–1 aralığında FAKE skoru, süre ve Grad-CAM gösterilir. Orijinal V3 decoder, EXIF düzeltmesi, RGB dönüşümü, `image_transform()` ve mevcut Grad-CAM fonksiyonu yeniden kullanılır; tahmin kuralı değiştirilmez.
- **Video analizi:** kısa MP4/M4V/MOV/WebM seçin. Süre/çözünürlük kontrolünden sonra önizleme hazırlanır. “Videoyu analiz et” ile önce tarihsel 6 kare işlenir; `0.4 < p6 < 0.6` ise 12 kareye tamamlanır. Eşik 0,5'tir. Eksik ilk karelerde mevcut güvenlik kuralı genişlemeyi zorlar. En az üç geçerli yüz karesi gerekir; eksik kareye skor uydurulmaz.
- “Oturumu temizle” yüklemeleri, önizlemeleri ve sonuçları arayüzden kaldırır. Sonuçlar kalıcı deney cache'lerine veya raporlarına yazılmaz.

## Model ve ortamlar

- Görüntü: `outputs/checkpoints/best_model_v3_robust.pth`, V3 EfficientNet-B0; ana `.venv` ortamında ayrı süreç.
- Video: SBI EfficientNet-B4, fine-tuned epoch 6; `outputs/video_research_v1/finetune12/main_20261014/checkpoints/best_video.pth`. SHA-256: `a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c`. Mevcut video ortamı `outputs/video_research_v1/env` kullanılır.
- Video adaptörü mevcut `AdaptiveVideo.predict(..., "adaptive")` akışını ve doğrulanmış SBI model yükleyicisini kullanır: RetinaFace, resmî crop, 380×380 RGB/255, FP32, yüz batch 1. Eski `video_predict.py` kullanılmaz.
- Her analiz ayrı bir süreçte model yükler; süreç bitince CUDA kaynakları bırakılır. Uygulama, aynı anda fotoğraf/video GPU analizi başlatmaz; oturumlar arası yerel kilit de vardır. Modeller RAM/VRAM'de süresiz cache'lenmez.
- Paket kurulumu, runtime birleştirme, eğitim, eşik değişikliği veya resmî test tekrarı yapılmaz. Video için CUDA gerekir; CPU'ya sessiz geçiş yoktur. Fotoğraf, eski davranış gibi otomatik CUDA/CPU seçer.

## Dosya kontrolleri ve sınırlar

- Fotoğraf: 10 MB, en fazla 16 megapiksel, gerçek JPEG/PNG içerik doğrulaması.
- Video: 100 MB, 60 saniye, Full HD (1920×1080 veya dikey karşılığı), en fazla 120 FPS / 7.200 kare; en az 6 kare. Uzantı yanında container başlığı ve decoder metadata'sı kontrol edilir.
- Desteklenmeyen codec, bozuk dosya, yüz bulunamaması, eksik kare, CUDA yokluğu ve bellek hataları Türkçe bildirilir. Analiz süreçleri fotoğrafta 120, videoda 300 saniyeyle sınırlandırılır.
- Analiz için yalnız özel bir OS geçici klasörü kullanılır; normal tamamlanma, hata ve timeout sonrasında çocuk süreç durdurulup dosyalar temizlenir. Kullanıcı dosya adları shell komutuna çevrilmez; sabit argv listeleri ve `shell=False` kullanılır.
- Sert sistem kapanışı sırasında OS temp altında bir klasör kalabilir. Bu uygulama şifreli saklama veya adli düzeyde güvenli silme iddiasında bulunmaz.

## Video önizlemesi

Mevcut FFmpeg bulunursa **yalnız önizleme** için H.264/AAC MP4 kopyası üretilir. Böylece Celeb-DF gibi tarayıcının doğrudan açamadığı MPEG-4 codec'leri de oynatılabilir. Önizleme CPU'da, zaman ve çözünürlük limitleri doğrulandıktan sonra oluşturulur; geçici dosyaları temizlenir.

**SBI her zaman yüklenen orijinal baytları analiz eder.** Önizleme sıkıştırması veya ölçeklemesi modele verilmez. FFmpeg bulunamaz/başlatılamazsa açık uyarıyla orijinal önizleme denenir; codec desteği tarayıcıya bağlı kalır. Projedeki mevcut FFmpeg kullanıldı; yeni paket kurulmadı.

## Sonuçları yorumlama

FAKE skoru kalibre edilmemiş model çıktısıdır; “% kesinlik” değildir. V3'ün %99,635 görsel test sonucu ile SBI'ın %97,4903 VIDEO-HOLDOUT sonucu farklı değerlendirme kümelerine aittir ve yeni yüklemelerde doğruluk garantisi oluşturmaz. Video sonucu kişi-bağımsız/cross-dataset genelleme kanıtı değildir. Grad-CAM, manipülasyon konumunu kesin olarak işaretlemez.

Arayüzdeki toplam süre süreç başlangıcı, model yükleme, video ısınması ve sonuç aktarımını da içerir. Video ayrıntılarında gerçek analiz/yükleme/ısınma süreleri ayrıca gösterilir. Her seferinde model yüklemek bir miktar gecikme ekler; 4 GB VRAM üzerinde belleğin güvenli bırakılması için bilinçli tercihtir.

## Kod yapısı

- `app_unified.py`: Türkçe iki sekmeli arayüz, önizleme, ilerleme, sonuç ve oturum durumu.
- `src/demo_media.py`: dosya sınırları, orijinal fotoğraf decoder'ına köprü, video metadata ve uyarılar.
- `src/demo_preview.py`: modele dokunmadan tarayıcı uyumlu video önizlemesi.
- `src/demo_services.py`: süreç adaptörü, oturumlar arası analiz kilidi, timeout ve geçici dosya yaşam döngüsü.
- `scripts/demo_worker.py`: mevcut fotoğraf/Grad-CAM veya SBI adaptif akışını çalıştıran tek işlik süreç.
- `scripts/start_deepfake_app.py`: var olan ana ortam ile yeni uygulamayı başlatır.
- `tests/test_unified_demo.py`: sentetik UI/adaptör, dosya sınırları, hata, süreç, önizleme ve temizlik testleri.

Gerçek kısa smoke kontrolleri yalnız sunum fotoğrafı ve kabul edilmiş validation kaydı `Celeb-real/id45_0001.mp4` ile yapıldı. Bu video resmî 518 test kohortunda değildir. Smoke sonuçları yeni başarı/genelleme ölçümü olarak yorumlanmaz. Denetim çıktıları: `outputs/unified_app_checks_20261018/`.
