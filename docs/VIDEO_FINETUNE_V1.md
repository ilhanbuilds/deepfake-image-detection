# Video kısmi fine-tuning V1

Bu yol fotoğraf V3 ve frozen video V1/V2 kodlarını/checkpointlerini değiştirmez.
Resmi test veya quarantine loader'ı oluşturmaz. Uzun eğitim başlatılmadı.

Yeni dosyalar: `src/video_finetune.py` (eğitim/resume),
`src/video_finetune_model.py` (son blok/BN koruması),
`src/video_finetune_data.py` (canlı crop tutarlılığı),
`src/video_crop_quality.py` (görsel rapor/CUDA smoke),
`tests/test_video_finetune.py` (23 yeni regresyon testi) ve bu kılavuz.
Tüm otomatik testler: **222 geçti**; Ruff lint/format, compileall ve
git diff whitespace kontrolü geçti. Korunan 3348 eski dosya değişmedi.

## Model ve başlangıç

`src/video_finetune_model.py`: ayrı `PartialVideoClassifier`. ImageNet B0 V1
backbone'u, V2 best epoch 6'daki frozen backbone ile SHA-256 olarak eşleşir.
GRU64/head aynı best checkpoint'ten başlar; yeni deneyin optimizer ve early
stopping durumu sıfırdan başlar. Bu, frozen GRU eğitimini resume etmek değildir.

Yalnız `backbone.features[8]` (son 320→1280 1×1 conv + BN affine + SiLU)
ve GRU/head açılır. Son MBConv stage `features[7]` dahil bütün prefix frozen.
Prefix eval/no_grad altında gerçek görüntüleri işler; son blok no_grad dışında
çalışır. Son bloğun gradienti gerçek yüz görüntülerinin aktivasyonlarından gelir.
Tüm BatchNorm running mean/variance/counters sabittir; son BN affine weight/bias
eğitilir. `train()` çağrısı bu korumayı yeniden uygular.

FP32, batch 1, 160×160, video başına başlangıçta altı eşit aralıklı kare;
chunk 2. GRU packed sequences geçerli 3–6 kareyi işler, padding loss'a veya
hidden state'e katılmaz. Frame başına supervision yok; tek video etiketi ve
tek weighted BCEWithLogits loss. Sonlu feature/logit/loss/gradient kontrolü,
strict norm clipping 1.0, optimizer step öncesinde yapılır.

AdamW: son blok LR **1e-5**, GRU/head LR **3e-4**, weight decay **0.01**.
Seed 42. Batch 1 ölçülen belleğe sığıyor; bu başlangıç için accumulation gerekli
değil. Sadece bu kısmi açma deneyi hazırlanır; ek bloklar otomatik açılmaz.

## Veri ve sınıf ağırlıkları

Eski feature cache salt okunur bütünlük kontrolünde kullanılır; **feature
tensorleri eğitim girdisi değildir**. Aynı Haar, IoU, margin, seçilmiş kareler
ve normalizasyon imzasıyla kabul edilmiş train/validation roster'ı korunur.
Her epoch gerçek videolar yeniden decode edilir; kare seçimi, crop kutuları,
kabul durumu veya failures değişirse `CropDriftError` eğitimi durdurur.
Video sessizce atılmaz; değişmiş kapsam üzerinde eski class weights kullanılmaz.

Kabul edilmiş train 2253: **420 REAL / 1833 FAKE**. N/(2*N_class) ağırlıkları
REAL **2.68214286**, FAKE **0.61456628**. Validation 572: 103/469.
Tarihsel 77 train / 10 validation ret raporu kapsamda ayrı loglanır, loss'a
girmez. Manifestteki 2330/582 ayrımı ve resmi 518 test izolasyonu korunur.

Bilinen filename kaynak aileleri yeniden denetlenir: 420 train / 103 val
aile, çakışma yok. Train REAL 252 Celeb-real / 168 YouTube-real, val 57/46.
En büyük aile train 28, validation 27 kabul edilmiş video içeriyor.
Class weights aile büyüklüğü veya domain dengesini düzeltmez. 58 ortak
kimlik/donor kodu var; kaynak/kimlik metadata'sı yok, **identity-disjoint değildir**.

## Bounded hazırlık kanıtı

`outputs/video_finetune_v1/preparation/82539908dda749a69640de10b2ab7b0f/`:
20 REAL + 20 FAKE accepted train örneği, 5 contact sheet, `report.json`,
`REPORT.md`, `VISUAL_REVIEW.md`, roster/config/preservation raporları.
40/40 video aynı kabul/crop kararını yeniden üretti; 233/240 kare geçerli.
7 eksik kare: 3 face-not-detected, 4 face-track-lost; tam görüntü fallback yok.

Gerçek GTX 1650 CUDA üzerinde 8 FP32 forward/backward/optimizer adımı,
her sınıfta 3/4/5/6 uzunluk (gerçek crop'ların diagnostic prefix'leri): tüm loss
ve gradientler sonlu. Son blok conv/BN affine, GRU ve head değişti; diğer
ağırlıklar ve bütün BN running stats bit düzeyinde aynı. NaN padding logit
farkı 0. Smoke ağırlıkları/checkpointleri kaydedilmedi.

PyTorch peak allocated **53.52 MiB**, reserved **62 MiB**; context/driver ve
diğer GPU uygulamaları hariç. 4 GiB GPU için başlangıç uygundur; tam epoch
peak belleği henüz ölçülmedi. Altı kareli optimizer adımı ortalama 0.0397 s;
validation forward 0.0311 s. Decode+crop REAL 0.1622 s, FAKE 0.2691 s/video.
2253 train + 572 val için bir epoch tahmini **13.5 dakika**, belirsizlik
aralığı yaklaşık **8–22 dakika**; başlangıç cache/source audit ve checkpoint
I/O eklenir. Bu tam epoch ölçümü değil, 40 crop / 8 adım extrapolation'ıdır.

## Güvenli tek epoch benchmark (henüz çalıştırılmadı)

Proje kökünden PowerShell:

```powershell
.\.venv\Scripts\python.exe -u -m src.video_finetune --run-training --accept-filename-families --device cuda --epochs 1 --backbone-lr 1e-5 --head-lr 3e-4 --run-dir "outputs/video_finetune_v1/benchmark_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

Komut tamamlanmış yeni bir epoch oluşturur. Makro-F1, balanced accuracy,
accuracy, REAL/FAKE precision/recall/F1, ROC-AUC, confusion matrix, loss,
coverage, numerics, epoch süresi/ETA ve CUDA peak bellek history/coverage
JSON'larında saklanır. Best seçimi ve patience=3 yalnız validation Macro-F1.
Train metrikleri online, train BCE class-weighted, val BCE unweighted.

Her epoch immutable checkpoint; best/last alias sadece bu yeni run'da.
Tam model, optimizer, disabled scaler, Python/NumPy/CPU/CUDA RNG, loader
generator, early-stopping ve history saklanır. Aynı run-dir/config ile
`--resume auto` tamamlanmış son epoch'tan devam eder; `--epochs` toplam tavandır.
OOM, numerical error veya crop drift tamamlanmamış epoch'u kaydetmez;
`interrupted.json` recovery durumunu bildirir. Early-stopped run yeniden açılmaz.
Yeni deneyi eski V1/V2 run-dir'lerine yazmak ve frozen checkpoint'i partial
resume olarak yüklemek reddedilir.

`experiment_record.json` gelecekte docs/EXPERIMENTS.md'ye aktarılabilir;
mevcut deney sonuçları dosyasına otomatik yazılmaz. Henüz fine-tuning başarımı
ölçülmedi veya doğruluk artışı iddia edilmedi.

Hazırlığı başka bağımsız rapor dizininde tekrar üretmek için:

```powershell
.\.venv\Scripts\python.exe -u -m src.video_finetune --prepare --accept-filename-families --device cuda
```
