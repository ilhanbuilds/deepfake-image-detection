# Video Modeli V1

Fotoğraf V3, eski checkpoint/sonuçlar, Grad-CAM ve Streamlit demo korunur.
Video modülleri ayrı giriş noktalarıdır. ResNet18 çalışması yoktur.
Kullanıcı train/validation feature cache işlemlerini tamamladı; aşağıdaki
denetim bu cache dosyalarını doğrular. Kullanıcının GRU V1 FP32 eğitimi
6. epoch'ta early stopping ile tamamlandı; best checkpoint epoch 3'tür.
GRU V2 eğitimi başlatılmadı. Sentetik otomatik testler gerçek eğitim sonucu değildir.

## Dataset ve split

12 Ekim 2026 inventory: Celeb-real 590, YouTube-real 300, Celeb-synthesis
5.639; toplam 6.529 MP4. Resmi liste formatı yerelde incelendi:
`<0|1> <klasör/video.mp4>` (tek boşluk). **1 REAL / 0 FAKE**, modeldeki
**0 REAL / 1 FAKE** sırasına dönüştürülür. 518 satırda eksik dosya, tekrar,
klasör-etiket uyuşmazlığı yok. Klasör anlamları
[dataset yazarlarının deposuyla](https://github.com/yuezunli/celeb-deepfakeforensics) uyumludur.

| Ayrım | REAL | FAKE | Toplam |
|---|---:|---:|---:|
| Train | 438 | 1.892 | 2.330 |
| Validation | 110 | 472 | 582 |
| Resmi test | 178 | 340 | 518 |
| Test ailesi karantinası | 164 | 2.935 | 3.099 |

`outputs/video_v1/manifest.json`: resmi test önce ayrılır.
`Celeb-real/idN_KKKK` ve `Celeb-synthesis/idN_idM_KKKK` aynı dosya adı
kaynak ailesidir. Testte bu aileden video varsa kalan üyeler karantinaya
alınır; resmi test genişletilmez. Kalan aileler seed 42 ile yaklaşık
%80/%20 train/val bölünür. 128 deterministik grup adayından sınıf adetleri
hedef orana en yakın olan seçilir; model/test skoru kullanılmaz.
Frame random split yok. Kaynak-aile çakışması 0.

**Sınır:** First-id + clip-index eşlemesi dosya adı çıkarımıdır; yerel kaynak,
donor eğitim videosu ve identity metadata yok. YouTube kaynak ilişkileri
doğrulanmadı. Identity-disjoint değildir: train–val 58, train–test 56,
val–test 55 ortak identity kodu vardır (donorlar dahil). Tüm videolarda
decode veya içerik/kopya denetimi yapılmadı. Dosya adı varsayımı yanlışsa
aile garantisi geçersiz olur; doğrulanmış kaynak metadata gerekir. Eğitim
bu sınır için ayrıca `--accept-filename-families` ister.

Manifest; resmi liste SHA-256, dosya boyut/mtime, sınıf/aile ve splitleri
tutar. Her komut değişiklikte devam etmeyi reddeder. Cache ayrıca gerçek
video içeriği, backbone ağırlıkları ve preprocessing imzasını SHA-256 ile
doğrular; değişmeyen kaynak her epoch yerine process başına bir kez hashlenir.

## Model ve ortak pipeline

6 farklı eşit aralıklı frame (ilk/son dahil), 160×160; 192 isteğe bağlı.
OpenCV sequential decoding → CPU frontal Haar detector → %20 margin crop →
RGB/PIL bilinear-antialias resize → ImageNet normalization. Haar detector
derin öğrenme değildir; video classifier derin öğrenmedir. Tek ek bağımlılık
OpenCV, ayrı `requirements-video.txt` içinde tutulur.

İlk yüz en büyük kutudur; sonraki örneklemelerde IoU ≥0,1 ile takip edilir.
Çok kişili video için doğru kişi garantisi veya landmark identity tracking
yoktur. Büyük hareket takip reddine neden olabilir. Tespit/takip/decode
hatalarında başka yüz, tam frame veya kopya frame kullanılmaz.
En az 3 geçerli yüz gereklidir. Geçerli frameler kronolojik sıralanır;
packed GRU padding'i yok sayar. Eksik frameler zaman aralığını değiştirebilir;
V1 ayrı timestamp özelliği kullanmaz. Yetersiz yüz varsa video reddedilir,
inference skor üretmez. Metrikler kabul edilen videolara aittir; reddedilen
videolar ve frame hataları kapsam raporunda ayrıca tutulur.

ImageNet `IMAGENET1K_V1` EfficientNet-B0 → 1.280 frame özelliği →
GRU hidden 128 → video başına tek binary logit. Video etiketiyle
`BCEWithLogitsLoss`; frame loss/supervision yok. Backbone ağırlıkları,
BatchNorm istatistikleri ve stochastic depth donuktur. Yalnız GRU/head,
AdamW LR 3e-4, gradient norm 1. Otomatik fine-tuning/ikinci aşama yoktur.

Kabul edilmiş train sınıfları %18,64 REAL / %81,36 FAKE; validation
%18,01 REAL / %81,99 FAKE.
Training BCE her **video** için `N_train / (2 * N_train_class)` ağırlığıyla
hesaplanır: REAL **2,682143**, FAKE **0,614566**. Train toplamında iki sınıfın
ağırlık katkıları eşittir. Validation/test bu ağırlıkları belirlemez ve
balanced sampling yapılmaz. Splitler, validation loss/metrik hesabı ve
threshold 0,5 korunur. Ağırlıklar checkpoint config'e kaydedilir; önceki
config ile resume uyumsuzluğu açıkça reddedilir. Ağırlık hesabı yalnız cache
denetiminden geçen **420 REAL + 1.833 FAKE** train videosuna dayanır.
77 train ve 10 validation reddi loss/metrik hesabına girmez; kapsam raporuna
ayrıca eklenir. Manifest ve cache dosyaları değiştirilmez.

CUDA FP32 varsayılan (`--precision fp32`), AMP isteğe bağlı (`--precision amp`);
batch 1–2, workers 0, seed 42; yalnız validation macro-F1 ile seçim,
patience 3 early stopping, her epoch checkpoint ve tam optimizer/scaler/RNG/
loader-generator resume. Epoch süresi/ETA loglanır. OOM'de örnek sessizce
atlanmaz; tamamlanmamış epoch bırakılır, son tam epoch korunur.
Resume aynı ayarları ister; batch/device/cache değişiminde yeni run kullanın.
Accuracy, balanced accuracy, macro-F1, REAL/FAKE precision/recall/F1,
ROC-AUC ve REAL/FAKE confusion matrix video seviyesindedir; threshold 0,5.
Validation'da iki sınıf kalmazsa durur. Train metrikleri online batch
tahminleridir; validation metrikleri epoch sonundaki sabit modelden gelir.
Resmi test train loader'a, model seçimine veya tuning'e girmez.

## Tamamlanmış cache ve GRU hazırlık denetimi

12 Ekim 2026: 2.912 cache dosyasının tamamı kayıt, pipeline/backbone imzası,
video içeriği SHA-256, feature şekli/sonluluğu ve kronolojik frame raporuyla
doğrulandı. Eksik/fazla cache ve train/validation video/kaynak-aile çakışması
yok. Resmi test videoları okunmadı/decode edilmedi. Manifestteki kimlik
ayrımı sınırı devam eder.

| Ayrım | Kabul REAL | Kabul FAKE | Kabul | Ret | Geçerli kare: 3 / 4 / 5 / 6 |
|---|---:|---:|---:|---:|---|
| Train | 420 | 1.833 | 2.253 | 77 | 33 / 31 / 199 / 1.990 |
| Validation | 103 | 469 | 572 | 10 | 11 / 8 / 40 / 513 |

`src.video_cache --check-cache` salt okunur train/validation denetimidir.
Eğitim giriş noktası da optimizer adımlarından önce bu denetimi zorunlu
yapar. Cache parmak izi checkpoint config'e girer; cache değişimiyle resume
reddedilir. 3–6 uzunluk birlikte padded batch'te packed GRU kullanır;
padding tahmini ve gradient'i etkilemez. Testler padding'i NaN ile de sınar.

İlk denetim döngüsü **19,426 s** sürdü (Python import/model hazırlığı hariç).
Batch 2 için bir epoch 1.127 train + 286 validation batch içerir.
GTX 1650 için hazırlık ve checkpoint yazımı dahil **1–3 dakika** ilk ön
tahmindi. Tamamlanan V1 run'da epoch döngüleri **15,34–17,02 saniye**,
6 epoch toplamı **98,72 saniye** ölçüldü; hazırlık ve checkpoint yazma
süresi bu epoch sürelerinin dışında kalır.

```powershell
.\.venv\Scripts\python.exe -m src.video_train --run-training --accept-filename-families --cache-dir outputs/video_v1/cache --run-dir outputs/video_v1/gru_benchmark_fp32_1epoch --epochs 1 --batch-size 2 --device cuda --precision fp32
```

Run dizinindeki `history.json` tüm train/validation metriklerini, epoch
süresini/ETA ve CUDA allocator peak allocated/reserved belleğini tutar.
`cache_audit.json`, her epoch kapsam raporu, immutable epoch checkpointleri
ve `best_video.pth`/`last_video.pth` ayrıca yazılır. Tamamlanınca
`experiment_record.json` mevcut `src.experiment_log.append_experiment`
şemasında oluşur: best validation sonuçları, sınıf adetleri/ağırlıkları,
tüm history, epoch süreleri ve çağrı hazırlık/toplam süresi içerir.
Bu kayıt daha sonra `docs/EXPERIMENTS.md` içerisine idempotent eklenebilir;
eğitim kodu bu dokümanı otomatik değiştirmez. Tamamlanmış V1 analizinin
skorları aşağıdaki bölümde ve ayrı analiz raporunda tutulur.

### Sayısal kararlılık düzeltmesi

GTX 1650 üzerindeki gerçek cache ile REAL-only batch'te eski AMP loss scale
65.536 iken finite logits/loss'a rağmen backward gradientleri Inf oldu.
Batch 1'de GRU/head, batch 2'de head gradientleri taştı. `unscale_` zaten
clipping'den önceydi; FP16 backward'da oluşmuş Inf değerini geri düzeltemez.
Sınıf ağırlığı REAL 2,682143 doğru hesaplanmıştır; taşmayı yüksek loss scale
ile azınlık sınıfına ait gradientin büyüklüğü birlikte tetikler.

Frozen feature-cache üzerinde FP32 GRU varsayılandır; AMP için başlangıç
scale 1.024'tür. BCE her iki modda FP32 hesaplanır. Ortak update yordamı
feature/logits/loss ve unscale sonrası tüm gradientleri denetler; sonrasında
`clip_grad_norm_(..., error_if_nonfinite=True)` ve optimizer adımı uygulanır.
Geçersiz gradientte optimizer çağrılmaz; gradientler temizlenir ve eğitim
durur. Run içinde `numerical_error.json` video yolları, uzunluklar, precision,
loss scale ve geçersiz parametre adlarını kaydeder; tamamlanmamış epoch için
checkpoint üretilmez. `history.json` maksimum gradient normunu da tutar.
Önceki precision/config ile resume reddedilir; eski çıktıları koruyarak yeni
run-dir seçin.

`src.video_cuda_smoke --smoke --precision fp32` yalnız train cache'ten
3–6 kareli 8 video / 4 batch ile gerçek forward/backward/optimizer kontrolü
yapar. Epoch/checkpoint ve validation/test değerlendirmesi yoktur; sonuçlar
ayrı `outputs/video_v1/cuda_smoke/<uuid>/report.json` içinde saklanır.

## PowerShell

Repository kökünde küçük **train'den 2 REAL + 2 FAKE** smoke (24 frame hedefi):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-video.txt
.\.venv\Scripts\python.exe -m src.video_preprocess --smoke --manifest outputs/video_v1/manifest.json --per-class 2 --image-size 160 --device cuda
```

Bu komut eğitim yapmaz. ImageNet ağırlığı torch hub cache'inde yoksa ilk
kullanımda indirilir. Yeni `outputs/video_v1/smoke/<uuid>/` klasörüne crop
PNG, features ve `report.json` (hatalar, süre, peak CUDA bellek) yazılır.
Rastgele GRU başarımı smoke ölçümü sayılmaz. CPU için `--device cpu`.

### Küçük preprocessing/cache ölçümü: 20 REAL + 20 FAKE

Mevcut train içinden seed 42 ile 40 farklı video seçer. Frozen ImageNet
backbone, aynı 6-frame yüz pipeline ve FP32 feature cache kullanılır.
Resmi test ve validation videoları bu komutla decode edilmez. Her tekrar
aynı örneklemi seçer; uyumlu mevcut cache dosyaları değiştirilmeden doğrulanır.

```powershell
.\.venv\Scripts\python.exe -m src.video_preprocess --sample-cache --per-class 20 --image-size 160 --cache-dir outputs/video_v1/cache --device cuda
```

12 Ekim 2026 gerçek GTX 1650 ölçümü: **40/40** kabul, **234/240** geçerli
yüz karesi, **6** frame hatası (5 videoda). Video reddi veya OOM yok.
Decode/crop/feature/hash/cache-write döngüsü **9,3879 s**, hazırlık **5,9068 s**
(Python import süresi hariç). 40 yeni cache yazıldı; cache hit yoktu.
PyTorch CUDA peak allocated **29,6411 MiB**, reserved **38 MiB**. Bunlar
allocator ölçümleridir; CUDA context/driver/diğer process belleği dahil değil.
Model başarımı veya GRU eğitim ölçümü değildir.

Ortalama süre REAL **0,21429 s/video**, FAKE **0,25303 s/video**.
`438 * REAL_ortalama + 1892 * FAKE_ortalama + hazırlık` ile 2.330 train
videosu için tahmin **578,49 s ≈ 9,64 dakika**. Küçük örneklem tahminidir;
video uzunlukları, disk ve sistem yükü nedeniyle değişebilir. Cache-hit
süreleri tahmine katılmaz; tamamı warm cache olan tekrarda yeni süre tahmini
üretilmez. Tam cache komutu bu 40 kaydı tekrar kullanır. Tam preprocessing
ve GRU eğitimi bu hazırlık aşamasında başlatılmadı.

Kanıt: `outputs/video_v1/cache_reports/a0fbc022251443ad8d581f41f4b4c1fd/report.json`.

Inventory tekrarında mevcut manifest ezilmez; yeni ad verin:

```powershell
.\.venv\Scripts\python.exe -m src.video_manifest --output outputs/video_v1/manifest_reaudit.json
```

Tam cache komutları kullanıcı tarafından çalıştırıldı; gerçek GRU eğitim
ve resume giriş noktaları:

```powershell
.\.venv\Scripts\python.exe -m src.video_preprocess --cache-split train --device cuda
.\.venv\Scripts\python.exe -m src.video_preprocess --cache-split val --device cuda
.\.venv\Scripts\python.exe -m src.video_train --run-training --accept-filename-families --cache-dir outputs/video_v1/cache --run-dir outputs/video_v1/training --batch-size 2 --device cuda
.\.venv\Scripts\python.exe -m src.video_train --run-training --accept-filename-families --cache-dir outputs/video_v1/cache --run-dir outputs/video_v1/training --batch-size 2 --device cuda --resume auto
```

Cache FP32, frame chunk 2; tamamlanan uyumlu kayıtlar tekrar kullanılır.
Eğitimle aynı çözünürlük gerekir. GRU eğitimi mevcut frozen feature cache'i
zorunlu kullanır; eğitim sırasında decode/feature extraction yapılmaz.
Dolu run dizininde yeni eğitim reddedilir; eski fotoğraf klasörünü kullanmayın.
Validation ile deney bitince sabit best checkpoint kullanılır:

```powershell
.\.venv\Scripts\python.exe -m src.video_evaluate --final-test --checkpoint outputs/video_v1/training/checkpoints/best_video.pth --device cuda
.\.venv\Scripts\python.exe -m src.video_predict --checkpoint outputs/video_v1/training/checkpoints/best_video.pth --video 'C:\videolar\ornek.mp4' --device cuda
```

Test sonucuna göre threshold/ayar/model seçimi yapılmamalıdır. Evaluation
ayrı UUID dizininde JSON ve confusion matrix üretir. Fotoğraf checkpoint'i
video inference/evaluation için kabul edilmez.

## Bellek ve doğrulama sınırı

6×160² RGB float32 video girdisi ~1,76 MiB; 6×1.280 feature ~30 KiB/video.
Backbone ~4 milyon donuk parametre (~16 MiB), GRU/head ~542 bin eğitilebilir
parametre (~2,1 MiB). Gradient/Adam, CUDA context, workspace ve allocator
rezervleri ayrıca gerekir. Chunk 2 ve batch 1–2 için **1–2 GiB VRAM bütçesi**,
cache ile GRU eğitimi için **0,5–1 GiB** tahmin edilir. 192² backbone activation
alanı %44 artar. Bunlar ilk aşama bütçe tahminleridir. Yukarıdaki küçük cache
testi gerçek PyTorch allocator peak değerlerini verir; tam GRU eğitiminin
belleği V1 run'da ölçüldü: peak allocated **50,40 MiB**, reserved **56 MiB**
(PyTorch allocator; CUDA context/driver ve başka GPU uygulamaları dahil değil).

Otomatik testler sentetik inventory/AVI, crop geometrisi, no-face reddi,
frozen BN/ağırlıklar, temporal order/padding, video loss, cache güvenliği,
metrik kapsamı, epoch resume ve OOM'u sınar. Gerçek dataset smoke ve model
başarımı ayrıca ölçülmelidir. Fotoğraf demo ve Grad-CAM kodu değiştirilmedi.

## Tamamlanan GRU V1 ve mean-pooling baseline analizi

`outputs/video_v1/gru_v1_fp32`: 6 epoch'un tüm history/coverage/epoch
checkpointleri doğrulandı. Best dosyası `epoch_003.pth` ile SHA-256 olarak
aynı; last epoch 6. Model seçimi yalnız validation macro-F1.

| Model | Train Macro-F1 (sabit model) | Val Macro-F1 | Val accuracy | Val ROC-AUC | REAL recall | FAKE recall |
|---|---:|---:|---:|---:|---:|---:|
| GRU best epoch 3 | 0,7942 | 0,6707 | 0,7990 | 0,7660 | 0,4854 | 0,8678 |
| Temporal mean + Logistic Regression | 0,9964 | 0,5351 | 0,5892 | 0,7198 | 0,6893 | 0,5672 |

`src/video_baseline.py` yalnız geçerli 3–6 kareyi ortalar; padding dahil
edilmez. StandardScaler ve class weights sadece kabul edilmiş train'den
gelir. Tek C=1 / lbfgs / seed 42 fit, 149 iterasyon, **0,322 s**; validation
parametre/threshold tuning yok. Eğitim/validation videoları GRU ile aynıdır.
Baseline katsayıları ve scaler `.npz`, tahminler JSON olarak ayrı analiz
dizininde saklanır. Yeni dependency yok.

`src/video_analysis.py` best GRU train/val metriklerini aynı sabit checkpoint
ile tekrar hesaplar. Epoch train logları online, train BCE class-weighted,
validation BCE unweighted: mutlak loss seviyeleri doğrudan eşdeğer değildir.
Grafiklerde bu fark açıkça belirtilir. 4–6. epoch validation gerilemesi ve
baseline'ın büyük train/val açığı genelleme/kapasite sorunuyla uyumludur.

Train 420 kaynak ailesi (FAKE 221 aile), validation 103 aile (FAKE 53 aile).
REAL Celeb-real/YouTube-real sayıları train 252/168, validation 57/46.
GRU REAL recall klasör bazında %33,33 / %67,39. Büyük aileler çok sayıda
manipülasyon içerir; sınıf ağırlıkları aile büyüklüğünü eşitlemez. Aile
çakışması yok; 58 identity/donor kodu ortaktır ve gerçek kimlik izolasyonu
doğrulanmamıştır. Domain etkisi olasılığı vardır; nedensellik kanıtlanmadı.

Gerçek cache üzerinde 3–6 uzunluk ve NaN padding mask kontrolü geçti,
maksimum logit farkı **2,38e-7**. Aynı 513 adet 6-kareli validation videosunda
3/4/5/6 prefix için GRU Macro-F1 **0,6575 / 0,6594 / 0,6575 / 0,6718**.
Prefix deneyi zaman kapsamını da değiştirir; doğal 3/4-kare grupları yalnız
11/8 video olduğundan geniş çıkarım yapılamaz. Ters sırada Macro-F1 0,6741,
ROC-AUC 0,7618: güçlü temporal sıra katkısı bu tanıda gösterilmiş sayılmaz.

Tek sonraki hipotez **hidden size 128 → 64**; dropout, AdamW weight decay
0,01, LR 3e-4, class weights, seed 42, FP32 ve splitleri koru. Kısa 6 epoch
tavanı, yeni run-dir. Bu deney henüz başlatılmadı.

Kanıtlar: `outputs/video_v1/analyses/5a78c4590243406bad187ffe1cb446bd/REPORT.md`,
`analysis.json`, `epoch_curves.png` ve `epoch_curves.pdf`. Analiz sırasında
3.245 eski dosyanın SHA-256 değerleri değişmedi; resmi test değerlendirilmedi.
`docs/EXPERIMENTS.md` ve eski checkpoint/log dosyaları değiştirilmedi.
