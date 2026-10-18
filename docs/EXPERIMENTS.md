# Fotoğraf Tabanlı Deepfake Tespiti — Deney Günlüğü

Bu belge final rapor ve sunumun kaynak günlüğüdür. Resmi proje geçmişi ilk
fotoğraf datasetiyle başlar. Mevcut ana model **V3 Robust EfficientNet-B0**,
karşılaştırma modelleri V1 ve V2'dir. Eğitim ve final/robustness değerlendirmeleri
tamamlandı. Checkpoint'ler sabittir; yeni eğitim veya tuning yapılmaz.

**Kaynak doğrulama tarihi:** 10 Ekim 2026. Aşağıdaki sıra proje aşamalarını
gösterir; eski çalıştırmaların kesin başlangıç/bitiş tarihleri kaydedilmediği
için son dosya değişiklik zamanlarından deney tarihi türetilmemiştir.
Sayısal sonuçlar JSON/CSV ve checkpoint metadata'sından doğrulanmıştır.
Dataset ve sonuçlar Git dışında olduğundan rapor tesliminde ilgili dosyalar
ayrıca korunmalıdır.

## 1. Proje özeti

**Proje:** Derin Öğrenme ile Deepfake Yüz Görüntülerinin Tespiti.
**Amaç:** REAL ve FAKE yüz fotoğraflarını EfficientNet-B0 ile sınıflandırmak;
model davranışını açıklanabilirlik ve görüntü kalitesi deneyleriyle incelemek.
**Etiket standardı:** `0 = REAL`, `1 = FAKE`. Precision, recall, F1 ve ROC-AUC
için pozitif sınıf FAKE'tir. Confusion matrix satırları gerçek, sütunları tahmin;
sıra REAL, FAKE'tir.

Python, PyTorch, torchvision, scikit-learn, matplotlib, Pillow ve pandas
kullanılır. Üç model de ImageNet `IMAGENET1K_V1` ağırlıklarıyla başlatılmış
EfficientNet-B0 ve iki çıkışlı sınıflandırıcı kullanır. Girdi RGB 224×224,
eğitim donanımı NVIDIA GTX 1650 4 GB'dır. Hazır yüz fotoğrafları kullanılır;
görüntünün tamamı ortak CLEAN dönüşümüyle sınıflandırılır.

Ortak preprocessing: EXIF yön düzeltme → RGB → bilinear/antialias 224×224 →
ImageNet normalizasyonu (`mean=[0.485,0.456,0.406]`,
`std=[0.229,0.224,0.225]`). Eğitimde ayrıca rastgele yatay çevirme uygulanır;
validation/testte rastgele augmentation yoktur. Test ayrımları eğitim ve
checkpoint seçimi için kullanılmamıştır.

## 2. İlk fotoğraf dataseti / V1

**Amaç:** Hafif bir fotoğraf sınıflandırma baseline'ı için dataset doğrulamak.
**Kaynak:** `data/raw/Real/`, `data/raw/Fake/`, `data/raw/dataset.csv`.

| Ayrım | REAL | FAKE | Toplam |
|---|---:|---:|---:|
| Train | 4.123 | 4.900 | 9.023 |
| Validation | 883 | 1.050 | 1.933 |
| Test | 884 | 1.050 | 1.934 |
| Genel | 5.890 (%45,69) | 7.000 (%54,31) | 12.890 |

Seed 42 ve yaklaşık %70/%15/%15 grup bazlı split kullanıldı. Tam dosya/piksel
kopyaları, yatay ayna kopyaları ve isimlerden bilinen augmentation aileleri
birlikte tutuldu; büyük gruplar nedeniyle oranlar yaklaşık olabilir.
`outputs/metrics/dataset_audit.json` bozuk görüntü ve yerel yol hatası
bulunmadığını, ayrımlar arası bilinen grup çakışması olmadığını gösterir.

6.445 augmentation görüntüsü vardır. Denetimde dört farklı birebir kopya çifti
bulunmuştur: `real_216_aug_2/3`, `real_271_aug_0/1`, `fake_104_aug_0/1`,
`fake_484_aug_1/2`. Aynı çift hem dosya hem piksel hash denetiminde
görünebildiğinden rapor kayıtları bağımsız kopyalar gibi toplanmamıştır.
Bilinen aileler aynı ayrımdadır; kopya olması tek başına ayrımlar arası sızıntı
anlamına gelmez. Orijinal kaynak/kişi eşlemesi eksiktir; tam kaynak/kişi
düzeyinde leakage garantisi verilemez.

**Karar:** Doğrulanmış grup ayrımlarıyla V1 eğitimine geçildi.
Dataset fingerprint:
`ebb19599105a5df4296dc072e4ec62b097bafd29366847f7a8ad5d89f9cd2a57`.
Denetimin işlem süresi kalıcı raporda bulunmuyor.
**Kaynak dosyalar:** `src/dataset.py`, `outputs/metrics/dataset_audit.json`.

## 3. V1 EfficientNet-B0 eğitimi

**Amaç:** Fotoğraf tabanlı baseline elde etmek.
**Model/ayarlar:** EfficientNet-B0 ImageNet pretrained, tüm model fine-tuning,
CrossEntropyLoss, AdamW, LR 3e-4, batch 8, 224×224, CUDA mixed precision,
seed 42, workers 0, maksimum 10 epoch, patience 3.
Model seçimi yalnızca validation FAKE F1 ile yapıldı.

| Epoch | Train loss | Val accuracy | Val precision | Val recall | Val F1 | Süre (sn) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0,23520581 | 0,89756855 | 0,87237762 | 0,95047619 | 0,90975387 | Kaydedilmedi |
| 2 | 0,11759913 | 0,92343508 | 0,92307692 | 0,93714286 | 0,93005671 | 179,30 |
| 3 | 0,08479325 | 0,92343508 | 0,90703971 | 0,95714286 | 0,93141798 | 182,79 |
| 4 | 0,07020385 | 0,93067770 | 0,89414802 | 0,98952381 | 0,93942134 | 169,33 |
| **5** | **0,05525940** | **0,95757889** | **0,97920792** | **0,94190476** | **0,96019417** | **164,21** |
| 6 | 0,04240190 | 0,89808588 | 0,84478577 | 0,99523810 | 0,91386095 | 164,10 |
| 7 | 0,04041329 | 0,94723228 | 0,94465291 | 0,95904762 | 0,95179584 | 166,25 |
| 8 | 0,04348953 | 0,93792033 | 0,91077739 | 0,98190476 | 0,94500458 | 164,92 |

**Sonuç:** Best epoch 5. Epoch 6–8 validation F1 best değerini aşmadı;
early stopping epoch 8'de durdurdu. Checkpoint durumu patience 3 ve
`bad_epochs=3` ile uyumludur. Train loss düşerken validation dalgalanmıştır.

**Süre:** Kaydedilmiş epoch 2–8 toplamı 1.190,91 saniye / 19,85 dakika.
Epoch 1 süre alanları boştur; **19,85 dakika tam sekiz epoch'un toplamı
değildir**, tam eğitim süresi bilinmiyor. Epoch 2 yaklaşık 2,99 dakikadır.
Ölçümler eğitim+validation döngüsünü kapsar; veri denetimi/kurulum/yazma dahil
değildir. Sonradan epoch süresi, toplam süre ve ETA saniye/dakika olarak
konsola, history ve checkpoint'e eklendi; eksik geçmiş süre uydurulmadı.

Her epoch checkpoint, atomik kaydetme ve model/optimizer/scaler/RNG/loader/
early-stopping durumlarıyla resume desteği vardır.
**İlgili dosyalar:** `src/train.py`, `outputs/checkpoints/best_model.pth`,
`outputs/checkpoints/last.pth`, `outputs/metrics/history.csv`.

## 4. V1 final test

**Amaç:** Seçilmiş epoch 5 modelini ayrılmış 1.934 fotoğrafta ölçmek.
**Checkpoint:** `outputs/checkpoints/best_model.pth`.
Validation'da seçim tamamlandıktan sonra final test yapıldı; testte tuning yok.

| Metrik | Sonuç |
|---|---:|
| Accuracy | %95,45 |
| Precision (FAKE) | %98,00 |
| Recall (FAKE) | %93,52 |
| F1 (FAKE) | %95,71 |
| ROC-AUC | 0,99185736 |
| Doğru / yanlış | 1.846 / 88 |

| Gerçek / Tahmin | REAL | FAKE |
|---|---:|---:|
| REAL | 864 | 20 |
| FAKE | 68 | 982 |

**Yorum:** Kendi fotoğraf test ayrımında güçlü baseline. Sonuç farklı fotoğraf
kaynaklarında aynı başarıyı garanti etmez; kaynak kimliği sınırlaması sürer.
Değerlendirme süresi sonuç JSON'unda bulunmuyor.
**Dosyalar:** `outputs/metrics/final_test_metrics.json`,
`outputs/figures/final_test_metrics_confusion_matrix.png`.

## 5. Grad-CAM

**Amaç:** Tek fotoğraf tahmininin uzamsal katkılarını açıklamak.
**Checkpoint:** V1 `best_model.pth` (best epoch 5).
**Target layer:** `features.8.0`; son uygun convolution otomatik seçilir,
uzamsal çıktı ve sınıflandırıcı kanal uyumu kontrol edilir.

REAL/FAKE tahmini, softmax confidence, original, heatmap, overlay,
comparison ve prediction JSON üretilir. Eğitim/optimizer adımı yoktur;
model ağırlıkları, buffer'lar ve mevcut gradient alanları korunur.

Mevcut çıktılar iki farklı örnek fotoğraf içerir: REAL örneği iki kez
çalıştırılmıştır (confidence %99,79); FAKE örneğinin confidence değeri %97,80'dir.
Görsellerde FAKE örneğinin göz/üst yanak/burun köprüsü çevresinde,
REAL örneğinin el/kıyafet ve arka plan bölgelerinde de katkı gözlenmiştir.
Bu, shortcut/background bias için **inceleme hipotezidir**; iki örnekten
genel veya nedensel bir sonuç çıkarılamaz. Confidence kalibre edilmiş doğruluk
değildir; Grad-CAM manipülasyon maskesi değildir.

Çalıştırma süreleri mevcut prediction JSON'larında yoktur.
**Dosyalar:** `src/explainability.py`, `scripts/generate_gradcam.py`,
`outputs/figures/gradcam/` altındaki `prediction.json` ve `comparison.png`
başta olmak üzere mevcut açıklama çıktıları.

## 6. İkinci 140k fotoğraf dataseti

**Dataset:** 140k Real and Fake Faces. İstenen yol `data/real_vs_fake/`;
yerelde mevcut yol `data/real-vs-fake/`tir. Loader mevcut ada yönlenir;
dosyalar taşınmaz. Kaynağın URL/sürüm/lisans bilgisi yerel sonuç dosyalarında
kayıtlı değildir; yayın raporunda kaynak kullanıcı tarafından tamamlanmalıdır.

| Ayrım | REAL | FAKE | Toplam | REAL / FAKE |
|---|---:|---:|---:|---:|
| Train | 50.000 | 50.000 | 100.000 | %50 / %50 |
| Validation (`valid`) | 10.000 | 10.000 | 20.000 | %50 / %50 |
| Test | 10.000 | 10.000 | 20.000 | %50 / %50 |
| Genel | 70.000 | 70.000 | 140.000 | %50 / %50 |

Hazır train/valid/test splitleri korunmuştur; yeniden split yapılmamıştır.
Önceki salt-okunur dataset doğrulamasında RGB 256×256 görüntüler, sıfır bozuk
görüntü, sıfır desteklenmeyen uzantı, sıfır birebir duplicate ve sıfır label
conflict raporlandı. Tüm 140k denetiminin ayrı kalıcı JSON raporu repoda yoktur;
bu özet önceki doğrulama oturumuna dayanır. Final test JSON'u ayrıca sıfır
test duplicate ve sıfır test–train/validation SHA-256 çakışmasını saklar.
Hash denetimi yakın kopya veya kişi düzeyinde bağımsızlık kanıtı değildir.

ImageFolder varsayılan `fake=0 / real=1` sırası proje standardıyla ters
olduğundan loader **REAL=0 / FAKE=1** olarak uyarlanmıştır. Samples, targets ve
getitem düzeyinde mapping test edilmiştir. Training loader yalnızca train/
valid kabul eder; test splitini reddeder. Denetim işlem süresi kaydedilmemiştir.
**Dosyalar:** `src/presplit_dataset.py`, `tests/test_presplit_training.py`.

## 7. V2 EfficientNet-B0 eğitimi

**Amaç:** Daha büyük ve dengeli fotoğraf datasetinde ikinci model geliştirmek.
**Ayarlar:** ImageNet pretrained EfficientNet-B0, 224×224, batch 8, AdamW,
LR 3e-4, CUDA mixed precision, GTX 1650 4 GB, workers 0, seed 42,
maksimum 10 epoch, early stopping patience 3. Seçim validation FAKE F1 ile.
Tek batch CUDA forward/backward kontrolü eğitim öncesinde yapıldı; smoke
test optimizer adımı/checkpoint oluşturmaz. Eğitim bir epoch benchmark ile
başladı, resume ile toplam dört epoch'a ulaştı.

| Epoch | Train loss | Val accuracy | Val precision | Val recall | Val F1 | Süre (sn / dk) |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 0,10062474 | 0,99415000 | 0,99174047 | 0,99660000 | 0,99416430 | 1.856,23 / 30,94 |
| 2 | 0,03437178 | 0,99130000 | 0,99130000 | 0,99130000 | 0,99130000 | 1.848,11 / 30,80 |
| **3** | **0,02489950** | **0,99480000** | **0,99688693** | **0,99270000** | **0,99478906** | **1.811,36 / 30,19** |
| 4 | 0,02000620 | 0,99215000 | 0,99989843 | 0,98440000 | 0,99208869 | 1.846,45 / 30,77 |

**Sonuç:** Best epoch 3. Epoch 3 F1, epoch 1'den yaklaşık 0,0625 yüzde puan
daha yüksektir; iyileşme küçük ama gerçektir. İlk epoch'tan sonraki validation
kazanımları sınırlı olduğundan kullanıcı daha fazla epoch çalıştırmadı.
Dört epoch'ta duruş early stopping değildir: son checkpoint'te
`bad_epochs=1`, patience 3; çalıştırılan toplam epoch sınırı 4'tür.
**Toplam ölçülmüş süre:** 7.362,15 saniye / 122,70 dakika.

Yeni model/sonuçlar ayrı kaydedildi; V1 dosyaları korundu. Her epoch checkpoint,
resume, epoch süresi/toplam süre/ETA desteği vardır. Epoch 1/4 ETA=0 kendi
çalıştırmalarının üst sınırına ulaşılmasıyla uyumludur.
**Dosyalar:** `src/train_v2.py`, `outputs/checkpoints/best_model_v2.pth`,
`outputs/checkpoints/image_v2/`, `outputs/metrics/image_v2/history.csv`.

## 8. V2 final test

**Amaç:** Sabit epoch 3 modelini 20.000 test fotoğrafında final değerlendirmek.
**Checkpoint:** `outputs/checkpoints/best_model_v2.pth`.
**Ayarlar:** CUDA, float32 inference, RGB 224×224 ve aynı ImageNet
normalizasyonu. Test eğitim/model seçimi/tuning için kullanılmadı.

| Metrik | Sonuç |
|---|---:|
| Best epoch | 3 |
| Accuracy | %99,41 |
| Precision (FAKE) | %99,57 |
| Recall (FAKE) | %99,25 |
| F1 (FAKE) | %99,41 |
| ROC-AUC | 0,99825422 |
| Doğru / yanlış | 19.882 / 118 |

| Gerçek / Tahmin | REAL | FAKE |
|---|---:|---:|
| REAL | 9.957 | 43 |
| FAKE | 75 | 9.925 |

Final test kaydında `test_development_hash_overlap=0`,
`test_duplicate_hashes=0`, `test_used_for_tuning=false`.
Train/valid fingerprint:
`9f035a11c082837f02cf5a4e66fb76fcd11df7441e8ca86041a77cd6dc3931be`.
Test fingerprint:
`4431a98958594392e4734aef40fc475efca7798a21b3d59c91124cbb3b0955e5`.
Checkpoint SHA-256:
`7fd27269ee44005d542978d0d404c26fea9432e98bbe61d50100444874bea520`.

**Süre:** Inference 76,63 saniye; doğrulama dahil toplam 297,94 saniye.
**Yorum:** V2 kendi temiz fotoğraf test ayrımında yüksek başarı elde etti.
Bu sonuç görüntü bozulmalarına dayanıklılığı henüz ölçmez.
**Dosyalar:** `outputs/metrics/final_test_metrics_v2.json`,
`outputs/figures/final_test_metrics_v2_confusion_matrix.png`.

## 9. V1 vs V2 karşılaştırması

| Ölçü | V1 | V2 |
|---|---:|---:|
| Dataset toplamı | 12.890 | 140.000 |
| Train / validation | 9.023 / 1.933 | 100.000 / 20.000 |
| Test | 1.934 | 20.000 |
| Tamamlanan epoch / best epoch | 8 / 5 | 4 / 3 |
| Accuracy | %95,45 | %99,41 |
| Precision (FAKE) | %98,00 | %99,57 |
| Recall (FAKE) | %93,52 | %99,25 |
| F1 (FAKE) | %95,71 | %99,41 |
| ROC-AUC | 0,99185736 | 0,99825422 |
| Kaydedilmiş eğitim süresi | 19,85 dk (epoch 1 hariç) | 122,70 dk (4 epoch) |

İki model farklı datasetlerin kendi test ayrımlarında ölçüldü. Bu tablo ortak
test setinde kontrollü model karşılaştırması değildir; artış yalnızca mimariye
veya dataset büyüklüğüne nedensel olarak atfedilemez. Bu aşamada ana model
V2 idi; V3 tamamlandıktan sonra ana model V3 olarak belirlendi. V1 ve V2
checkpoint'leri korunur. Yeniden eğitim/tuning yapılmayacaktır.

## 10. Robustness analizi

**Durum:** Gerçek 20.000 fotoğraf üzerinde kullanıcı tarafından tamamlandı.
9 Ekim 2026 21:55:18 +03:00 tarihli çalıştırmanın tüm metrikleri ve yorumları
belgenin sonundaki **V2 fotoğraf kalite dayanıklılığı** deney kaydındadır.
JSON/CSV ve grafik mevcut; CLEAN sonuçları final test referansıyla birebir eşleşti.
Hazırlık doğrulamasında tüm 70 test (19 yeni robustness/günlük testi dahil)
geçti; lint, formatter, Python compile, dependency ve diff kontrolleri temizdi.
Mevcut 41 çıktı dosyasının SHA-256 özeti değişmedi.
**Amaç:** Sabit V2 modelinin görüntü kalitesi düşüşlerine duyarlılığını ölçmek.
**Checkpoint:** `outputs/checkpoints/best_model_v2.pth` (epoch 3).
**Dataset:** Aynı test 10.000 REAL + 10.000 FAKE; tüm koşullarda aynı dosyalar
ve etiketler, batch 8, workers 0, seed 42, CUDA float32 inference.

| Koşul | Protokol |
|---|---|
| CLEAN | Mevcut test preprocessing'iyle aynı, ek bozulma yok |
| JPEG_Q70 | RGB 224×224 üzerinde JPEG quality 70, 4:2:0 subsampling |
| JPEG_Q50 | RGB 224×224 üzerinde JPEG quality 50, 4:2:0 subsampling |
| BLUR_SIGMA1 | Gaussian sigma 1, kernel 7×7, reflect padding |
| BLUR_SIGMA2 | Gaussian sigma 2, kernel 13×13, reflect padding |
| NOISE_STD0.01 | RGB [0,1] üzerinde Gaussian noise std 0,01, clamp [0,1] |
| NOISE_STD0.03 | RGB [0,1] üzerinde Gaussian noise std 0,03, clamp [0,1] |
| RESIZE_112 | 224→112→224, bilinear ve antialias |

Bozulmalar **224'e resize sonrası, ImageNet normalizasyonu öncesi** uygulanır;
ardından normalizasyon yapılır. JPEG decode bellektedir; dosya yazılmaz.
Noise her göreli dosya yolu+seed için deterministiktir, batch sırasından
bağımsızdır; iki noise koşulu aynı standart normal örneği farklı ölçekler.
Bu protokol 224 boyutundaki model girdisinin kalite duyarlılığını ölçer.

Her koşul accuracy, precision, recall, F1, ROC-AUC, confusion matrix üretir.
Düşüş CLEAN−koşul olarak hesaplanır; accuracy/F1 için ayrıca yüzde puan
kaydedilir. Negatif düşüş iyileşme demektir. CLEAN metrikleri final test
referansıyla ayrıca karşılaştırılır; olası fark raporlanır, tuning yapılmaz.
Checkpoint SHA-256, test dosya sayısı/dağılımı/fingerprint, label mapping ve
best epoch doğrulanmadan inference başlamaz. Splitler değiştirilmez;
train/valid fotoğrafları okunmaz. Model/ağırlıklar ve orijinal fotoğraflar
değiştirilmez; eski sonuçların üzerine yazılmaz. Var olan robustness
çıktılarında yeniden çalıştırma yeni output dizini gerektirir.

Kullanıcının açıkça başlatacağı komut:

```powershell
.\.venv\Scripts\python.exe -u scripts\evaluate_robustness_v2.py --run-evaluation --device cuda
```

**Üretilen dosyalar:** `outputs/metrics/robustness_v2.json`,
`outputs/metrics/robustness_v2.csv`, `outputs/figures/robustness_v2.png`.
JSON toplam/koşul süreleri, preprocessing, fingerprint ve referans farklarını
saklar; grafik accuracy/F1/ROC-AUC'yi karşılaştırır. Rapor toplam süresi
471,02 saniye / 7,85 dakikadır. Günlük yazımı dahil otomatik kayıt süresi
471,13 saniyedir. Tamamlanan/başarısız çalıştırma bu günlüğe otomatik eklenir.
**Kod:** `src/robustness.py`, `scripts/evaluate_robustness_v2.py`,
`tests/test_robustness.py`.

## 11. Bundan sonra yapılacak fotoğraf tabanlı deneyler

V2 robustness, V3 robust eğitimi ve V3 final/robustness ölçümleri tamamlandı;
tarihli kayıtlar aşağıdadır. Yeni eğitim/tuning yapılmamıştır. Araştırılabilecek
diğer fotoğraf deneyleri: V3 hata analizi, bağımsız fotoğraf
kaynağında sabit checkpoint değerlendirmesi, kaynak/kişi metadata'sıyla
bağımsızlık denetimi. Bunlar öneridir; yapılmış deney veya izin verilmiş
yeniden eğitim olarak sunulmaz. Test sonuçlarına göre V2 tuning yapılmaz.

Her yeni fotoğraf deneyi, iyi/kötü sonuç veya çalıştırma hatası dahil,
`src.experiment_log.append_experiment` ile belge sonuna kaydedilmelidir.
Robustness entrypoint'i bunu başarı/hata durumunda otomatik yapar. Gelecekteki
deney entrypoint'leri aynı helper'ı kullanmalıdır; mevcut tamamlanmış eğitim
pipeline'ları bu hazırlıkta değiştirilmemiştir.

Kayıt alanları: benzersiz id, deney adı, gerçek tarih, durum, model,
checkpoint, dataset, train/val/test sayıları, amaç, ayarlar (image size,
batch, LR, epoch, seed, preprocessing), accuracy/precision/recall/F1/ROC-AUC/
confusion matrix veya başarısızlıkta hata, saniye cinsinden süre, gözlem,
yorum ve ilgili metrics/figures/checkpoint yolları. Uygulanmayan eğitim
ayarları açıkça belirtilir. Olmayan sonuç/süre uydurulmaz.

Helper aynı id'yi iki kez eklemez; kilit ve atomik replace ile mevcut metni
korur. Kayıt hatası susturulmaz. Sentetik test kayıtları geçici test klasörlerine
yazılır; bu günlüğe gerçek sonuç olarak girmez. Sonraki gerçek kayıtlar aşağıya
eklenecektir.

<!-- experiment:robustness-v2-d2f6c5e397ed4c2192f680119365ad0b -->
## V2 fotoğraf kalite dayanıklılığı

**Tarih:** 2026-10-09T21:55:18+03:00

**Durum:** completed

**Model:** EfficientNet-B0

**Checkpoint:** C:\Users\kilic\Desktop\deepfake-image-detection\outputs\checkpoints\best_model_v2.pth

**Dataset:** C:\Users\kilic\Desktop\deepfake-image-detection\data\real_vs_fake

**Train/Val/Test sayıları:**

```json
{
  "train": "kullanılmadı",
  "val": "kullanılmadı",
  "test": 20000
}
```

**Amaç:** Sabit modelin aynı fotoğraflarda kalite bozulmasına dayanıklılığı

**Ayarlar:**

```json
{
  "image_size": 224,
  "batch_size": 8,
  "seed": 42,
  "learning_rate": "uygulanmaz",
  "epoch": "uygulanmaz",
  "conditions": [
    "CLEAN",
    "JPEG_Q70",
    "JPEG_Q50",
    "BLUR_SIGMA1",
    "BLUR_SIGMA2",
    "NOISE_STD0.01",
    "NOISE_STD0.03",
    "RESIZE_112"
  ],
  "preprocessing": "RGB/EXIF -> bilinear 224 -> corruption -> ImageNet normalization",
  "jpeg": {
    "qualities": [
      70,
      50
    ],
    "subsampling": "4:2:0"
  },
  "blur": {
    "sigmas": [
      1,
      2
    ],
    "kernels": [
      7,
      13
    ],
    "padding": "reflect"
  },
  "noise": {
    "stds": [
      0.01,
      0.03
    ],
    "domain": "RGB [0,1], clamp [0,1]",
    "seed": "SHA256(seed:relative_path:gaussian_noise), CPU generator"
  },
  "resize": "224 -> 112 -> 224, bilinear, antialias"
}
```

**Sonuç:**

```json
{
  "CLEAN": {
    "accuracy": 0.9941,
    "precision": 0.9956861958266453,
    "recall": 0.9925,
    "f1": 0.9940905448717948,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        9957,
        43
      ],
      [
        75,
        9925
      ]
    ],
    "roc_auc": 0.9982542199999999,
    "duration_seconds": 39.273195099992336,
    "accuracy_drop": 0.0,
    "accuracy_drop_percentage_points": 0.0,
    "f1_drop": 0.0,
    "f1_drop_percentage_points": 0.0,
    "roc_auc_drop": 0.0
  },
  "JPEG_Q70": {
    "accuracy": 0.8569,
    "precision": 0.8145047585477617,
    "recall": 0.9243,
    "f1": 0.8659359190556493,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        7895,
        2105
      ],
      [
        757,
        9243
      ]
    ],
    "roc_auc": 0.9347699300000001,
    "duration_seconds": 66.94360420000066,
    "accuracy_drop": 0.1372,
    "accuracy_drop_percentage_points": 13.719999999999999,
    "f1_drop": 0.12815462581614556,
    "f1_drop_percentage_points": 12.815462581614556,
    "roc_auc_drop": 0.06348428999999978
  },
  "JPEG_Q50": {
    "accuracy": 0.7634,
    "precision": 0.8391707442698945,
    "recall": 0.6517,
    "f1": 0.7336485421591804,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        8751,
        1249
      ],
      [
        3483,
        6517
      ]
    ],
    "roc_auc": 0.8574376850000001,
    "duration_seconds": 66.29082089999429,
    "accuracy_drop": 0.23070000000000002,
    "accuracy_drop_percentage_points": 23.07,
    "f1_drop": 0.2604420027126144,
    "f1_drop_percentage_points": 26.04420027126144,
    "roc_auc_drop": 0.1408165349999998
  },
  "BLUR_SIGMA1": {
    "accuracy": 0.50955,
    "precision": 1.0,
    "recall": 0.0191,
    "f1": 0.037484054557943285,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        10000,
        0
      ],
      [
        9809,
        191
      ]
    ],
    "roc_auc": 0.861511815,
    "duration_seconds": 41.24604380000483,
    "accuracy_drop": 0.48455000000000004,
    "accuracy_drop_percentage_points": 48.455000000000005,
    "f1_drop": 0.9566064903138516,
    "f1_drop_percentage_points": 95.66064903138516,
    "roc_auc_drop": 0.1367424049999999
  },
  "BLUR_SIGMA2": {
    "accuracy": 0.50005,
    "precision": 1.0,
    "recall": 0.0001,
    "f1": 0.00019998000199980003,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        10000,
        0
      ],
      [
        9999,
        1
      ]
    ],
    "roc_auc": 0.588402,
    "duration_seconds": 43.14790050000374,
    "accuracy_drop": 0.49405,
    "accuracy_drop_percentage_points": 49.405,
    "f1_drop": 0.993890564869795,
    "f1_drop_percentage_points": 99.3890564869795,
    "roc_auc_drop": 0.4098522199999999
  },
  "NOISE_STD0.01": {
    "accuracy": 0.94895,
    "precision": 0.9693674856246733,
    "recall": 0.9272,
    "f1": 0.9478149757219525,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        9707,
        293
      ],
      [
        728,
        9272
      ]
    ],
    "roc_auc": 0.99107987,
    "duration_seconds": 57.72981280001022,
    "accuracy_drop": 0.04515000000000002,
    "accuracy_drop_percentage_points": 4.515000000000002,
    "f1_drop": 0.04627556914984232,
    "f1_drop_percentage_points": 4.627556914984233,
    "roc_auc_drop": 0.007174349999999885
  },
  "NOISE_STD0.03": {
    "accuracy": 0.75455,
    "precision": 0.7228789072760704,
    "recall": 0.8256,
    "f1": 0.7708323607674712,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        6835,
        3165
      ],
      [
        1744,
        8256
      ]
    ],
    "roc_auc": 0.840072175,
    "duration_seconds": 57.71756240000241,
    "accuracy_drop": 0.23954999999999993,
    "accuracy_drop_percentage_points": 23.95499999999999,
    "f1_drop": 0.22325818410432363,
    "f1_drop_percentage_points": 22.325818410432362,
    "roc_auc_drop": 0.15818204499999988
  },
  "RESIZE_112": {
    "accuracy": 0.50185,
    "precision": 0.9743589743589743,
    "recall": 0.0038,
    "f1": 0.007570475146926985,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        9999,
        1
      ],
      [
        9962,
        38
      ]
    ],
    "roc_auc": 0.77869271,
    "duration_seconds": 39.503988600001776,
    "accuracy_drop": 0.49224999999999997,
    "accuracy_drop_percentage_points": 49.224999999999994,
    "f1_drop": 0.9865200697248678,
    "f1_drop_percentage_points": 98.65200697248679,
    "roc_auc_drop": 0.2195615099999999
  }
}
```

**Süre (saniye):** 471.1332255000002

**Önemli gözlem:** Blur ve downscale/upscale altında FAKE örnekler büyük ölçüde
REAL olarak sınıflandırıldı. Blur sigma 1: 10.000 FAKE'in 9.809'u REAL
(%98,09), FAKE recall %1,91. Blur sigma 2: 9.999'u REAL (%99,99), recall
%0,01. Resize 224→112→224: 9.962'si REAL (%99,62), recall %0,38.
Bu koşulların precision değerleri yüksek görünse de çok az FAKE tahmini
üretildiğinden recall/F1 ile birlikte yorumlanmalıdır.

**Sonuç / yorum:** CLEAN accuracy %99,41 olmasına rağmen kalite bozulmalarında
başarı belirgin düştü. Özellikle blur ve downscale, yüksek frekanslı doku ve
ince görsel ipuçlarını zayıflattığı için sonuçlar modelin bu ipuçlarına
bağımlı olabileceği hipotezini destekler; nedensel veya frekans analiziyle
kanıtlanmış bir sonuç değildir. Ayrıca karar dağılımı REAL'e kaymaktadır.
Blur sigma 1 ROC-AUC 0,86151182 ve resize ROC-AUC 0,77869271 olduğundan
yaklaşık %50 accuracy tüm koşullarda rastgele sıralama anlamına gelmez.
Değerlendirmede threshold/model tuning veya ağırlık değişikliği yapılmadı.

**Karşılaştırmalı özet:** Precision/recall/F1 pozitif FAKE sınıfı için;
düşüşler CLEAN−koşul, accuracy/F1 için yüzde puan (yp).

| Koşul | Accuracy % | Precision % | Recall % | F1 % | ROC-AUC | Acc düşüş (yp) | F1 düşüş (yp) | AUC düşüş | Süre (sn) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CLEAN | 99,410 | 99,569 | 99,250 | 99,409 | 0,99825422 | 0 | 0 | 0 | 39,27 |
| JPEG Q70 | 85,690 | 81,450 | 92,430 | 86,594 | 0,93476993 | 13,720 | 12,815 | 0,06348429 | 66,94 |
| JPEG Q50 | 76,340 | 83,917 | 65,170 | 73,365 | 0,85743769 | 23,070 | 26,044 | 0,14081653 | 66,29 |
| Blur sigma 1 | 50,955 | 100,000 | 1,910 | 3,748 | 0,86151182 | 48,455 | 95,661 | 0,13674240 | 41,25 |
| Blur sigma 2 | 50,005 | 100,000 | 0,010 | 0,020 | 0,58840200 | 49,405 | 99,389 | 0,40985222 | 43,15 |
| Noise std 0,01 | 94,895 | 96,937 | 92,720 | 94,781 | 0,99107987 | 4,515 | 4,628 | 0,00717435 | 57,73 |
| Noise std 0,03 | 75,455 | 72,288 | 82,560 | 77,083 | 0,84007218 | 23,955 | 22,326 | 0,15818204 | 57,72 |
| Resize 112 | 50,185 | 97,436 | 0,380 | 0,757 | 0,77869271 | 49,225 | 98,652 | 0,21956151 | 39,50 |

Tam hassasiyetli metrikler ve tüm confusion matrix'ler yukarıdaki JSON kaydında
korunmuştur. Satırlar gerçek/sütunlar tahmin, sıra REAL/FAKE.
Toplam 20.000 fotoğraf × 8 koşul = 160.000 tahmin. Test fingerprint ve
checkpoint hash sabit; referans CLEAN accuracy/F1/AUC farkları sıfırdır.

**İlgili dosyalar:**

```json
[
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\metrics\\robustness_v2.json",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\metrics\\robustness_v2.csv",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\figures\\robustness_v2.png"
]
```

## V3 Robust EfficientNet-B0 — hazırlık aşamasının kaydı

**Durum:** Yalnızca ayrı kod/test pipeline'ı hazır; eğitim başlatılmadı,
checkpoint veya V3 başarı metriği üretilmedi. **Amaç:** Train fotoğraflarının
kalitesini rastgele değiştirerek incelikli görsel ipuçlarına aşırı bağımlılığı
azaltmayı araştırmak. Bunun başarı sağlayacağı henüz ölçülmemiştir.

Bu paragraf 9 Ekim'deki kod hazırlığının durumunu kaydeder. Sonraki tamamlanmış
eğitim ve değerlendirmeler aşağıdaki tarihli deney kayıtlarında yer alır.

**Dataset/etiket:** Aynı hazır 100.000 train / 20.000 valid / 20.000 test,
REAL=0 / FAKE=1. Splitler yeniden oluşturulmaz. Training loader yalnızca
train ve valid okur; test klasörüne erişmez. Validation ve gelecekteki test
preprocessing'i önceki CLEAN dönüşümüyle aynı ve deterministiktir.

| Yalnızca TRAIN augmentation | Bağımsız olasılık | Parametre |
|---|---:|---|
| JPEG compression | %50 | Uniform tam sayı quality 50–95, 4:2:0 |
| Gaussian blur | %15 | Uniform sigma 0,1–1,5; Pillow GaussianBlur |
| Gaussian noise | %15 | Uniform std 0–0,02; RGB [0,1], clamp [0,1] |
| Downscale/upscale | %30 | Uniform tam sayı 112–224 → 224, bilinear/antialias |
| Horizontal flip | %50 | Yatay çevirme |

İşlem sırası: EXIF/RGB → 224 → flip → JPEG → blur → downscale/upscale →
tensor/noise → ImageNet normalizasyonu. Gate'ler bağımsızdır; bozulmalar
birlikte uygulanabilir veya uygulanmayabilir. Fotoğraflar bellekte işlenir,
orijinal dosyalar değiştirilmez. Blur için Pillow, Gaussian'ı yaklaşıklayan
genişletilmiş box filter dizisi kullanır; radius standart sapmadır. Bu motor
GTX 1650 yanında workers=0 CPU preprocessing maliyetini düşük tutmak için
seçildi; robustness ölçümündeki torchvision Gaussian kernel ile aynı
sayısal uygulama olduğu iddia edilmez. Tüm rastgele seçimler Torch RNG'dedir;
RNG ve loader durumları checkpoint resume'da geri yüklenir.

**Eğitim ayarları:** ImageNet pretrained EfficientNet-B0 (V2'den warm-start
yapılmaz), 224×224, batch 8, AdamW, LR 3e-4, CUDA mixed precision,
maksimum 6 epoch, early stopping patience 2, seed 42, workers 0.
Checkpoint seçimi yalnızca CLEAN validation FAKE F1 ile yapılır. Her epoch
checkpoint, resume, epoch süresi/toplam süre/ETA ve otomatik deney kaydı vardır.
Resume augmentation/config veya train/valid içeriği değişirse reddedilir;
V1/V2 checkpoint'leri resume olarak kabul edilmez.

**Yeni çıktı yolları:** `outputs/checkpoints/best_model_v3_robust.pth`,
`outputs/checkpoints/image_v3_robust/`, `outputs/metrics/image_v3_robust/history.csv`.
Mevcut V1/V2 ağırlıkları, sonuçları ve robustness dosyaları korunur.

**Deney tasarımı sınırı:** V2 testindeki bozulma sonuçları V3 augmentation
tasarımına yön verdi. Aynı 20.000 fotoğrafta ileride yapılacak V3 değerlendirmesi,
geliştirme boyunca bütünüyle görülmemiş yeni test iddiasıyla sunulmamalıdır.
V3 eğitim/model seçimi sırasında bu dosyalar okunmayacaktır. Bağımsız fotoğraf
testi daha güçlü genelleme kanıtı sağlayabilir.

**Tamamlanmış eğitim protokolü:** İlk çalıştırma bir epoch benchmark ile başladı.
Benchmark sonrasında aynı ayarlarla toplam altı epoch'a resume ile devam edildi.
**Kod/test:** `src/robust_augmentation.py`, `src/train_v3.py`,
`tests/test_robust_training.py`. Ölçülmüş süre aşağıdaki eğitim kaydındadır.

<!-- experiment:v3-robust-training-6a2573c879f6489789a3dfbf98922dea -->
## V3 Robust EfficientNet-B0 eğitim çalıştırması

**Doğrulanmış eğitim özeti (checkpoint + history CSV):** Altı epoch tamamlandı;
en iyi CLEAN validation FAKE F1 epoch 5'te 0,99614093, validation accuracy
0,99615'tir. Epoch 6'da `bad_epochs=1`, patience 2 olduğundan duruş early
stopping değildir; toplam altı epoch sınırına ulaşıldı. Test klasörü eğitimde
ve checkpoint seçiminde okunmadı. Ölçülmüş eğitim+validation döngüsü
11.434,72 saniye / 190,58 dakika; aşağıdaki otomatik çalıştırma süresi
11.571,32 saniye / 192,86 dakika olup hazırlık/yazma maliyetlerini de içerir.

| Epoch | Train loss | CLEAN val accuracy % | CLEAN val F1 % | Süre (sn) |
|---|---:|---:|---:|---:|
| 1 | 0,17033523 | 99,120 | 99,1175 | 2.020,43 |
| 2 | 0,06943234 | 98,460 | 98,4822 | 1.885,03 |
| 3 | 0,05120369 | 99,425 | 99,4226 | 1.873,54 |
| 4 | 0,04269859 | 99,145 | 99,1390 | 1.858,99 |
| **5** | **0,03659786** | **99,615** | **99,6141** | **1.917,48** |
| 6 | 0,03371896 | 99,515 | 99,5151 | 1.879,26 |

Train augmentation, CLEAN validation, ImageNet başlangıcı ve ayrı V3
checkpoint yolları hazırlık protokolüyle eşleşir. Tam ayarlar ve epoch
metrikleri aşağıdaki otomatik kayıtta korunmuştur. Kaynaklar:
`outputs/metrics/image_v3_robust/history.csv`,
`outputs/checkpoints/best_model_v3_robust.pth`,
`outputs/checkpoints/image_v3_robust/last.pth`.

**Tarih:** 2026-10-09T22:25:49+03:00

**Durum:** completed

**Model:** EfficientNet-B0 / ImageNet IMAGENET1K_V1

**Checkpoint:** C:\Users\kilic\Desktop\deepfake-image-detection\outputs\checkpoints\best_model_v3_robust.pth

**Dataset:** C:\Users\kilic\Desktop\deepfake-image-detection\data\real_vs_fake

**Train/Val/Test sayıları:**

```json
{
  "train": {
    "REAL": 50000,
    "FAKE": 50000,
    "total": 100000
  },
  "valid": {
    "REAL": 10000,
    "FAKE": 10000,
    "total": 20000
  },
  "test": "erişilmedi"
}
```

**Amaç:** Train kalite augmentation ile CLEAN validation FAKE F1 seçimi

**Ayarlar:**

```json
{
  "batch_size": 8,
  "workers": 0,
  "learning_rate": 0.0003,
  "seed": 42,
  "amp": true,
  "device_type": "cuda",
  "patience": 2,
  "min_delta": 0.0,
  "optimizer": "AdamW",
  "augmentation": {
    "jpeg_probability": 0.5,
    "jpeg_quality": [
      50,
      95
    ],
    "blur_probability": 0.15,
    "blur_sigma": [
      0.1,
      1.5
    ],
    "noise_probability": 0.15,
    "noise_std": [
      0.0,
      0.02
    ],
    "resize_probability": 0.3,
    "resize_size": [
      112,
      224
    ],
    "flip_probability": 0.5
  },
  "augmentation_version": 1,
  "blur_engine": "Pillow.GaussianBlur",
  "image_size": 224,
  "epochs": 6,
  "preprocessing": "EXIF/RGB -> 224 -> TRAIN quality augmentation -> ImageNet normalization; valid CLEAN"
}
```

**Sonuç:**

```json
{
  "history": [
    {
      "epoch": 1,
      "train_loss": 0.1703352331864834,
      "val_accuracy": 0.9912,
      "val_precision": 0.9939662107803701,
      "val_recall": 0.9884,
      "val_f1": 0.99117529081428,
      "epoch_duration": 2020.431388,
      "epoch_duration_minutes": 33.673856,
      "total_training_duration": 2020.431388,
      "total_training_duration_minutes": 33.673856,
      "eta": 10102.156942,
      "eta_minutes": 168.369282
    },
    {
      "epoch": 2,
      "train_loss": 0.06943233823180199,
      "val_accuracy": 0.9846,
      "val_precision": 0.9708511465215701,
      "val_recall": 0.9992,
      "val_f1": 0.9848216045732309,
      "epoch_duration": 1885.026404,
      "epoch_duration_minutes": 31.417107,
      "total_training_duration": 3905.457792,
      "total_training_duration_minutes": 65.090963,
      "eta": 7810.915585,
      "eta_minutes": 130.181926
    },
    {
      "epoch": 3,
      "train_loss": 0.05120368553876877,
      "val_accuracy": 0.99425,
      "val_precision": 0.998386608853484,
      "val_recall": 0.9901,
      "val_f1": 0.9942260380579404,
      "epoch_duration": 1873.539432,
      "epoch_duration_minutes": 31.225657,
      "total_training_duration": 5778.997224,
      "total_training_duration_minutes": 96.31662,
      "eta": 5778.997224,
      "eta_minutes": 96.31662
    },
    {
      "epoch": 4,
      "train_loss": 0.04269858548283577,
      "val_accuracy": 0.99145,
      "val_precision": 0.9983774465064396,
      "val_recall": 0.9845,
      "val_f1": 0.9913901616232819,
      "epoch_duration": 1858.985493,
      "epoch_duration_minutes": 30.983092,
      "total_training_duration": 7637.982717,
      "total_training_duration_minutes": 127.299712,
      "eta": 3818.991359,
      "eta_minutes": 63.649856
    },
    {
      "epoch": 5,
      "train_loss": 0.036597864696979523,
      "val_accuracy": 0.99615,
      "val_precision": 0.9984929167085301,
      "val_recall": 0.9938,
      "val_f1": 0.9961409311882925,
      "epoch_duration": 1917.476456,
      "epoch_duration_minutes": 31.957941,
      "total_training_duration": 9555.459173,
      "total_training_duration_minutes": 159.257653,
      "eta": 1911.091835,
      "eta_minutes": 31.851531
    },
    {
      "epoch": 6,
      "train_loss": 0.03371895802497864,
      "val_accuracy": 0.99515,
      "val_precision": 0.9949025487256372,
      "val_recall": 0.9954,
      "val_f1": 0.9951512121969508,
      "epoch_duration": 1879.259883,
      "epoch_duration_minutes": 31.320998,
      "total_training_duration": 11434.719056,
      "total_training_duration_minutes": 190.578651,
      "eta": 0.0,
      "eta_minutes": 0.0
    }
  ],
  "best_epoch": 5,
  "best_validation_metrics": {
    "accuracy": 0.99615,
    "precision": 0.9984929167085301,
    "recall": 0.9938,
    "f1": 0.9961409311882925,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        9985,
        15
      ],
      [
        62,
        9938
      ]
    ]
  },
  "new_epochs": 6
}
```

**Süre (saniye):** 11571.3160939

**Önemli gözlem:** Validation CLEAN; test klasörü okunmadı.

**Sonuç / yorum:** Bu çalıştırma tamamlandı; final test veya robustness sonucu üretilmedi.

**İlgili dosyalar:**

```json
[
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\checkpoints\\best_model_v3_robust.pth",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\checkpoints\\image_v3_robust\\last.pth",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\metrics\\image_v3_robust\\history.csv"
]
```

<!-- experiment:v3-final-robustness-879d710f180a4048bf09f05085371f7d -->
## V3 CLEAN final test ve V2 ile eşleştirilmiş robustness

**Doğrulanmış sonuç özeti:** Mevcut V3 best epoch **5** checkpoint'i üzerinde
10.000 REAL + 10.000 FAKE fotoğrafta önce ayrı CLEAN final test, ardından
V2'deki aynı sekiz koşul ölçüldü. Training augmentation kullanılmadı.
Robustness CLEAN metrikleri ayrı final testle birebir eşleşti. Test dosya
parmak izi V2 ile aynı; seed 42, batch 8, workers 0, CUDA float32 ve ortak
`src.robustness.corrupt_batch` kullanıldı. Orijinal 53 çıktı dosyasının
SHA-256 özeti korundu; ağırlık, threshold veya split değişmedi.

| CLEAN final test | Sonuç |
|---|---:|
| Accuracy | %99,635 |
| Precision (FAKE) | %99,789347 |
| Recall (FAKE) | %99,480 |
| F1 (FAKE) | %99,634433 |
| ROC-AUC | 0,99992647 |
| Doğru / yanlış | 19.927 / 73 |
| Best epoch | 5 |

Confusion matrix (gerçek satır / tahmin sütun, sıra REAL/FAKE):
`[[9979, 21], [52, 9948]]`.

**V2 vs V3:** Accuracy/F1 yüzde; Δ = V3−V2, yüzde puan.

| Koşul | V2 Accuracy % | V3 Accuracy % | Δ Acc (yp) | V2 F1 % | V3 F1 % | Δ F1 (yp) |
|---|---:|---:|---:|---:|---:|---:|
| CLEAN | 99,410 | 99,635 | +0,225 | 99,4091 | 99,6344 | +0,2254 |
| JPEG Q70 | 85,690 | 99,220 | +13,530 | 86,5936 | 99,2180 | +12,6245 |
| JPEG Q50 | 76,340 | 98,525 | +22,185 | 73,3649 | 98,5169 | +25,1521 |
| Blur sigma 1 | 50,955 | 98,885 | +47,930 | 3,7484 | 98,8753 | +95,1269 |
| Blur sigma 2 | 50,005 | 91,030 | +41,025 | 0,0200 | 90,2224 | +90,2024 |
| Noise std 0,01 | 94,895 | 99,430 | +4,535 | 94,7815 | 99,4292 | +4,6477 |
| Noise std 0,03 | 75,455 | 97,850 | +22,395 | 77,0832 | 97,8457 | +20,7625 |
| Resize 224→112→224 | 50,185 | 98,230 | **+48,045** | 0,7570 | 98,2045 | **+97,4475** |

**Blur/resize gözlemi:** FAKE→REAL hataları blur sigma 1'de 9.809→198,
sigma 2'de 9.999→1.723, resize'da 9.962→319 düştü. FAKE recall sırasıyla
%1,91→%98,02, %0,01→%82,77, %0,38→%96,81 oldu. En büyük accuracy ve F1
iyileşmesi resize'dadır. V3'te en zayıf koşul hâlâ blur sigma 2: accuracy
%91,03, F1 %90,2224; sigma 2 eğitimdeki 0,1–1,5 aralığının dışındadır.

**Yorum:** V3 bu kalite bozulmalarında belirgin biçimde daha dayanıklıdır ve
CLEAN başarıyı da korumuştur. Yüksek frekanslı ipuçlarına aşırı bağımlılığın
azalmış olabileceğiyle uyumludur; modelin hangi frekansları kullandığını
doğrudan kanıtlamaz. Eğitim epoch sayısı da değiştiğinden augmentation'ın
tek başına nedensel etkisini ölçen izole bir ablation değildir. V2 testindeki
bulgular V3 tasarımında daha önce görüldü; aynı testteki karşılaştırma
tamamen görülmemiş bağımsız test olarak sunulmaz. Bu değerlendirmede yeni
eğitim, checkpoint seçimi veya threshold/model tuning yapılmadı.

**Süre:** CLEAN geçişi (loader dahil) 315,74 saniye, sekiz koşullu geçiş
553,87 saniye; denetimler dahil rapor toplamı 915,53 saniye / 15,26 dakika.
Batch dönüşüm+model hesaplarının birikimli süreleri tam sonuç kaydındadır.
CLEAN sırasında kod testleri de çalıştırıldığından bu süreler V2 ile hız
benchmark'ı olarak yorumlanmaz. Tüm 95 test, lint, formatter, Python compile,
dependency ve diff kontrolleri geçti. Karşılaştırma grafiği görsel olarak
doğrulandı. Kod: `src/evaluate_v3.py`, `tests/test_v3_evaluation.py`.

Tam precision/recall/F1/ROC-AUC ve tüm confusion matrix'ler aşağıdaki otomatik
JSON kaydında ve `outputs/metrics/robustness_v3.json` / `.csv` içindedir.
Grafik: `outputs/figures/robustness_v2_vs_v3.png`.

**Tarih:** 2026-10-10T15:02:38+03:00

**Durum:** completed

**Model:** EfficientNet-B0 V3 Robust

**Checkpoint:** C:\Users\kilic\Desktop\deepfake-image-detection\outputs\checkpoints\best_model_v3_robust.pth

**Dataset:** C:\Users\kilic\Desktop\deepfake-image-detection\data\real_vs_fake

**Train/Val/Test sayıları:**

```json
{
  "train": "okunmadı",
  "val": "okunmadı",
  "test": 20000,
  "test_classes": {
    "REAL": 10000,
    "FAKE": 10000
  }
}
```

**Amaç:** Sabit V3 modelini CLEAN test ve aynı 8 koşulda V2 ile karşılaştırmak

**Ayarlar:**

```json
{
  "image_size": 224,
  "batch_size": 8,
  "seed": 42,
  "learning_rate": "uygulanmaz",
  "epoch": "uygulanmaz",
  "conditions": [
    "CLEAN",
    "JPEG_Q70",
    "JPEG_Q50",
    "BLUR_SIGMA1",
    "BLUR_SIGMA2",
    "NOISE_STD0.01",
    "NOISE_STD0.03",
    "RESIZE_112"
  ],
  "preprocessing": "RGB/EXIF -> bilinear 224 -> corruption -> ImageNet normalization",
  "jpeg": {
    "qualities": [
      70,
      50
    ],
    "subsampling": "4:2:0"
  },
  "blur": {
    "sigmas": [
      1,
      2
    ],
    "kernels": [
      7,
      13
    ],
    "padding": "reflect"
  },
  "noise": {
    "stds": [
      0.01,
      0.03
    ],
    "domain": "RGB [0,1], clamp [0,1]",
    "seed": "SHA256(seed:relative_path:gaussian_noise), CPU generator"
  },
  "resize": "224 -> 112 -> 224, bilinear, antialias",
  "best_epoch": 5,
  "device": "cuda",
  "evaluation_precision": "float32"
}
```

**Sonuç:**

```json
{
  "clean_final_test": {
    "accuracy": 0.99635,
    "precision": 0.9978934697562444,
    "recall": 0.9948,
    "f1": 0.9963443337172617,
    "positive_class": "FAKE",
    "class_names": [
      "REAL",
      "FAKE"
    ],
    "confusion_matrix": [
      [
        9979,
        21
      ],
      [
        52,
        9948
      ]
    ],
    "roc_auc": 0.99992647,
    "duration_seconds": 70.82206860000389,
    "date": "2026-10-10T15:02:38+03:00",
    "best_epoch": 5,
    "checkpoint": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\checkpoints\\best_model_v3_robust.pth",
    "checkpoint_sha256": "c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c",
    "dataset_root": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test",
    "test_fingerprint": "4431a98958594392e4734aef40fc475efca7798a21b3d59c91124cbb3b0955e5",
    "test_image_count": 20000,
    "test_class_counts": {
      "REAL": 10000,
      "FAKE": 10000
    },
    "label_mapping": {
      "REAL": 0,
      "FAKE": 1
    },
    "device": "cuda",
    "image_size": 224,
    "evaluation_precision": "float32",
    "seed": 42,
    "training_validation_fingerprint": "9f035a11c082837f02cf5a4e66fb76fcd11df7441e8ca86041a77cd6dc3931be",
    "test_used_for_training_or_checkpoint_selection": false,
    "test_used_for_tuning_in_this_evaluation": false,
    "test_previously_inspected_for_v3_augmentation_design": true,
    "split": "test",
    "correct_predictions": 19927,
    "incorrect_predictions": 73,
    "inference_duration_seconds": 315.7370853
  },
  "robustness": {
    "CLEAN": {
      "accuracy": 0.99635,
      "precision": 0.9978934697562444,
      "recall": 0.9948,
      "f1": 0.9963443337172617,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9979,
          21
        ],
        [
          52,
          9948
        ]
      ],
      "roc_auc": 0.99992647,
      "duration_seconds": 40.42695570000319,
      "accuracy_drop": 0.0,
      "accuracy_drop_percentage_points": 0.0,
      "f1_drop": 0.0,
      "f1_drop_percentage_points": 0.0,
      "roc_auc_drop": 0.0
    },
    "JPEG_Q70": {
      "accuracy": 0.9922,
      "precision": 0.9946733668341708,
      "recall": 0.9897,
      "f1": 0.9921804511278195,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9947,
          53
        ],
        [
          103,
          9897
        ]
      ],
      "roc_auc": 0.9996577300000001,
      "duration_seconds": 70.62696860000392,
      "accuracy_drop": 0.004149999999999987,
      "accuracy_drop_percentage_points": 0.4149999999999987,
      "f1_drop": 0.004163882589442225,
      "f1_drop_percentage_points": 0.4163882589442225,
      "roc_auc_drop": 0.0002687399999998785
    },
    "JPEG_Q50": {
      "accuracy": 0.98525,
      "precision": 0.9905975128905066,
      "recall": 0.9798,
      "f1": 0.985169171987331,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9907,
          93
        ],
        [
          202,
          9798
        ]
      ],
      "roc_auc": 0.99893779,
      "duration_seconds": 69.90430929999854,
      "accuracy_drop": 0.011099999999999999,
      "accuracy_drop_percentage_points": 1.1099999999999999,
      "f1_drop": 0.011175161729930716,
      "f1_drop_percentage_points": 1.1175161729930716,
      "roc_auc_drop": 0.000988679999999964
    },
    "BLUR_SIGMA1": {
      "accuracy": 0.98885,
      "precision": 0.9974559886028289,
      "recall": 0.9802,
      "f1": 0.9887527109497151,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9975,
          25
        ],
        [
          198,
          9802
        ]
      ],
      "roc_auc": 0.99960162,
      "duration_seconds": 43.46768489999829,
      "accuracy_drop": 0.007499999999999951,
      "accuracy_drop_percentage_points": 0.7499999999999951,
      "f1_drop": 0.0075916227675466574,
      "f1_drop_percentage_points": 0.7591622767546657,
      "roc_auc_drop": 0.00032485000000004316
    },
    "BLUR_SIGMA2": {
      "accuracy": 0.9103,
      "precision": 0.9914949688548155,
      "recall": 0.8277,
      "f1": 0.9022236756049705,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9929,
          71
        ],
        [
          1723,
          8277
        ]
      ],
      "roc_auc": 0.990515345,
      "duration_seconds": 44.97523120000221,
      "accuracy_drop": 0.08604999999999996,
      "accuracy_drop_percentage_points": 8.604999999999997,
      "f1_drop": 0.0941206581122912,
      "f1_drop_percentage_points": 9.41206581122912,
      "roc_auc_drop": 0.009411124999999965
    },
    "NOISE_STD0.01": {
      "accuracy": 0.9943,
      "precision": 0.9956879261933413,
      "recall": 0.9929,
      "f1": 0.9942920088123373,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9957,
          43
        ],
        [
          71,
          9929
        ]
      ],
      "roc_auc": 0.9997976249999999,
      "duration_seconds": 60.619929000005186,
      "accuracy_drop": 0.0020499999999999963,
      "accuracy_drop_percentage_points": 0.20499999999999963,
      "f1_drop": 0.002052324904924463,
      "f1_drop_percentage_points": 0.2052324904924463,
      "roc_auc_drop": 0.00012884500000009957
    },
    "NOISE_STD0.03": {
      "accuracy": 0.9785,
      "precision": 0.9804216867469879,
      "recall": 0.9765,
      "f1": 0.9784569138276553,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9805,
          195
        ],
        [
          235,
          9765
        ]
      ],
      "roc_auc": 0.997513675,
      "duration_seconds": 60.461401100000444,
      "accuracy_drop": 0.01784999999999992,
      "accuracy_drop_percentage_points": 1.7849999999999921,
      "f1_drop": 0.017887419889606404,
      "f1_drop_percentage_points": 1.7887419889606404,
      "roc_auc_drop": 0.002412795000000023
    },
    "RESIZE_112": {
      "accuracy": 0.9823,
      "precision": 0.9963976945244957,
      "recall": 0.9681,
      "f1": 0.9820450395617772,
      "positive_class": "FAKE",
      "class_names": [
        "REAL",
        "FAKE"
      ],
      "confusion_matrix": [
        [
          9965,
          35
        ],
        [
          319,
          9681
        ]
      ],
      "roc_auc": 0.99921374,
      "duration_seconds": 40.97330410000427,
      "accuracy_drop": 0.014050000000000007,
      "accuracy_drop_percentage_points": 1.4050000000000007,
      "f1_drop": 0.014299294155484477,
      "f1_drop_percentage_points": 1.4299294155484477,
      "roc_auc_drop": 0.0007127300000000503
    }
  },
  "v2_vs_v3": {
    "CLEAN": {
      "v2_accuracy": 0.9941,
      "v2_f1": 0.9940905448717948,
      "v3_accuracy": 0.99635,
      "v3_f1": 0.9963443337172617,
      "accuracy_improvement_percentage_points": 0.22499999999999742,
      "f1_improvement_percentage_points": 0.22537888454668975
    },
    "JPEG_Q70": {
      "v2_accuracy": 0.8569,
      "v2_f1": 0.8659359190556493,
      "v3_accuracy": 0.9922,
      "v3_f1": 0.9921804511278195,
      "accuracy_improvement_percentage_points": 13.529999999999998,
      "f1_improvement_percentage_points": 12.624453207217023
    },
    "JPEG_Q50": {
      "v2_accuracy": 0.7634,
      "v2_f1": 0.7336485421591804,
      "v3_accuracy": 0.98525,
      "v3_f1": 0.985169171987331,
      "accuracy_improvement_percentage_points": 22.185,
      "f1_improvement_percentage_points": 25.15206298281506
    },
    "BLUR_SIGMA1": {
      "v2_accuracy": 0.50955,
      "v2_f1": 0.037484054557943285,
      "v3_accuracy": 0.98885,
      "v3_f1": 0.9887527109497151,
      "accuracy_improvement_percentage_points": 47.93000000000001,
      "f1_improvement_percentage_points": 95.12686563917718
    },
    "BLUR_SIGMA2": {
      "v2_accuracy": 0.50005,
      "v2_f1": 0.00019998000199980003,
      "v3_accuracy": 0.9103,
      "v3_f1": 0.9022236756049705,
      "accuracy_improvement_percentage_points": 41.025,
      "f1_improvement_percentage_points": 90.20236956029707
    },
    "NOISE_STD0.01": {
      "v2_accuracy": 0.94895,
      "v2_f1": 0.9478149757219525,
      "v3_accuracy": 0.9943,
      "v3_f1": 0.9942920088123373,
      "accuracy_improvement_percentage_points": 4.535,
      "f1_improvement_percentage_points": 4.647703309038476
    },
    "NOISE_STD0.03": {
      "v2_accuracy": 0.75455,
      "v2_f1": 0.7708323607674712,
      "v3_accuracy": 0.9785,
      "v3_f1": 0.9784569138276553,
      "accuracy_improvement_percentage_points": 22.395,
      "f1_improvement_percentage_points": 20.762455306018413
    },
    "RESIZE_112": {
      "v2_accuracy": 0.50185,
      "v2_f1": 0.007570475146926985,
      "v3_accuracy": 0.9823,
      "v3_f1": 0.9820450395617772,
      "accuracy_improvement_percentage_points": 48.044999999999995,
      "f1_improvement_percentage_points": 97.44745644148503
    }
  }
}
```

**Süre (saniye):** 915.7020566000001

**Önemli gözlem:** {"blur_resize_fake_to_real_counts": {"BLUR_SIGMA1": {"v2_fake_to_real": 9809, "v3_fake_to_real": 198}, "BLUR_SIGMA2": {"v2_fake_to_real": 9999, "v3_fake_to_real": 1723}, "RESIZE_112": {"v2_fake_to_real": 9962, "v3_fake_to_real": 319}}, "largest_accuracy_improvement": "RESIZE_112", "largest_f1_improvement": "RESIZE_112"}

**Sonuç / yorum:** Aynı testte eşleştirilmiş karşılaştırma. Ağırlık/threshold değiştirilmedi. Test V3 augmentation tasarımı öncesinde görülmüştü; tamamen görülmemiş bağımsız test değildir.

**İlgili dosyalar:**

```json
[
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\metrics\\final_test_metrics_v3.json",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\metrics\\robustness_v3.json",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\metrics\\robustness_v3.csv",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\figures\\robustness_v2_vs_v3.png",
  "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\outputs\\figures\\final_test_metrics_v3_confusion_matrix.png"
]
```

<!-- experiment:v3-gradcam-local-demo-20261010 -->
## V3 Grad-CAM ve lokal fotoğraf demosu

**Tarih:** 2026-10-10T15:37:08+03:00

**Durum:** completed

**Model:** V3 Robust EfficientNet-B0

**Checkpoint:** C:\Users\kilic\Desktop\deepfake-image-detection\outputs\checkpoints\best_model_v3_robust.pth

**Dataset:** Hazır fotoğraf test ayrımı

**Train/Val/Test sayıları:**

```json
{
  "train": "kullanılmadı",
  "val": "kullanılmadı",
  "test_examples": 10,
  "REAL": 5,
  "FAKE": 5
}
```

**Amaç:** Sabit V3 tahminlerini Grad-CAM ile açıklamak ve lokal sunum demosu hazırlamak.

**Ayarlar:**

```json
{
  "seed": 42,
  "best_epoch": 5,
  "device": "cuda",
  "image_size": 224,
  "preprocessing": "Ortak image_transform(training=False); EXIF/RGB/bilinear-antialias/ImageNet normalization",
  "selection": "Seed 42; her sınıfta ilk 5 doğru tahmin; confidence sıralaması yok.",
  "learning_rate": "uygulanmaz",
  "epoch": "eğitim yapılmadı",
  "target_layer": "features.8.0",
  "checkpoint_sha256": "c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c",
  "test_fingerprint": "4431a98958594392e4734aef40fc475efca7798a21b3d59c91124cbb3b0955e5"
}
```

**Sonuç:**

```json
{
  "correct_REAL": 5,
  "correct_FAKE": 5,
  "examined": {
    "REAL": 5,
    "FAKE": 5
  },
  "empty_heatmaps": 0,
  "outputs": [
    "original.png",
    "heatmap.png",
    "overlay.png",
    "comparison.png",
    "prediction.json"
  ],
  "examples": [
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\real\\26663.jpg",
      "true_label": "REAL",
      "prediction": "REAL",
      "confidence_percent": 99.96285438537598
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\real\\47131.jpg",
      "true_label": "REAL",
      "prediction": "REAL",
      "confidence_percent": 99.98236298561096
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\real\\51124.jpg",
      "true_label": "REAL",
      "prediction": "REAL",
      "confidence_percent": 99.99526739120483
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\real\\05283.jpg",
      "true_label": "REAL",
      "prediction": "REAL",
      "confidence_percent": 99.9884843826294
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\real\\26700.jpg",
      "true_label": "REAL",
      "prediction": "REAL",
      "confidence_percent": 99.9594509601593
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\fake\\Z1LQ3KT084.jpg",
      "true_label": "FAKE",
      "prediction": "FAKE",
      "confidence_percent": 99.99991655349731
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\fake\\FAYO6BLDAU.jpg",
      "true_label": "FAKE",
      "prediction": "FAKE",
      "confidence_percent": 99.99980926513672
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\fake\\4XLPOENFNA.jpg",
      "true_label": "FAKE",
      "prediction": "FAKE",
      "confidence_percent": 100.0
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\fake\\8YHA16TPTU.jpg",
      "true_label": "FAKE",
      "prediction": "FAKE",
      "confidence_percent": 99.9992847442627
    },
    {
      "image": "C:\\Users\\kilic\\Desktop\\deepfake-image-detection\\data\\real-vs-fake\\test\\fake\\BXHRLT8ROJ.jpg",
      "true_label": "FAKE",
      "prediction": "FAKE",
      "confidence_percent": 99.99985694885254
    }
  ]
}
```

**Süre (saniye):** None

**Önemli gözlem:**

```json
[
  "Confidence sıralaması yok; seçilmiş doğru örnekler genel başarı ölçümü değildir.",
  "Bir REAL örneğinde arka plan/kenar aktivasyonu belirgin; Grad-CAM yüz maskesi veya sahte bölge ground truth değildir.",
  "Orijinal fotoğraflar, mevcut checkpoint ve eski çıktılar korunur."
]
```

**Sonuç / yorum:** Yeni eğitim, checkpoint seçimi veya tuning yapılmadan V3 açıklamaları üretildi. Demo yüz fotoğrafı sınıflandırır; her AI düzenlemesini tespit ettiği iddia edilmez. Süre kalıcı olarak ölçülmedi.

**İlgili dosyalar:**

```json
[
  "app.py",
  ".streamlit/config.toml",
  "scripts/generate_gradcam_v3.py",
  "src/explainability.py",
  "tests/test_demo.py",
  "outputs\\figures\\gradcam_v3\\gradcam_26663_onqcco2k\\selection_summary.json"
]
```

**Demo ve açıklanabilirlik doğrulaması (10 Ekim 2026):** 105 test geçti
(10 yeni demo/seçim kontrolü dahil). Lint, formatter, Python compile,
dependency ve Git diff kontrolleri geçti. Gerçek V3 checkpoint ile bir REAL
ve bir FAKE fotoğrafı, dış bağlantı açılmasını reddeden AppTest ortamında
analiz edildi; prediction/confidence seçim kaydıyla eşleşti ve iki görsel
üretildi. Tarayıcıda REAL tahmini %99,96 güven ve overlay ile doğrulandı;
yerel sağlık endpoint'i `ok` döndürdü. Mevcut 58 checkpoint/sonuç/Grad-CAM
çıktısının SHA-256 özeti değişmedi. Yeni klasörde 40 PNG, 10 prediction JSON
ve bir seçim özeti vardır. Dataset veya checkpoint değişmedi; yeni eğitim
ve test setiyle tuning yapılmadı.

## Video Modeli V1

**Tarih:** 2026-10-12. **Durum:** altyapı ve split hazır; gerçek video
preprocessing/smoke ve dataset eğitimi başlatılmadı. Fotoğraf V3, checkpointler,
metrikler, Grad-CAM ve Streamlit korunur. ResNet18 beklemede.

**Mimari:** ImageNet EfficientNet-B0 frozen özellik çıkarıcı → zamansal GRU
hidden 128 → tek video binary logit/loss. 6 eşit aralıklı yüz frame, 160×160;
ortak OpenCV yüz/crop pipeline. AMP, batch 1–2, seed 42, disk feature cache,
OOM koruması, epoch checkpoint/resume, early stopping ve ETA desteği.
Fine-tuning otomatik başlatılmaz; resmi test model seçiminde kullanılmaz.

| Video ayrımı | REAL | FAKE | Toplam |
|---|---:|---:|---:|
| Train | 438 | 1892 | 2330 |
| Validation | 110 | 472 | 582 |
| Resmi test | 178 | 340 | 518 |
| Test ailesi karantinası | 164 | 2935 | 3099 |

**Veri güvenliği:** Resmi test parser'ı gerçek yerel `<1 REAL / 0 FAKE> <path>`
formatıyla doğrulandı. Frame random split yok. Dosya adı kaynak aileleri
train/validation/test arasında çakışmıyor; testin diğer aile üyeleri dışlandı.
Kaynak/identity metadata yok, aile eşlemesi first-id + clip-index çıkarımı;
doğrulanmış source/identity-disjoint iddiası yok. Train–val 58, train–test 56,
val–test 55 ortak identity kodu mevcut. Video içerik/kopya ve tüm decode
kontrolleri henüz yapılmadı. Manifest: `outputs/video_v1/manifest.json`.

**Ölçümler:** Gerçek video accuracy, precision, recall, F1, ROC-AUC,
confusion matrix, detector coverage, eğitim süresi ve VRAM peak **ölçülmedi**.
Sentetik otomatik test sonuçları model başarımı olarak sunulmaz. Donuk
backbone chunk 2/batch 1–2 için 1–2 GiB bellek bütçesi tahmindir.
Komutlar, split sınırları ve raporlama: [VIDEO_V1.md](VIDEO_V1.md).

**İlk aşama doğrulaması:** Tam proje paketi **145 test geçti** (31 video testi
dahil). Son checkpoint kesinti raporu düzeltmesinden sonra ilgili resume/OOM
testlerinin **2'si tekrar geçti**. Ruff lint ve formatter (36 Python dosyası),
compileall, pip check ve git diff --check başarılı. Sandbox altında Streamlit
testi takıldığı için tam paket izinli ortamda çalıştırıldı; video testleri
workspace içindeki yeni geçici dizinlerde de geçti. Mevcut **309** checkpoint,
sonuç ve görsel dosyasının SHA-256 karşılaştırmasında değişen/eksik dosya **0**.
Kanıt: `outputs/video_v1/photo_preservation.json`. Bu kontroller yapay test
verisiyle gerçekleştirildi; gerçek video preprocessing/smoke, eğitim ve resmi
test değerlendirmesi yapılmadı.

## Video Modeli V1 — sınıf dengesi ve küçük cache ölçümü

**Tarih:** 2026-10-12. **Durum:** eğitim hazırlığı; yalnız 20 REAL + 20 FAKE
train videosunda preprocessing/feature cache ölçümü yapıldı. Tam preprocessing,
GRU dataset eğitimi ve resmi test evaluation başlatılmadı.

Sabit splitler korunur: train 438 REAL / 1892 FAKE; validation 110 REAL /
472 FAKE. Training video-level BCE ağırlıkları yalnız train manifestinden
`N / (2 * sınıf_adedi)` ile hesaplanır: REAL **2,6598173516**, FAKE
**0,6157505285**. Balanced sampling eklenmedi. Validation/test loss/metrik
hesabı ve threshold korunur. Ağırlıklar checkpoint config'e kaydedilir;
farklı loss/config ile resume reddedilir. Frame supervision yoktur.

Küçük cache testi seed 42 ile train'den seçildi. Frozen ImageNet EfficientNet-B0,
6 frame, 160×160, aynı detector/crop pipeline, FP32 chunk 2. Önceki başarılı
4-video smoke ile backbone ağırlık hash'i eşleşir. **40/40** video kabul,
**234/240** geçerli kare; 5 videoda toplam 6 yüz/frame hatası, video reddi
ve OOM yok. 40 yeni cache kaydı üretildi; mevcut dosya ezilmedi.

GTX 1650 4 GB: işlem döngüsü **9,3879015 s**, hazırlık **5,9067806 s**;
Python import süresi dahil değil. PyTorch peak CUDA allocated **29,6411 MiB**,
reserved **38 MiB**; CUDA context/driver/diğer process belleğini ölçmez.
REAL ortalama **0,21429138 s/video**, FAKE **0,253025195 s/video**.
Sabit train dağılımına göre 2330 video tahmini **578,49 s / 9,64 dakika**;
bu küçük örneklemden tahmindir, tam preprocessing süresi henüz ölçülmedi.
Cache-hit süreleri tahminde kullanılmaz; tam komut mevcut 40 kaydı tekrar kullanır.

**Doğrulama:** Tam paket **162 test geçti** (48 video testi). Ruff lint,
formatter (37 Python dosyası), compileall, pip check ve git diff --check başarılı.
316 korunan dosyanın (309 eski çıktı, manifest, resmi liste ve fotoğraf kodu)
SHA-256 karşılaştırmasında değişiklik **0**. Resmi 518 video decode/cache edilmedi;
train/validation manifesti değişmedi. Bunlar model accuracy ölçümü değildir.

**Kanıt:**
`outputs/video_v1/cache_reports/a0fbc022251443ad8d581f41f4b4c1fd/report.json`,
`outputs/video_v1/preparation_verification.json`.
Komutlar: [VIDEO_V1.md](VIDEO_V1.md).

## Final rapor ve sunum paketi

Model geliştirme tamamlandı. Mevcut kayıtların akademik özeti [FINAL_SUMMARY.md](FINAL_SUMMARY.md) dosyasındadır. Sunum görselleri `outputs/figures/final/` altında hazırlanmıştır; kaynak ve görsel hash'leri `package_manifest.json` içinde kayıtlıdır. Bu paket yeni bir model deneyi değildir; eğitim, inference, model seçimi veya tuning yapılmamıştır. Checkpointler ve önceki metrikler korunmuştur. V1 karşılaştırmasının farklı test setinde olması ve V2 gözlemlerinin V3 tasarımına etkisi özette açıkça belirtilmiştir.
