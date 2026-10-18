# Derin Öğrenme ile Deepfake Yüz Görüntülerinin Tespiti

## 1. Problem

Bu çalışma, yüz ağırlıklı fotoğrafları REAL ve FAKE olarak sınıflandıran bir
derin öğrenme sistemi geliştirmeyi amaçlar. Etiket sırası `0 = REAL`,
`1 = FAKE` olarak sabittir. Precision, recall, F1 ve ROC-AUC için pozitif
sınıf FAKE'tir. Sistem genel amaçlı AI-generated image detector değildir.

## 2. Datasetler

| Dataset | Train | Validation | Test | Toplam |
|---|---:|---:|---:|---:|
| İlk fotoğraf dataseti / V1 | 9.023 | 1.933 | 1.934 | 12.890 |
| 140k Real and Fake Faces / V2–V3 | 100.000 | 20.000 | 20.000 | 140.000 |

İlk dataset 5.890 REAL ve 7.000 FAKE görüntü içerir. Bilinen augmentation
aileleri ve birebir kopyalar aynı ayrımda tutularak seed 42 ile yaklaşık
%70/%15/%15 grup ayrımı yapılmıştır. Kaynak/kişi kimlikleri eksik olduğundan
tam kişi bağımsızlığı kanıtlanamamıştır.

İkinci datasetin hazır train/valid/test ayrımları değiştirilmemiştir. Her
ayrım %50 REAL / %50 FAKE dengelidir: train 50.000+50.000; validation ve
test 10.000+10.000. V2 final test kaydı, test içinde sıfır birebir SHA-256
kopyası ve test–train/validation arasında sıfır SHA-256 çakışması bildirir.
Bu kontrol yakın kopya veya kişi düzeyinde bağımsızlık kanıtı değildir.
Datasetler `data/raw/` ve `data/real_vs_fake/` yollarıyla kullanılır;
yerelde ikinci klasörün mevcut adı `data/real-vs-fake/`tir.

## 3. EfficientNet-B0

Üç model de torchvision EfficientNet-B0 mimarisini ve iki çıkışlı son
sınıflandırma katmanını kullanır. Standart CLEAN dönüşümü EXIF yön düzeltme,
RGB dönüşümü, bilinear/antialias 224×224 yeniden boyutlandırma ve ImageNet
normalizasyonudur. Eğitim NVIDIA GTX 1650 4 GB üzerinde batch size 8 ve
mixed precision ile gerçekleştirilmiştir.

## 4. Transfer learning

Modeller ImageNet `IMAGENET1K_V1` pretrained ağırlıklarıyla başlatılmış;
son classification katmanı REAL/FAKE için uyarlanmıştır. AdamW, başlangıç
learning rate `3e-4`, seed 42 ve Windows uyumlu workers 0 kullanılmıştır.
Checkpoint seçimi yalnızca CLEAN validation FAKE F1 üzerinden yapılmıştır.

## 5. V1

İlk fotoğraf datasetinde eğitilen baseline'ın best epoch'u 5'tir. Early
stopping epoch 8'de eğitimi durdurmuştur. Ayrılmış 1.934 test görüntüsünde
accuracy %95.450, F1 %95.712 ve ROC-AUC 0.99185736 elde edilmiştir.
Bu sonuç ilk datasetin test ayrımıyla sınırlıdır.

## 6. V2

140k Real and Fake Faces ile eğitilen V2'nin best epoch'u 3'tür. Dört epoch
eğitim logu vardır; bu çalıştırma early stopping olarak raporlanmaz.
20.000 CLEAN test görüntüsünde accuracy %99.410 ve F1 %99.409 elde edilmiştir.

| Ölçü | V1 | V2 | V3 |
|---|---:|---:|---:|
| Best epoch | 5 | 3 | 5 |
| Accuracy (%) | 95.450 | 99.410 | 99.635 |
| Precision FAKE (%) | 98.004 | 99.569 | 99.789 |
| Recall FAKE (%) | 93.524 | 99.250 | 99.480 |
| F1 FAKE (%) | 95.712 | 99.409 | 99.634 |
| ROC-AUC | 0.99185736 | 0.99825422 | 0.99992647 |

V1 farklı bir datasetin test ayrımında ölçülmüştür. Bu tablo ortak benchmark
veya mimari/dataset büyüklüğünün nedensel etkisi olarak yorumlanamaz.
V2 ve V3 aynı 20.000 test görüntüsünde değerlendirilmiştir.

## 7. Robustness problemi

V2'nin yüksek CLEAN başarısı kalite bozulmalarına dayanıklılık anlamına
gelmemiştir. Blur sigma 1/2 ve 224→112→224 resize altında accuracy yaklaşık
%50'ye düşmüş, FAKE örnekler büyük ölçüde REAL olarak sınıflandırılmıştır.
Yüksek frekanslı görsel ipuçlarına duyarlılık olası bir açıklamadır;
gözlem tek başına nedensel kanıt değildir.

## 8. V3 augmentation yaklaşımı

V3 Robust EfficientNet-B0, aynı mimari ve dataset üzerinde yalnızca TRAIN
splitinde kalite augmentation'larıyla eğitilmiştir. Validation ve final test
CLEAN kalmıştır.

| Augmentation | Olasılık | Aralık |
|---|---:|---|
| JPEG compression | %50 | Quality 50–95 |
| Gaussian blur | %15 | Sigma 0.1–1.5 |
| Gaussian noise | %15 | RGB [0,1], std 0–0.02 |
| Downscale/upscale | %30 | 224→rastgele 112–224→224 |
| Horizontal flip | %50 | Yatay çevirme |

Maksimum 6 epoch ve early stopping patience 2 kullanılmış; altı epoch
tamamlanmıştır. Best epoch 5'tir; epoch sınırında patience dolmadığından
duruş early stopping değildir.

### V2 → V3 robustness

| Koşul | V2 accuracy (%) | V3 accuracy (%) | Artış (yüzde puan) |
|---|---:|---:|---:|
| CLEAN | 99.410 | 99.635 | +0.225 |
| JPEG Q70 | 85.690 | 99.220 | +13.530 |
| JPEG Q50 | 76.340 | 98.525 | +22.185 |
| Blur sigma 1 | 50.955 | 98.885 | +47.930 |
| Blur sigma 2 | 50.005 | 91.030 | +41.025 |
| Noise 0.01 | 94.895 | 99.430 | +4.535 |
| Noise 0.03 | 75.455 | 97.850 | +22.395 |
| Resize 224→112→224 | 50.185 | 98.230 | +48.045 |

En büyük accuracy artışı resize koşulunda +48.045 yüzde puandır. Blur sigma 1
ve sigma 2 altında artış sırasıyla +47.930 ve +41.025 yüzde puandır; sigma 2
hâlâ en zor koşuldur. FAKE→REAL hataları resize için 9.962→319, blur sigma 1
için 9.809→198 ve blur sigma 2 için 9.999→1.723 olarak azalmıştır.

## 9. Final test sonuçları

**V3, 140k Real and Fake Faces test setinde %99.635 accuracy elde etmiştir.**
Bu ölçüm 10.000 REAL ve 10.000 FAKE olmak üzere 20.000 CLEAN test görüntüsüne
aittir; tüm deepfake görüntüler için genelleme iddiası değildir.

| Ölçü | V3 final |
|---|---:|
| Accuracy | %99.635 |
| Precision FAKE | %99.7893 |
| Recall FAKE | %99.480 |
| F1 FAKE | %99.6344 |
| ROC-AUC | 0.99992647 |
| Doğru / toplam | 19.927 / 20.000 |
| Yanlış | 73 |
| Best epoch | 5 |

Confusion matrix, satır gerçek / sütun tahmin ve sıra REAL, FAKE:
`[[9979, 21], [52, 9948]]`. Ana model
`outputs/checkpoints/best_model_v3_robust.pth` dosyasıdır. SHA-256:
`c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c`.

## 10. Grad-CAM

Grad-CAM, tahmin edilen sınıfın softmax öncesi logitini ve son convolution
katmanını (`features.8.0`) kullanır. Harita sınıf kararına uzamsal katkıları
gösterir; manipüle edilmiş piksellerin ground-truth segmentasyonu değildir.
Final pakette önceki seed-42 doğru örnek sırasından ilk iki REAL ve ilk iki
FAKE açıklaması yer alır; confidence'a göre sıralama yapılmamıştır.
Seçilmiş doğru örnekler genel model başarısını veya nedensel karar gerekçesini
kanıtlamaz. Gösterilen confidence, kalibre edilmemiş softmax skorudur.

## 11. Sınırlılıklar

- Başarı belirli fotoğraf datasetleri ve ölçülen kalite koşullarıyla sınırlıdır;
  bağımsız fotoğraf kaynaklarında aynı başarı garanti edilmez.
- V2 test/robustness gözlemleri V3 augmentation tasarımına yön vermiştir.
  Test eğitim veya checkpoint seçiminde kullanılmamıştır; buna rağmen V3
  sonuçları geliştirme boyunca hiç görülmemiş bağımsız test olarak sunulamaz.
- Bilinen hash çakışmalarının olmaması kişi/kaynak bağımsızlığını kanıtlamaz.
  İkinci datasetin kaynak URL, sürüm ve lisans künyesi yerel kayıtlarda yoktur.
- Demo, eğitim dağılımına benzer yüz ağırlıklı görüntüleri sınıflandırır.
  Genel amaçlı AI-generated image detector değildir; görüntünün tamamı
  standart CLEAN pipeline'dan geçirilir.
- Loglarda train accuracy ve validation loss bulunmadığından bu eğriler
  üretilmemiştir; mevcut train loss ve validation metrikleri çizilmiştir.

## 12. Sonuç

V3, 140k Real and Fake Faces test setindeki CLEAN başarısını korurken ölçülen
sıkıştırma, blur, noise ve resize koşullarında V2'ye göre daha yüksek başarı
elde etmiştir. Çalışmanın ana katkısı yalnızca yüksek CLEAN accuracy değil,
aynı test fotoğraflarında kalite dayanıklılığının iyileştirilmesidir.
Model geliştirme tamamlanmıştır; bu paket hazırlanırken yeni eğitim,
inference, model seçimi veya tuning yapılmamıştır.

## Kaynak kayıtları ve sunum dosyaları

Sayısal kaynaklar `outputs/metrics/final_test_metrics*.json`,
`robustness_v2.json`, `robustness_v3.json` ve üç mevcut `history.csv` dosyasıdır.
Özet [README](../README.md) ve [deney günlüğü](EXPERIMENTS.md) ile uyumludur.
Sunum görselleri `outputs/figures/final/` altında 10 PNG ve 6 SVG olarak
hazırdır; kaynak/file hash'leri `package_manifest.json` içinde bulunur.
Görseller üretilmiş çıktı oldukları için Git dışında tutulur; rapor/sunum
tesliminde bu klasör ayrıca paylaşılmalıdır.
