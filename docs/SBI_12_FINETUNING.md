# SBI EfficientNet-B4: 12 kare fine-tuning

14 Ekim 2026. Uzun preprocessing veya eğitim çalıştırılmadı. Hazırlık smoke'u
yalnızca 6 train ve 3 validation videosu içerir. Tam eğitim sonrası başarı henüz
ölçülmedi; smoke sonucu model kalitesi olarak yorumlanmaz.

## Sabit deney koşulları

- Mevcut kabul edilmiş ayrım: 2.223 train (417 REAL / 1.806 FAKE), 602 validation
  (106 REAL / 496 FAKE). Manifest ve kabul edilmiş roster değişmez.
- Örnekleme doğrudan `src.video_frame_study.nested_indices(...)[12]` çağrısıdır.
  Tarihsel altı indeks korunur; en geniş aralığın aşağı yuvarlanmış orta noktası
  eklenir, eşitlikte erken aralık seçilir. Rastgele veya adaptif örnekleme yoktur.
- Resmî RetinaFace, bbox crop, 380×380 RGB uint8 → FP32 / 255 korunur.
  Cache kayıpsız zlib sıkıştırması kullanır; yeniden açılan pikseller aynıdır.
  Validation cache'indeki her yüzün bbox, boyut ve piksel SHA-256'sı tamamlanmış
  frame-study kaydıyla karşılaştırılır. Sapmada işlem durur.
- Yüzsüz/decode edilemeyen kareler açık kaydedilir; başka kareyle doldurulmaz.
  Geçerli kare sayısı üçten azsa işlem durur; kabul edilmiş video sessizce düşürülmez.
  Kısa videodaki eksik kare sayısı cache raporunda görünür. Gerçek smoke'taki dokuz
  videonun her birinde 12 geçerli kare vardır.
- Başlangıç yalnızca resmî FF-raw: SHA-256
  `e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2`.
  Dosya okunur; güncellenmez. FF-c23 veya eski fine-tuning başlangıcı reddedilir.
- Eğitim amacı mevcut sınıf ağırlıklı, video başına eşit ağırlıklı kare CE'sidir.
  Eğitimde her karede en büyük yüz; validation'da korunmuş yüzlerin maksimum
  FAKE olasılığı, ardından karelerin aritmetik ortalaması kullanılır. Eşik 0,5.
- Frame batch=1, accumulation=4 video (tam grupta 48 ardışık backward), FP32.
  Son üç videoluk grupta doğru payda 3'tür. AMP kullanılmaz. İlk epoch yalnızca
  head; ikinci epoch'tan itibaren blok 24 ve sonrası, projection ve head açılır.
  BatchNorm sabit; dropout/drop-connect kapalı. AdamW head LR=1e-4, backbone
  LR=1e-5, weight decay=0,01; gradient clip=1. Varsayılan bütçe 8 epoch.
- Early stopping validation Macro-F1 üzerindedir: patience=3, min_delta=0.
  Eşit skor iyileşme sayılmaz. En iyi fine-tuning checkpoint'i ayrı saklanır;
  zero-shot baseline'ın yerini otomatik almaz.
- Resmî test/quarantine video içerikleri kullanılmaz. Görüntü V3, eski video
  çalışmaları ve 6/12/24/32 frame-study çıktıları korunur. Adaptif 6→12 ayrı kalır.

## A — Train ve validation cache hazırlama

Proje klasöründe PowerShell:

```powershell
.\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_sbi12 --prepare-cache --resume auto
```

Bu komut uzun işlemi **kullanıcı çalıştırınca** başlatır: önce 2.223 train,
ardından gerekli 602 validation videosu hazırlanır. Ayrı konum:
`outputs/video_sbi12_v1/crops/<protokol-hash>/`. Eski altı-kare cache'i yazılmaz.

Kesintide **aynı komutu tekrar çalıştırın**. Tamamlanmış her video source SHA-256,
record, protokol, indeks, metadata ve piksel hash'iyle doğrulanıp atlanır.
Yarım atomik dosya tamamlanmış video sayılmaz; o video yeniden işlenir.
Bozuk/tutarsız cache otomatik silinmez veya üstüne yazılmaz; hata ile durur.
İşletim sistemi kilidi eşzamanlı iki cache yazıcısını engeller; süreç kapanınca
kilit serbest kalır, `.lock` dosyasını elle silmek gerekmez.

Dokuz videoluk soğuk pilot: 4,82–6,51 sn/video; tam hazırlık nokta tahmini
**4,05 saat**, pratik planlama **3,5–5,5 saat**. Cache yaklaşık **10,55 GiB**.
Süre ve boyut küçük pilotun ekstrapolasyonudur; video/yüz sayısına göre değişir.
Kaynak videolar, eski sonuçlar veya checkpoint'ler alan açmak için silinmez.

## B — SBI 12 kare fine-tuning

A başarıyla bittikten sonra, aynı proje klasöründe:

```powershell
.\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_forensic_train --run-training --frames 12 --run-dir outputs/video_research_v1/finetune12/main_20261014 --epochs 8 --seed 42 --resume auto
```

`--resume auto` yeni dizinde FF-raw'dan başlar; mevcut dizinde en son atomik,
tamamlanmış `epoch_NNN.pth` dosyasından devam eder. Başlamadan train/validation
kaynakları ve tüm crop cache'i doğrulanır. Eksik cache varsa eğitim başlamaz;
bu komut hiçbir zaman otomatik crop çıkarmaz.

Kesintide **aynı komutu tekrar çalıştırın**. Model, AdamW momentleri, RNG,
shuffle generator, history, en iyi epoch ve early stopping yüklenir. Yarım
epoch son tamamlanmış epoch'tan yeniden oynatılır; batch ortasından devam yoktur.
İlk epoch tamamlanmadan kesildiyse aynı config ile FF-raw'dan epoch 1 tekrarlanır.
Kod, veri, runtime, örnekleme veya eğitim ayarı değişmişse resume reddedilir.
Smoke dizinleri eğitim başlangıcı olarak kullanılamaz. Early stopping zaten
gerçekleşmişse yeniden çalıştırmak ek epoch başlatmaz.

İki kısa CUDA pilotu cache okuma ve validation dahil **28,52 ve 35,53 dk/epoch**
tahmini verdi. Sekiz epoch için nokta tahminleri **3,80–4,74 saat**, pratik
planlama **30–45 dk/epoch, toplam 4–6 saat**;
early stopping süreyi kısaltabilir. Başlangıç bütünlük taraması ve checkpoint
yazımı ayrıca süre alır. Sekiz epoch + best/last kopyaları yaklaşık **1,57 GiB**;
atomik yazım sırasında ek geçici alan gerekir. Cache + checkpoint için yaklaşık
13 GiB ve işletim sistemine ek boş alan ayırın.

## Raporlar ve kıyas

Her epoch `epoch_NNN_validation.json` ve `history.json` içinde accuracy,
Macro-F1, ROC-AUC, REAL/FAKE recall, confusion matrix, kapsam, kaynak başına
recall/hata ve uygun REAL kaynakları için Celeb-synthesis'e karşı AUC kaydedilir.
Tek sınıflı Celeb-synthesis klasörü için tek başına AUC tanımlanmaz.

`baseline_comparison.json` tüm epoch'ların farklarını ve validation Macro-F1 ile
seçilen epoch'u gösterir. Rapor yazılmadan kesinti olursa son epoch checkpoint'i
resume sırasında raporları yeniden oluşturur. `best_video.pth` ve
`last_video.pth`, `checkpoints/` altındadır; her epoch dosyası ayrıca korunur.

Referans `outputs/video_frames_v1/full.json` içindeki aynı 602 videoluk 12 kare
zero-shot sonucudur; yuvarlanmış değerler yerine tam hassasiyet kullanılır:

| Metrik | Zero-shot 12 |
|---|---:|
| Accuracy | 0,9202657807 |
| ROC-AUC | 0,9495397139 |
| Macro-F1 | 0,8615694165 |
| REAL recall | 0,7641509434 |
| FAKE recall | 0,9536290323 |

Fine-tuning henüz yapılmadığından daha iyi sonuç iddiası yoktur. Aynı validation
hem kare seçimi hem model seçimi için kullanılmıştır; bu kıyas bağımsız test
başarısı değildir.

## Kısa doğrulama kanıtı

İlk gerçek smoke raporu:
`outputs/video_research_v1/finetune12/smoke_20261014/smoke.json`.
GTX 1650 üzerinde dört optimizer update; hem head hem açılan MBConv katmanları
güncellendi. Dondurulmuş parametreler ve BN buffer'ları değişmedi, gradient'ler
sonlu kaldı. Gerçek model/optimizer/RNG checkpoint replay'i bit düzeyinde aynıydı.
Üç validation videosunun 12 kare olasılıkları eski zero-shot ile delta=0 verdi;
crop pikselleri ve bbox'lar da eşitti. Tail aşaması tepe CUDA allocated
392,22 MiB / reserved 436 MiB ölçüldü; bunlar sistemin toplam GPU kullanımı değildir.

Son kodla ikinci smoke:
`outputs/video_research_v1/finetune12/smoke_final_20261014/smoke.json`.
Dokuz cache'in tamamı yeniden çıkarılmadan doğrulanıp kullanıldı. İki kısa smoke
toplam sekiz optimizer update yaptı; tam eğitim epoch'u başlatılmadı.

Tam regresyon **354 test başarılı**: `outputs/video_sbi12_v1/regression.xml`.
Ruff lint, 73 dosya format kontrolü, compileall ve git diff kontrolü geçti.
Koruma hash'leri
`outputs/video_sbi12_v1/protected_before.json` altındadır. Yeni testler kayıpsız
cache, kesintili atomik yazma, cache yeniden kullanımı, eksik cache'de durma,
12 kare/1–3–4 video accumulation matematiği, ilk epoch resume, checkpoint kontrol
durumu, early stopping ve üretim epoch döngüsünün kesintiden kurtarılmasını kapsar.
