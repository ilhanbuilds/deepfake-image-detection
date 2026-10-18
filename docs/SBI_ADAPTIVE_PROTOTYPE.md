# SBI epoch 6: kontrollü adaptif 6→12 prototipi

15 Ekim 2026. Ek eğitim yapılmadı. Ana uygulama ve eski inference modülleri
değişmedi. Resmî test video/listesi açılmadı. Uzun benchmark çalıştırılmadı.

## Sabit aday kural ve metodolojik kapsam

Doğrulanmış model:
`outputs/video_research_v1/finetune12/main_20261014/checkpoints/best_video.pth`.
SHA-256: `a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c`.

Tarihsel altı indeks, eski `sample_indices(..., 6)` ve frame-study
`nested_indices` fonksiyonlarıyla korunur. Kare skoru korunmuş yüzlerin maksimum
FAKE olasılığı, video skoru geçerli karelerin aritmetik ortalamasıdır. İlk altı
karenin ortalaması p6; `|p6−0,5| < 0,10` ise ek altı kare işlenir ve 12 karenin
ortalaması kullanılır. Diğer durumda p6 ile karar verilir. Karar eşiği 0,5.
Kod, kesin sınır dışlaması için eşdeğer `0,4 < p6 < 0,6` aralığını kullanır;
0,4/0,6 değerlerinde binary çıkarma yuvarlamasıyla yanlış genişletme önlenir.
CLI'da ayarlanabilir gate eşiği veya eşik taraması yoktur.

**0,10 bu validation bulgularından seçilmiştir.** Offline sonucun iyi görünmesi
bağımsız başarı kanıtı değildir. Validation kare bütçesi, epoch ve aday gate
seçiminde kullanıldı. Gerçek kişi/donor kaynak klibi ve YouTube kişi/kanal/session
bağımsızlığı doğrulanmadı; celebrity validation tek bağlı bileşende 10 kod içerir.
Dataset-source shortcut riski sürer. Ayrı kalibrasyon ve bağımsız değerlendirme
öncesi model/eşik değiştirilmemeli; resmî test sonucuna bakarak yöntem ayarlanmamalı.

## Offline simülasyon

Çıktı: `outputs/video_adaptive_v1/offline_20261015/offline.json`.
Yalnızca kayıtlı epoch 6 frame-level skorları kullanıldı; 602/602 video ve
6/12 sabit referansların tam eşleşmesi doğrulandı. Yeni inference yoktur.

| Metrik | Sabit 6 | Sabit 12 | Adaptif aday |
|---|---:|---:|---:|
| Doğru / 602 | 589 | 595 | 596 |
| Accuracy | 0,97840532 | 0,98837209 | 0,99003322 |
| ROC-AUC | 0,99600578 | 0,99830721 | 0,99610088 |
| Macro-F1 | 0,96292454 | 0,97988765 | 0,98269618 |
| Balanced accuracy | 0,96464166 | 0,97810788 | 0,97911595 |
| REAL recall | 0,94339623 | 0,96226415 | 0,96226415 |
| FAKE recall | 0,98588710 | 0,99395161 | 0,99596774 |
| Yanlış alarm | 6 | 4 | 4 |
| Ortalama işlenen kare | 6 | 12 | 6,16943522 |
| Medyan işlenen kare | 6 | 12 | 6 |
| 12 kare kullanılan video | 0 | 602 | 17 |

Adaptif genişletme 17/602 = %2,82392. Adaptif confusion matrix
`[[102,4],[2,494]]`; satır/sütun sırası REAL/FAKE. Adaptif aday sabit 12'den
daha yüksek AUC vermiyor; adayın AUC'si belirgin biçimde daha düşük.
Kare sayısı azalması ölçülmüş gecikme/hızlanma değildir. Sonuçlar aynı validation
üzerinde seçilmiş adayın keşif amaçlı simülasyonudur.

Tekrar üretim (yeni run-dir gerekir):

```powershell
.\outputs\video_research_v1\env\Scripts\python.exe -B -u -m scripts.benchmark_video_adaptive --offline --run-dir outputs/video_adaptive_v1/offline_repeat
```

## Gerçek video akışı

Yeni modül `src/video_adaptive.py`; komut arayüzü
`scripts/benchmark_video_adaptive.py`. Eski SBI preprocessing fonksiyonu
`src.video_frame_study.process_frame` doğrudan kullanılır: resmî RetinaFace,
resmî bbox crop, en büyük alanın en az yarısına sahip yüzlerin korunması,
380×380 RGB/255, FP32, face batch=1 ve karede maksimum yüz skoru.

Adaptif yol ilk geçişte yalnız tarihsel altı kareyi detector/model'den geçirir.
Genişletme olursa video ikinci kez sıralı decode edilir, yalnız ek indeksler
retrieve/detect/classify edilir. İlk altının sonuçları video çağrısı içindeki
RAM sözlüğünde tutulur; aynı indeksin başarılı veya başarısız işlemi tekrar
denenmez. Videolar/model karşılaştırmaları arasında skor cache'i paylaşılmaz.
Tarihsel cache'lere yazılmaz; yeni crop cache'i oluşturulmaz.

İlk altı indeks videonun son karesini de içerir; ilk geçiş zaten tüm zaman
aralığını dolaşır. İkinci geçiş ek decode maliyeti getirir. On iki ham kareyi
veya tüm videoyu RAM'de tutmak yerine tek decode edilmiş kare ve geçici yüz
crop'ları kullanılır. GPU'da model ve detector bulunur; classifier frame batch=1.
Sabit 12 referansı adil karşılaştırma için **tek** sıralı decode geçişi kullanır.

Güvenli özel durumlar:

- Altı benzersiz kareden kısa video açıkça reddedilir; kare çoğaltılmaz.
- 6–11 kareli videoda mevcut benzersiz kareler kullanılır; 12 bütçesinin eksiği
  `budget_shortfall` ve uyarı ile bildirilir. 12 kareye tamamlandığı iddia edilmez.
- İlk altıda yüz/decode eksikliği varsa, marjdan bağımsız güvenlik geri dönüşü
  olarak mevcut ek kareler denenir. Bu, eksiksiz 602 videoluk offline kuralı
  değiştirmez; gerçek eksik veride kapsam kaybını görünür yönetir.
- Üçten az geçerli yüzlü kare varsa karar/olasılık üretilmez; video reddedilir.
  En az üç geçerli kare varsa yalnız bunların ortalaması kullanılır ve eksik
  kare uyarıları korunur. Nötr skor veya başka kareyle doldurma yoktur.
- Sistemsel CUDA/provenance/nonfinite hataları sessizce yutulmaz; çalışma hata
  kaydıyla durur. Dosya açılmama/retrieve/EOF sorunları kare veya video raporudur.

Tek kullanıcı videosu için örnek (kullanıcının gerçek video yolu verilmelidir):
`--predict` modu bir dosya alır. Dataset kökü içinde yalnız mevcut kabul edilmiş
train/validation yollarına izin verir; test/quarantine yollarını decode öncesi
reddeder. Dataset dışındaki kullanıcı videosu için de aynı pipeline çalışır.

## Gerçek pilot benchmark komutu — henüz çalıştırılmadı

Offline çalışmada kaydedilen `pilot_plan.json`, üç kaynaktan SHA256(path) sırasıyla
üçer video seçer: **9 video, 6 REAL / 3 FAKE**. Skor/hata durumuna göre seçim
yapılmaz. Plan benchmark öncesi yeniden doğrulanır; farklı liste kabul edilmez.
Bu kaynak dengeli küçük pilot, 602 videonun sınıf dağılımını temsil etmez.

Proje klasöründe PowerShell:

```powershell
.\outputs\video_research_v1\env\Scripts\python.exe -B -u -m scripts.benchmark_video_adaptive --benchmark --plan outputs/video_adaptive_v1/offline_20261015/pilot_plan.json --repeats 3 --run-dir "outputs/video_adaptive_v1/benchmark_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

Bu komut kullanıcı çalıştırınca 9 video × 3 yöntem × 3 tekrar = **81 kısa video
çalıştırması** yapar. Tam cohort modu yoktur. Tekrar sınırı 1–5'tir. Önceki
pipeline maliyetleriyle birkaç dakikalık iş beklenir; bu bir ölçüm veya garanti
değildir. Çıktı dizini mevcutsa üzerine yazmaz.

Model+detector yükleme ve sentetik ısınma süreleri `environment.json` içinde
ayrı kaydedilir. Sentetik ısınma üç classifier forward ve bir detector çağrısıdır;
herhangi bir test videosu açmaz. Kaynak byte hash doğrulama preflight'ı ayrı
raporlanır. Video başına süre video açma, decode, detector, crop, classifier,
gate ve varsa ikinci decode geçişini içerir; model yükleme, ısınma, preflight ve
JSON yazımı dışında tutulur. Senkronizasyon ve Python işlem giderleri süreye dahildir.

Her video/tekrarda yöntem sırası döndürülür; her yöntem üç tekrarda bir kez ilk
sıraya gelir. Yöntemler arasında crop/skor paylaşılmaz. Dosya sistemi cache'i ve
CUDA allocator geçmişi tamamen izole değildir; tekrarlar ısınmış ortam ölçümleridir.

`benchmark.json` doğruluk için ilk tekrarın **9 benzersiz videosunu** kullanır.
Tekrarlar yalnız gecikme ve bellek dağılımını ölçer; doğruluk örneklemini 27 diye
şişirmez. Tekrarlar arasında karar/skor/rota değişirse işlem durur. Raporlar:
accuracy, AUC, Macro-F1, REAL/FAKE recall, yanlış alarm, kaynak sonuçları, kapsam,
ortalama/medyan işlenen kare, genişletme oranı, ortalama/medyan/p95 gerçek gecikme,
decode süre/geçiş sayıları, yüz sınıflandırma çağrıları ve GPU bellek tepeleri.
GPU allocated/reserved PyTorch süreç allocator değerleridir; tüm sistem GPU
belleği veya ayrı allocator benchmark'ı değildir. Başarısız/eksik videolar
raporda kalır; ortak tam-kapsam doğruluğu varmış gibi sunulmaz.

## Tamamlanan kısa CUDA smoke

`outputs/video_adaptive_v1/smoke_20261015/`: GTX 1650 4 GB üzerinde iki bilinen
branch örneği; bu seçim doğruluk veya tarafsız hız pilotu değildir.

- `Celeb-real/id45_0001.mp4`: 12 farklı kare, iki decode geçişi; genişledi.
- `Celeb-synthesis/id47_id42_0001.mp4`: 6 farklı kare, tek geçiş; erken durdu.
- Her iki videoda kare skor farkı **0**, resmî crop pikselleri/bbox'lar birebir aynı.
- Model+detector yükleme 3,094 sn; sentetik ısınma 1,348 sn.
- Bu iki farklı videoda süreler sırasıyla 5,778 ve 2,791 sn. Bunlar eşleşmiş
  yöntem benchmark'ı değildir; hızlanma oranı olarak kullanılamaz.
- Tepe CUDA allocated 708,32 MiB; reserved 1114 MiB. Eğitim/backward yapılmadı.

Üç koruma katmanı vardır: sabit checkpoint hash'i, eski preprocessing/eğitim
kodu hash'lerinin kontrolü ve çalışma öncesi/sonrası korunan girdi hash'leri.
Checkpoint, manifest, eski sonuç ve ana uygulama dosyaları aynı kaldı.

## Son kontroller

388/388 proje testi geçti; çıktı `outputs/video_adaptive_v1/regression.xml`.
Yeni testler strict gate sınırlarını, video başına tek frame işlemini, ikinci
decode maliyetini, eksik yüz/EOF/retrieve/çoklu yüz/kısa video durumlarını,
offline referans eşleşmesini ve test/quarantine yol kilidini kapsar. Benchmark
döngüsü sentetik decoder ile test edildi: doğruluk 9 benzersiz video, süre 27
tekrar üzerinden raporlanıyor; gerçek pilot bu test sırasında çalıştırılmadı.
Ruff lint, 76 dosya format kontrolü, compileall ve git diff whitespace kontrolü geçti.
