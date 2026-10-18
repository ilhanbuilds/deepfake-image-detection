# Kilitli SBI nihai video testi hazırlığı

18 Ekim 2026. Bu aşamada **resmî test çalıştırılmadı**. Test listesi, bireysel test
etiketleri veya test videoları açılmadı. Yeni eğitim, validation optimizasyonu
veya gerçek GPU inference yapılmadı. Birim testleri sentetik verilerle çalıştı.

## Sabit protokol

- Model: SBI EfficientNet-B4, fine-tuning epoch 6.
- Checkpoint: `outputs/video_research_v1/finetune12/main_20261014/checkpoints/best_video.pth`.
- Checkpoint SHA-256: `a4d8232ed450ce7f09ca8fc82298a40fad4af92c425e43de0798d055c01ab98c`.
- Yöntemler: tarihsel sabit 6, iç içe sabit 12 ve adaptif 6→12.
- Adaptif aday: `|p6−0,5| < 0,10`; eşdeğer strict aralık `0,4 < p6 < 0,6`.
- Karar eşiği 0,5. Karede korunmuş yüzlerin maksimum FAKE olasılığı; video için
  geçerli karelerin aritmetik ortalaması.
- Resmî RetinaFace/SBI crop, 380×380 RGB/255, FP32, face batch=1.
- İlk altıda yüz/decode eksikliği varsa önceki prototipin güvenlik geri dönüşü
  korunur: mevcut ek kareleri dene; skor doldurma veya aynı kareyi tekrar sayma.
  Altı benzersiz kareden kısa video reddedilir; 6–11 karede eksik bütçe görünürdür.
  Üçten az geçerli yüzlü karede karar yoktur. Eksikler sessizce cohort'tan düşürülmez.

Konfigürasyon:
`outputs/video_final_v1/protocol_20261018/protocol.json`.

**Protokol SHA-256:**
`a14a2ae31482caf1b57afb4bc621f9f84a0c3a5b06d822c90e4b339d4a9f7eab`

Dosya yalnız bir kez oluşturulur. Komut yukarıdaki dışarıdan sabitlenmiş hash'i
doğrular; konfigürasyon, kaynak kodu, runtime paketleri, checkpoint, preprocessing
varlıkları veya manifest değişirse evaluation/resume reddedilir. Bu, hash ile
değişiklik tespiti ve yazılım düzeyinde kilittir; yöneticinin dosya sistemini
değiştiremeyeceğine ilişkin fiziksel write-once iddiası değildir.

Hazırlık yalnız manifestin train/validation bölümünü okur ve ilk test kaydından
önce durur. Manifest dosyalarının tamamı içerikleri ayrıştırılmadan opaque
SHA-256 için okunur; test listesi dosyasının kendisi açılmaz. Beklenen resmî liste
hash'i önceden var olan manifest başlığından alınır. Test sınıf dağılımı veya
performansı hazırlık raporuna çıkarılmaz.

## Bilinen engel ve zorunlu denetim

Eski `outputs/video_identity_v1/REPORT.md` kimlik denetimi, aynı resmî testin
train/validation ile kimlik kodu bakımından ayrık olmadığını zaten bildiriyor.
Bu hazırlıkta test kodları/etiketleri yeniden incelenmedi. Sabit checkpoint veya
split değiştirilmedi; örtüşme riski otomatik kabul edilmedi.

Kullanıcı nihai komutu çalıştırınca, **model yüklenmeden önce**:

1. Resmî liste hash'i, benzersiz tam 518 üyelik ve kilitli manifest üyeliği doğrulanır.
2. Test ile tüm eski train/validation kayıtları karşılaştırılır: dosya yolu,
   çözülmüş/case-insensitive yol, filename kaynak ailesi ve first/second/donor
   rol ayrımı olmadan herhangi bir kimlik kodu.
3. Bunlardan biri örtüşürse `leakage_blocked.json` yazılır, çıkış kodu **3** ile
   durulur. Bu aşamada MP4 içerikleri açılmaz, etiketler hedeflere dönüştürülmez,
   model/GPU yüklenmez ve test metriği üretilmez.
4. Yalnız metadata denetimi geçerse train/validation source SHA-256'ları eğitim
   cache'iyle doğrulanır; tarihsel cache fingerprint'i kontrol edilir. Test
   kaynakları kilitli envanter boyut/mtime bilgileriyle ve tam MP4 SHA-256 ile
   denetlenir. Development/test veya test içi birebir içerik tekrarları durdurur.
5. İlk başarılı hash denetimi mühürlenir. Resume yeni test byte'larını, metadata
   değişikliğini veya farklı protokolü kabul etmez. Ancak bundan sonra hedef
   etiketler dönüştürülür ve model yüklenebilir.

**Bu kilitli manifestteki eski örtüşme doğrulanırsa mevcut 518 set üzerinde skor
üretilmesi beklenmez.** Bypass, risk kabulü veya test alt kümesi seçme bayrağı
yoktur. Resmî holdout olmak gerçek kişi/kaynak bağımsızlığı garantisi değildir.
Hash'ler yeniden kodlanmış/ortak video segmentlerini dışlayamaz. Gerçek kişi
alias'ları, YouTube kaynakları ve pretraining kaynakları ayrıca doğrulanmış değildir.

## Ortak inference cache'i ve süre yorumu

Yeni çıktı kökü `outputs/video_final_v1/final_test_20261018/`.
Önce altı kare işlenir; p6, sabit 6 ve adaptif erken-durma sonucu elde edilir.
Ardından sabit 12 karşılaştırması için kalan kareler her video için bir kez
işlenir. Adaptif genişletilecekse bu aynı ek skorları kullanır. Yeni, atomik ve
mühürlü JSON cache checkpoint/protokol hash'i, kaynak hash'i, video yolu ve indeks
ile bağlıdır. Eski crop/cache dosyalarına yazılmaz; aynı kaydedilmiş kareye yeniden
GPU inference yapılmaz.

İşleme tek ham kare ve geçici crop'larla sınırlıdır. İkinci sıralı decode
maliyeti ölçüme dahildir. Üç yöntem için kaynak metrikleri, Accuracy, ROC-AUC,
Macro-F1, REAL/FAKE recall, confusion matrix, yanlış alarm, kapsam ve ortalama/
medyan kare sayıları raporlanır. Ortak kabul edilen cohort ayrıca gösterilir;
reddedilen videolar toplam 518 paydasında görünür kalır.

`actual_shared_execution_seconds` gerçek ortak yürütme süresidir.
`actual_shared_completion_timing` aynı altı-önce zaman çizelgesindeki gerçek
tamamlanma noktalarını gösterir. **Bu, üç ayrı bağımsız uçtan uca benchmark
değildir.** Sabit 12 burada iki decode geçişini paylaşır; daha önceki tek-geçişli
sabit 12 pilotuyla aynı hız ölçümü sayılmaz. Üç completion süresi toplanamaz.
Yöntemlerin kare sayıları kendi karar bütçeleridir; gerçek ortak çalışma her
videonun 12 benzersiz karesini en fazla bir kez hesaplar. Model/ısınma süreleri
ayrı kaydedilir; per-frame compute süreleri atanmış maliyet olarak etiketlidir.

Sonuç dosyası `final_results.json` mühürlü bir `payload` içinde raporlanır.
Başarı halinde test sonuçlarıyla model, eşik veya parametre değiştirilmez.

## Tek PowerShell komutu

Proje kökünde, kullanıcı uzun çalışmayı başlatmak istediğinde:

```powershell
.\outputs\video_research_v1\env\Scripts\python.exe -B -u -m scripts.run_video_final_test --run-final-test --protocol outputs/video_final_v1/protocol_20261018/protocol.json --protocol-sha256 a14a2ae31482caf1b57afb4bc621f9f84a0c3a5b06d822c90e4b339d4a9f7eab --resume auto
```

Denetim geçerse GTX 1650 için önceki validation ölçümlerine dayalı yaklaşık
**45–90 dakika**, ayrıca birkaç dakikalık hash denetimi planlanabilir. Bu süre
test videolarından ölçülmüş değildir. Kimlik örtüşmesi doğrulanırsa inference
başlamadan kısa denetimde durur; bu durumda uzun çalışma süresi geçerli olmaz.

## Resume ve tek seferlik çalışma

Aynı komut yeniden kullanılır. Global `official_518_once.json` protokol/run
eşleşmesini kilitler; OS dosya kilidi iki eşzamanlı çalışmayı engeller. Tamamlanmış
test tekrar değerlendirilmez. Örtüşme nedeniyle bloklanmış denemeye bypass yoktur.

Kesintide önce tüm denetimler tekrar geçmelidir. Tamamlanmış video sonuçları ve
tam atomik yazılmış frame skorları yeniden kullanılır; yarım/bozuk dosyalar
başarılı cache sayılmaz. Bozuk cache otomatik silinmez; işlem durur. GPU işi
bitmiş ama henüz atomik kaydedilmeden kesilmiş kare yeniden işlenebilir.

Kısmi frame cache'inden devam edilen videonun orijinal karşılaştırılabilir
gecikmesi yeniden oluşturulmaz: completion timing alanı boş bırakılır ve bu
eksiklik sayılır. Mevcut denemenin gerçek wall süresi ayrı kaydedilir. Kesinti
öncesi bilinmeyen süre soğuk başlangıç benchmark'ı gibi tahmin edilmez.

## Kontroller

Sentetik final-test ve mevcut adaptif testleri: **54 geçti**. Lint, 80 dosya
format kontrolü ve ilgili Python derleme kontrolleri tamamlandı. Testler gerçek
resmî listeyi veya videoları kullanmaz. Kimlik/hash bloklama, test öncesi etiket
kilidi, mühürlü cache, protokol drift'i, tek-sefer kaydı, frame/video resume ve
ortak zaman çizelgesi raporlaması doğrulandı.
