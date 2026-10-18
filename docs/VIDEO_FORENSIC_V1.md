# Deepfake video güçlendirme raporu

14 Ekim 2026. Uzun eğitim başlatılmadı. Gerçek zero-shot validation, kontrollü kalite karşılaştırması, üç-seed bozulma pilotu ve gerçek CUDA backward/resume deneyi yapıldı.

## 1. Son karar

**Ana strateji: resmî SBI FF-raw EfficientNet-B4 + resmî yüz ön işlemesi + video düzeyinde basit ortalama.** Kalite mekanizması aynı backbone üzerinde sınanan katkı adayıdır; başarısı kanıtlanmadan varsayılan karar yöntemi yapılmamalıdır. Ayrı SBI kopyasında kontrollü fine-tuning altyapısı hazırdır; zero-shot checkpoint korunur. Fine-tuning ancak validation ve kaynak bazlı ölçümlerle üstünse seçilmelidir.

602 kabul edilmiş videodan 602 değerlendirildi; coverage %100.00. Bu sonuçlar resmî 518 videoluk test setinin sonuçları değildir.

| Model | ROC-AUC | Macro-F1 | Balanced acc. | Accuracy | REAL recall | FAKE recall |
|---|---:|---:|---:|---:|---:|---:|
| B0 + GRU, natural/weighted | 0.69739 | 0.62053 | 0.63839 | 0.75249 | 0.46226 | 0.81452 |
| B0 + GRU, balanced | 0.62837 | 0.56310 | 0.59234 | 0.68272 | 0.45283 | 0.73185 |
| B0 spatial mean | 0.61753 | 0.58903 | 0.57656 | 0.81561 | 0.20755 | 0.94556 |
| SBI FF-raw zero-shot mean | 0.94353 | 0.83744 | 0.83869 | 0.90532 | 0.73585 | 0.94153 |
| SBI + kalite adayı | 0.94309 | 0.83296 | 0.83667 | 0.90199 | 0.73585 | 0.93750 |


Eski üç deneyin best checkpoint epoch'ları, history ve checkpoint içerikleriyle doğrulandı. GRU/natural epoch 2, balanced epoch 1, spatial epoch 3. Ön eğitim, crop, giriş çözünürlüğü, kapasite ve aggregation birlikte değişti; tabloda görülen fark yalnız “SBI eğitiminin” nedensel etkisini ölçmez. Aynı cohort üzerindeki sistem karşılaştırmasıdır. Teste veya V3'e erişerek model seçilmedi.

![Karşılaştırma](C:/Users/kilic/Desktop/deepfake-image-detection/outputs/video_research_v1/comparison.png)

## 2. Önceki model neden zorlanıyor?

Kanıt: `PartialVideoClassifier` ve spatial partial model yalnız `features[8]` projection bloğunu eğitiyor; MBConv katmanları frozen. REAL recall düşük, kaynaklar arası fark belirgin; balanced örnekleme ve spatial mean deneyleri mevcut GRU'dan daha iyi çıkmadı. Bunlar tek başına sorun teşhisi değildir.

Hipotezler: ImageNet özelliklerinin adli izlere yetersiz adaptasyonu, kaynak kısayolları, Haar crop/160 çözünürlük etkisi, az ve seyrek kare, eğitim/regularization stratejisi. Mevcut deneyler bunların bağımsız nedensel etkilerini ayırmıyor. SBI sonucu daha güçlü bir başlangıç olup olmadığını gösterir; başarısızlığın tek nedenini kanıtlamaz.

## 3. SBI araştırması ve lisans

[SBI makalesi](https://arxiv.org/pdf/2204.08376) tek gerçek görüntünün dönüştürülmüş iki sürümünü blend ederek genel sahtecilik izleri üretir. Asıl yenilik EfficientNet veya GRU değildir; bu sentetik eğitim örnekleridir. B4, ImageNet başlangıcından SAM ile 100 epoch, batch 32, LR 0,001 ve video başına sekiz eğitim karesiyle eğitilmiştir. FF++ gerçek görüntülerinden üretilen SBI'ler kullanılır; Celeb-DF yayınlanan protokolde çapraz veri seti değerlendirmesidir.

[Resmî depo](https://github.com/mapooon/SelfBlendedImages) FF-raw ve FF-c23 ağırlıklarını yayımlar; ikisi de gerçekten indirildi. Depoda verilen CDF AUC değerleri %93,82/%92,87 yalnız literatür referansıdır; bizim altı-kare/602-video protokolümüze aktarılmaz. FF-c23 bu görevde ana validation için denenmedi; sonuçlara bakarak checkpoint taraması yapılmadı.

[SBI lisansı](https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE) ticari olmayan iç araştırma kullanımını kapsar; genel MIT lisansı değildir. Kod/ağırlıkların dağıtımı kısıtlıdır. İndirilen upstream dosyaları ve ağırlıklar git tarafından ignore edilen outputs altında kaldı. SBI atfı: Shiohara ve Yamasaki, CVPR 2022, “Detecting Deepfakes with Self-Blended Images”. RetinaFace paketinin kendi lisansı MIT; SBI lisansının yerine geçmez.

Resmî checkpoint'in FF++ dışındaki tüm internet kaynaklarıyla örtüşmediğine dair bağımsız içerik/kimlik denetimi yoktur. Celeb-DF üzerinde bilinen doğrudan eğitim sızıntısı yayınlanan protokolde görünmüyor; bütün bilinmeyen kaynak örtüşmelerinin dışlandığı iddia edilmez.

| Dosya | SHA-256 |
|---|---|
| FFraw.tar | e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2 |
| FFc23.tar | 2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267 |
| RetinaFace extracted | b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2 |

SBI kaynak revizyonu: `d6d9a351f32334ae9a3a250872c05836e0dd0f2c`. Setup script'i dosya hash'lerini doğrular; rastgele checkpoint veya değiştirilmiş upstream crop dosyası kabul edilmez.

## 4. Ön işleme ve gerçek validation

[Resmî inference](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/inference_video.py) iki logitten softmax'ın FAKE sütununu alır. [Resmî crop](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/preprocess.py) RetinaFace bbox'ına her yönde %12,5 margin uygular; cv2 linear resize ile 380×380 RGB, float32 / 255 kullanılır. ImageNet mean/std normalizasyonu uygulanmaz. Detector kendi normalization'ını ayrıca uygular.

Mevcut altı örnekleme zamanı korunur; yüz detector/crop ve geçerli kareler değişebilir. Çoklu yüzde resmî alan filtresi ve kare başına maksimum FAKE skoru, zaman boyunca mean kullanıldı. Makaledeki 32 kare yerine altı kare ve en az üç geçerli kare şartı kullanılması protokol farkıdır. Yüzü olmayan videolara sahte başarı veya otomatik 0,5 ataması yoktur. Threshold sürekli 0,5 kaldı; calibration/threshold tuning yapılmadı.

| Kaynak | Video | Recall | FAKE skor ort. | Synthesis karşısında AUC |
|---|---:|---:|---:|---:|
| Celeb-real | 63 | 0.77778 | 0.35516 | 0.9471646185355862 |
| YouTube-real | 43 | 0.67442 | 0.37008 | 0.9382033008252063 |
| Celeb-synthesis | 496 | 0.94153 | 0.82103 | Tek sınıfta tanımsız |


REAL precision=0.72897; FAKE precision=0.94343. Confusion matrix satırlar gerçek, sütunlar tahmin; sıra REAL/FAKE: `[[78, 28], [29, 467]]`.

Süre 28.09 dakika; video başına 2.799 saniye (decode, detector, cache yazımı ve sınıflandırma dahil). Süre ölçümü sırasında bazı CPU regresyon kontrolleri de yürüdü; saf GPU benchmark değildir. Tepe allocated 978.4 MiB, reserved 2080.0 MiB. Reddedilen videolar: 0. Kare hata ayrıntıları JSON içindeki her video raporunda saklanır.

## 5. Diğer uygulanabilir seçenekler

- B0 son MBConv stage'lerini açmak: mevcut kod/cache ile en düşük entegrasyon riski; fakat güçlü forensic ön eğitim sağlamaz. Gerekirse kontrollü yedek deneydir, bu görevin ana stratejisi değildir.
- V3 görüntü transferi: görüntü test başarısı video genellemesini kanıtlamaz; tam görüntü/yüz crop domain farkı vardır. Orijinal V3 değiştirilmedi; bu görevde video transferi çalıştırılmadı.
- [DFDC EfficientNet çözümü](https://github.com/selimsef/dfdc_deepfake_challenge): resmî model indirme script'i ve MIT kod vardır; sunulan B7 ensemble eğitimi çok GPU/12+ GB odaklıdır. 4 GB'a yeniden uyarlama ve farklı crop protokolü ek teslim riski getirir; ağırlık lisansını ayrı incelemeden entegre edilmedi.
- [LipForensics](https://github.com/ahaliassos/LipForensics): resmî FF++ checkpoint ve değerlendirme kodu vardır, fakat 68 landmark, ağız crop'u ve temporal hazırlık ister. Altı seyrek Haar karesine takılabilecek bir model değildir. Bu cihazda ölçülmedi; “çalışmaz” iddiası yok, entegrasyon maliyeti yüksek olduğu için seçilmedi. FTCN/TALL için de ölçülmemiş 4 GB uygunluk iddiası kurulmadı.

SBI seçimi; gerçek yerel ölçüm, düşük inference VRAM'i, anlaşılır tek CNN + birleştirme mimarisi ve 9 Kasım teslimine kadar sınırlı ek geliştirme gereksinimine dayanır. Daha büyük model otomatik başarı garantisi değildir.

## 6. Özgün katkı ve ölçümü

Tam matematik, gerekçe, AFW farkı, sınırlar ve sabit deney koşulları [protokolde](VIDEO_FORENSIC_PROTOCOL.md). Özet: modelden bağımsız göreli keskinlik ve bbox kenar payı korunma oranı; kalite ağırlıkları eşit ortalamayla karıştırılır. Her karenin ağırlığı `2/(3T)` ile `2/T` arasındadır. Global veri seti/validation normalizasyonu veya fake skorundan güven üretimi yoktur.

| Birleştirme | ROC-AUC | Macro-F1 |
|---|---:|---:|
| mean | 0.94353 | 0.83744 |
| full | 0.94309 | 0.83296 |
| sharpness | 0.94250 | 0.84205 |
| boundary | 0.94353 | 0.83744 |


Full kalite adayının mean'e farkı: **AUC -0.000437, Macro-F1 -0.004476**. Bu fark tek başına kesin iyileşme değildir. Bilinen filename aileleriyle 1.000 tekrarlı bootstrap, seed 42 için AUC farkı %95 aralığı `[-0.0019426528688498146, 0.0008321308272110691]`, F1 farkı `[-0.010912471340372907, 0.0]`. Seed 123 ve 2026 sonuçları JSON'dadır. Kaynak/kişi bağımlılıklarının tamamını gidermeyen koşullu validation aralıklarıdır.

Ham ortalama keskinliğin etiket AUC'si `0.42730523432744977`; yalnız REAL örneklerde YouTube/Celeb kaynak AUC'si `0.6112956810631229`. Bunlar kaynak kısayolu riskinin tanılarıdır, alternatif sınıflandırıcı başarısı veya nedensel kanıt değildir.

**Karar: full kalite yöntemi temiz validation'da mean'den kötü çıktı; kullanıma alınmadı.** Yalnız keskinlik ablation'ındaki küçük F1 artışı, AUC düşüşüyle birlikte ele alınmalıdır; sonuçları gördükten sonra bunu yeni kazanan diye seçmiyoruz. Sınır oranı ablation'ı bu cohort'ta skoru değiştirmedi. Faydalı bir özgün yöntem elde edildiği henüz kanıtlanmış değildir.

Bu çalışmada bize ait olan SBI değildir. Katkı adayı, ölçülebilir kaliteye dayalı sınırlı karar düzeltmesi ve bunun aynı checkpoint/kareler üzerinde ablation/robustness denetimidir. Akademik katkının fayda iddiası elde edilen sonuçlarla sınırlıdır; küçük veya olumsuz farklar saklanmamalıdır.

## 7. Üç seed ile bozulma pilotu

| Koşul | Mean AUC | Kalite AUC | Mean Macro-F1 | Kalite Macro-F1 |
|---|---:|---:|---:|---:|
| clean | 0.96528 | 0.96528 | 0.74825 | 0.77500 |
| blur | 0.96875 | 0.96991 | 0.73924 | 0.73924 |
| jpeg | 0.94444 | 0.93981 | 0.76503 | 0.77394 |
| drop_frame | 0.96412 | 0.96181 | 0.76608 | 0.77499 |
| crop_translation | 0.96181 | 0.96181 | 0.77499 | 0.78391 |


Tablo, 42/123/2026 seed'lerinin ortalamasıdır. Her kaynaktan önceden belirlenen 12 video; 36 video, 24 REAL/12 FAKE. Aynı model/kareler üzerinde iki yöntem eşleştirildi. Bunlar üç eğitim seed'i değildir. Detector tekrar çalıştırılmadı; sentetik crop ötelemesi gerçek bbox jitter testi değildir. JPEG görüntü sıkıştırması gerçek video codec deneyi değildir. Keskinlik clean dahil bütün pilot koşullarında 380×380 üzerinde yeniden hesaplandı; ana 602-video karşılaştırmasındaki ham crop keskinliğiyle aynı ölçüm ölçeği değildir. Her koşulun tek tek sonuçları ve tahminleri `robustness_pilot.json` içinde. Pilot, tüm 602 videoda veya başka veri setlerinde robustness kanıtı olarak sunulmamalıdır.

## 8. GTX 1650 uygulanabilirliği ve eğitim süresi

Gerçek CUDA FP32 backward yapıldı; uzun eğitim epoch'u başlamadı. Toplam B4 parametresi **17.552.202**. İlk aşama **3.586**, ikinci aşama **12,235,982** trainable parametre. İkinci aşama `_blocks[24:32]`, projection ve classifier; frozen prefix/BN state değişmedi.

Frame batch=1, dört video accumulation. İkinci aşamada 24 gerçek kare / dört video güncellemesi **1.242 s**; allocated **499.5 MiB**, reserved **1272.0 MiB**. NaN/Inf oluşmadı. Checkpoint'e model/AdamW/RNG kaydedilip aynı sonraki CUDA güncellemesi yeniden oynatıldı; bit düzeyinde eşitlik: **True**.

Küçük pilot üzerinden cache hazırken train+validation epoch tahmini **13.2 dakika**; sekiz epoch üst bütçe yaklaşık **1.77 saat**. İlk crop çıkarma, disk I/O ve checkpoint yazımı ek süredir; train videoları daha uzun olabilir. Bu tam epoch ölçümü değildir. Early stopping daha erken bitirebilir. İlk train crop hazırlığı için 2.223 video × mevcut uçtan uca süre yaklaşık 104 dakika mertebesinde kaba ek bütçe düşünülmelidir; detector-only ölçümü değildir.

FP32 seçildi; B4 ve detector cihazda sığdı. AMP/gradient checkpointing eklenmedi. Pilot, önerilen LR=1e-5/1e-4'ün kısa vadede sonlu olduğunu gösterir; uzun vadeli optimal LR garantisi vermez. Artış beklentisinin gerekçesi güçlü forensic başlangıcını gerçek MBConv adaptasyonuyla hedef alana uyarlayabilmektir; catastrophic forgetting veya overfit yüzünden başarının düşmesi de mümkündür.

## 9. Güvenli PowerShell komutları

Proje kökünde, yalnız uzun eğitime başlamak istediğinde:

```powershell
$python = '.\outputs\video_research_v1\env\Scripts\python.exe'
$run = 'outputs/video_research_v1/finetune/sbi_seed42_20261014'
& $python -u -m src.video_forensic_train --run-training --run-dir $run --epochs 8 --seed 42
```

Aynı run üzerinden devam:

```powershell
$python = '.\outputs\video_research_v1\env\Scripts\python.exe'
$run = 'outputs/video_research_v1/finetune/sbi_seed42_20261014'
& $python -u -m src.video_forensic_train --run-training --run-dir $run --epochs 8 --seed 42 --resume auto
```

Resume son tamamlanan epoch'tan devam eder; yarım epoch tekrar oynatılır. İlk epoch tamamlanmadıysa resume edilebilir checkpoint yoktur; yeni run adı gerekir. Erken durma tamamlandıysa ek epoch çalışmaz. Seed, config, kod, ortam, kaynak/cache hash'leri değişirse resume reddedilir. Mevcut run adı yeni başlangıç için yeniden kullanılamaz. Üç bağımsız eğitim gerektiğinde seed 123/2026 ve ayrı run adları kullanılmalıdır; bu görevde çalıştırılmadı.

Çıktılar: run altında `config.json`, `history.json`, `epoch_XXX_validation.json`, `interrupted.json`; `checkpoints/epoch_XXX.pth`, `best_video.pth`, `last_video.pth`. Best seçim Macro-F1, eşik 0,5. Zero-shot başlangıç ağırlıkları ayrı kalır. Eğitim çıktısını değerlendirmek için aynı `src.video_forensic --checkpoint .../best_video.pth --output YENI_JSON` arayüzü kullanılabilir.

Ortamı yeniden kurmak veya train/val cache audit'ini tekrarlamak:

```powershell
.\.venv\Scripts\python.exe scripts/setup_video_forensic.py --download-and-install --audit-train-val
```

## 10. Doğrulamalar, önemli dosyalar ve kalan sınırlar

Yeni çekirdek modüller: `src/video_forensic.py`, `video_forensic_train.py`, `video_quality_fusion.py`, `video_forensic_robustness.py`. Setup/rapor script'leri, izole gereksinim listesi, `tests/test_video_forensic.py` ve `pytest.ini` eklendi. Test keşfi proje test klasörüne sınırlandı; outputs içindeki izole ortam paketlerinin testleri proje testi sayılmaz.

Regresyon sonuçları `regression.xml`; lint/format/compile kontrolleri `checks.json` dosyasında. Sandbox'taki Streamlit socketpair engeli, testleri değiştirmeden sandbox dışı çalıştırmayla çözüldü. İzole ortam `pip check` eski RetinaFace'in torch 1.9 / torchvision 0.10 / albumentations 1.0 pinleri nedeniyle başarısızdır. Ana .venv güncellenmedi. Windows/Python 3.12 + torch 2.11 CUDA çalışma uyumluluğu gerçek inference ve backward ile doğrulandı; eski ortamla bit düzeyinde eşdeğerlik iddiası yoktur.

Koruma hash kontrolü: **True**. `app.py`, V3 checkpoint, manifest ve eski best video checkpoint'leri aynı. Eski feature cache tekrar audit ile kontrol edildi. Resmî 518 video ve quarantine videoları açılmadı; split değiştirilmedi. Commit/push yapılmadı.

Validation daha önce birçok kez kullanıldı; seçime bağlı aşırı uyum riski vardır. Gerçek kişi düzeyinde ayrım garantisi yoktur. Tam 602-video robustness, üç-seed uzun eğitim, başka kaynaklarda genelleme ve kilitli resmî test bu görevde tamamlanmış sayılmaz.

## 11. Hocaya kısa anlatım

“Videodan belirli zamanlarda yüzleri çıkarıyoruz. Literatürdeki SBI yöntemiyle önceden eğitilmiş modeli kullanarak her kare için sahtecilik skoru hesaplıyoruz. Bizim araştırdığımız katkı, bulanıklık ve crop sınır kaybı gibi modelden bağımsız ölçülerle karelerin etkisini sınırlandırılmış biçimde değiştirmek. Aynı model ve karelerde düz ortalamayla karşılaştırıp hangi koşullarda işe yaradığını, hangi koşullarda yaramadığını ölçüyoruz. SBI'yi kendi buluşumuz olarak sunmuyoruz.”
