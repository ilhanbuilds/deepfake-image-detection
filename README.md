# Deepfake Analiz

## Güncel uygulama teslimi ve UBYS

[9 Kasım 2026 teslim sürümü](https://github.com/ilhanbuilds/deepfake-image-detection/releases/tag/teslim-2026-11-09) iki ayrı dosya içerir: **UBYS_Kaynak_Kod.zip**, 100 MB sınırına uygun kaynak kod teslimidir; **Deepfake_Analiz_Modeller_Dahil.zip**, V3, SBI ve RetinaFace modellerini de içeren yaklaşık 291 MiB kurulum paketidir. Rapor ve sunum ayrı teslim edilir. Kaynak ZIP'i veya GitHub'ın otomatik **Source code** arşivi model içermez; analiz için modeller dahil paketi indirin.

Modeller dahil ZIP'i tamamen çıkarın. Python 3.12 kurulu olmalıdır; ilk kurulumda internet gerekir. Windows/CUDA için **Deepfake Analiz Windows.cmd**, yalnız fotoğraf için **Windows Fotograf CPU.cmd**, Apple Silicon Mac'te yerel ARM64 Python ile fotoğraf için **Deepfake Analiz Teslim Mac.command** kullanılır. Mac video analizi desteklenmez. Tarayıcı ve masaüstü aynı Studio arayüzünü kullanır. Farklı yerel klasörde fotoğraf, CPU Grad-CAM ve CUDA video smoke kontrolleri geçti; gerçek Mac ve temiz bilgisayarda sıfırdan internet kurulumu henüz doğrulanmadı.

Depo private olduğundan release dosyalarını yalnız erişim verilen GitHub hesapları indirebilir. Linke sahip olmak tek başına erişim vermez. [UBYS teslim ve erişim açıklaması](docs/TESLIM_UBYS.md).

Güncel akademik teslim: [17 sayfalık final raporu, 12 slaytlık sunum ve kaynak tablolar](teslim/README.md). [87 video envanter farkının doğrulanmış açıklaması](teslim/SON_KONTROL_TR.md) aynı pakettedir. Bu paket özel depoda paylaşılır; içerdiği üçüncü taraf yüz görsellerinin kamuya yeniden dağıtım hakkı doğrulanmamıştır. Depoyu veya teslim belgelerini izin incelemesi olmadan kamuya açmayın.

Üniversite Derin Öğrenme dersi kapsamında geliştirilen, yerel çalışan fotoğraf ve video deepfake araştırma uygulaması. Ana Windows arayüzü **Studio** (`scripts/studio_app.py`); fotoğrafta **V3 Robust EfficientNet-B0**, videoda **Celeb-DF üzerinde fine-tune edilmiş SBI EfficientNet-B4 epoch 6** kullanır.

**Model skoru, kalibre edilmiş kesinlik olasılığı değildir.** Sistem bütün AI düzenlemelerini tespit etmez. Yüksek tarihsel test başarısı, yeni kaynaktan gelen bir fotoğrafın güvenilir biçimde sınıflandırılacağını garanti etmez. Özellikle aşağıdaki harici face-swap deneyinde iki model de sekiz sahtenin tamamını kaçırmıştır.

> Bu GitHub deposu kaynak kodu deposudur; model, dataset veya deney arşivi değildir. Yalnız klonlamak tam analiz için yeterli olmaz. Ağırlıklar, provenance dosyaları ve ayrı çalışma ortamları ayrıca gerekir. Veri ve ağırlıkların yeniden dağıtım izni varsayılmaz.

## Windows Studio: başlatma ve kullanım

Yerel modeller ve üç ortam hazırsa proje kökünde:

```powershell
.\.desktop-venv\Scripts\pythonw.exe -B .\scripts\studio_app.py
```

Çift tıklama için `Deepfake Analiz.vbs` Studio'yu açar. Yeni kısayol oluşturmak için:

```powershell
.\scripts\create_desktop_shortcut.ps1
```

Bu komut **Deepfake Analiz Studio.lnk** oluşturur; mevcut bir kısayolu ezmez. Eski `Deepfake Analiz.lnk` korunur ve eski kabuğu açabilir. VBScript kapalıysa Studio kısayolunu veya PowerShell komutunu kullanın; Windows güvenlik ayarlarını değiştirmeniz gerekmez.

1. Tek bir fotoğraf veya video seçin ya da alana bırakın. Tür içerik ve uzantı kontrolüyle belirlenir.
2. Orijinal önizlemeyi kontrol edin, analizi başlatın. Gerçek iş adımları ve sonuç gösterilir.
3. Fotoğrafta Grad-CAM açıklamasını; videoda altı kareden on ikiye genişleme ayrıntısını açabilirsiniz.
4. Dosyayı değiştirince önceki sonuç temizlenir. Çalışan analiz iptal edilebilir; pencereyi kapatınca yalnız uygulamanın sahip olduğu süreçler kapanır.

Arayüz pywebview + Edge WebView2, yerel HTML/CSS/JavaScript ve Python HTTP sunucusudur. Harici CDN kullanmaz. Pencere 1180×760, minimum 640×460'tır. Tek pencere 4 GB kart için önerilir. Sunucu yalnız `127.0.0.1` üzerinde işletim sisteminin seçtiği boş porta bağlanır; oturuma özel URL/token, Origin denetimi ve Windows Job Object kullanır. Başka uygulamanın portuna/sürecine müdahale edilmez.

| Girdi | Sınır |
|---|---|
| Fotoğraf | JPEG/JPG/PNG; en fazla 10 MiB, 16 megapiksel |
| Video | MP4/M4V/MOV/WebM; en fazla 100 MiB, 60 sn, 120 FPS, 7.200 kare; Full HD, en uzun kenar ≤1920; en az altı kare |
| Analiz zaman aşımı | Fotoğraf 120 sn; video 300 sn |

Bozuk içerik, desteklenmeyen codec, CUDA/model doğrulama hatası, yüz bulunamaması ve yetersiz geçerli kare açıkça bildirilir. FFmpeg varsa yalnız tarayıcı önizlemesi için H.264/AAC kopyası üretilir; **SBI yüklenen orijinal video baytlarını analiz eder**. FFmpeg yoksa orijinal önizleme denenir ve codec desteği sınırlı olabilir.

Yüklemeler özel OS geçici klasörlerinde işlenir. Normal bitiş, hata, timeout ve kapanışta yalnız kayıtlı uygulama geçicileri temizlenir; bağlantı hedefleri izlenmez. Sert sistem kapanışı artık bırakabilir; şifreli saklama veya adli güvenli silme iddiası yoktur. Oturum logları `outputs/desktop_studio_sessions/<oturum>/` altındadır ve Git dışındadır. Her iş ayrı model sürecidir; modeller GPU'da aynı anda tutulmaz. Model yükleme gecikmesi toplam süreye dahildir. GPU kilidi oturumlar arasında da eşzamanlı analizi engeller.

## İki ayrı analiz hattı

Mac'te yerel tarayıcı arayüzü için proje kökündeki **Deepfake Analiz Mac.command** dosyasını kullanın. Bir defalık ortam kurulumu ve gereken V3 ağırlığı [Mac kurulum kılavuzunda](docs/MACOS.md) açıklanır. Fotoğraf analizi CPU kullanabilir; mevcut CUDA video hattı Mac'te desteklenmez.

### Fotoğraf: V3 Robust

`REAL=0`, `FAKE=1`. EfficientNet-B0, iki çıkışlı classifier, best epoch 5. Ortak CLEAN dönüşümü: EXIF yön düzeltme → RGB → bilinear/antialias 224×224 → tensor → ImageNet normalizasyonu (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). **Görüntünün tamamı** sınıflandırılır; Studio fotoğrafta otomatik SBI yüz analizi yapmaz. CUDA varsa kullanılır, yoksa mevcut fotoğraf akışı CPU seçer.

Grad-CAM tahmin edilen sınıfın softmax öncesi logitini son convolution katmanında (`features.8.0`) açıklar. Optimizer güncellemesi yoktur. Isı haritası manipüle edilmiş alanın maskesi, nesnel doğruluk kanıtı veya nedensel açıklama değildir; arka planı da vurgulayabilir.

### Video: SBI epoch 6, adaptif 6→12

SBI (Self-Blended Images) Shiohara ve Yamasaki'nin CVPR 2022 yöntemidir; projenin özgün icadı olarak sunulmaz. Resmî FF-raw başlangıcından yerel Celeb-DF fine-tuning uygulanmıştır. Runtime RetinaFace + hash ile doğrulanan resmî crop kodunu kullanır: bbox çevresinde %12,5 marj, RGB 380×380, **FP32 / 255**; ImageNet mean/std uygulanmaz. Video için CUDA gerekir; sessiz CPU fallback yoktur.

Her karede en büyük yüzün alanının en az yarısı olan kutular korunur; kare skoru tutulan yüzlerin maksimum FAKE skorudur. Video skoru geçerli karelerin aritmetik ortalamasıdır; eşik 0,5. Tarihsel eşit aralıklı altı indeks korunur. `0.4 < p6 < 0.6` ise iç içe örneklemeyle toplam on iki benzersiz kareye genişlenir; sınırlar strict'tir. İlk altı kare eksikse güvenlik gereği ek kareler denenir. Üçten az geçerli yüzlü karede karar verilmez; eksik kareye nötr skor veya kopya atanmaz. Kısa videoda eksik bütçe görünür tutulur.

Video modeli ve preprocessing, fotoğraf araştırmasındaki detector B varyantından ayrıdır. Fotoğraf B, küçük görüntüyü gereksiz yere 2048'e büyütmez; bu değişiklik **video hattına veya Studio fotoğraf kararına taşınmamıştır**. Önceden doğrulanmış yüz crop'larının doğrudan SBI yolu da yalnız araştırmadadır; `no_face` sonucu REAL sayılmaz.

## Veri, kaynak ve kullanım durumu

Aşağıdaki lisans bilgileri korunan yerel inceleme kayıtlarının özetidir; yeni bir izin veya güncel hukuki doğrulama değildir. Kullanıcı, Kaggle indirmelerini ve ticari olmayan yerel üniversite çalışmasını beyan etmiştir; bu beyan resmî lisans kabul belgesi değildir. `download_source_verified` ve `license_verified` ayrı tutulur. Bekleyen doğrulama alanları değiştirilmemiştir; fotoğraf/video/veri/ağırlıklar kamuya dağıtılmamaktadır.

| Veri / model | Kaynak | Kayıtlı durum ve sınır |
|---|---|---|
| V1 yüz fotoğrafları | `data/raw/` | 12.890 dosya; özgün üretim/kişi ve bazı augmentation etiket kökenleri doğrulanamamıştır. Dosya adından gerçeklik çıkarılmaz. |
| 140K Real and Fake Faces | [Kaggle/xhlulu](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | İndirme kaynağı doğrulandı; lisans alanı **Other (specified in description)**. `license_verified=false`, `verification_pending`. Yayıncı 70K FFHQ/Flickr gerçek ve Bojan StyleGAN koleksiyonundan 70K sahteyi 256px'e dönüştürdüğünü bildirir. |
| FFHQ/Flickr kaynak zinciri | [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset#licenses) | Dosya başına Flickr lisansı/yazar/atıf kaydı gerekir. Seçilen REAL dosyalar bu metadata ile eşlenmiş değildir; FFHQ metadata koşulları görselin bireysel lisansının yerine geçmez. |
| Aday StyleGAN koleksiyonu | [tunguz/1-million-fake-faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces) | Kayıtlı alan CC BY-NC 4.0; 140K'nin tartışma bağlantısı ile koleksiyon/sürüm/dosya eşlemesi beklemede. StyleGAN kod lisansı tek başına koleksiyon izni değildir. |
| Celeb-DF v2 | [Kaggle/reubensuju](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2), [resmî proje](https://github.com/yuezunli/celeb-deepfakeforensics) | Ayna lisansı **Unknown**; indirme kaynağı doğrulandı, resmî kabul belgesi yok. Ticari olmayan araştırma, kurumsal sorumluluk, atıf, kopyalama/dağıtım ve türev veri kısıtları kayıtlıdır. Ayna ve mevcut crop kullanımı için belge belirsizliği sürer. |
| SBI | [SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages), [makale](https://arxiv.org/abs/2204.08376) | Ticari olmayan iç araştırma ve yeniden dağıtım kısıtları yerel kayıtta mevcut; genel MIT izni varsayılmaz. Kaynak revizyonu `d6d9a351f32334ae9a3a250872c05836e0dd0f2c`. FF++ ön eğitim kaynak örtüşmesi bütünüyle denetlenmiş değildir. |
| Harici fotoğraf geliştirme | [Sowaiba01/Deepfake](https://huggingface.co/datasets/Sowaiba01/Deepfake) | Veri kartı MIT bildirir; özgün görüntü hak zinciri/üretim logları doğrulanmadı. `license_verified=false`, `label_verified=false`; sağlayıcı etiketleriyle hata analizi. |

Celeb-DF erişimi için resmî projenin yönlendirdiği form/akademik e-posta kabulü ve kurum/araştırmacı, tarih, sürüm, amaç kaydı gerekir; Kaggle kopyası/crop kapsamı gerektiğinde veri sağlayıcısından yazılı açıklığa kavuşturulmalıdır. Bu depo kullanıcı adına başvuru yapmaz. Kanıtlar: `outputs/image_v4_pilot_v1/dataset_provenance.json`, `celeb_df_official_access_terms.json`, `execution_authorization.json`. Eski hazırlık raporunda görülen eğitim engeli tarihsel durumdur; sonraki yetkili yerel pilotun çalıştığını bu ayrı yürütme kaydı belgeler. Lisans bayrakları bu sırada verified yapılmamıştır.

## Ayrımlar ve bilimsel kapsam

| Çalışma | Train | Validation / geliştirme | Test / kapsam |
|---|---:|---:|---|
| V1 fotoğraf | 9.023 | 1.933 | 1.934; seed42, bilinen augmentation/kopya aileleri birlikte |
| V2–V3 / 140K | 100.000 | 20.000 | 20.000; hazır ayrımlar korunur, her split %50 REAL/FAKE |
| SBI kabul edilmiş video roster'ı | 2.223 (417 REAL / 1.806 FAKE) | 602 (106 REAL / 496 FAKE) | Resmî holdout 518 (178 REAL / 340 FAKE); ilişkili 3.099 video karantinada |
| V4 / V4.1 fotoğraf pilotu | 4.800: 2.400 GAN/REAL + 2.400 Celeb-DF crop | 1.200: 600 tarihsel GAN validation + 600 crop (100 video) | Yeni bağımsız final test yok |
| Harici fotoğraf deneyi | Eğitim yok | 8 eşli REAL/FAKE aile, toplam 16 görüntü | Hata analizi; bağımsız nihai test iddiası yok |

SBI manifest inventory train/val toplamları 2.306/606; crop kabulünden sonraki 2.223/602 sayılarıyla karıştırılmamalıdır. Kaynak-video aileleri ve kod bağlı bileşenleri split içinde tutulur; crop'lar bağımsız kişi sayılmaz. Video holdout'ta bilinen yol/hash/taşıyıcı aile çakışması bulunmaması **kişi bağımsızlığı değildir**: test filename kodları train ile 46, validation ile 10 ortak kod taşır; donor/YouTube/ön eğitim kaynakları tam doğrulanmamıştır.

İlk kimlik-ayrık final-test tasarımı bu örtüşme nedeniyle bloklanmıştır. Sonraki **VIDEO-HOLDOUT** protokolü kişi bağımsızlığı iddia etmeden ayrı mühürlü değerlendirmedir; `video_final_v1` ve `video_holdout_v1` kayıtları birbirinin yerine kullanılmaz. Nihai test yeniden kurulum veya smoke kontrolü değildir; tekrar çalıştırılmamalıdır.

V2 test/robustness gözlemleri V3 augmentation tasarımına yön verdi. Test optimizer'a veya checkpoint seçimine girmemiş olsa da V3 sonucu geliştirme boyunca hiç görülmemiş dış test değildir. SHA-256 çakışması yokluğu yeniden kodlanmış kopyaları, aynı kişiyi veya ortak kaynağı dışlamaz.

## Fotoğraf geliştirme geçmişi

| Sürüm | Yapılan çalışma | Sonuç / kullanım |
|---|---|---|
| V1 | ImageNet EfficientNet-B0; ilk 12.890 görüntü; best ep5, early stopping ep8 | Kendi 1.934 testinde accuracy %95,450; FAKE F1 %95,712; AUC 0,99185736. Farklı dataset, ortak benchmark değil. |
| V2 | 140K hazır ayrımı; dört epoch, best ep3 | 20K CLEAN accuracy %99,410; blur/resize altında yaklaşık %50'ye gerileme. |
| **V3 Robust** | Train-only kalite augmentation; altı epoch, best ep5 | Kullanılan fotoğraf modeli. CLEAN başarı ve ölçülmüş kalite dayanıklılığı iyileşti; face-swap genellemesi zayıf. |
| V4 | V3 başlangıcı; features7/8 + classifier; aynı 4.800/1.200 pilot, iki epoch | Celeb video BA %50, FAKE recall %16, REAL recall %84; AUC 0,5558. Kabul koşulları geçmedi. |
| V4.1 | Yalnız features6 da açıldı; yine V3 başlangıcı, iki epoch | Celeb video BA %55; FAKE recall %38, REAL recall %72; AUC 0,5738. REAL yanlış alarm maliyeti nedeniyle kabul edilmedi. |

V3 eğitimi batch8, AdamW LR3e-4, AMP, seed42, workers0, patience2 ile altı epoch tamamlandı; duruş epoch sınırıdır. Best yalnız CLEAN validation FAKE F1 ile seçildi. Train augmentation: flip %50; JPEG %50 (Q50–95); blur %15 (sigma0,1–1,5); noise %15 (std0–0,02); resize %30 (224→112–224→224). Validation/inference CLEAN kalır.

V4'te finite loss'a rağmen AMP scale65536 backward overflow üretti; unscale zaten clipping'den önceydi. Aynı batch FP32 ve scale1024 ile finite kaldı. Ayrı sürümlü düzeltme scale1024, finite kontrolü, strict clipping ve kontrollü GradScaler skip/backoff kullandı. İki epoch 1.200 başarılı update, sıfır skip ile tamamlandı. V4.1 de aynı augmentation/LR/batch/seed/BN ve sayısal kuralları korudu; frozen tensor/BN hash'leri değişmedi. İki pilotun da seçilen epoch'u 2'dir. Bunlar Studio'ya alınmadı.

Küçük CLEAN train alt kümesinde V4 Celeb kare AUC0,5596, FAKE recall%17,71; development AUC0,5165, recall%22,92. Eğitimde güçlü öğrenip yalnız geliştirmede kaybettiği gösterilemedi. V4.1 video AUC farkı +0,0180 için aile bootstrap %95 aralığı [-0,0072,+0,0456]; güvenilir ayrım kazanımı kanıtlanmadı. Önceden sabit kabul şartları FAKE recall/Macro-F1 artışı, V3'e göre REAL recall kaybı ≤2 puan, GAN BA kaybı ≤1 puandı; REAL şartı sağlanmadı. V4/V4.1 GAN geliştirme BA sırasıyla %99,83/%99,67; bu sayılar dış test değildir.

## Doğrulanmış tarihsel sonuçlar

### V3 / 20.000 CLEAN fotoğraf

Kaynak: `outputs/metrics/final_test_metrics_v3.json`; yeniden inference yapılmadan okunmuştur.

| Accuracy | FAKE precision | FAKE recall | FAKE F1 | ROC-AUC | Doğru / yanlış |
|---:|---:|---:|---:|---:|---:|
| **%99,635** | %99,7893 | %99,480 | %99,6344 | 0,99992647 | 19.927 / 73 |

Confusion matrix, satır gerçek/sütun tahmin, sıra REAL–FAKE: `[[9979,21],[52,9948]]`.

| Kalite koşulu | V2 accuracy % | V3 accuracy % |
|---|---:|---:|
| CLEAN | 99,410 | 99,635 |
| JPEG Q70 / Q50 | 85,690 / 76,340 | 99,220 / 98,525 |
| Blur sigma1 / sigma2 | 50,955 / 50,005 | 98,885 / 91,030 |
| Noise std0,01 / std0,03 | 94,895 / 75,455 | 99,430 / 97,850 |
| Resize224→112→224 | 50,185 | 98,230 |

Kaynak: `outputs/metrics/robustness_v2.json`, `robustness_v3.json`. V1 epoch1 süre kaydı yoktur; V1 için kaydedilen 19,85 dakika tam sekiz epoch toplamı olarak sunulmaz. Tarihsel raporlarda train accuracy/validation loss olmayan yerlerde eğri uydurulmamıştır.

### SBI / 518 video VIDEO-HOLDOUT

Kaynak: `outputs/video_holdout_v1/final_test_20261018/final_results.json`. Tüm yöntemlerde 518/518 kapsam, video reddi yoktur.

| Yöntem | Accuracy % | REAL recall % | FAKE recall % | Macro-F1 | ROC-AUC | Ort. kare | Genişleme % |
|---|---:|---:|---:|---:|---:|---:|---:|
| Sabit6 | 97,2973 | 94,9438 | 98,5294 | 0,96988 | 0,997224 | 6 | 0 |
| Sabit12 | 97,4903 | 96,0674 | 98,2353 | 0,97215 | 0,997274 | 12 | 0 |
| **Adaptif6→12** | **97,4903** | 95,5056 | 98,5294 | 0,97207 | 0,997555 | 6,23166 | 3,861 |

Sabit12 ve adaptif aynı accuracy'ye rağmen farklı hatalar yapar. Ortak inference cache'inin 3.252,605 sn toplam duvar süresi bağımsız yöntem benchmark'ı veya doğrudan hızlanma ölçümü değildir. Daha önceki 602-video validation, kare sayısı ve epoch seçiminde kullanılmıştır. **Fotoğraf %99,635 ile video %97,49 farklı örneklem/görevlerdir; doğrudan model üstünlüğü karşılaştırması yapılamaz.** [Korunan kısa holdout özeti ve dosya hash'leri](docs/VIDEO_HOLDOUT_RESULTS.md).

Video araştırma geçmişindeki Haar/160px B0+GRU, balanced sampling ve spatial mean referansları aktif SBI hattıyla karıştırılmamalıdır. Aynı 602 validation'da natural GRU AUC0,69739, balanced0,62837, spatial0,61753; SBI FF-raw zero-shot altı kare0,94353 ölçüldü. Birçok bileşen birlikte değiştiği için fark tek bir mimari etkiyi kanıtlamaz. Kalite ağırlıklı aday basit ortalamayı tutarlı iyileştirmedi. SBI12 eğitim hazırlık belgesinin “henüz eğitilmedi” ifadesi tarihsel aşamadır; tamamlanmış run ve epoch6 checkpoint'i güncel sonuç için esastır.

## Fotoğrafta dış kaynak ve veri kalitesi bulguları

Küçük tarihsel 64-görüntü ablation'ında RetinaFace A küçük fotoğrafı 2048'e büyütüyordu. Fotoğraf B bunu kaldırınca 32 Celeb görüntüsünde BA%50→%96,875, yanlış REAL→FAKE15→1 oldu. B, 16 GAN sahtenin hiçbirini yakalamadı. Bu tarihsel geliştirme sonucudur; B video preprocessing değişikliği değildir.

Daha geniş 256-görüntü çalışması üç birebir aynı RGB/zıt etiketli çift ve dört logo crop'u ortaya çıkardı. Kaynak geliştirme karelerinin sınırlı incelemesinde dört logo yanlış nesne kırpımı; kalan aynı-yüz çiftinde çok kişili sahnede seçilen yüzün değişmemesi görüldü. İncelenen cache'lerin kendi kaynak/bbox bağlantısı yeniden üretildi; yanlış kaynak eşlemesi kanıtlanmadı. Video FAKE etiketi her crop'un manipüle olduğunu garanti etmez. Orijinal etiket/dosya değiştirilmeden `invalid_face_crop` / `unverified_frame_label` işaretleri eklendi; kalite filtresi skor bağımsız, sonuçları yalnız ikincil teşhistir.

### Harici sekiz face-swap çifti: ortak başarısızlık

Sabit HF revision `d8b189507fc026b3e0c794a06c318cb13d0c8743` içinden metadata sırasıyla seçilen sekiz farklı çift; yalnız 16 JPG (411.020 bayt). Metadata5.352 çift, kart5.426 çift bildirir. Yöntem sağlayıcıya göre `inswapper_128`; kamera orijinali/üretim logu/kişi kimliği bağımsız doğrulanmamıştır. Sınıf başına yalnız sekiz örnek: bir hata12,5 puan değiştirir; varyantlar/çiftler bağımsız kaynak sayılmaz.

| Model | REAL doğru | FAKE doğru | FP / FN | BA | Macro-F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| V3 | 8/8 | 0/8 | 0 / 8 | %50 | 0,3333 | 0,6094 |
| SBI fotoğraf B | 8/8 | 0/8 | 0 / 8 | %50 | 0,3333 | 0,5938 |

İkisinin de CM'si `[[8,0],[8,0]]`; anlaşmazlık0/16, SBI no_face0/16. Tam yüz kutusu bulmak manipülasyon sınıflandırmasının doğru olduğunu göstermedi. Bu kaynakta modeller **birbirini tamamlamadı**. Önceki Celeb-DF içi sonuç SBI'nin aynı kaynaktaki eğitimi/model seçimine bağlıdır; bağımsız dış başarı kanıtı değildir. Otomatik OR/AND, skor ortalaması veya yeni eşik kullanılmamıştır.

Kanıtlar yerelde `outputs/real_photo_research_v1/external_hf_v1/` (rapor, registry bağlantıları, eşli galeri), `outputs/dual_photo_detector_research_v1/`, `outputs/sbi_photo_detector_ablation_v1/`, `outputs/real_photo_research_v1/photo_evaluation_manifest.json` altındadır. Fotoğraflar/kişisel kaynak kayıtları GitHub'a gönderilmez.

## Kurulum ve geri yükleme

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

Klon sonrası bu dosyalar yoksa analiz çalışmaz. Fine-tune checkpointlerini GitHub'dan otomatik temin eden genel indirme URL'si yoktur; proje sahibinin uygun yetkili yedeği gerekir. Upstream kaynak/revizyon/hash listesi `scripts/setup_video_forensic.py` içindedir; script kurulum/indirmeyi yalnız açık bayrakla yapar. Günlük başlatma eğitim, veri indirme veya resmî test başlatmaz.

**Taşınabilirlik doğrulaması:** Eğitim config'inin `implementation_sha256` alanındaki tarihsel Windows yolları, `src/runtime_provenance.py` tarafından mevcut proje kökündeki dokuz bilinen kaynak kimliğine eşlenir. Özgün SHA-256 değerlerinin tamamı doğrulanır; config, model ve preprocessing değiştirilmez. İlgili kaynakların satır sonları `.gitattributes` ile korunur. Farklı klasörde gerçek fotoğraf ve CUDA video smoke kontrolü aynı skorları üretti. Bu, gerçek Mac veya farklı donanımda kabul testi değildir. Git yedeği büyük bilimsel dosyaların fiziksel bağımsız yedeği değildir.

## Kod haritası ve korunan araştırma arşivi

```text
assets/studio/                  Yerel Studio frontend
assets/desktop/                 Windows simgesi
scripts/studio_app.py           Ana masaüstü giriş noktası
src/studio_server.py            Oturum, yükleme, önizleme ve analiz yaşam döngüsü
src/studio_worker_gate.py       Sahiplik atandıktan sonra worker başlatma kapısı
src/desktop_job.py              Yalnız sahip olunan Windows süreç ağacını kapatma
scripts/demo_worker.py          Ortak V3/Grad-CAM ve adaptif SBI işçisi
src/demo_{media,preview,services}.py  Dosya kontratı, önizleme ve GPU kilidi
src/model.py, predict.py, explainability.py  Fotoğraf inference ve Grad-CAM
src/video_adaptive.py, video_forensic.py    Aktif video kontrolü ve SBI yükleyicisi
src/train*.py, video_*.py        Korunan tarihsel eğitim/değerlendirme protokolleri
tests/                         Dar testler ve tarihsel regresyonlar
docs/                          Değiştirilmemiş tarihsel protokol/deney kayıtları
research/                      Küçük, fotoğraf içermeyen sayısal kanıt özeti
outputs/                       Yerel bilimsel arşiv, ortamlar, ağırlıklar; Git dışı
data/                          Datasetler; Git dışı
```

Eski `app.py` decoder'ı `src/demo_media.py` üzerinden Studio'da hâlâ kullanılır. `src/desktop_runtime.py` Studio'nun sahipli geçici dosya temizliğini sağlar. `app_unified.py`, `scripts/desktop_app.py`, `src/desktop_server.py` eski arayüzler ve yaşam döngüsü testleri/restore için korunmuştur. Bunlar güncel Studio giriş noktası değildir. Masaüstüyle aynı Studio arayüzünü tarayıcıda açmak için:

```powershell
.\.venv\Scripts\python.exe scripts/start_deepfake_app.py
```

Studio tarayıcı sunucusu yalnız `127.0.0.1:8502` üzerinde çalışır; oturuma özel tam adres Terminal'de yazılır ve varsayılan tarayıcıda açılır. Masaüstüyle aynı `assets/studio/` dosyaları ve `src/studio_server.py` analiz akışı kullanılır. Terminal'de Control+C ile durdurulur. Port doluysa başka uygulamaya müdahale edilmez; boş port için `.\.venv\Scripts\python.exe -B -m src.studio_browser --port 0` kullanılabilir. Eski Streamlit arayüzü gerektiğinde `python -m streamlit run app_unified.py` ile ayrıca açılabilir. Bilimsel scriptlerin import ve sabit hash bağımlılıkları nedeniyle geniş taşıma/yeniden adlandırma yapılmamıştır. `docs/EXPERIMENTS.md` bazı değerlendirme/eğitim scriptlerinin günlük hedefidir; silinmez.

Güncel kurulum/kullanım/restore açıklamaları bu README'de birleştirilmiştir. `docs/` geçmiş aşamaların kaydıdır; “henüz çalışmadı” ifadeleri yazıldıkları aşamayı anlatır. Önemli kayıtlar:

| Kayıt | İçerik |
|---|---|
| [EXPERIMENTS](docs/EXPERIMENTS.md), [FINAL_SUMMARY](docs/FINAL_SUMMARY.md) | V1–V3 günlüğü ve tarihsel sunum özeti |
| [VIDEO_V1](docs/VIDEO_V1.md), [fine-tuning](docs/VIDEO_FINETUNE_V1.md) | Haar/GRU/cache ve eski sayısal güvenlik protokolleri |
| [Identity split](docs/VIDEO_IDENTITY_SPLIT_V1.md), [balanced](docs/VIDEO_BALANCED_SAMPLING_V1.md), [spatial](docs/VIDEO_SPATIAL_V1.md) | Kod bağlı bileşenleri ve referans deneyler |
| [Forensic protokol](docs/VIDEO_FORENSIC_PROTOCOL.md), [rapor](docs/VIDEO_FORENSIC_V1.md) | SBI başlangıcı, kalite adayı ve kaynak sınırlamaları |
| [Kare protokolü](docs/VIDEO_FRAME_PROTOCOL.md), [kare çalışması](docs/VIDEO_FRAME_STUDY.md) | İç içe6/12/24/32 ve eşli belirsizlik |
| [SBI12 hazırlığı](docs/SBI_12_FINETUNING.md), [adaptif](docs/SBI_ADAPTIVE_PROTOTYPE.md) | Eğitim/resume ve adaptif kuralın tarihsel kaydı |
| [İlk kilitli final protokol](docs/SBI_FINAL_TEST_PROTOCOL.md), [holdout](docs/VIDEO_HOLDOUT_RESULTS.md) | Kimlik kontrolüyle bloklanan tasarım ve ayrı tamamlanmış holdout ayrımı |

`outputs/image_v4_pilot_v1/`, `image_v4_learning_diagnosis_v1/`, `image_v4_1_unfreeze6_v1/`, `image_v4_preparation_v1/` ve fotoğraf araştırma klasörlerindeki script/manifest/hash'ler yerel arşivin parçasıdır. Araştırma adaptörleri bu klasörleri import edebilir; yalnız GitHub clone'unda bu fotoğraf deneylerinin tamamı yeniden çalışmaz. Tam arşiv için izinlere uygun ayrı yedek gerekir. Final sunum paketi `outputs/figures/final/` içindeki10 PNG/6 SVG ve `package_manifest.json` ile korunur; boş yeni hedefe rapor üretimi için `scripts/build_final_report_pack.py` vardır, yeni inference yapmaz.

## Güvenli doğrulama ve katkı

Dar kontroller; resmî veri/test veya uzun eğitim çalıştırmaz:

```powershell
.\.venv\Scripts\python.exe -B -m pytest tests/test_studio.py tests/test_unified_demo.py tests/test_desktop.py tests/test_desktop_windows.py tests/test_video_adaptive.py tests/test_explainability.py -q -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check src scripts tests app.py app_unified.py
.\.venv\Scripts\python.exe -m ruff format --check src scripts tests app.py app_unified.py
git diff --check
```

Tüm tarihsel testleri körlemesine çalıştırmayın: bazı eğitim fixture'ları büyük dosya oluşturabilir. Gerçek Windows Job/soket testleri uygun yerel izin ister. Python projesidir; TypeScript/typecheck veya frontend paket build'i yoktur. JavaScript dağıtımı yerel statik dosyalardır. Otomatik viewport/süreç kontrolleri fiziksel Windows DPI, codec ve insan tarafından pencere kullanım kabulünün yerine geçmez.

Yeni eğitim, checkpoint/manifest/eşik değişikliği ve resmî20K/518test tekrarı ayrı kapsamdır. Bilimsel kayıtlar, veri, crop cache ve sanal ortamlar temizlik amacıyla silinmez. `.gitignore` geçmişte commit edilmiş bir dosyayı geçmişten kaldırmaz; force push veya history rewrite yapılmaz. Gizli anahtarlar, `.env`, Streamlit secrets, kişisel fotoğraflar, ortamlar ve ağırlıklar gönderilmez.

Sonraki araştırma önceliği, üretim kökeni/etiketi doğrulanmış farklı face-swap kaynaklarında kişi ve türev ailelerini birlikte tutan geliştirme verisidir. Bağımsız nihai test baştan ayrılmalı; geliştirmede açılmamalıdır. Gerçek fotoğrafta yanlış alarm ve kaçırılan sahte maliyetleri birlikte değerlendirilmeden iki skoru tek karara dönüştürmek uygun değildir. Studio entegrasyonu, dış doğrulama ve olası onedir/installer paketleme ayrı kabul adımlarıdır; tüm ML ortamlarını tek EXE'ye gömmek bu sürümün kapsamı değildir.
