# Araştırma ve değerlendirme notları

Bu belge model geliştirme geçmişini, kayıtlı sonuçları ve değerlendirme sınırlarını bir araya getirir. Güncel kurulum için [proje ana sayfasına](../README.md) bakın. Yerel `outputs/` yolları sonuçların kökenini gösterir; deney arşivinin tamamı dağıtım paketinde bulunmaz.

## İki ayrı analiz hattı

### Fotoğraf: V3 Robust

`REAL=0`, `FAKE=1`. EfficientNet-B0, iki çıkışlı classifier, best epoch 5. Ortak CLEAN dönüşümü: EXIF yön düzeltme → RGB → bilinear/antialias 224×224 → tensor → ImageNet normalizasyonu (`mean=[0.485,0.456,0.406]`, `std=[0.229,0.224,0.225]`). **Görüntünün tamamı** sınıflandırılır; Studio fotoğrafta otomatik SBI yüz analizi yapmaz. CUDA varsa kullanılır, yoksa mevcut fotoğraf akışı CPU seçer.

Grad-CAM tahmin edilen sınıfın softmax öncesi logitini son convolution katmanında (`features.8.0`) açıklar. Optimizer güncellemesi yoktur. Isı haritası manipüle edilmiş alanın maskesi, nesnel doğruluk kanıtı veya nedensel açıklama değildir; arka planı da vurgulayabilir.

### Video: SBI epoch 6, adaptif 6→12

SBI (Self-Blended Images) Shiohara ve Yamasaki'nin CVPR 2022 yöntemidir; projenin özgün icadı olarak sunulmaz. Resmî FF-raw başlangıcından yerel Celeb-DF fine-tuning uygulanmıştır. Runtime RetinaFace + hash ile doğrulanan resmî crop kodunu kullanır: bbox çevresinde %12,5 marj, RGB 380×380, **FP32 / 255**; ImageNet mean/std uygulanmaz. Video için CUDA gerekir; sessiz CPU fallback yoktur.

Her karede en büyük yüzün alanının en az yarısı olan kutular korunur; kare skoru tutulan yüzlerin maksimum FAKE skorudur. Video skoru geçerli karelerin aritmetik ortalamasıdır; eşik 0,5. Tarihsel eşit aralıklı altı indeks korunur. `0.4 < p6 < 0.6` ise iç içe örneklemeyle toplam on iki benzersiz kareye genişlenir; sınırlar strict'tir. İlk altı kare eksikse güvenlik gereği ek kareler denenir. Üçten az geçerli yüzlü karede karar verilmez; eksik kareye nötr skor veya kopya atanmaz. Kısa videoda eksik bütçe görünür tutulur.

Video modeli ve preprocessing, fotoğraf araştırmasındaki detector B varyantından ayrıdır. Fotoğraf B, küçük görüntüyü gereksiz yere 2048'e büyütmez; bu değişiklik **video hattına veya Studio fotoğraf kararına taşınmamıştır**. Önceden doğrulanmış yüz crop'larının doğrudan SBI yolu da yalnız araştırmadadır; `no_face` sonucu REAL sayılmaz.

## Veri, kaynak ve kullanım durumu

Aşağıdaki tablo, veri hazırlığı sırasında kaydedilmiş kaynak ve doğrulama durumlarını gösterir.

| Veri / model | Kaynak | Kayıtlı durum ve sınır |
|---|---|---|
| V1 yüz fotoğrafları | `data/raw/` | 12.890 dosya; özgün üretim/kişi ve bazı augmentation etiket kökenleri doğrulanamamıştır. Dosya adından gerçeklik çıkarılmaz. |
| 140K Real and Fake Faces | [Kaggle/xhlulu](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | İndirme kaynağı doğrulandı; lisans alanı **Other (specified in description)**. `license_verified=false`, `verification_pending`. Yayıncı 70K FFHQ/Flickr gerçek ve Bojan StyleGAN koleksiyonundan 70K sahteyi 256px'e dönüştürdüğünü bildirir. |
| FFHQ/Flickr kaynak zinciri | [NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset#licenses) | Dosya başına Flickr lisansı/yazar/atıf kaydı gerekir. Seçilen REAL dosyalar bu metadata ile eşlenmiş değildir; FFHQ metadata koşulları görselin bireysel lisansının yerine geçmez. |
| Aday StyleGAN koleksiyonu | [tunguz/1-million-fake-faces](https://www.kaggle.com/datasets/tunguz/1-million-fake-faces) | Kayıtlı alan CC BY-NC 4.0; 140K'nin tartışma bağlantısı ile koleksiyon/sürüm/dosya eşlemesi beklemede. StyleGAN kod lisansı tek başına koleksiyon izni değildir. |
| Celeb-DF v2 | [Kaggle/reubensuju](https://www.kaggle.com/datasets/reubensuju/celeb-df-v2), [resmî proje](https://github.com/yuezunli/celeb-deepfakeforensics) | Ayna lisansı **Unknown**; indirme kaynağı doğrulandı, resmî kabul belgesi yok. Ticari olmayan araştırma, kurumsal sorumluluk, atıf, kopyalama/dağıtım ve türev veri kısıtları kayıtlıdır. Ayna ve mevcut crop kullanımı için belge belirsizliği sürer. |
| SBI | [SelfBlendedImages](https://github.com/mapooon/SelfBlendedImages), [makale](https://arxiv.org/abs/2204.08376) | Ticari olmayan iç araştırma ve yeniden dağıtım kısıtları yerel kayıtta mevcut; genel MIT izni varsayılmaz. Kaynak revizyonu `d6d9a351f32334ae9a3a250872c05836e0dd0f2c`. FF++ ön eğitim kaynak örtüşmesi bütünüyle denetlenmiş değildir. |
| Harici fotoğraf geliştirme | [Sowaiba01/Deepfake](https://huggingface.co/datasets/Sowaiba01/Deepfake) | Veri kartı MIT bildirir; özgün görüntü hak zinciri/üretim logları doğrulanmadı. `license_verified=false`, `label_verified=false`; sağlayıcı etiketleriyle hata analizi. |

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

Sabit12 ve adaptif aynı accuracy'ye rağmen farklı hatalar yapar. Ortak inference cache'inin 3.252,605 sn toplam duvar süresi bağımsız yöntem benchmark'ı veya doğrudan hızlanma ölçümü değildir. Daha önceki 602-video validation, kare sayısı ve epoch seçiminde kullanılmıştır. **Fotoğraf %99,635 ile video %97,49 farklı örneklem/görevlerdir; doğrudan model üstünlüğü karşılaştırması yapılamaz.** [Korunan kısa holdout özeti ve dosya hash'leri](VIDEO_HOLDOUT_RESULTS.md).

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
