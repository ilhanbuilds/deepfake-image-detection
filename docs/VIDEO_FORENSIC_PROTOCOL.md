# Video forensic deney protokolü — 14 Ekim 2026

Bu belge, kalite birleştirme karşılaştırmasının sonuçları hesaplanmadan önce sabitlenen tasarımı tanımlar. Başarı veya dünya literatüründe ilk olma iddiası değildir. Nihai ölçümler `VIDEO_FORENSIC_V1.md` raporundadır.

## Korunan sınırlar

V3 görüntü modeli, `app.py`, identity manifest, eski video checkpoint'leri ve eski feature cache değişmez. Resmî test ve quarantine videoları açılmaz. Yalnız mevcut kabul edilmiş train/validation listeleri kullanılır. Manifest bütünlüğü kontrolünde manifest metadata'sı okunur; test videosu veya resmî test listesi dosyası okunmaz. Eğitim ayrı `--run-training` bayrağı gerektirir.

## Ana baseline

Resmî SBI FF-raw EfficientNet-B4; RetinaFace bbox, resmî test crop fonksiyonu, 380×380 RGB, float32 / 255, iki sınıflı softmax'ın 1 numaralı FAKE sütunu. Mean aggregation ve eşik 0,5. Model score'u kalibre edilmiş gerçek dünya olasılığı olarak sunulmaz.

602 video, tarihsel cache'teki aynı altı örnekleme zamanı üzerinden işlenir. Yüz crop'ları ve geçerli kare kümeleri Haar ile aynı olmak zorunda değildir. Her karede alanı en büyük yüzün en az yarısı olan yüzler tutulur; kare skoru bunların maksimum FAKE skorudur. En az üç geçerli zaman noktası istenir; başarısız video reddedilir ve coverage paydasında tutulur. Yüz bulunmayan videoya sessizce 0,5 atanmaz. Bu, resmî makaledeki 32 kare/test protokolünün aynısı değildir.

Eski GRU ve spatial sonuçlarıyla fark; ön eğitim, kapasite, çözünürlük, crop ve aggregation birlikte değiştiği için tek bileşenin nedensel etkisi diye yorumlanamaz. Başarısız SBI videoları olursa tam roster coverage'ına ek olarak ortak başarılı video kesişiminde eski modellerin metrikleri yeniden hesaplanmalıdır.

## Proje katkısı adayı: sınırlandırılmış göreli kalite birleştirmesi

Kare skoru `p_i` modelden gelir. Kalite özellikleri model skorundan ve etiketten bağımsızdır:

1. `s_i`: crop gri görüntüsünün Laplacian varyansının `log(1+variance)` dönüşümü. Çoklu yüz durumunda tutulan yüzlerin minimum değeri kullanılır.
2. `b_i`: resmî bbox'un beklenen %12,5 kenar payıyla oluşacak crop alanına göre gerçekten elde edilen alan oranı; `[0,1]` aralığına kırpılır. Bu yalnız sınırda kesilme göstergesidir, yüz görünürlüğü veya calibrated detector confidence değildir.

`r_i = (average_rank(s_i)-1)/(T-1)`. Bütün değerler eşitse veya T=1 ise `r_i=0,5`.

`q_i = (r_i+b_i)/2`, `u_i=0,5+q_i`, `w_i=0,5/T + 0,5*u_i/sum(u)`.

Video skoru `P=sum(w_i*p_i)`; karar `P>=0,5` ise FAKE.

Burada `u_i∈[0,5;1,5]`, `sum(w)=1`, `2/(3T) <= w_i <= 2/T`. Hiçbir geçerli kare atılmaz; kalite kararın tamamını ele geçiremez. Kare sayısı azaltıldığında ağırlıklar yeniden normalize edilir.

Keskinlik sıra dönüşümü video içinde yapılır. Dolayısıyla global training/validation normalizasyon parametresi öğrenilmez; validation'dan ölçek, eşik veya katsayı seçilmez. Ortak pozitif ölçek ve ofset keskinlik sıralamasını değiştirmez. Bu özellik kaynak farkını tamamen gidermeyi garanti etmez. Bbox hareketi ceza olarak kullanılmaz: seyrek örneklenen gerçek kafa/kamera hareketi güvenilmezlik değildir.

En yakın yöntem [Automatic Face Weighting (Montserrat vd., CVPRW 2020)](https://arxiv.org/pdf/2004.12027), EfficientNet özelliklerinden hem logit hem öğrenilebilir yüz ağırlıkları üretir; ağırlıklı logit ortalamasını sigmoid ile dönüştürür ve ayrıca GRU kullanır. Buradaki fark; etiketle eğitilmeyen ölçümler, video içi sıralama, açık ağırlık sınırı ve olasılık ortalaması üzerindeki sınırlı düzeltmedir. Ağırlıklı ortalamanın kendisi bizim buluşumuz değildir. Katkı, bu kısıtların problem için tasarlanıp aynı checkpoint/kareler üzerinde kontrollü sınanmasıdır.

Beklenen fayda, bazı kareler bulanık/kesilmiş olduğunda daha güvenilir karelerin etkisini sınırlı artırmaktır. Olası zararlar: sahte görüntülerde keskinlik bir adli iz olabilir; bütün kareler kötü olduğunda sıralama anlamsız ayrımlar yaratabilir; crop sınırları manipülasyon izi taşıyabilir; keskin yanlış yüz veya arka plan yüksek kalite sayılabilir. Bu yüzden temiz veri ve bozulma sonuçları birlikte değerlendirilir.

## Önceden sabitlenmiş karşılaştırmalar

| Deney | Değişen unsur | Sabit kalanlar |
|---|---|---|
| Mean baseline | Eşit ağırlık | Checkpoint, kareler, crop, eşik |
| Full kalite | Keskinlik + sınır oranı | Aynı skorlar |
| Sharpness ablation | Yalnız keskinlik | Aynı skorlar, mixing=0,5 |
| Boundary ablation | Yalnız sınır oranı | Aynı skorlar, mixing=0,5 |

602 videoda ROC-AUC, Macro-F1, balanced accuracy, accuracy, iki sınıf precision/recall/F1, confusion matrix; kaynak recall'ları ve Celeb-real / YouTube-real vs synthesis AUC raporlanır. Ham ortalama keskinliğin REAL/FAKE AUC'si ve yalnız REAL videolarda Celeb/YouTube kaynak AUC'si tanısaldır; kaliteyi tuning için kullanılmaz. AUC'nin 0,5'ten her iki yönde uzaklığı kaynak ilişkisine işaret edebilir; nedensel kanıt değildir.

## Bozulma pilotu ve nihai robustness

İlk pilot: her kaynaktan `sha256(path)` sırasıyla 12 video, toplam 36; seçim skora bağlı değildir. 24 REAL/12 FAKE oranı nedeniyle bu pilotun accuracy/F1 değerleri 602 video değerleriyle doğrudan kıyaslanmaz.

Seed'ler 42, 123, 2026. Aynı video için her iki karar yöntemine aynı bozulma uygulanır. Sınıfa göre şiddet seçilmez. Koşullar:

- Clean.
- Rastgele seçilen zaman noktalarının yarısında Gaussian blur, sigma=1,5.
- Aynı seçim kuralıyla JPEG kalite=50; video codec sıkıştırmasıyla eşdeğer olduğu iddia edilmez.
- En az üç geçerli kare kalacak şekilde bir zaman noktasını çıkarma.
- Crop içinde ±19 piksel (%5) sentetik öteleme ve yansıtılmış kenar doldurma.

Pilot önceden çıkarılmış crop'larda çalışır. Detector tekrar çalıştırılmaz; crop translation gerçek videodaki bbox jitter/face detection başarısızlığı testi değildir. Keskinlik tüm koşullarda, clean dahil, 380×380 üzerinde yeniden hesaplanır. Bbox alan oranı sabittir. Böylece clean/corrupt kalite ölçeği eşleşir; bu pilotun temiz ağırlıkları ham crop keskinliği kullanan ana clean deneyden farklı olabilir.

Nihai robustness: aynı protokolü tüm 602 videoya yay; ayrıca orijinal video karelerinde gerçek bbox jitter ve codec sıkıştırmasıyla detector coverage'ını tekrar ölç. Sabitlenmiş iki şiddet seviyesi ve üç seed kullan. Her seed için eşleştirilmiş AUC/F1 farkı ve bozulmaya bağlı düşüşü raporla. Video bağımsız bootstrap tek başına yeterli değildir; bilinen filename aileleriyle grup bootstrap ve mümkünse kaynak/kimlik gruplanması kullan. Kimlik bağlantıları yüzünden bağımsız grup sayısı düşükse belirsizliği açıkça belirt. Üç bozulma seed'i üç bağımsız model eğitimi değildir.

## Eğitim deneyi

Ana zero-shot model korunur. Ayrı bir kopya üzerinde ilk epoch yalnız classifier; ikinci epoch'tan itibaren B4 `_blocks[24:]`, projection ve classifier eğitilebilir. Bunlar gerçek MBConv bloklarını içerir. FP32, frame batch=1, dört video gradient accumulation, doğal örnekleme, training roster'ından sınıf ağırlıkları, AdamW LR=1e-5/1e-4, clip=1, patience=3, üst bütçe=8 epoch. BN istatistikleri sabit; dropout/drop-connect kapalıdır.

Training her geçerli karedeki en büyük yüzü, video etiketinden zayıf denetimli örnek olarak kullanır. Loss, önce karelerde sonra videolarda eşit ağırlıklı CE'dir; doğrudan mean video probability BCE'si değildir. Validation resmî çoklu yüz maksimumu ve zaman ortalamasını kullanır. Bir training/validation video kabulünü kaybederse run durur; sessizce yeni cohort oluşturulmaz.

Pilot yalnız uygulanabilirlik/bellek/nümerik kararlılık göstergesidir. Seçilen LR'nin saatler boyunca istikrarlı veya optimal olduğu henüz bilinmez. Sabit seed ve deterministik CUDA; her epoch model/optimizer/RNG/loader generator/history/early-stop state; best/last ve immutable epoch dosyaları. Resume, son tamamlanan epoch'tan eksik epoch'u yeniden oynatır. İlk epoch tamamlanmadan kesinti varsa `resume auto` mümkün değildir; yeni run adıyla başlanır. Ortam/config/cache/source hash değişirse resume reddedilir.

Fine-tuning, zero-shot'ın validation sonucunu ve özellikle Celeb-real davranışını iyileştirmiyorsa kullanılmaz. Nihai eğitim karşılaştırması üç seed ile yapılmalıdır. Güncel görevde uzun eğitim veya üç-seed eğitim başlatılmaz.

## Bilimsel sınırlar

Mevcut split yalnız bilinen dosya adı aileleri/kodlarını ayırır; tüm gerçek insan kimliklerinin ayrıldığını garanti etmez. FF++ eğitim kaynağı ile yerel Celeb-DF kaynak videolarının tam hash/kişi karşılaştırması yoktur: yayınlanmış protokolde Celeb-DF eğitim verisi değildir, ama bilinmeyen internet kaynak örtüşmesi dışlanmış değildir. Çok sayıda önceki validation denemesi seçim yanlılığı yaratabilir. Resmî test yalnız yöntem, checkpoint ve karar protokolü kesinleştikten sonra bir kez açılmalıdır; bu görevde açılmaz.
