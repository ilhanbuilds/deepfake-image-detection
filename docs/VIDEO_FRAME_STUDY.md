# SBI kare sayısı deneyi — 14 Ekim 2026

602/602 sabit validation videosu; resmî FF-raw checkpoint, FP32, RetinaFace, 380×380 RGB/255, karede maksimum yüz skoru ve videoda basit ortalama. Eşik her koşulda 0.5. Eğitim adımı: **0**. Resmî test ve quarantine video içeriğine erişim: **0**.

## Ölçülen karşılaştırma

| Kare | ROC-AUC | Macro-F1 | REAL Recall | FAKE Recall | Celeb-real AUC | Video başına ham süre* |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.94353 | 0.83744 | 0.73585 | 0.94153 | 0.94716 | 3.065 s |
| 12 | 0.94954 | 0.86157 | 0.76415 | 0.95363 | 0.95312 | 5.832 s |
| 24 | 0.95371 | 0.86209 | 0.72642 | 0.96573 | 0.95897 | 11.373 s |
| 32 | 0.95192 | 0.85251 | 0.70755 | 0.96371 | 0.95677 | 16.313 s |

*Süre, ölçülen ortak decode süresi + seçilen karelerin ölçülen yüz bulma/kırpma/skorlama süreleri toplamıdır. Dört ayrı bağımsız uçtan uca benchmark değildir. Model yükleme, dosya hash kontrolü, eski crop doğrulaması ve JSON yazımı bu tabloda yoktur. Tüm koşullar aynı son kareye ulaşır. Altı kare önce doğrulandığı için ek karelerde ikinci sıralı decode geçişi gerekir; eski altı kare yeniden retrieve/detect/classify edilmez. Bu ikinci decode deneyin gerçek toplam süresine dahildir, koşul maliyetine iki kez eklenmez. GPU ısınması, işletim sistemi ve ardışık fazlar küçük zaman farkları yaratabilir.

32-kare ham ortalaması ayrıca tek karedeki yaklaşık 757 saniyelik olağandışı beklemeyi içerir. O karenin ayrı yeniden ölçümüyle hesaplanan ortalama maliyet tahmini **15.057 s/video**; ham değer ve tüm skorlar korunmuştur. Ayrıntılar süre duyarlılığı bölümündedir.

En yüksek ROC-AUC: **24 kare**. En yüksek Macro-F1: **24 kare**. Bu ayrım yalnızca bu validation üzerinde gözlenen sonuçtur; istatistiksel üstünlük iddiası değildir.

![Karşılaştırma](../outputs/video_frames_v1/comparison.png)

## Altı kare yeniden üretimi ve kontrol edilen değişkenler

602 videonun altı karesi MP4 dosyalarından yeniden çıkarıldı. Eski bbox koordinatları ve yeniden boyutlandırılmış crop piksel SHA-256 değerleri birebir eşleşti. En büyük yüz skoru farkı: **0**. Yeniden üretilen altı kare metrikleri eski sonuç sözlüğüyle birebir eşit: **True**.

Tarihsel seçim `np.linspace(0, frame_count-1, 6).round().astype(int)` idi. Örnek: 315 kareli `Celeb-real/id39_0001.mp4` → `[0, 63, 126, 188, 251, 314]`. Ek kareler en geniş aralığın aşağı yuvarlanan orta noktasından eklenir; eşit aralıklarda erken zaman seçilir. 6 ⊂ 12 ⊂ 24 ⊂ 32. Resmî SBI'nin bağımsız 32 noktalı linspace seçimiyle aynı örnekleme değildir: burada amaç tarihsel altı kareyi koruyan kontrollü karşılaştırmadır.

Çoklu yüzde en büyük yüz alanının en az yarısı kadar olan yüzler korunur; bunların en yüksek FAKE softmax olasılığı kare skoru olur. Resmî crop fonksiyonu, 12.5% bbox marjları, resize, checkpoint ve sınıf sırası değişmedi. İndeksler, bbox'lar, crop hash'leri, yüz skorları ve hata nedenleri ayrı frame cache'inde saklanır. Skorla ağırlıklandırma, threshold tuning veya video eleme yapılmadı.

## Tam metrikler, kaynaklar ve kapsam

| Kare | Accuracy | Balanced accuracy | FAKE precision | REAL precision | Confusion [[TN,FP],[FN,TP]] |
|---:|---:|---:|---:|---:|:---|
| 6 | 0.90532 | 0.83869 | 0.94343 | 0.72897 | [[78, 28], [29, 467]] |
| 12 | 0.92027 | 0.85889 | 0.94980 | 0.77885 | [[81, 25], [23, 473]] |
| 24 | 0.92359 | 0.84607 | 0.94291 | 0.81915 | [[77, 29], [17, 479]] |
| 32 | 0.91860 | 0.83563 | 0.93910 | 0.80645 | [[75, 31], [18, 478]] |

| Kare | Celeb-real recall (63) | YouTube-real recall (43) | YouTube-real AUC | Değerlendirilen | Geçerli yüz karesi | Yüz sayısı | Hatalı / yetersiz kare |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 6 | 0.77778 | 0.67442 | 0.93820 | 602/602 | 3612 | 3724 | 0 / 0 |
| 12 | 0.79365 | 0.72093 | 0.94430 | 602/602 | 7224 | 7438 | 0 / 0 |
| 24 | 0.76190 | 0.67442 | 0.94599 | 602/602 | 14448 | 14911 | 0 / 0 |
| 32 | 0.77778 | 0.60465 | 0.94481 | 602/602 | 19264 | 19876 | 0 / 0 |

Kaynak AUC'leri, ilgili REAL kaynak ile aynı 496 Celeb-synthesis videosunu karşılaştırır. Celeb-synthesis tek başına tek sınıftır; AUC hesaplanamaz. Bozuk veya yüzsüz kareler açık hata olarak tutulur; başka kare/video ile doldurulmaz. Tüm video tahminleri sonuç JSON'unda yer alır.

## Kazanç, kayıp ve doygunluk

| Değişim | ΔAUC | ΔMacro-F1 | Düzeltilen video | Bozulan video | REAL yanlış alarm | Süre oranı (6'ya göre) |
|:---|---:|---:|---:|---:|---:|---:|
| 6→12 | +0.00601 | +0.02413 | 12 | 3 | 25 (önce 28) | 1.90× |
| 6→24 | +0.01018 | +0.02465 | 15 | 4 | 29 (önce 28) | 3.71× |
| 6→32 | +0.00839 | +0.01507 | 16 | 8 | 31 (önce 28) | 5.32× |

| Ardışık adım | ΔAUC | ΔMacro-F1 | Ek saniye/video |
|:---|---:|---:|---:|
| 6→12 | +0.00601 | +0.02413 | 2.767 |
| 12→24 | +0.00417 | +0.00052 | 5.541 |
| 24→32 | -0.00179 | -0.00959 | 4.940 |

Düzeltilen ve bozulan videoların tüm yolları, gerçek etiketleri ve eski/yeni skorları `analysis.json → transitions` alanındadır. Daha fazla kare her video için iyileşme garantilemez.

Gözlenen doygunluk ölçüte bağlı: Macro-F1 12 kareden sonra hemen hemen sabit kalıyor (12→24 farkı yalnızca +0.00052); AUC 24 karede en yüksek noktaya ulaşıyor. 32 karede AUC ve Macro-F1 birlikte geriliyor. Bu noktalar tek iç içe örnekleme düzenindeki gözlemlerdir; evrensel optimum kanıtı değildir.

REAL yanlış alarmlar 6/12/24/32 için 28/25/29/31: daha fazla kare bunları monoton azaltmıyor. Celeb-real AUC 24 karede en yüksekken Celeb-real recall 12 karede en yüksek. YouTube-real recall 12 karede 0.72093 iken 32 karede 0.60465'e düşüyor; sıralama başarısı (AUC) ile 0.5 eşiğindeki karar başarısı ayrı değerlendirilmelidir.

## Eşleştirilmiş belirsizlik

106 bilinen dosya-adı ailesi üzerinden eşleştirilmiş cluster bootstrap; 42, 123, 2026 tohumlarında ayrı ayrı 1000 çekiliş. Aşağıdaki tablo tohum 42'nin yüzdelik %95 aralıklarını gösterir; diğer iki tohum JSON'dadır.

| Değişim | ΔAUC %95 aralığı | ΔMacro-F1 %95 aralığı |
|:---|:---|:---|
| 6→12 | [-0.00131, +0.01361] | [+0.00345, +0.04528] |
| 6→24 | [+0.00091, +0.02070] | [-0.00338, +0.05314] |
| 6→32 | [-0.00104, +0.01860] | [-0.02124, +0.04623] |
| 12→24 | [-0.00251, +0.01103] | [-0.02355, +0.02437] |
| 24→32 | [-0.00535, +0.00195] | [-0.02969, +0.00722] |

Bunlar aynı validation'a koşullu, keşif amaçlı aralıklardır. Dosya ailesi gerçek kişi bağımsızlığını tam garanti etmez. Daha önce aynı validation üzerinde çok sayıda deney yapılması seçim yanlılığı yaratır; bootstrap bunu veya çoklu karşılaştırmayı düzeltmez. Sıfırı içeren küçük farklar kesin başarı olarak yorumlanmamalıdır. Resmî test açılarak seçim yapılmadı.

Üç bootstrap tohumunda da 6→12 Macro-F1 farkının aralığı pozitiftir; AUC farkının aralığı sıfırı içerir. 6→24 AUC aralıkları pozitiftir; Macro-F1 aralığı sıfır sınırı çevresinde tohuma duyarlıdır. 12→24 ve 24→32 için iki ölçütün aralıkları da üç tohumda sıfırı içerir. Dolayısıyla 24'ün 12'ye küçük nokta avantajını veya 32'nin düşüşünü kesin genelleme sonucu olarak sunmuyoruz.

## Pilot, gerçek çalışma süresi ve VRAM

Pilot: her kaynaktan SHA-256(path) sırasıyla 9 video, toplam **27**; seçim skor/hata bilgisine dayanmaz. Üç kaynağı kapsar, tam cohort sınıf oranını temsil etmez. Pilot ek-kare fazı duvar süresi **5.51 dk**. Bu 27 videonun altı kare + ek kare ölçülen inference geçişleri toplamı **6.88 dk**. Pilot cache'i tam deneyde yeniden inference yapılmadan kullanıldı.

602 videonun baseline doğrulama fazı **31.36 dk**; tamamlama fazı **128.37 dk**. Üç fazın kayıtlı duvar süreleri toplamı **165.24 dk** (süreç/model başlatma hariç). Benzersiz video inference geçişleri toplamı **164.47 dk**; ikinci decode geçişlerinin maliyeti **31.91 s**. Fazlar arası rapor/test çalışmaları bu sürelere dahil değildir.

| Kare | Atfedilen toplam süre (dk) | Peak allocated MiB | Peak reserved MiB |
|---:|---:|---:|---:|
| 6 | 30.75 | 978.4 | 2080.0 |
| 12 | 58.51 | 978.4 | 2680.0 |
| 24 | 114.11 | 978.4 | 2680.0 |
| 32 | 163.68 | 978.4 | 2680.0 |

### Olağandışı bekleme ve süre duyarlılığı

Tam çalışma sırasında izleme bağlantısı zaman aşımına uğradı; inference süreci ve cache korunarak devam etti. Windows güç günlüğünde 14:01:09 uyku (Button or Lid) ve 14:01:12 uyanma olayı var. Bu kayıtlar tek başına yaklaşık 757 saniyelik beklemenin tamamının nedenini kanıtlamaz. `power_events.json` ve ham `timing_incident.json` korunur.

Aşağıdaki kareler için bir ısınma + üç ölçümlü tekrar yapıldı. Bbox, crop piksel hash'i ve yüz skorları özgün kayıtlarla karşılaştırıldı. Hiçbir başarı skoru, tahmin, video veya ham süre değiştirilmedi. Yeniden ölçüm sütunu yalnızca işaretli kare süresini üç tekrarın medyanıyla değiştiren ek bir maliyet tahminidir; bağımsız tam-cohort benchmark'ı değildir.

| Video / kare | Ham kare süresi | Tekrar medyanı | Maksimum skor farkı | Etkilenen bütçe |
|:---|---:|---:|---:|:---|
| Celeb-synthesis/id46_id39_0004.mp4 / 177 | 756.915 s | 0.462 s | 0 | [32] |

| Kare | Ham ortalama s/video | Ham medyan s/video | Tek-kare yeniden ölçümüyle ortalama tahmini |
|---:|---:|---:|---:|
| 6 | 3.065 | 2.901 | 3.065 |
| 12 | 5.832 | 5.443 | 5.832 |
| 24 | 11.373 | 10.558 | 11.373 |
| 32 | 16.313 | 13.982 | 15.057 |

Bellek, ilgili seçili karelerin işlemleri sırasında CUDA allocator'dan ölçülen tepe değerdir; her fazda aynı model/detektör birlikte yüklenir ve koşullar ortak ölçümleri paylaşır. Reserved değeri allocator geçmişini içerir, dört izole koşulun bağımsız VRAM benchmark'ı değildir. Kareler sırayla, sınıflandırıcı batch=1 çalışır; 32 kare aynı anda GPU'ya yığılmaz.

## Adaptif kare araştırması için tanı

Bu deney temporal bir model öğrenmez; aynı görüntü sınıflandırıcısının zamana yayılmış skorlarını ortalar. İlk altı kare son kareyi de içerdiği için olası adaptif mekanizma tüm videoyu okuma maliyetini ortadan kaldırmaz. Tasarruf ek karelerin detector/sınıflandırıcı işlemlerinde olur; mevcut tasarımda genişletme için ikinci decode geçişi veya aday kareleri RAM'de tutma maliyeti ayrıca hesaba katılmalıdır.

Altı kare skorunun 0.5'e uzaklığına göre önceden sabitlenmiş aralıklar; bunlar bir yönlendirme sistemi veya optimize edilmiş eşik değildir. Her hücre `video sayısı / düzeltilen / bozulan` biçimindedir.

| Altı kare marjı | 12 kare | 24 kare | 32 kare |
|:---|:---|:---|:---|
| [0, 0.05) | 35 / 9 / 3 | 35 / 9 / 2 | 35 / 9 / 3 |
| [0.05, 0.10) | 31 / 2 / 0 | 31 / 2 / 1 | 31 / 2 / 2 |
| [0.10, 0.20) | 89 / 1 / 0 | 89 / 3 / 1 | 89 / 4 / 2 |
| [0.20, 0.50] | 447 / 0 / 0 | 447 / 1 / 0 | 447 / 1 / 1 |

0.5'e yakınlık kalibre edilmiş belirsizlik değildir: model kendinden emin biçimde yanlış olabilir. Adaptif kare sayısı bilinen bir araştırma fikridir; yalnızca bu mekanizmayı eklemek dünya-ilki veya tek başına özgün bilimsel katkı oluşturmaz. Örneğin [AdaFrame (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html) girdi başına kare seçimini, [FrameExit (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Ghodrati_FrameExit_Conditional_Early_Exiting_for_Efficient_Video_Recognition_CVPR_2021_paper.html) kolay videolarda az kareyle erken çıkışı araştırır. Bunlar genel video tanıma çalışmalarıdır; bu SBI/deepfake cohort'unda başarı garantisi vermez. Araştırma, hata/işlem maliyeti dengesini ayrı politika geliştirme verisinde belirleyip sabit bütçe ve rastgele ek-kare kontrolleriyle bağımsız değerlendirmede karşılaştırmalıdır. Bu görevde ana sisteme adaptif mekanizma eklenmedi.

Somut araştırma gerekçesi: altı kare marjı <0.10 olan **66/602 video (%10.96)**, 6→12'nin düzelttiği **11/12 videoyu** içeriyor; aynı grupta **3 tahmin bozuluyor**. Bu yoğunlaşma önce 6, gerekirse 12 ve daha sonra 24 kareyi araştırmayı destekliyor. Ancak 24 karenin bazı düzeltmeleri bu dar grubun dışında; yalnızca 0.5'e uzaklık yeterli olmayabilir. Bu tablo bir adaptif sistemin ölçülmüş başarısı veya doğrulanmış durma eşiği değildir.

## Karar: sabit bütçe, fine-tuning ve özgün katkı

**En yüksek gözlenen başarı: 24 kare.** ROC-AUC 0.95371 ve Macro-F1 0.86209 ile iki ölçütte de en yüksek nokta; yaklaşık 11.37 s/video. Özellikle FAKE recall öncelikliyse bir doğruluk referansı olarak tutulabilir. 32 kare, ek maliyetine karşı gözlenen genel başarı avantajı sağlamıyor.

**En iyi hız/başarı dengesi ve bundan sonraki sabit bütçe önerisi: 12 kare.** Macro-F1 0.86157 ile 24 kareye çok yakın, yaklaşık 5.83 s/video ile neredeyse iki kat hızlı; REAL recall ve balanced accuracy de dört koşul içinde en yüksek. 6→12: AUC +0.00601, Macro-F1 +0.02413, 12 düzelme / 3 bozulma, REAL yanlış alarm 28→25. Ek maliyet yaklaşık 2.77 s/video. 12→24: AUC +0.00417 ve Macro-F1 yalnızca +0.00052; maliyet yaklaşık +5.54 s/video ve REAL yanlış alarm 25→29. Bu, hız ve yanlış alarm hedefleri için 12 kareyi daha makul yapıyor. Mevcut ana SBI inference dosyasının altı-kare varsayılanı bu araştırma görevinde değiştirilmedi.

**SBI fine-tuning planı: başlangıçta 12 kare/video ve sabit 12-kare validation protokolü.** Bu seçim zero-shot deneyden gelen pratik başlangıç bütçesidir; fine-tuned model için kanıtlanmış optimum değildir. 24 kare ek maliyetli karşılaştırma olarak korunabilir. Eğitimden önce 12-kare train cache'i, veri sınırları ve CUDA smoke ayrı doğrulanmalı; inference VRAM değeri eğitim VRAM'i sayılmaz. Mevcut eğitim kodunun altı-kare davranışı değiştirilmedi ve fine-tuning başlatılmadı.

**Adaptif kare sayısı araştırmaya değer; tek başına özgünlük iddiası yeterli değil.** İlk araştırma hattı 6→12→24 olabilir; 32'nin zorunlu son basamak olması bu sonuçlarla desteklenmiyor. Katkı adayı, deepfake tespitinde yanlış alarm/kayıp maliyeti ve hesaplama bütçesini birlikte kontrol eden, ayrı geliştirme verisinde sabitlenmiş bir durma kuralının kaynak değişimi altında değerlendirilmesidir. Özgünlük ayrıca ilgili literatürle karşılaştırılmalı; mevcut validation üzerinde yeni eşikler seçip aynı veride başarı ilan edilmemelidir.

## Koruma ve yeniden çalıştırma

Korunan 17 dosyanın SHA-256 kontrolü geçti: app.py, V3 ve eski video checkpoint'leri, FFraw/FFc23, RetinaFace, manifestler, önceki SBI/kalite/robustness sonuçları ve mevcut SBI inference modülü değişmedi. Eski cache'ler yalnızca okundu. Yeni dependency yok; mevcut izole SBI runtime kullanıldı. Legacy RetinaFace paket metadata sürüm uyuşmazlığı önceki kurulumdan sürüyor; gerçek inference bu ortamda doğrulandı.

```powershell
.\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_frame_study baseline
.\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_frame_study pilot
.\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_frame_study full
.\outputs\video_research_v1\env\Scripts\python.exe -m scripts.recheck_video_frame_timing
.\.venv\Scripts\python.exe -m scripts.report_video_frames
```

Tamamlanmış fazların sonuçlarını ezme girişimi engellenir. Kesilen bir faz aynı komutla tekrar başlatılabilir; hash/provenance kontrolünden geçen video cache'leri yeniden kullanılır. Cache video başına atomik kaydedilir; kesilme anında tamamlanmamış videonun kareleri tekrar hesaplanabilir. Fazlar mutlaka baseline → pilot → full sırasıyla çalışır.

## Yazılım doğrulaması

Tam regresyon: **327 test, 0 failure, 0 error, 0 skip**. Yeni testler; iç içe/benzersiz örnekleme, tarihi indekslerin korunması, kısa video, decode hatası, sabit eşik, cache bütünlüğü, çoklu yüz maksimumu, crop/skor sapması, süre duyarlılığı ve eşleştirilmiş istatistik kontrollerini kapsar.

`ruff check`, `ruff format --check`, `compileall` ve `git diff --check` geçti. Python projesi; TypeScript kontrolü yok. Testler için gereken yerel soket/atomik geçici dosya işlemleri Windows sandbox'ta engellendiğinden tam pytest normal yerel izinlerle çalıştırıldı. Bu test ortamı kısıtı için uygulama kodu değiştirilmedi. Son kayıtlar: `outputs/video_frames_v1/checks.json` ve `regression.xml`.
