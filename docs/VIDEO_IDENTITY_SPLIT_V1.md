# Video filename kimlik/kaynak ayrımı V1

Yeni manifest: `outputs/video_identity_v1/manifest.json`. Eski manifest,
fotoğraf V3, frozen GRU V1/V2, feature cache ve fine-tuning benchmarkları
korunur. Yeni eğitim veya büyük preprocessing başlatılmadı.

## Eski protokolün sınırı ve çözüm

Eski source-family grupları first-code + clip-index bazındaydı; aynı klibin
bilinen manipülasyonlarını koruyor, aynı filename kişi/donor kodunun başka
klipler üzerinden validation'a geçmesini engellemiyordu. 58 ortak kodun
detayları `shared_codes.csv` ve `split_report.json` içinde. Aynı rolün iki
splitte görülmesi: 39 REAL kodu, 37 FAKE first-code, 57 FAKE second-code.
342 yönlü manipülasyon kod çifti hem train hem validation'da yer alıyordu.
Dosya adındaki first/second rollerinin gerçek kişi/carrier/donor metadata'sı
olduğu doğrulanmadı; rapor bu yüzden rol adlarını filename düzeyinde tutar.

[Celeb-DF makalesi](https://arxiv.org/html/1909.12962) gerçek kaynakları ve
kişi çiftleri arasında yüz değiştirmeyi açıklar. Yerel dosyalarda kaynak kişi,
donor klibi, YouTube URL/kanal/session eşlemesi yok. Kimlik veya detector
değişimi yapmadan bu eksikliği kapattığımız iddia edilmez.

`src/video_identity_split.py` her FAKE videonun **iki kodunu birlikte**
bağlayan graph bileşenlerini kurar. Resmi test/karantinanın yalnız filename
metadata'sı bilinen ilişkileri korumak için graph'a dahil edilir; video
içerikleri ve test model sonuçları kullanılmaz. Bileşenlerin tamamı tek splitte
kalır. Bileşen seçimi yalnız eligible inventory'nin REAL/FAKE validation
oranını yaklaşık %20 tutar; model başarımı veya cache kalite oranıyla tuning yok.

| Bileşen | Kod sayısı | Eligible REAL | Eligible FAKE | Yeni split |
|---|---:|---:|---:|---|
| Büyük bağlı grup | 38 | 186 | 1424 | Train |
| id36 | 1 | 5 | 0 | Train |
| id39–id48 | 10 | 63 | 497 | Validation |
| id49–id58 | 10 | 64 | 443 | Train |

Kalan 230 YouTube video adı seed 42 ile video düzeyinde ayrılır: train 184,
validation 46. YouTube identity/source disjoint olduğu **doğrulanmamıştır**.
59 bilinen Celeb filename kodu 49/10 ayrılır; train/val ortak kod ve kaynak
ailesi sıfır, 2912 eligible videodan hiçbiri kaybedilmez.

## Sayılar ve korunmuş holdout

| Split | Inventory toplam | REAL | FAKE | Kabul | Kabul REAL | Kabul FAKE | Ret |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 2306 | 439 | 1867 | 2223 | 417 | 1806 | 83 |
| Validation | 606 | 109 | 497 | 602 | 106 | 496 | 4 |

Inventory kaynak dağılımı Celeb-real / YouTube-real / Celeb-synthesis:
train **255 / 184 / 1867**, validation **63 / 46 / 497**.
Kabul edilmiş dağılım train **246 / 171 / 1806**, validation **63 / 43 / 496**.
Train/val acceptance oranı farklıdır; crop/pose/domain kalite etkisi devam edebilir.
Validation yalnız 10 Celeb kodlu bir bileşenden geldiği için kişi/domain kapsamı
sınırlıdır; eski validation skoru ile doğrudan deney karşılaştırması yapılmaz.

Resmi test **518 = 178 REAL + 340 FAKE**, eski karantina
**3099 = 164 REAL + 2935 FAKE**; kayıtlar byte-for-byte metadata eşitliğiyle
eski manifestle karşılaştırıldı, split üyelikleri değişmedi. Bilinen eski
first-code kaynak-aile holdout izolasyonu korunur. Resmi test filename kişi
kodları train ile 46, validation ile 10 ortaktır; **test identity-disjoint değildir**.

## Cache ve temiz başlangıç

Record alanları değiştirilmediği için aynı content-addressed cache dosyaları
salt okunur kullanılır. 2912 cache kaydı, pipeline/backbone/source SHA ve
3–6 kare raporları denetlendi. Yeni train/val arasında birebir aynı MP4 SHA-256
yok; yeniden encode edilmiş/ortak segment veya farklı isimli aynı kişi/kaynak
bu denetimle tespit edilmiş sayılmaz. Cache yeniden üretilmedi.

Weighted video BCE ağırlıkları sadece **2223 kabul edilmiş train** üzerinden:
REAL **2.6654676259**, FAKE **0.6154485050**. Validation ağırlık hesabına girmez.
Fine-tuning girdisi hâlâ gerçek crop görüntüleridir; cache tensorleri bu yolda
backbone'a eğitim girdisi olmaz, yalnız roster/crop metadata'sı kullanılır.

Yeni validation'daki 606 videonun **500'ü eski train** üyesiydi. Eski V1/V2
ve fine-tuning learned GRU/head ağırlıkları bu validation'a maruz kaldı;
warm-start veya cross-manifest resume yapılmaz. `--fresh-head` ImageNet B0 V1
backbone + seed42 random GRU64/head başlatır; eski video checkpoint'i yüklemez.
Yeni splitte bu bayrak olmadan fine-tuning hazırlığı reddedilir. Aynı yeni
manifest/config üzerindeki kendi checkpointlerinden gerçek resume korunur.

Manifest loader yeni fingerprint'i doğrular, korunmuş orijinal parent/inventory
üzerinden deterministik split'i yeniden kurar. Rehashed yanlış üyelik, test
veya karantina değişimi de reddedilir; yalnız self-hash kontrolüyle yetinilmez.

## Gradient clipping denetimi

Benchmark 563.202391 s, train Macro-F1 0.78760148, validation Macro-F1
0.58010795 / ROC-AUC 0.73333057, REAL recall 0.70873786, FAKE recall 0.62899787.
1889.797241 değeri **clipping öncesi** normdur; PyTorch
[clip_grad_norm_](https://docs.pytorch.org/docs/2.11/generated/torch.nn.utils.clip_grad_norm_.html)
normu hesaplar, gradientleri yerinde sınırlar ve önceki toplam normu döndürür.
Unscale/finite-check → strict clip(max_norm=1) → optimizer sırası doğru.

Yeni helper clipping sonrası gerçek toplam normu da hesaplar; norm sonlu
değilse veya 1.0001 toleransını aşarsa optimizer'a göndermez. Eski
`max_gradient_norm` alanı pre-clip olarak korunur; yeni pre/post max alanları
ve peak batch path/label/length/logit/loss bilgisi loglanır.

GTX 1650 CUDA'da 1889.797 sentetik gradient **1.0**'a sınırlandı. İki mevcut
train videosunda gerçek image forward/backward sonlu; normlar 0.021022 ve
0.776363, bu örnekler eşik altında. **Optimizer step yapılmadı**; ağırlıklar
ve BN değişmedi. Benchmark BN stats orijinal ImageNet/frozen V2 ile aynı.
Eski zirve batch/parametre kaydı olmadığı için 1889'un hangi videodan veya
parametre grubundan geldiği kanıtlanamaz; clipping başarısızlığı demek değildir.
`gradient_audit.json` ayrıntılı kanıtı tutar.

## Sonraki tek epoch benchmark (başlatılmadı)

Önce yeni splitte temiz başlangıçla bir epoch kısmi fine-tuning benchmark:
aynı batch1/FP32/160/6 kare, last-block LR1e-5, GRU/head LR3e-4,
weight decay0.01, seed42, validation Macro-F1 / patience3.

```powershell
.\.venv\Scripts\python.exe -u -m src.video_finetune --run-training --accept-filename-families --fresh-head --manifest outputs/video_identity_v1/manifest.json --device cuda --epochs 1 --backbone-lr 1e-5 --head-lr 3e-4 --run-dir "outputs/video_finetune_v1/identity_fresh_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

Bu komut mevcut checkpointleri resume etmez, ayrı run oluşturur. Yeni protokol
sonuçları ancak aynı yeni split ve temiz initialization ile karşılaştırılır.
Eski V2 Macro-F1 0.68135 bu yeni validation için baseline olarak yeniden
ölçülmez: learned head veriye maruz kalmıştır.

Split üretimini yeniden farklı bağımsız dizinde hazırlamak için:

```powershell
.\.venv\Scripts\python.exe -u -m src.video_identity_split --prepare-split --output-dir outputs/video_identity_v1_recheck
```

## Doğrulama ve dosyalar

**240 otomatik test geçti**; 18 yeni test component/holdout, rehashed manifest
tampering, transitive protected relationships, cache/source duplicate,
class weights, fresh-head zorunluluğu ve optimizer öncesi clipping'i sınar.
Lint, formatter, compileall ve git diff whitespace kontrolü geçti.

Yeni kod: `src/video_identity_split.py`, `tests/test_video_identity_split.py`.
Dar değişiklikler: `video_manifest.py` (yeni schema dispatch), `video_cache.py`
(yeni split için kod/MP4 content denetimi), `video_finetune.py` (fresh-head),
`video_train.py` (post-clip guard ve log). Eski fotoğraf kodlarına dokunulmadı.
Raporlar `outputs/video_identity_v1/REPORT.md`, `split_report.json`,
`analysis.json`, `shared_codes.csv`, `manipulation_pairs.csv`,
`cache_content_audit.json`, `training_readiness.json`, `gradient_audit.json`.
`docs/EXPERIMENTS.md` ve eski model sonuçları değiştirilmedi.
