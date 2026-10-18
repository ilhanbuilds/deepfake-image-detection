# Spatial-only CNN video referansı ve balanced deney analizi

## Doğrulanmış balanced sonuç

İncelenen run: `outputs/video_finetune_v1/balanced_20261012_221120_937`.
`best_fixed_evaluation.json` tahminlerinden train/validation metrikleri yeniden
hesaplandı; roster, etiketler ve eşik 0.5 doğrulandı. `best_video.pth` epoch 1
checkpoint'iyle birebir aynı; SHA-256
`e208b8371ebe558cf7c02662ba50e013f23c1473998ce6a4a28c1a2883809f8c`.
Natural referansın best checkpoint'i epoch 2 ve önceki hash'i değişmedi.

`comparison.json` **current epoch 3** karşılaştırmasını içerir; best epoch 1
sonucu için `best_fixed_evaluation.json` kullanılmalıdır.

| Validation, fixed best | Natural + weighted, epoch 2 | Balanced + unweighted, epoch 1 |
|---|---:|---:|
| Macro-F1 | 0.62053 | 0.56310 |
| ROC-AUC | 0.69739 | 0.62837 |
| Balanced accuracy | 0.63839 | 0.59234 |
| REAL recall | 0.46226 | 0.45283 |
| FAKE recall | 0.81452 | 0.73185 |
| Celeb-real recall | 0.25397 (16/63) | 0.30159 (19/63) |
| YouTube-real recall | 0.76744 (33/43) | 0.67442 (29/43) |
| Celeb-real vs Celeb-synthesis AUC | 0.57879 | 0.57255 |
| YouTube-real vs Celeb-synthesis AUC | 0.87116 | 0.71015 |

FAKE klasörü tek sınıflıdır; kendi ROC-AUC'si tanımlı değildir. Yukarıdaki iki
AUC ayrı REAL kaynaklarının aynı 496 Celeb-synthesis videoyla karşılaştırmasıdır.
Celeb-real recall kazancı +3 videoyla sınırlı; YouTube-real'de -4 ve FAKE'te
-41 doğru video kaybı var. Toplam REAL doğru sayısı 49'dan 48'e düştü.
Balanced fixed train Macro-F1 0.71487 / AUC 0.83700; fixed validation
0.56310 / 0.62837. Online train metrikleri bu değerlendirmeye karıştırılmadı.

| Balanced epoch | Val Macro-F1 | AUC | REAL recall | FAKE recall |
|---|---:|---:|---:|---:|
| 1 | 0.56310 | 0.62837 | 0.45283 | 0.73185 |
| 2 | 0.42258 | 0.62626 | 0.73585 | 0.37903 |
| 3 | 0.44042 | 0.64185 | 0.77358 | 0.39315 |

## Kanıt ve hipotezlerin ayrımı

**Gözlem:** sonraki checkpoint'ler validation'da REAL yönüne kayıyor; FAKE
recall yaklaşık %39'a düşüyor. Celeb-synthesis medyan FAKE skoru epoch 1'de
0.93515, epoch 3'te 0.01348. AUC de referanstan düşük olduğu için sorun yalnız
0.5 eşikteki karar kaymasıyla açıklanamaz. Bu tur threshold tuning yapılmadı.

**Gözlem:** ilk epoch REAL draw 1.110, unique 387/417; 305 REAL video tekrarlandı,
723 ek tekrar draw var. FAKE unique 832/1.806 (%46.07). Üç epoch birleşiminde
417/417 REAL ve 1.513/1.806 FAKE görüldü; 293 FAKE video hiç örneklenmedi.
Natural sampling her epoch tüm 2.223 kabul edilmiş train videosunu görür.

**Hipotez:** REAL tekrarları ve azalan FAKE çeşitliliği genellemeyi etkiliyor
olabilir. Daha fazla REAL optimizer adımı, video sırası, gradient clipping ve
AdamW momentleri skorları değiştirebilir. Bunlar gözlenen performans farkının
kanıtlanmış tek nedeni değildir. Natural weighted BCE de beklenen sınıf
katkılarını dengelemeyi hedefler; yalnız "%50 prior değişti" açıklaması yeterli
değildir. Sampling ve loss birlikte değiştiği için etkileri ayrı tanımlanamaz.

**Gözlem:** balanced ilk epoch REAL clipping %65.41, FAKE %57.95; epoch 3'te
%31.62 / %26.14. Clipping sonrası norm maksimum yaklaşık 1.0; NaN/Inf yok.
Bu, clipping'in çalıştığını gösterir. Natural geçmiş online sınıf bazlı clipping
oranları yok; önceki fixed-model probe'u bu online oranlarla eşdeğer değildir.

**Hipotez:** kaynak/identity/crop dağılımı ayrımı sınıf dengesizliğinden daha
temel olabilir. Celeb-only AUC her iki yöntemde yaklaşık 0.57–0.58; YouTube
karşılaştırması daha kolay. Tek seed ve üç epoch ile evrensel yöntem üstünlüğü,
kalıcı başarısızlık veya nedensel sonuç çıkarılmamalı.

## CNN video mimarisi

`SpatialVideoClassifier`: ImageNet EfficientNet-B0 V1 -> her geçerli yüz için
1280 özellik -> lengths 3–6 ile **masked feature mean** -> fresh Linear(1280,1)
-> video logit -> tek weighted BCEWithLogits loss.

Padding önce `torch.where` ile sıfırlanır; NaN padding ortalamaya karışmaz.
Yalnız geçerli kare sayısına bölünür. Model kare sırasına invarianttır.
GRU/RNN yok; kare başına bağımsız REAL/FAKE loss veya kesin FAKE supervision yok.
Bu, uzamsal kanıtı ölçen referans modeldir; özgün kalite-aware katkı olarak
sunulmaz. Daha küçük linear head, GRU'ya göre kapasiteyi de değiştirir; farkı
saf temporal nedensellik diye yorumlamayın.

### İki backbone seçeneğinin amacı

- **Partial (ana karşılaştırma):** aynı ImageNet başlangıcı, yalnız features[8]
  ve fresh linear head eğitilir. Gerçek canlı yüz görüntülerinden son bloğa
  gradient geçer. Frozen prefix, stochastic depth ve tüm BN running stats sabit.
  GRU referansıyla aynı backbone adaptasyonu altında aggregation/head karşılaştırması.
- **Frozen (ek tanı):** tüm backbone frozen; mevcut doğrulanmış ImageNet feature
  cache'i model girdisidir, yalnız linear head eğitilir. ImageNet uzamsal
  özelliklerinin ayrım gücünü ucuz ölçer. Bu seçenek backbone fine-tuning yapmaz;
  partial GRU ile doğrudan temporal ablation sayılmaz. İstenirse ayrı run'da çalışır.

Önerilen ilk deney **partial** moddur. Frozen seçenek bu tur tam eğitilmedi.

## Adil deney koşulları

- Aynı identity-aware manifest ve kabul edilmiş train 2.223 (417/1.806),
  validation 602 (106/496); aynı cache/source hash'leri ve crop kabul/ret roster'ı.
- Aynı Haar/IoU crop, 160px, başlangıçta 6 kare, geçerli 3–6 kare, ImageNet V1
  normalizasyonu. Partial modda cache özellikleri model girdisi değildir.
- Seed 42, fresh spatial head; eski GRU/photo learned ağırlıkları yüklenmez.
- Natural sampling, replacement yok. Her epoch tüm 2.223 video, batch 1,
  FP32, workers 0, tam 2.223 optimizer adımı; en fazla 3 epoch / 6.669 adım.
- Weighted video BCE: kabul edilmiş train'den REAL 2.6654676 / FAKE 0.6154485;
  validation loss unweighted, eşik 0.5 ve validation doğal roster sabit.
- AdamW: features[8] LR 1e-5, head LR 3e-4, weight decay 0.01, clip norm 1.0.
  Early stopping patience=3, best yalnız validation Macro-F1.
- Config, checkpoint/resume, RNG/optimizer/scaler/early stopping, epoch ETA,
  VRAM, overall metrics, kaynak recall/score ve ayrı kaynak çiftleri AUC loglanır.
- Eğitim bitiminde best checkpoint'in tekrarsız tüm train/validation fixed
  değerlendirmesi yapılır. Bir epoch pilotu ile üç epoch referansı tam karşılaştırmayın.

Resmi 518 test ve ilişkili quarantine loader'a alınmaz. Kod/aile ve içerik
izolasyonu kontrol edilir; gerçek identity-disjoint garanti edilmez.

## Hazırlık ve sonraki komut

Gerçek GTX 1650 CUDA partial smoke: 4 weighted optimizer adımı, sonlu
loss/logits/gradient; yalnız features[8]/head değişti, frozen prefix/BN sabit.
3/4/5/6 masking farkları 0, kare sırası farkları yaklaşık 1.49e-8.
PyTorch peak allocated 48.99 MiB, reserved 60 MiB; context/driver hariç.
Smoke parametreleri/checkpoint kaydedilmedi; tam eğitim başlatılmadı.

Referans epoch süreleri yaklaşık 9.6–9.7 dakika; decode/crop hâlâ baskın.
Partial CNN için başlangıç tahmini **9–12 dakika/epoch**, üç epoch **27–36
dakika**; başlangıç audit ve son fixed evaluation dahil **40–50 dakika**.
CNN'nin kesin daha hızlı olduğu iddia edilmez; gerçek süre ilk epoch'ta loglanır.

Proje kökünden güvenli ilk **partial, üç epoch** komutu; bu tur çalıştırılmadı:

```powershell
.\.venv\Scripts\python.exe -u -m src.video_spatial --run-training --accept-filename-families --backbone-mode partial --manifest outputs/video_identity_v1/manifest.json --cache-dir outputs/video_v1/cache --reference-run outputs/video_finetune_v1/identity_continue_20261012_210232 --device cuda --epochs 3 --run-dir "outputs/video_spatial_v1/partial_$(Get-Date -Format yyyyMMdd_HHmmss_fff)"
```

`--epochs 1` yalnız pilot/benchmark olarak kullanılabilir; sonra aynı literal
run-dir ile `--resume auto --epochs 3` toplam bütçeye tamamlanır. Resume için
timestamp'i yeniden üretmeyin. Yeni spatial pipeline GRU checkpoint'lerini
resume olarak kabul etmez; eski çıktı dizinlerine yazamaz. Her epoch checkpoint,
history ve evaluation saklanır. Fixed değerlendirmeler epoch/bütçe bazında
ayrı dosyalarda korunur; güncel alias `best_fixed_evaluation.json`'dır.

Yeni dosyalar: `src/video_spatial_model.py`, `src/video_spatial_evaluate.py`,
`src/video_spatial.py`, `src/video_spatial_smoke.py`, `tests/test_video_spatial.py`,
`scripts/analyze_video_balanced.py` ve bu kılavuz. Önceki kaynak dosyaları,
fotoğraf modelleri, cache, splitler ve deney kayıtları değiştirilmedi.
Analiz kanıtı `outputs/video_spatial_v1/readiness_20261014/balanced_analysis.json`;
CUDA kanıtı `outputs/video_spatial_v1/cuda_prepare_20261014/smoke.json`.

Son doğrulama: **279 test geçti** (19 yeni spatial test dahil), Ruff lint/format
(60 Python dosyası), compileall ve git diff whitespace kontrolü geçti.
Eğitim/resume entegrasyonu sentetik test fixture'ında 2.223 adım/epoch,
weighted loss, optimizer/RNG/state ve immutable fixed değerlendirme çıktıları
ile doğrulandı; gerçek veri üzerinde yalnız dört CUDA smoke adımı yapıldı.
Korunan **17.326 mevcut dosyanın SHA-256 değeri değişmedi**. Uzun eğitim yok.
