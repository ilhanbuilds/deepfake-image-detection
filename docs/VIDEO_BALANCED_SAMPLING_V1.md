# Video balanced sampling karşılaştırması

Identity-aware manifest ve kabul edilmiş roster değişmez: train 2.223
(417 REAL / 1.806 FAKE), validation 602 (106 REAL / 496 FAKE).
Resmi 518 test ve ilişkili quarantine videoları loader'a girmez.

Tek değiştirilen deney faktörü sınıf dengeleme politikasıdır:
natural sampling + weighted BCE yerine inverse-frequency
`WeightedRandomSampler(replacement=True)` + **unweighted video BCE**.
Loss'a sınıf ağırlığı verilmez. Örnekleme frame değil video seviyesindedir.
Beklenen sınıf oranı %50/%50; gerçekleşen oran rastgele ve loglanır.

ImageNet EfficientNet-B0 V1 + seed 42 fresh GRU64/head, FP32, batch 1,
6 seçilmiş kare, 160px, AdamW, features[8] LR 1e-5, GRU/head LR 3e-4,
weight decay 0.01, clip norm 1.0 korunur. Diğer feature blokları ve tüm BN
running statistics frozen. Aynı Haar/IoU crop pipeline canlı görüntü üretir;
feature cache yalnız kabul/ret ve crop metadata tutarlılık denetimidir.
3–6 geçerli kare packed GRU ile maskelenir; tek video logit/loss kullanılır.

Referans `identity_continue_20261012_210232`, best epoch 2:
validation Macro-F1 0.620533, AUC 0.697390, balanced accuracy 0.638390,
REAL recall 0.462264, FAKE recall 0.814516, Celeb-real recall 0.253968,
YouTube-real recall 0.767442. Komut başlangıçta referans ayarlarını,
manifest/cache fingerprint ve best checkpoint metriklerini doğrular.

Her epoch **2.223 optimizer adımı**, en fazla **3 epoch / 6.669 adım**.
Validation doğal tam roster, eşik 0.5. Best ve patience=3 yalnız validation
Macro-F1 üzerinden. Sampler ve DataLoader aynı seed42 generator'ını paylaşır;
epoch checkpoint'indeki RNG restore, sonraki epoch örneklemesini de korur.
Model, AdamW, scaler, RNG, early stopping ve history birlikte kaydedilir.
Natural mod eski config/resume davranışını korur.

## Loglar

- `history.json`: online train metrikleri, optimizer adım sayısı,
  REAL/FAKE örnekleme oranları, unique coverage, repeated video sayısı,
  repeat draw sayısı ve video başına multiplicity; sınıf bazlı pre/post norm
  ortalama/medyan/maksimum ve clipping oranları; validation metrikleri/kaynak
  recall'ları, epoch süresi/ETA ve CUDA peak memory.
- `epoch_NNN_coverage.json`: gerçek sample draw kapsamı ve bütün doğal
  validation tahminleri. Sample coverage, unique roster coverage ile aynı değildir.
- `comparison.json`: güncel epoch ve referans validation metrikleri/farkları.
- `best_fixed_evaluation.json`: eğitim tamamlandıktan sonra seçilmiş best
  checkpoint'in doğal, tekrarsız train/validation değerlendirmesi ve referansa
  validation farkları. Online train metrikleriyle karıştırılmamalı.
- Immutable epoch checkpoint'leri, `best_video.pth`, `last_video.pth`,
  `config.json`, `roster_audit.json`, `experiment_record.json`.

Clipping rate, norm > 1 olan **video optimizer adımlarının** sınıf içi oranıdır.
Pre/post norm AdamW parametre güncelleme normu değildir. Sampler, sık REAL
tekrarına ve daha düşük FAKE unique coverage'a yol açar; metriklerle birlikte
bu fark raporlanmalıdır. Bu deneyde dengeleme politikasının toplam etkisi
ölçülür; sampling ve loss değişikliğinin bağımsız etkileri ayrıştırılmaz.

## Tam epoch başlatmayan CUDA kontrolü

`outputs/video_finetune_v1/balanced_readiness_20261012/cuda_smoke/smoke.json`:
8 gerçek CUDA FP32 unweighted güncelleme sonlu; yalnız son blok/GRU/head
güncellendi, frozen prefix ve BN değişmedi. 3/4/5/6 NaN padding farkı 0.
PyTorch peak allocated 51.94 MiB, reserved 62 MiB; CUDA context/driver ve
diğer GPU uygulamaları hariç. Smoke ağırlıkları/checkpoint'i kaydedilmedi.

İlk epoch'un salt örnekleme önizlemesi, decode/eğitim yapmadan:

| Sınıf | Draw | Unique / mevcut | Unique coverage | Tekrarlı videolar | Ek tekrar draw |
|---|---:|---:|---:|---:|---:|
| REAL | 1.110 | 387 / 417 | %92.81 | 305 | 723 |
| FAKE | 1.113 | 832 / 1.806 | %46.07 | 221 | 281 |

Sekiz smoke adımı ilk draw'lardır; sınıf oranı küçük örneklemde 6/2 oldu.
Önizleme tam epoch ölçümü değildir. Eğitimde her epoch gerçekleşen değerler
yeniden loglanır; sonraki epoch'lar RNG ilerlemesine bağlı değişir.

## Güvenli eğitim komutu — henüz çalıştırılmadı

Proje kökünden PowerShell:

```powershell
.\.venv\Scripts\python.exe -u -m src.video_finetune --run-training --sampling balanced --fresh-head --accept-filename-families --manifest outputs/video_identity_v1/manifest.json --cache-dir outputs/video_v1/cache --reference-run outputs/video_finetune_v1/identity_continue_20261012_210232 --device cuda --epochs 3 --backbone-lr 1e-5 --head-lr 3e-4 --run-dir "outputs/video_finetune_v1/balanced_$(Get-Date -Format yyyyMMdd_HHmmss_fff)"
```

Referans epoch süreleri yaklaşık 578–582 saniye. Üç epoch eğitim için
yaklaşık **30 dakika**, başlangıç cache/source audit, checkpoint I/O ve
eğitim sonu fixed best değerlendirmesi dahil toplam **35–45 dakika** ayırın.
Balanced sampling REAL/FAKE decode karışımını değiştirdiğinden bu tahmin
kesin ölçüm değildir; gerçek süre/ETA her epoch loglanır.

Resume için ilk komuttaki **aynı run-dir** literal değerini kullanın ve
`--resume auto` ekleyin; timestamp'i yeniden üretmeyin. Balanced/natural
politika veya referans/config değişirse resume reddedilir. Var olan farklı
run üzerine fresh eğitim yazılmaz. OOM/numerical/crop drift sessizce video
atlamaz; tamamlanmamış epoch checkpoint'i üretilmez. Eğitim sonu fixed
değerlendirme kesilirse aynı resume komutu, tamamlanmış eğitimde yeni epoch
açmadan değerlendirmeyi tekrarlar.

Yeni video mimarisi veya kalite-aware aggregation geliştirilmedi. Gerçek
identity-disjoint garantisi verilmez; filename kodları ve bilinen ailelerin
izolasyonu korunur. Resmi test model seçimi veya karşılaştırmada kullanılmaz.

## Otomatik doğrulama

260 test geçti (20 yeni sampling/regresyon testi dahil). Ruff lint/format
(54 Python dosyası), compileall ve git diff whitespace kontrolü geçti.
Streamlit testleri normal süreç ortamında çalıştırıldı; sandbox denemesi
ilk demo testinde takılıp durduruldu. Korunan 17.306 mevcut dosyanın SHA-256
değeri değişmedi. Kontrol kanıtı `balanced_readiness_20261012/checks.json`.

Yeni modüller `src/video_sampling.py`, `src/video_balanced_smoke.py`;
eğitim bağlantısı/logları `src/video_finetune.py`, `src/video_train.py`;
regresyonlar `tests/test_video_sampling.py`. Fotoğraf kaynakları ve modelleri,
eski video checkpointleri, cache, manifestler ve deney çıktıları değiştirilmedi.
