# Derin Öğrenme ile Deepfake Yüz Görüntülerinin Tespiti

Üniversite Derin Öğrenme dersi için Python/PyTorch projesi. Amaç, bir yüz
görüntüsünü **REAL** veya **FAKE** olarak sınıflandırmaktır. Bu aşamada yalnızca
altyapı hazırlanmıştır; gerçek dataset ile eğitim başlatılmamıştır.

## Proje yapısı

```text
data/raw/                 # Dataset; Git'e eklenmez
src/
    __init__.py
    dataset.py            # Veri keşfi, ayrımlar, dönüşümler, DataLoader
    model.py              # EfficientNet-B0, checkpoint ve ortam kontrolü
    train.py              # Açık komutla eğitim veya yalnızca veri kontrolü
    evaluate.py           # Test metrikleri ve confusion matrix
    predict.py            # Tek görüntü tahmini
outputs/
    checkpoints/          # best_model.pth, last.pth, epoch_XXX.pth; Git dışı
    figures/              # Confusion matrix
    metrics/              # Test JSON ve eğitim geçmişi CSV
tests/test_pipeline.py
requirements.txt
.gitignore
```

## Kurulum (Windows / PowerShell)

Python 3.12 kullanın. GTX 1650 4 GB için CUDA 12.8 paketleriyle eşleşen
PyTorch 2.11.0 / torchvision 0.26.0 kullanılır. Paket kurulumu internet ve birkaç
GB disk alanı gerektirir. Sürücüde `nvidia-smi` tarafından gösterilen CUDA
sürümü ile PyTorch paketinin CUDA runtime sürümü farklı olabilir.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Aktivasyon zorunlu değildir; aşağıdaki komutlar doğrudan sanal ortam Python'unu
kullanır. Genel Python ortamında CPU sürümü olsa da proje ortamı bağımsızdır.
CPU üzerinde çalışmak için ilk pip komutunda `cu128` yerine `cpu` kullanın.
Kurulum kaynağı: [resmi PyTorch sürüm tablosu](https://pytorch.org/get-started/previous-versions/).

## Ortamı kontrol etme (eğitim yapmaz)

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -m src.model
```

Çıktıda Python/PyTorch sürümleri, `cuda_available`, CUDA runtime, GPU adı ve
VRAM GiB yer alır. `cuda_available: false` ise kurulum kaynağını ve NVIDIA
sürücüsünü kontrol edin. `--device cuda` GPU yoksa açık hata verir; `auto`
CUDA kullanılabiliyorsa GPU'yu, aksi durumda CPU'yu seçer.

## Dataset yerleşimi

```text
data/raw/
    Real/
        face001.jpg
    Fake/
        face002.jpg
```

Klasör adları **Real** ve **Fake** olmalıdır; sınıf içindeki alt klasörler
desteklenir. JPEG, PNG, BMP, WebP ve TIFF okunur. Etiket eşlemesi `REAL=0`,
`FAKE=1` olarak sabittir. Boş/eksik sınıflar ve bozuk görüntüler açık hata verir.
Gri tonlama/RGBA görüntüler RGB'ye çevrilir, EXIF yönü uygulanır.

Seed **42** ile sınıf başına **%70 eğitim / %15 validation / %15 test** hedeflenir.
Tam dosya SHA256, RGB piksel özeti, yatay ayna ve dosya kimliği kopyaları ile
`real_N_aug_M` / `fake_N_aug_M` aileleri aynı grupta tutulur. Gruplar bölünmeden
sınıf dağılımını koruyacak şekilde ayrılır; büyük gruplar nedeniyle oranlar
yaklaşık olabilir. Aynı içerik farklı sınıflarda bulunursa eğitim engellenir.
Dosya içeriği değişmezse ayrımlar tekrarlanır; her ayrım iki sınıfı da içermelidir.
Her görüntü **224×224** boyutuna dönüştürülür ve ImageNet ortalama/standart
sapmasıyla normalize edilir. Yalnızca eğitimde rastgele yatay çevirme uygulanır.

Dataset geldiğinde önce eğitim yapmadan tüm görüntülerin okunmasını kontrol edin:

```powershell
.\.venv\Scripts\python.exe -m src.train --check-data
```

Bu komut tüm görüntüleri decode eder, hash/kopya/yol/grup ayrımını ve varsa
CSV'nin dosya adları üzerinden yerel etiket eşleşmesini kontrol eder. Eski Kaggle
mutlak CSV yolları dosya okumak için kullanılmaz. Model ağırlığı indirmez,
checkpoint oluşturmaz. Aynı denetim gerçek eğitimden önce de yapılır.

**Veri sızıntısı:** Aynı kişi/video/kaynak görüntüden türetilmiş kareler veya
kopyalar farklı ayrımlara düşerse sonuç yanıltıcı olabilir. Bilinen augmentation
aileleri ve tam kopyalar korunur; farklı sıkıştırılmış/renk değiştirilmiş yakın
kopyalar ile aynı kişi/video otomatik tanınmaz. Augmentation indekslerinin
orijinal dosyalarla eşlemesi sağlanmadığından kaynak/kişi düzeyinde tam leakage
garantisi yoktur. Bu eşleme sağlandığında kaynak grupları genişletilmelidir.
Checkpoint ayrımlarını ve dataset içeriğinin SHA256 parmak izini saklar;
resume/değerlendirme içerik değişikliğini reddeder. Dataset'i sabit tutun.
Kod hazır yüz görüntülerini bekler;
otomatik yüz tespiti/kırpma yapmaz.

## Eğitim (yalnızca sonraki aşamada)

**Şu anda çalıştırmayın.** Veri kontrolü ve dataset incelemesi bittikten sonra:

```powershell
.\.venv\Scripts\python.exe -u -m src.train --run-training --device cuda --batch-size 8 --learning-rate 3e-4 --epochs 10 --patience 3 --workers 0 --seed 42 --resume auto
```

Model [torchvision EfficientNet-B0](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.efficientnet_b0)
ve `IMAGENET1K_V1` ağırlıklarını kullanır; ağırlıklar torch önbelleğinde yoksa
internetten indirilir. Gece eğitimi hazırlığında ImageNet ağırlıkları eğitim
yapılmadan önbelleğe alınmıştır. Son Linear katmanı iki çıkışa
uyarlanır; CrossEntropyLoss ve AdamW ile modelin tamamı fine-tune edilir.

Varsayılanlar: batch size **8**, epoch **10**, learning rate **0.0003**, Windows
için workers **0**. CUDA'da `torch.amp.autocast` ve `GradScaler` ile mixed
precision açıktır; CPU'da kapalıdır. `--no-amp` ile devre dışı bırakılabilir.
Bellek hatasında `--batch-size 4` veya `2` kullanın. GTX 1650 üzerinde gerçek
eğitim belleği ve süresi henüz ölçülmemiştir; batch 8 başlangıç ayarıdır.

En iyi validation **FAKE F1** modeli `outputs/checkpoints/best_model.pth`
olarak kaydedilir. Her tamamlanan epoch ayrıca `epoch_001.pth` vb. dosyasına
atomik olarak yazılır; `last.pth` güncellenir. Validation F1 **3 epoch** boyunca
artmazsa early stopping çalışır. Test ayrımı eğitim/model seçiminde kullanılmaz.
`outputs/metrics/history.csv` epoch kaybı ve validation metriklerini içerir.
Her epoch sonunda süre konsola saniye ve dakika olarak yazılır. CSV ve checkpoint
geçmişine `epoch_duration`, `total_training_duration`, `eta` (saniye) ve bunların
`_minutes` alanları eklenir. Süre eğitim + validation döngüsünü kapsar;
veri denetimi, model hazırlığı ve checkpoint/log yazımı dahil değildir.
Toplam süre resume sırasında önceki ölçülmüş epoch'lardan devam eder; kesinti
arasında geçen zaman sayılmaz. Eski checkpoint'lerde ölçülmemiş süreler eklenmez.
ETA ölçülmüş epoch'ların ortalamasına ve toplam epoch üst sınırına dayanır;
early stopping veya son epoch'ta sıfırdır.

Yukarıdaki tek komut, checkpoint yoksa yeni eğitim başlatır; varsa en son
tamamlanan epoch'tan devam eder. Model, AdamW momentleri, AMP scaler,
early stopping, geçmiş, RNG ve loader generator durumları geri yüklenir.
Epoch ortasında kesintide o epoch yeniden yapılır; `--epochs 10` ek 10 epoch
değil toplam üst sınırdır. Early stopping ile bitmiş deney yeniden başlamaz.
Resume için batch/LR/seed/AMP/workers/early stopping ayarları aynı kalmalıdır.
Tekrarlanabilir resume `workers=0` gerektirir. Yeni deney için
`--output-dir outputs/experiment_02` kullanın. Eski checkpoint'ler izinsiz ezilmez.
Resume/AMP yaklaşımı [resmi PyTorch checkpoint](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
ve [AMP](https://docs.pytorch.org/tutorials/recipes/amp_recipe.html) belgeleriyle uyumludur.

## Değerlendirme ve tahmin (eğitilmiş checkpoint gerekir)

```powershell
.\.venv\Scripts\python.exe -m src.evaluate --checkpoint outputs/checkpoints/best_model.pth
.\.venv\Scripts\python.exe -m src.predict path/to/face.jpg --checkpoint outputs/checkpoints/best_model.pth
```

Accuracy, precision, recall ve F1 hesaplanır; precision/recall/F1 için pozitif
sınıf **FAKE**'tir. Confusion matrix satırları gerçek, sütunları tahmin;
sıra **REAL, FAKE**. Sonuçlar `outputs/metrics/test.json` ve
`outputs/figures/test_confusion_matrix.png` dosyalarına yazılır. Tahmin çıktısı
etiket ve softmax olasılıklarıdır; olasılıklar kesin doğruluk garantisi değildir.

## Test ve kod kontrolü

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m ruff format --check src tests
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m pip check
```

Testler geçici küçük görüntülerle ayrımları, veri dönüşümlerini, etiket sırasını,
modelin iki çıkışını, checkpoint okumasını, tahmini ve metrikleri doğrular.
Model testleri `pretrained=False` kullanır; ImageNet indirmesi, optimizasyon
adımı veya gerçek dataset eğitimi yapılmaz. Bu testler model başarısını ölçmez.
Python projesi olduğu için TypeScript/typecheck veya frontend build yoktur.
Dataset, sanal ortam, çıktılar ve `.pt/.pth/.ckpt` dosyaları Git dışında tutulur;
boş dizinlerin sürümlenebilmesi için yalnızca `.gitkeep` dosyaları izlenebilir.
