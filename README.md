# Deepfake Analiz

Derin öğrenme tabanlı, yerel çalışan fotoğraf ve video deepfake analiz uygulaması. **Studio** arayüzü tarayıcıda veya Windows masaüstü penceresinde kullanılır; fotoğraflarda Grad-CAM, videolarda adaptif kare analizi sunar.

**[Modeller dahil uygulamayı indir · yaklaşık 291 MiB](https://github.com/ilhanbuilds/deepfake-image-detection/releases/download/teslim-2026-11-09/Deepfake_Analiz_Modeller_Dahil.zip)**

## Kurulum ve çalıştırma

1. Paketi indirip ZIP'in tamamını yazılabilir bir klasöre çıkarın.
2. **Python 3.12 (64 bit)** kurulu olmalı. Apple Silicon Mac'te yerel **ARM64** Python kullanılmalı.
3. Bilgisayarınıza uygun başlatıcıya çift tıklayın:

| Bilgisayar | Başlatıcı | Destek |
|---|---|---|
| Windows + CUDA destekli NVIDIA GPU | `Deepfake Analiz Windows.cmd` | Fotoğraf ve video |
| Windows | `Windows Fotograf CPU.cmd` | Fotoğraf ve Grad-CAM |
| Apple Silicon Mac | `Deepfake Analiz Teslim Mac.command` | Fotoğraf ve Grad-CAM |

İlk açılışta internetten bağımlılıklar indirilir ve ortamlar paket klasöründe kurulur. Studio varsayılan tarayıcıda açılır. Terminal penceresini açık tutun; durdurmak için **Control+C** kullanın. Mac'te video analizi desteklenmez. [Mac başlatma yardımı](docs/MACOS.md).

GitHub'ın **Code → Download ZIP** seçeneği yalnız kaynak kodu içerir. Analiz için yukarıdaki modeller dahil paketi kullanın; V3, SBI ve RetinaFace ağırlıkları bu pakette hazırdır. Dataset indirmeniz veya yeniden eğitim yapmanız gerekmez.

## Modeller

- **Fotoğraf:** V3 Robust EfficientNet-B0; görüntü sınıflandırması ve Grad-CAM açıklaması.
- **Video:** Celeb-DF üzerinde fine-tune edilmiş SBI EfficientNet-B4, epoch 6; RetinaFace yüz algılama ve adaptif 6→12 kare analizi.

SBI, Shiohara ve Yamasaki'nin yöntemine dayanır. Projedeki çalışma yerel fine-tuning, veri/değerlendirme protokolleri ve uygulama entegrasyonunu içerir.

## Kayıtlı sonuçlar

| Değerlendirme | Örnek sayısı | Doğruluk |
|---|---:|---:|
| V3 tarihsel CLEAN fotoğraf testi | 20.000 | %99,635 |
| SBI adaptif video VIDEO-HOLDOUT | 518 | %97,4903 |

Bu sonuçlar farklı görev ve veri kümelerine aittir; doğrudan model karşılaştırması değildir. V3 testi geliştirme sürecinde görülmüştür; video holdout tam kişi bağımsız değildir. Harici sekiz face-swap çiftinde iki model de sekiz sahteyi kaçırmıştır. Model skoru kalibre edilmiş kesinlik olasılığı değildir. [Deney ayrıntıları ve sınırlamalar](docs/ARASTIRMA_NOTLARI.md).

## Kod yapısı

| Klasör | İçerik |
|---|---|
| `src/` | Modeller, eğitim/değerlendirme ve uygulama servisleri |
| `scripts/` | Kurulum, başlatma ve bilimsel yardımcı araçlar |
| `assets/studio/` | Ortak HTML/CSS/JavaScript arayüzü |
| `tests/` | Birim ve entegrasyon testleri |
| `docs/` | Kurulum, mimari ve araştırma belgeleri |

124 dar regresyon testi, lint/format ve farklı klasörde gerçek fotoğraf/CPU Grad-CAM/CUDA video kontrolleri geçti. Gerçek Mac ve temiz bilgisayarda sıfırdan kurulum henüz doğrulanmadı. [Geliştirici kılavuzu ve doğrulama komutları](docs/GELISTIRICI_KILAVUZU.md).

## Belgeler ve kaynaklar

[Belge dizini](docs/README.md) · [UBYS teslim notu](docs/TESLIM_UBYS.md) · [Sayısal sonuç kaydı](research/verified_results.json)

[SBI makalesi](https://arxiv.org/abs/2204.08376) · [SBI kaynak kodu](https://github.com/mapooon/SelfBlendedImages) · [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics) · [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
