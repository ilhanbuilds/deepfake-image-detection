# UBYS üzerinden kaynak kod ve çalışan uygulama teslimi

UBYS'nin kullanıcı tarafından bildirilen 100 MB dosya sınırı nedeniyle kaynak kod ve modeller dahil uygulama ayrı dağıtılır.

- **UBYS_Kaynak_Kod.zip**: kaynak kodlar, arayüz, testler, bağımlılık listeleri ve indirme/kurulum açıklaması. Model, dataset, Python ortamı veya rapor/sunum içermez; 100 MB'ın altındadır.
- **Deepfake_Analiz_Modeller_Dahil.zip**: aynı uygulama, V3 fotoğraf modeli, SBI video modeli, RetinaFace ağırlığı ve gerekli config/preprocessing kayıtları. Yaklaşık 291 MiB; GitHub Release dosyası olarak dağıtılır. Dataset veya hazır sanal ortam içermez.

## İndirme

[Teslim sürümü ve dosyalar](https://github.com/ilhanbuilds/deepfake-image-detection/releases/tag/teslim-2026-11-09)

GitHub'ın otomatik **Source code (zip)** dosyasını çalışan paketle karıştırmayın. Analiz için release içindeki **Deepfake_Analiz_Modeller_Dahil.zip** dosyasını indirin ve tamamen çıkarın.

## Başlatma

Python 3.12 gerekir. İlk açılışta internetten bağımlılıklar indirilir ve paketin kendi klasöründe ortamlar kurulur.

| Bilgisayar | Başlatıcı | Kapsam |
|---|---|---|
| Windows, CUDA destekli NVIDIA GPU | `Deepfake Analiz Windows.cmd` | Fotoğraf ve video |
| Windows, GPU gerektirmeyen kullanım | `Windows Fotograf CPU.cmd` | Fotoğraf ve Grad-CAM |
| Apple Silicon Mac, yerel ARM64 Python | `Deepfake Analiz Teslim Mac.command` | Fotoğraf ve Grad-CAM; gerçek Mac testi yapılmadı |

Tarayıcı ve masaüstü aynı Studio arayüzünü kullanır. Açılan Terminal penceresi uygulama boyunca açık kalmalıdır; Control+C ile durdurulur. Mac'te mevcut CUDA video akışı desteklenmez.

## Öğretim üyesinin erişimi

Depo private tutulur. Öğretim üyesinin GitHub hesabının depoya erişimi olmalı ve daveti kabul etmelidir; aynı erişim release dosyaları için de geçerlidir. GitHub'da anonim, yalnız bağlantıya sahip olanlara açık bir depo görünürlüğü yoktur. Public depolar herkese; private depolar yalnız yetkili hesaplara açıktır.

Erişim doğrulandıktan sonra UBYS'ye küçük kaynak ZIP'i yüklenip release bağlantısı açıklamaya eklenebilir. Oturum çerezi, token veya kişisel erişim anahtarı teslim dosyasına konmaz.

Kaynaklar: [GitHub depo görünürlüğü](https://docs.github.com/en/repositories/creating-and-managing-repositories/about-repositories#about-repository-visibility), [büyük dosyaların release ile dağıtımı](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github#distributing-large-binaries).

## Doğrulama kapsamı

124 dar regresyon testi, Ruff lint/format, JavaScript ve Bash sözdizimi kontrolleri geçti. Modellerin ve dokuz provenance kaynağının özgün hash'leri korundu. Farklı yerel kökte gerçek fotoğraf, CPU Grad-CAM ve CUDA video smoke analizi başarılı; GPU skorları önceki demo kayıtlarıyla aynı. Eğitim veya resmi final test yeniden çalıştırılmadı. Gerçek Mac, yeni donanım ve internetten tamamen temiz kurulum ayrıca kabul testi gerektirir.
