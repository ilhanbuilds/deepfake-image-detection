# Deepfake Analiz teslim paketi

**Proje:** Derin Öğrenme Tabanlı Deepfake Görüntü ve Video Tespiti

Önce rapor ve sunumun kapaklarındaki üniversite, bölüm, öğrenci/numara ve ders sorumlusu alanlarını doldurun. Yerel dosyalarda zorunlu bir üniversite rapor şablonu bulunmadı; standart A4 akademik düzen kullanıldı.

| Dosya / klasör | Kullanım |
|---|---|
| [Final raporu PDF](rapor/FINAL_RAPOR_TR.pdf) | Öğretim üyesine gönderilecek 17 sayfalık okunabilir rapor |
| [Düzenlenebilir rapor](rapor/FINAL_RAPOR_TR.docx) | Kapak ve son metin düzeltmeleri; değiştirdikten sonra PDF'yi de güncelleyin |
| [Markdown kaynak](rapor/FINAL_RAPOR_TR.md) | Raporun metin ve yerel görsel kaynakları |
| [Sunum](sunum/Deepfake_Analiz_Sunum.pptx) | 12 slayt, düzenlenebilir grafikler/tablolar ve gömülü konuşmacı notları |
| [Konuşmacı notları](sunum/SUNUM_NOTLARI_TR.md) | Kaynaklar, sınırlamalar ve kısa demo akışı |
| gorseller/ | 10 temel şekil + 3 ana karşılaştırma tablo görseli; PNG ve uygun grafiklerde SVG |
| tablolar/ | 8 tablonun Markdown ve UTF-8 CSV sürümleri; birim, dipnot, yerel kaynak yolları |
| [Teslim durumu](TESLIM_DURUMU_TR.md) | Koruma ve doğrulama kapsamı, eksikler, GitHub önerisi |
| SHA256SUMS.txt | Dosya bütünlüğü özeti; kendisi dahil değil |

## Sonuçların kapsamı

Fotoğrafta V3 Robust, videoda SBI epoch 6 ve adaptif 6→12 korunmuştur. V4/V4.1/V5 kabul edilmedi. Fotoğraf ve video sayıları farklı görevlerde ölçülmüştür. V3 tarihsel testi dış kaynak başarısı değildir; VIDEO-HOLDOUT tam kişi bağımsız değildir. Harici sekiz face-swap sahtenin tamamının kaçırılması raporda açıklanır. Skorlar kalibre edilmiş kesinlik olasılığı değildir.

## Demo

Proje kökünde PowerShell:

```powershell
.\.desktop-venv\Scripts\pythonw.exe -B .\scripts\studio_app.py
```

Mevcut geliştirme fotoğrafını seç → analiz → Grad-CAM → kısa geliştirme videosunu seç → adaptif ayrıntılar. Kullanılabilecek kayıtlı girdiler konuşmacı notlarında ve doğrulama kaydındadır. Bu paket uygulama kurulumu, dataset veya model ağırlığı içermez; yerel çalışma ortamı korunmalıdır.

## Paylaşım

Depo görünürlüğü GitHub API üzerinden **PRIVATE** olarak doğrulandı. Kullanıcının açık isteğiyle en son 17 sayfalık rapor, 12 slaytlık sunum ve bu teslim materyalleri özel depoda paylaşılır. Paket, ekran görüntülerine gömülü üçüncü taraf yüzleri içerir; bunların kamuya yeniden dağıtım hakkı doğrulanmamıştır. Private paylaşım, lisans veya kamuya dağıtım izni doğrulaması olarak gösterilmez. Depoyu veya PDF/PPTX/DOCX ve yüz içeren görselleri izin incelemesi olmadan kamuya açmayın. Dataset, ağırlık, ortam, yedek ve cache kopyaları bu pakette bulunmaz.

Paket mevcut `refactor/project-cleanup-20261022` branch'i ve [PR #1](https://github.com/ilhanbuilds/deepfake-image-detection/pull/1) üzerinden paylaşılır. Main'e merge yapılmaz. Studio ve model kaynakları önceki commit'teki sürümle korunur; `.gitignore` kapsamındaki veriler ve ortamlar gönderilmez. `SON_KONTROL_TR.md` ve `TESLIM_DURUMU_TR.md` önceki hazırlık/doğrulama aşamalarının tarihsel kayıtlarıdır.

Son akademik/görsel kontrol: [değişiklikler ve 87 video açıklaması](SON_KONTROL_TR.md). Ayrıntı CSV’si tablolar/09_tarihsel_ret_87.csv içindedir.
