# Apple Silicon Mac'te başlatma

[Proje ana sayfasındaki modeller dahil paketi](../README.md) indirip tamamen çıkarın. Yerel ARM64 **Python 3.12** ve ilk kurulum için internet gerekir.

**Deepfake Analiz Teslim Mac.command** dosyasına çift tıklayın. Başlatıcı ortamı kurar ve Studio'yu tarayıcıda açar. V3 model dosyası bu pakette hazırdır.

Finder çalıştırma izni vermezse paket klasöründeki Terminal'de:

```bash
bash "Deepfake Analiz Teslim Mac.command"
```

Çift tıklama için izin vermek isterseniz:

```bash
chmod +x "Deepfake Analiz Teslim Mac.command"
```

Terminal'i açık tutun; Control+C ile durdurun. Masaüstüne erişim için Finder'da başlatıcının bir takma adını (alias) oluşturabilirsiniz. Başlatıcıyı paket klasöründen taşımayın; Windows sanal ortamını Mac'e kopyalamayın.

Mac'te fotoğraf ve Grad-CAM CPU yolunu kullanır; mevcut CUDA video akışı desteklenmez. Gerçek Mac donanımında doğrulama henüz yapılmadı.
