# Mac'te yerel tarayıcı arayüzü

Windows `.lnk` ve `.vbs` dosyaları yerine proje kökündeki **Deepfake Analiz Mac.command** dosyasını kullanın. Proje klasörünün tamamı gerekir; başlatıcıyı tek başına başka klasöre taşımayın. Masaüstüne erişim için Finder'da dosyanın bir takma adını (alias) oluşturabilirsiniz.

## Bir defalık kurulum

Python 3.12 kurulu olmalıdır. Terminal'de proje klasörüne geçip çalıştırın:

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
chmod +x "Deepfake Analiz Mac.command"
```

`.venv` Mac üzerinde oluşturulmalıdır; Windows'taki `.venv` veya `.desktop-venv` klasörlerini kopyalamayın. Tarayıcı sürümü için `requirements-desktop.txt` kurulmaz. Kaynak kod ZIP'i `.command` çalıştırma iznini korur; başka aktarım yöntemlerinde yukarıdaki `chmod` komutu gerekebilir.

Fotoğraf analizi için projenin özgün V3 ağırlığını aynı konuma yerleştirin:

```text
outputs/checkpoints/best_model_v3_robust.pth
```

Beklenen SHA-256: `c7e2c8f06a21a88bfed15f0d63d05b7a345dcae25f874cad3136313331c2ca0c`.

Bu dosya kaynak kod paketinde bulunmaz; yetkili proje yedeğinden ayrıca gerekir. Yerine farklı bir ağırlık koymayın. Dataset indirmek veya yeniden eğitim yapmak gerekmez.

## Başlatma ve durdurma

**Deepfake Analiz Mac.command** dosyasına çift tıklayın. Terminal'de Studio sunucusu başlar ve varsayılan tarayıcıda masaüstü uygulamasıyla aynı arayüz açılır. Adres `http://127.0.0.1:8502/<oturum>/` biçimindedir; tarayıcı otomatik açılmazsa Terminal'de yazılan tam adresi açın. Terminal penceresini açık tutun; durdurmak için **Control+C** kullanın. Tarayıcı sekmesini kapatmak sunucuyu durdurmaz.

Finder başlatmayı engellerse proje klasöründeki Terminal'den çalıştırabilirsiniz:

```bash
bash "Deepfake Analiz Mac.command"
```

8502 portu kullanımdaysa başlatıcı hata verir; mevcut uygulamanın sürecini kapatmaz. Önce önceki Deepfake Terminal oturumunu Control+C ile durdurun.

## Destek kapsamı

- Fotoğraf V3 ve Grad-CAM mevcut CPU yolunu kullanır. İlk model yükleme ve analiz Windows GPU kullanımından daha uzun sürebilir.
- Mevcut SBI video analizi CUDA destekli NVIDIA GPU gerektirir; bu akış Mac üzerinde desteklenmez. Mac'in Apple GPU'suna veya CPU'ya otomatik video geçişi yapılmaz.
- Tarayıcı ve Windows Studio aynı HTML/CSS/JavaScript arayüzünü ve Studio analiz sunucusunu kullanır. Model, preprocessing ve karar eşikleri korunmuştur.
- Başlatıcı yerel bilgisayara bağlanır; internet üzerinde yayın yapmaz.

Bu değişiklik gerçek Mac donanımında henüz doğrulanmamıştır; Mac'teki kurulum ve ilk fotoğraf analizi ayrıca kontrol edilmelidir.
