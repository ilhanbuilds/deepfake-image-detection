# Konuşmacı notları ve demo akışı

## Slayt 1

Amaç: fotoğraf ve videoyu iki ayrı model hattıyla yerel Windows uygulamasında analiz etmek. Öğrenci, numara, bölüm ve ders sorumlusu bilgilerini sunumdan önce kapakta tamamlayın. Bu sunum yeni bir test çalışması içermez.

## Slayt 2

Sentetik yüz üretimi ile gerçek çekime yüz değiştirme uygulanması farklı dağılımlardır. EfficientNet mimarisi: Tan ve Le, ICML 2019, https://proceedings.mlr.press/v97/tan19a.html . Dış fotoğraf hataları outputs/real_photo_research_v1/external_hf_v1/comparison.json içindedir.

## Slayt 3

Kaynak: scripts/demo_worker.py, src/studio_server.py, src/studio_worker_gate.py. SBI fotoğraf araştırması final uygulamaya entegre değildir. Skorların ortalaması veya otomatik OR/AND kararı yoktur.

## Slayt 4

Kaynaklar: outputs/metrics/final_test_metrics_v3.json; docs/VIDEO_IDENTITY_SPLIT_V1.md; outputs/video_holdout_v1/final_test_20261018/final_results.json. Celeb-DF: Li vd. CVPR2020, https://openaccess.thecvf.com/content_CVPR_2020/html/Li_Celeb-DF_A_Large-Scale_Challenging_Dataset_for_DeepFake_Forensics_CVPR_2020_paper.html . Kabul edilen crop kapsamı ile orijinal manifest sayıları farklıdır. İndirme kaynağı lisans belgesi değildir.
Envanter uzlaştırması: 6529 = (2223 kabul + 83 ret) eğitim + (602 kabul + 4 ret) validation + 518 holdout + 3099 karantina. Retler tarihsel Haar/altı kare cache kabulündendir: 86 insufficient_valid_faces, 1 invalid_or_short_frame_count. Kaynak: outputs/video_identity_v1/manifest.json ve cache_audit.json; SBI accepted_roster aynı kayıtları kullanır. Yeni inference yapılmadı.

## Slayt 5

Kaynak: outputs/metrics/robustness_v3.json, outputs/metrics/final_test_metrics_v3.json ve V3 eğitim kayıtları. Aynı tarihsel test augmentation tasarımında daha önce incelenmiştir. Test optimizer veya checkpoint seçimine katılmamıştır.

## Slayt 6

Kaynak: outputs/metrics/final_test_metrics_v3.json; robustness_v2.json ve robustness_v3.json. CM satırları gerçek etiket, sütunları tahmindir. 20.000 görüntü; 10.000 REAL,10.000 FAKE. V2/V3 farkı yalnız augmentation etkisini izole etmez.
Görsel düzenleme: tablo ve grafik yazıları büyütüldü; veriler ve eşik değiştirilmedi. Grafikteki beş koşul kayıtlı robustness ölçümlerinin seçilmiş gösterimidir; rapor tam koşul tablosunu içerir.

## Slayt 7

SBI: Shiohara ve Yamasaki, CVPR2022, https://openaccess.thecvf.com/content/CVPR2022/html/Shiohara_Detecting_Deepfakes_With_Self-Blended_Images_CVPR_2022_paper.html . RetinaFace: Deng vd., CVPR2020. Yerel config outputs/video_research_v1/finetune12/main_20261014/config.json. SBI özgün yönteminin katkısı yazarlarına aittir; yerel katkı kontrollü uyarlama ve uygulamadır.

## Slayt 8

Kaynak: src/video_adaptive.py; outputs/video_holdout_v1/final_test_20261018/final_results.json. Katı aralık 0.4<p6<0.6; eksik ilk altılı da genişletir. Son eşik 0.5. Kare sayısındaki %48.0695 azalma uçtan uca hızlanma değildir.

## Slayt 9

Kaynak: outputs/video_holdout_v1/final_test_20261018/final_results.json. Eşik0.5; bütün518video değerlendirildi. Sabit12 CM[[171,7],[6,334]], adaptif CM[[170,8],[5,335]]. Eşit accuracy farklı hataları gizlememelidir. Test dosya/aile bağımsızlığıyla sınırlıdır; tam kişi bağımsız değil.
Kanıt: 18 Ekim 2026 sabit VIDEO-HOLDOUT kaydı; dosya SHA-256 8f443470e1b3c260d6e08a8ca7bc497ed743bb7b318cfb365df69aedfff64f76. Sonuçlar yeniden ölçülmedi.

## Slayt 10

Görseller: outputs/desktop_studio_v1/03_photo_result.png ve outputs/figures/final/gradcam_real_01_26663.png. Güncel Studio Chromium görünümü; native pencere/DPI kabul kanıtı değildir. Grad-CAM: Selvaraju vd. ICCV2017, https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html . Gerçek kayıtlı çıktı; manipülasyon maskesi veya kesinlik kanıtı değil.

## Slayt 11

Kaynak: outputs/real_photo_research_v1/external_hf_v1/comparison.json; outputs/image_v5_final_pilot_v1/comparison.json; V4 ve V4.1 comparison.json. Farklı geliştirme kümeleri ortak benchmark değildir. SBI fotoğraf B referansı video holdout sonucu ile karıştırılmamalıdır.

## Slayt 12

Demo: .\.desktop-venv\Scripts\pythonw.exe -B .\scripts\studio_app.py . 1) Mevcut geliştirme fotoğrafını seç, analiz ve Grad-CAM göster. 2) Mevcut kısa geliştirme videosunda6→12 ayrıntısını göster. 3) Karar eşiği ve skorun kalibrasyon sınırını açıkla. Resmî test veya eğitim başlatma. Çalışma yerel akademik prototiptir.
