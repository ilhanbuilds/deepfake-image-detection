# Son akademik ve görsel kontrol

## Düzenlemeler

- Word ve PDF: 18 sayfalık önceki sürüm 17 sayfaya düzenlendi. Ana metin 11,5 pt, tablolar 10,5 pt. Kapak ve kaynakça dışında zorunlu bölüm sayfa sonları kaldırıldı. Grafikler büyütüldü; tablo başlıkları gövdelerinden kopmayacak biçimde gruplandı. Kapaktaki üniversite, bölüm, öğrenci/numara, ekip ve ders sorumlusu alanları aynen korundu; bilinmeyen bilgi eklenmedi.
- Yeni sayfa düzeninde s.2–4 giriş/veri/model metni sürekli akar. S.5–6 fotoğraf sonuçları ve robustness daha büyük şekil/tablo içerir. S.7–8 veri ayrımı ve 87 video uzlaştırmasıdır. S.9–12 video/Grad-CAM şekilleri, s.13–14 Studio görselleri büyütülmüştür. S.15 model seçimi, s.16 sonuç, s.17 ayrılmış kaynakça/deney kanıtlarıdır.
- PPTX: yalnız 6, 9 ve 10. slaytların görsel nesneleri değişti. Slayt 6: CM ve grafik yazıları büyüdü, uzun eksen etiketleri açılı düzenlendi. Slayt 9: metrik tablosu yazıları büyüdü. Slayt 10: Studio ve Grad-CAM görselleri büyüdü. Diğer dokuz slaydın XML ve ilişki parçaları bayt düzeyinde aynı. Slayt 4’ün yalnız konuşmacı notuna envanter açıklaması eklendi.
- Akademik yayınlar/veri kaynakları ile yerel deney kayıtları ayrıldı. PDF’de tıklanabilir kaynak bağlantıları, deney dosya yolları ve önemli SHA-256 kayıtları var. Sunumun 6/9. slaytlarında görünür deney dosyası bilgisi; tam kaynaklar konuşmacı notlarında bulunur. Grafik değerleri ve gömülü iki Excel veri dosyası korundu.

## 87 video farkı

Tam envanter 6.529 videodur. Eğitim manifesti 2.306 = 2.223 kabul + 83 tarihsel ret; validation manifesti 606 = 602 kabul + 4 tarihsel ret. Holdout 518 ve karantina 3.099 değişmedi.

**6.529 = 2.223 + 83 + 602 + 4 + 518 + 3.099.**

87 video kayıp değildir: 25 REAL, 62 FAKE. Eski cache raporunda 86 insufficient_valid_faces, 1 invalid_or_short_frame_count kayıtlıdır. Bu retler tarihsel Haar/altı kare hattına aittir. SBI12 kabul edilmiş aynı roster’ı kullanmıştır; 87 videonun güncel RetinaFace/SBI ile yeniden denenmesi veya aynı nedenle reddedilmesi iddia edilmez. Tam liste tablolar/09_tarihsel_ret_87.csv içindedir.

Kanıt: outputs/video_identity_v1/manifest.json, cache_audit.json; outputs/video_research_v1/roster_audit.json; finetune12/main_20261014/config.json; src/video_forensic.py accepted_roster ve src/video_sbi12.py. Manifest fingerprint ve kabul edilen kayıtların eşitliği denetlendi. Hiçbir split, etiket veya ham dosya değiştirilmedi.

## Kontroller ve geri alma

DOCX açıldı, Word’den PDF üretildi; 17 sayfanın tamamı görsel olarak kontrol edildi. PPTX paket/yerleşim/font, iki gömülü grafik çalışma kitabı ve birinci taraf import kontrolünden geçti; PowerPoint’te salt okunur açılıp 12 slaydın tamamı dışa aktarılarak kontrol edildi. Türkçe karakter, kesilen grafik, taşan metin ve yalnız kalan tablo başlığı kontrol edildi.

Görev öncesi 48 teslim dosyasının doğrulanmış kopyası outputs/final_academic_review_v1/teslim_before.zip içindedir. Önceki sürüm bu arşivden geri alınabilir. Arşiv, SHA-256 listesinin her dosya baytıyla eşleşmesiyle doğrulandı. Önce/sonra kaynak ve checkpoint hash kontrolleri ayrı kanıt klasöründedir. Uygulama, modeller, datasetler ve önceki sonuçlar değiştirilmedi. Eğitim, inference, nihai test, commit/push/merge yapılmadı.
