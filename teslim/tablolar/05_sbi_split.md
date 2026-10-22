# SBI video veri ayrımları ve kapsam uzlaştırması

|Ayrım / video|REAL|FAKE|Toplam|
|---|---:|---:|---:|
|Eğitim kabul edilen|417|1806|2223|
|Eğitim tarihsel ret|22|61|83|
|Validation kabul edilen|106|496|602|
|Validation tarihsel ret|3|1|4|
|VIDEO-HOLDOUT|178|340|518|
|Karantina|164|2935|3099|
|Tam kaynak envanteri|890|5639|6529|

Not: Tam envanter = 2.223 + 83 + 602 + 4 + 518 + 3.099 = 6.529. Ret: tarihsel Haar/altı kare cache kabulü; yeni SBI/RetinaFace başarısızlık ölçümü değildir. Tam kaynak envanteri satırı genel toplamdır, diğer satırlara tekrar eklenmez.

Kaynak: outputs/video_identity_v1/manifest.json; outputs/video_identity_v1/cache_audit.json; outputs/video_research_v1/roster_audit.json; outputs/video_research_v1/finetune12/main_20261014/config.json
