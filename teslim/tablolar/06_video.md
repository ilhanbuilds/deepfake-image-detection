# SBI holdout stratejileri

|Strateji|Accuracy %|REAL recall %|FAKE recall %|ROC-AUC|Ort. kare|
|---|---|---|---|---|---|
|Sabit 6|97,2973|94,944|98,529|0.997224|6.0000|
|Sabit 12|97,4903|96,067|98,235|0.997274|12.0000|
|Adaptif 6→12|97,4903|95,506|98,529|0.997555|6.2317|

Not: n=518; 178 REAL, 340 FAKE; eşik 0,5. Sabit12 ve adaptif aynı sayıda doğru karara rağmen farklı hatalar yapar.

Kaynak: `outputs/video_holdout_v1/final_test_20261018/final_results.json`
