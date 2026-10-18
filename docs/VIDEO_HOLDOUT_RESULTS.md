# Tamamlanmış SBI VIDEO-HOLDOUT özeti

Yeni eğitim, inference veya metrik optimizasyonu yapılmadan mevcut mühürlü sonuçtan aktarılmıştır. Bireysel video adları/skorları ve ham veri paylaşılmaz.

| Yöntem | Accuracy | ROC-AUC | Macro-F1 | REAL recall | FAKE recall | Ort./medyan kare | Genişleme |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.97297297 | 0.99722406 | 0.96987972 | 0.94943820 | 0.98529412 | 6.00000 / 6 | 0.00000% |
| 12 | 0.97490347 | 0.99727363 | 0.97214535 | 0.96067416 | 0.98235294 | 12.00000 / 12 | 0.00000% |
| adaptive | 0.97490347 | 0.99755453 | 0.97206956 | 0.95505618 | 0.98529412 | 6.23166 / 6 | 3.86100% |

Her üç yöntemde 518/518 video değerlendirildi; reddedilen video yok. Ortak frame inference cache kullanıldı. Ortak cache tamamlanma süreleri bağımsız yöntem gecikmesi veya hızlanma ölçümü değildir.

Kayıtlı çağrıların toplam duvar süresi: 3252.605 saniye (54.21 dakika); 2 çağrı kaydı. Pilot benchmark süreleriyle birleştirilmedi.

Sabit model: SBI EfficientNet-B4 epoch 6, eşik 0.5, adaptif marj 0.10. Seçimler validation üzerinde geliştirildi; resmî test sonucuna göre değiştirilmedi.

Bu Celeb-DF içinde video-holdout değerlendirmesidir; kişi-bağımsız veya cross-dataset başarı kanıtı değildir. Bilinen yol, SHA-256 ve taşıyıcı aile örtüşmesi yoktur; kişi/donor kodu ortaklığı vardır. Taşıyıcı eşlemesi geliştirme örnekleriyle desteklenir; tüm kaynak ilişkileri ve ön eğitim örtüşmeleri doğrulanmış değildir.

## Korunan yerel kanıtlar

`outputs/video_holdout_v1/` ve önceki protokol/audit klasörleri haricen arşivlenmelidir. Bu kısa özet tam bilimsel kanıt yedeğinin yerine geçmez.

- `outputs/video_holdout_v1/final_test_20261018/final_results.json` dosya SHA-256: `8f443470e1b3c260d6e08a8ca7bc497ed743bb7b318cfb365df69aedfff64f76`
- `outputs/video_holdout_v1/final_test_20261018/execution_cost.json` dosya SHA-256: `986febb1f7bada57fd22b34ae97a8407e7fce1ebfc90a5200b2feffecfb9fce3`
- `outputs/video_holdout_v1/protocol_20261018/protocol.json` dosya SHA-256: `a0afab06c299b8e675dd81a920e4f62d1a97578f76c67fb0dca881c42a0dbc09`
