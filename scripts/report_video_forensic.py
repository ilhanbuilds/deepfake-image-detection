"""Render the Turkish research report strictly from measured local artifacts."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    from src.video_forensic import ROOT, read_json

    measured = read_json(ROOT / "sbi_ffraw_validation.json")
    fusion = read_json(ROOT / "quality_comparison.json")
    pilot = read_json(ROOT / "robustness_pilot.json")
    smoke = read_json(ROOT / "finetune/prepare_20261014/smoke.json")
    old = read_json(ROOT / "verified_baselines.json")
    preservation = read_json(ROOT / "preservation_check.json")
    metrics, sources = measured["metrics"], measured["sources"]
    mean = fusion["methods"]["mean"]["metrics"]
    quality = fusion["methods"]["full"]["metrics"]
    auc_delta = quality["roc_auc"] - mean["roc_auc"]
    f1_delta = quality["macro_f1"] - mean["macro_f1"]
    table = "| Model | ROC-AUC | Macro-F1 | Balanced acc. | Accuracy | REAL recall | FAKE recall |\n|---|---:|---:|---:|---:|---:|---:|\n"
    labels = [
        "B0 + GRU, natural/weighted",
        "B0 + GRU, balanced",
        "B0 spatial mean",
        "SBI FF-raw zero-shot mean",
        "SBI + kalite adayı",
    ]
    all_metrics = [r["metrics"] for r in old] + [metrics, quality]
    for label, item in zip(labels, all_metrics, strict=True):
        table += f"| {label} | {item['roc_auc']:.5f} | {item['macro_f1']:.5f} | {item['balanced_accuracy']:.5f} | {item['accuracy']:.5f} | {item['per_class']['REAL']['recall']:.5f} | {item['per_class']['FAKE']['recall']:.5f} |\n"
    source_table = "| Kaynak | Video | Recall | FAKE skor ort. | Synthesis karşısında AUC |\n|---|---:|---:|---:|---:|\n"
    for source, row in sources.items():
        auc = row.get("roc_auc_vs_celeb_synthesis")
        source_table += f"| {source} | {row['support']} | {row['recall']:.5f} | {row['fake_score_mean']:.5f} | {auc if auc is not None else 'Tek sınıfta tanımsız'} |\n"
    ablation_table = "| Birleştirme | ROC-AUC | Macro-F1 |\n|---|---:|---:|\n"
    for mode, result in fusion["methods"].items():
        ablation_table += f"| {mode} | {result['metrics']['roc_auc']:.5f} | {result['metrics']['macro_f1']:.5f} |\n"
    robustness_table = "| Koşul | Mean AUC | Kalite AUC | Mean Macro-F1 | Kalite Macro-F1 |\n|---|---:|---:|---:|---:|\n"
    for condition in ("clean", "blur", "jpeg", "drop_frame", "crop_translation"):
        rows = [pilot["conditions"][f"{seed}:{condition}"] for seed in pilot["seeds"]]
        values = [
            np.mean([r[m]["metrics"][metric] for r in rows])
            for m, metric in [
                ("mean", "roc_auc"),
                ("quality", "roc_auc"),
                ("mean", "macro_f1"),
                ("quality", "macro_f1"),
            ]
        ]
        robustness_table += (
            "| " + condition + " | " + " | ".join(f"{v:.5f}" for v in values) + " |\n"
        )
    stage = smoke["stages"][-1]
    ci = fusion["paired_family_bootstrap"]["seeds"]["42"]
    figure = ROOT / "comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(labels))
    short = [
        "GRU\nweighted",
        "GRU\nbalanced",
        "B0\nspatial",
        "SBI\nmean",
        "SBI\nquality",
    ]
    for ax, key, title in zip(
        axes, ("roc_auc", "macro_f1"), ("Video ROC-AUC", "Video Macro-F1"), strict=True
    ):
        bars = ax.bar(
            x,
            [m[key] for m in all_metrics],
            color=["#8696a6"] * 3 + ["#137c8b", "#df9629"],
        )
        ax.set(xticks=x, xticklabels=short, ylim=(0, 1.08), title=title)
        ax.bar_label(bars, fmt="%.3f", padding=3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "602 accepted validation videos · threshold 0.5\nSBI changes pretraining, crop and resolution; not a single-component ablation",
        fontsize=11,
    )
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    report = f"""# Deepfake video güçlendirme raporu

14 Ekim 2026. Uzun eğitim başlatılmadı. Gerçek zero-shot validation, kontrollü kalite karşılaştırması, üç-seed bozulma pilotu ve gerçek CUDA backward/resume deneyi yapıldı.

## 1. Son karar

**Ana strateji: resmî SBI FF-raw EfficientNet-B4 + resmî yüz ön işlemesi + video düzeyinde basit ortalama.** Kalite mekanizması aynı backbone üzerinde sınanan katkı adayıdır; başarısı kanıtlanmadan varsayılan karar yöntemi yapılmamalıdır. Ayrı SBI kopyasında kontrollü fine-tuning altyapısı hazırdır; zero-shot checkpoint korunur. Fine-tuning ancak validation ve kaynak bazlı ölçümlerle üstünse seçilmelidir.

602 kabul edilmiş videodan {measured["evaluated"]} değerlendirildi; coverage %{100 * measured["coverage"]:.2f}. Bu sonuçlar resmî 518 videoluk test setinin sonuçları değildir.

{table}

Eski üç deneyin best checkpoint epoch'ları, history ve checkpoint içerikleriyle doğrulandı. GRU/natural epoch 2, balanced epoch 1, spatial epoch 3. Ön eğitim, crop, giriş çözünürlüğü, kapasite ve aggregation birlikte değişti; tabloda görülen fark yalnız “SBI eğitiminin” nedensel etkisini ölçmez. Aynı cohort üzerindeki sistem karşılaştırmasıdır. Teste veya V3'e erişerek model seçilmedi.

![Karşılaştırma]({figure.resolve().as_posix()})

## 2. Önceki model neden zorlanıyor?

Kanıt: `PartialVideoClassifier` ve spatial partial model yalnız `features[8]` projection bloğunu eğitiyor; MBConv katmanları frozen. REAL recall düşük, kaynaklar arası fark belirgin; balanced örnekleme ve spatial mean deneyleri mevcut GRU'dan daha iyi çıkmadı. Bunlar tek başına sorun teşhisi değildir.

Hipotezler: ImageNet özelliklerinin adli izlere yetersiz adaptasyonu, kaynak kısayolları, Haar crop/160 çözünürlük etkisi, az ve seyrek kare, eğitim/regularization stratejisi. Mevcut deneyler bunların bağımsız nedensel etkilerini ayırmıyor. SBI sonucu daha güçlü bir başlangıç olup olmadığını gösterir; başarısızlığın tek nedenini kanıtlamaz.

## 3. SBI araştırması ve lisans

[SBI makalesi](https://arxiv.org/pdf/2204.08376) tek gerçek görüntünün dönüştürülmüş iki sürümünü blend ederek genel sahtecilik izleri üretir. Asıl yenilik EfficientNet veya GRU değildir; bu sentetik eğitim örnekleridir. B4, ImageNet başlangıcından SAM ile 100 epoch, batch 32, LR 0,001 ve video başına sekiz eğitim karesiyle eğitilmiştir. FF++ gerçek görüntülerinden üretilen SBI'ler kullanılır; Celeb-DF yayınlanan protokolde çapraz veri seti değerlendirmesidir.

[Resmî depo](https://github.com/mapooon/SelfBlendedImages) FF-raw ve FF-c23 ağırlıklarını yayımlar; ikisi de gerçekten indirildi. Depoda verilen CDF AUC değerleri %93,82/%92,87 yalnız literatür referansıdır; bizim altı-kare/602-video protokolümüze aktarılmaz. FF-c23 bu görevde ana validation için denenmedi; sonuçlara bakarak checkpoint taraması yapılmadı.

[SBI lisansı](https://github.com/mapooon/SelfBlendedImages/blob/master/LICENSE) ticari olmayan iç araştırma kullanımını kapsar; genel MIT lisansı değildir. Kod/ağırlıkların dağıtımı kısıtlıdır. İndirilen upstream dosyaları ve ağırlıklar git tarafından ignore edilen outputs altında kaldı. SBI atfı: Shiohara ve Yamasaki, CVPR 2022, “Detecting Deepfakes with Self-Blended Images”. RetinaFace paketinin kendi lisansı MIT; SBI lisansının yerine geçmez.

Resmî checkpoint'in FF++ dışındaki tüm internet kaynaklarıyla örtüşmediğine dair bağımsız içerik/kimlik denetimi yoktur. Celeb-DF üzerinde bilinen doğrudan eğitim sızıntısı yayınlanan protokolde görünmüyor; bütün bilinmeyen kaynak örtüşmelerinin dışlandığı iddia edilmez.

| Dosya | SHA-256 |
|---|---|
| FFraw.tar | e477fc1789f8c1d2147cffe86a47eb37a24adedae10b106b7d47c1eeb73777e2 |
| FFc23.tar | 2b05c2ba36ccb9e9f4f9e1aae9acd443ae2e6400ce725f56104fdb175d3c3267 |
| RetinaFace extracted | b9fdd58c474d809b67166130b42d5913f00b48f912a3491565bc56b8a8a796f2 |

SBI kaynak revizyonu: `d6d9a351f32334ae9a3a250872c05836e0dd0f2c`. Setup script'i dosya hash'lerini doğrular; rastgele checkpoint veya değiştirilmiş upstream crop dosyası kabul edilmez.

## 4. Ön işleme ve gerçek validation

[Resmî inference](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/inference_video.py) iki logitten softmax'ın FAKE sütununu alır. [Resmî crop](https://github.com/mapooon/SelfBlendedImages/blob/master/src/inference/preprocess.py) RetinaFace bbox'ına her yönde %12,5 margin uygular; cv2 linear resize ile 380×380 RGB, float32 / 255 kullanılır. ImageNet mean/std normalizasyonu uygulanmaz. Detector kendi normalization'ını ayrıca uygular.

Mevcut altı örnekleme zamanı korunur; yüz detector/crop ve geçerli kareler değişebilir. Çoklu yüzde resmî alan filtresi ve kare başına maksimum FAKE skoru, zaman boyunca mean kullanıldı. Makaledeki 32 kare yerine altı kare ve en az üç geçerli kare şartı kullanılması protokol farkıdır. Yüzü olmayan videolara sahte başarı veya otomatik 0,5 ataması yoktur. Threshold sürekli 0,5 kaldı; calibration/threshold tuning yapılmadı.

{source_table}

REAL precision={metrics["per_class"]["REAL"]["precision"]:.5f}; FAKE precision={metrics["per_class"]["FAKE"]["precision"]:.5f}. Confusion matrix satırlar gerçek, sütunlar tahmin; sıra REAL/FAKE: `{metrics["confusion_matrix"]}`.

Süre {measured["seconds"] / 60:.2f} dakika; video başına {measured["seconds"] / 602:.3f} saniye (decode, detector, cache yazımı ve sınıflandırma dahil). Süre ölçümü sırasında bazı CPU regresyon kontrolleri de yürüdü; saf GPU benchmark değildir. Tepe allocated {measured["peak_allocated_mib"]:.1f} MiB, reserved {measured["peak_reserved_mib"]:.1f} MiB. Reddedilen videolar: {len(measured["rejected"])}. Kare hata ayrıntıları JSON içindeki her video raporunda saklanır.

## 5. Diğer uygulanabilir seçenekler

- B0 son MBConv stage'lerini açmak: mevcut kod/cache ile en düşük entegrasyon riski; fakat güçlü forensic ön eğitim sağlamaz. Gerekirse kontrollü yedek deneydir, bu görevin ana stratejisi değildir.
- V3 görüntü transferi: görüntü test başarısı video genellemesini kanıtlamaz; tam görüntü/yüz crop domain farkı vardır. Orijinal V3 değiştirilmedi; bu görevde video transferi çalıştırılmadı.
- [DFDC EfficientNet çözümü](https://github.com/selimsef/dfdc_deepfake_challenge): resmî model indirme script'i ve MIT kod vardır; sunulan B7 ensemble eğitimi çok GPU/12+ GB odaklıdır. 4 GB'a yeniden uyarlama ve farklı crop protokolü ek teslim riski getirir; ağırlık lisansını ayrı incelemeden entegre edilmedi.
- [LipForensics](https://github.com/ahaliassos/LipForensics): resmî FF++ checkpoint ve değerlendirme kodu vardır, fakat 68 landmark, ağız crop'u ve temporal hazırlık ister. Altı seyrek Haar karesine takılabilecek bir model değildir. Bu cihazda ölçülmedi; “çalışmaz” iddiası yok, entegrasyon maliyeti yüksek olduğu için seçilmedi. FTCN/TALL için de ölçülmemiş 4 GB uygunluk iddiası kurulmadı.

SBI seçimi; gerçek yerel ölçüm, düşük inference VRAM'i, anlaşılır tek CNN + birleştirme mimarisi ve 9 Kasım teslimine kadar sınırlı ek geliştirme gereksinimine dayanır. Daha büyük model otomatik başarı garantisi değildir.

## 6. Özgün katkı ve ölçümü

Tam matematik, gerekçe, AFW farkı, sınırlar ve sabit deney koşulları [protokolde](VIDEO_FORENSIC_PROTOCOL.md). Özet: modelden bağımsız göreli keskinlik ve bbox kenar payı korunma oranı; kalite ağırlıkları eşit ortalamayla karıştırılır. Her karenin ağırlığı `2/(3T)` ile `2/T` arasındadır. Global veri seti/validation normalizasyonu veya fake skorundan güven üretimi yoktur.

{ablation_table}

Full kalite adayının mean'e farkı: **AUC {auc_delta:+.6f}, Macro-F1 {f1_delta:+.6f}**. Bu fark tek başına kesin iyileşme değildir. Bilinen filename aileleriyle 1.000 tekrarlı bootstrap, seed 42 için AUC farkı %95 aralığı `{ci["roc_auc_delta_ci95"]}`, F1 farkı `{ci["macro_f1_delta_ci95"]}`. Seed 123 ve 2026 sonuçları JSON'dadır. Kaynak/kişi bağımlılıklarının tamamını gidermeyen koşullu validation aralıklarıdır.

Ham ortalama keskinliğin etiket AUC'si `{fusion["quality_source_diagnostic"]["label_auc"]}`; yalnız REAL örneklerde YouTube/Celeb kaynak AUC'si `{fusion["quality_source_diagnostic"]["real_source_auc"]}`. Bunlar kaynak kısayolu riskinin tanılarıdır, alternatif sınıflandırıcı başarısı veya nedensel kanıt değildir.

**Karar: full kalite yöntemi temiz validation'da mean'den kötü çıktı; kullanıma alınmadı.** Yalnız keskinlik ablation'ındaki küçük F1 artışı, AUC düşüşüyle birlikte ele alınmalıdır; sonuçları gördükten sonra bunu yeni kazanan diye seçmiyoruz. Sınır oranı ablation'ı bu cohort'ta skoru değiştirmedi. Faydalı bir özgün yöntem elde edildiği henüz kanıtlanmış değildir.

Bu çalışmada bize ait olan SBI değildir. Katkı adayı, ölçülebilir kaliteye dayalı sınırlı karar düzeltmesi ve bunun aynı checkpoint/kareler üzerinde ablation/robustness denetimidir. Akademik katkının fayda iddiası elde edilen sonuçlarla sınırlıdır; küçük veya olumsuz farklar saklanmamalıdır.

## 7. Üç seed ile bozulma pilotu

{robustness_table}

Tablo, 42/123/2026 seed'lerinin ortalamasıdır. Her kaynaktan önceden belirlenen 12 video; 36 video, 24 REAL/12 FAKE. Aynı model/kareler üzerinde iki yöntem eşleştirildi. Bunlar üç eğitim seed'i değildir. Detector tekrar çalıştırılmadı; sentetik crop ötelemesi gerçek bbox jitter testi değildir. JPEG görüntü sıkıştırması gerçek video codec deneyi değildir. Keskinlik clean dahil bütün pilot koşullarında 380×380 üzerinde yeniden hesaplandı; ana 602-video karşılaştırmasındaki ham crop keskinliğiyle aynı ölçüm ölçeği değildir. Her koşulun tek tek sonuçları ve tahminleri `robustness_pilot.json` içinde. Pilot, tüm 602 videoda veya başka veri setlerinde robustness kanıtı olarak sunulmamalıdır.

## 8. GTX 1650 uygulanabilirliği ve eğitim süresi

Gerçek CUDA FP32 backward yapıldı; uzun eğitim epoch'u başlamadı. Toplam B4 parametresi **17.552.202**. İlk aşama **3.586**, ikinci aşama **{stage["trainable_parameters"]:,}** trainable parametre. İkinci aşama `_blocks[24:32]`, projection ve classifier; frozen prefix/BN state değişmedi.

Frame batch=1, dört video accumulation. İkinci aşamada {stage["frames"]} gerçek kare / dört video güncellemesi **{stage["seconds"]:.3f} s**; allocated **{stage["peak_allocated_mib"]:.1f} MiB**, reserved **{stage["peak_reserved_mib"]:.1f} MiB**. NaN/Inf oluşmadı. Checkpoint'e model/AdamW/RNG kaydedilip aynı sonraki CUDA güncellemesi yeniden oynatıldı; bit düzeyinde eşitlik: **{smoke["checkpoint_replay_bitwise_equal"]}**.

Küçük pilot üzerinden cache hazırken train+validation epoch tahmini **{smoke["cached_epoch_estimate_minutes"]:.1f} dakika**; sekiz epoch üst bütçe yaklaşık **{smoke["full_budget_estimate_hours"]:.2f} saat**. İlk crop çıkarma, disk I/O ve checkpoint yazımı ek süredir; train videoları daha uzun olabilir. Bu tam epoch ölçümü değildir. Early stopping daha erken bitirebilir. İlk train crop hazırlığı için 2.223 video × mevcut uçtan uca süre yaklaşık {2223 * measured["seconds"] / 602 / 60:.0f} dakika mertebesinde kaba ek bütçe düşünülmelidir; detector-only ölçümü değildir.

FP32 seçildi; B4 ve detector cihazda sığdı. AMP/gradient checkpointing eklenmedi. Pilot, önerilen LR=1e-5/1e-4'ün kısa vadede sonlu olduğunu gösterir; uzun vadeli optimal LR garantisi vermez. Artış beklentisinin gerekçesi güçlü forensic başlangıcını gerçek MBConv adaptasyonuyla hedef alana uyarlayabilmektir; catastrophic forgetting veya overfit yüzünden başarının düşmesi de mümkündür.

## 9. Güvenli PowerShell komutları

Proje kökünde, yalnız uzun eğitime başlamak istediğinde:

```powershell
$python = '.\\outputs\\video_research_v1\\env\\Scripts\\python.exe'
$run = 'outputs/video_research_v1/finetune/sbi_seed42_20261014'
& $python -u -m src.video_forensic_train --run-training --run-dir $run --epochs 8 --seed 42
```

Aynı run üzerinden devam:

```powershell
$python = '.\\outputs\\video_research_v1\\env\\Scripts\\python.exe'
$run = 'outputs/video_research_v1/finetune/sbi_seed42_20261014'
& $python -u -m src.video_forensic_train --run-training --run-dir $run --epochs 8 --seed 42 --resume auto
```

Resume son tamamlanan epoch'tan devam eder; yarım epoch tekrar oynatılır. İlk epoch tamamlanmadıysa resume edilebilir checkpoint yoktur; yeni run adı gerekir. Erken durma tamamlandıysa ek epoch çalışmaz. Seed, config, kod, ortam, kaynak/cache hash'leri değişirse resume reddedilir. Mevcut run adı yeni başlangıç için yeniden kullanılamaz. Üç bağımsız eğitim gerektiğinde seed 123/2026 ve ayrı run adları kullanılmalıdır; bu görevde çalıştırılmadı.

Çıktılar: run altında `config.json`, `history.json`, `epoch_XXX_validation.json`, `interrupted.json`; `checkpoints/epoch_XXX.pth`, `best_video.pth`, `last_video.pth`. Best seçim Macro-F1, eşik 0,5. Zero-shot başlangıç ağırlıkları ayrı kalır. Eğitim çıktısını değerlendirmek için aynı `src.video_forensic --checkpoint .../best_video.pth --output YENI_JSON` arayüzü kullanılabilir.

Ortamı yeniden kurmak veya train/val cache audit'ini tekrarlamak:

```powershell
.\\.venv\\Scripts\\python.exe scripts/setup_video_forensic.py --download-and-install --audit-train-val
```

## 10. Doğrulamalar, önemli dosyalar ve kalan sınırlar

Yeni çekirdek modüller: `src/video_forensic.py`, `video_forensic_train.py`, `video_quality_fusion.py`, `video_forensic_robustness.py`. Setup/rapor script'leri, izole gereksinim listesi, `tests/test_video_forensic.py` ve `pytest.ini` eklendi. Test keşfi proje test klasörüne sınırlandı; outputs içindeki izole ortam paketlerinin testleri proje testi sayılmaz.

Regresyon sonuçları `regression.xml`; lint/format/compile kontrolleri `checks.json` dosyasında. Sandbox'taki Streamlit socketpair engeli, testleri değiştirmeden sandbox dışı çalıştırmayla çözüldü. İzole ortam `pip check` eski RetinaFace'in torch 1.9 / torchvision 0.10 / albumentations 1.0 pinleri nedeniyle başarısızdır. Ana .venv güncellenmedi. Windows/Python 3.12 + torch 2.11 CUDA çalışma uyumluluğu gerçek inference ve backward ile doğrulandı; eski ortamla bit düzeyinde eşdeğerlik iddiası yoktur.

Koruma hash kontrolü: **{preservation["all_unchanged"]}**. `app.py`, V3 checkpoint, manifest ve eski best video checkpoint'leri aynı. Eski feature cache tekrar audit ile kontrol edildi. Resmî 518 video ve quarantine videoları açılmadı; split değiştirilmedi. Commit/push yapılmadı.

Validation daha önce birçok kez kullanıldı; seçime bağlı aşırı uyum riski vardır. Gerçek kişi düzeyinde ayrım garantisi yoktur. Tam 602-video robustness, üç-seed uzun eğitim, başka kaynaklarda genelleme ve kilitli resmî test bu görevde tamamlanmış sayılmaz.

## 11. Hocaya kısa anlatım

“Videodan belirli zamanlarda yüzleri çıkarıyoruz. Literatürdeki SBI yöntemiyle önceden eğitilmiş modeli kullanarak her kare için sahtecilik skoru hesaplıyoruz. Bizim araştırdığımız katkı, bulanıklık ve crop sınır kaybı gibi modelden bağımsız ölçülerle karelerin etkisini sınırlandırılmış biçimde değiştirmek. Aynı model ve karelerde düz ortalamayla karşılaştırıp hangi koşullarda işe yaradığını, hangi koşullarda yaramadığını ölçüyoruz. SBI'yi kendi buluşumuz olarak sunmuyoruz.”
"""
    destination = PROJECT / "docs/VIDEO_FORENSIC_V1.md"
    destination.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            dict(
                report=str(destination), figure=str(figure.resolve()), metrics=metrics
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
