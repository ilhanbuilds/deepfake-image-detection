"""Report measured SBI nested-frame results, paired uncertainty and routing diagnostics."""

from collections import Counter
from itertools import pairwise
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src.video_forensic import accepted_roster, read_json, summarize, write_json
from src.video_frame_study import BUDGETS, STUDY
from src.video_manifest import digest_json
from src.video_pipeline import file_sha256


def macro_f1(labels, scores):
    predictions = scores >= 0.5
    return float(
        np.mean(
            [
                2
                * np.sum((labels == label) & (predictions == label))
                / max(1, np.sum(labels == label) + np.sum(predictions == label))
                for label in (0, 1)
            ]
        )
    )


def paired_uncertainty(methods, records, repeats=1000):
    rows = methods["6"]["rows"]
    expected = [r["path"] for r in rows]
    if any([r["path"] for r in methods[str(n)]["rows"]] != expected for n in BUDGETS):
        raise ValueError("Paired analysis requires identical ordered cohorts.")
    if any(r["status"] != "accepted" for n in BUDGETS for r in methods[str(n)]["rows"]):
        raise ValueError(
            "Do not hide rejected videos in a paired full-cohort analysis."
        )
    family_by_path = {r["path"]: r["family"] for r in records}
    families = {}
    for index, path in enumerate(expected):
        families.setdefault(family_by_path[path], []).append(index)
    groups = [np.asarray(v) for v in families.values()]
    y = np.asarray([r["target"] for r in rows])
    scores = {
        n: np.asarray([r["fake_probability"] for r in methods[str(n)]["rows"]])
        for n in BUDGETS
    }
    pairs = [(6, 12), (6, 24), (6, 32), (12, 24), (24, 32)]
    result = {}
    for seed in (42, 123, 2026):
        rng = np.random.default_rng(seed)
        deltas = {f"{a}_to_{b}": [] for a, b in pairs}
        for _ in range(repeats):
            sample = np.concatenate(
                [groups[i] for i in rng.integers(len(groups), size=len(groups))]
            )
            target = y[sample]
            if len(np.unique(target)) != 2:
                continue
            measured = {
                n: np.array(
                    [
                        roc_auc_score(target, scores[n][sample]),
                        macro_f1(target, scores[n][sample]),
                    ]
                )
                for n in BUDGETS
            }
            for a, b in pairs:
                deltas[f"{a}_to_{b}"].append(measured[b] - measured[a])
        result[str(seed)] = {
            name: dict(
                valid_draws=len(values),
                roc_auc_delta_ci95=np.quantile(
                    np.asarray(values)[:, 0], [0.025, 0.975]
                ).tolist(),
                macro_f1_delta_ci95=np.quantile(
                    np.asarray(values)[:, 1], [0.025, 0.975]
                ).tolist(),
            )
            for name, values in deltas.items()
        }
    return dict(
        family_count=len(groups),
        repeats_per_seed=repeats,
        seeds=result,
        limitation="Known filename families, not guaranteed independent identities. Conditional exploratory intervals do not correct repeated-validation selection or multiple comparisons. Seeds resample the same inference, not independent model runs.",
    )


def transitions(methods):
    baseline = methods["6"]["rows"]
    result = {}
    for budget in BUDGETS[1:]:
        rows = methods[str(budget)]["rows"]
        corrected, harmed, details = [], [], []
        for old, new in zip(baseline, rows, strict=True):
            if old["path"] != new["path"] or old["target"] != new["target"]:
                raise ValueError("Paired transition alignment failed.")
            was_correct = old["prediction"] == old["target"]
            is_correct = new["prediction"] == new["target"]
            detail = dict(
                path=old["path"],
                target=old["target"],
                six_score=old["fake_probability"],
                new_score=new["fake_probability"],
                six_margin=abs(old["fake_probability"] - 0.5),
                was_correct=was_correct,
                is_correct=is_correct,
            )
            details.append(detail)
            if not was_correct and is_correct:
                corrected.append(detail)
            elif was_correct and not is_correct:
                harmed.append(detail)
        bins = []
        for low, high in pairwise((0.0, 0.05, 0.1, 0.2, 0.5000001)):
            selected = [r for r in details if low <= r["six_margin"] < high]
            bins.append(
                dict(
                    margin_lower_inclusive=low,
                    margin_upper_exclusive=high,
                    videos=len(selected),
                    corrected=sum(
                        not r["was_correct"] and r["is_correct"] for r in selected
                    ),
                    harmed=sum(
                        r["was_correct"] and not r["is_correct"] for r in selected
                    ),
                )
            )
        result[str(budget)] = dict(
            corrected=corrected,
            harmed=harmed,
            corrected_by_source=dict(
                Counter(r["path"].split("/")[0] for r in corrected)
            ),
            harmed_by_source=dict(Counter(r["path"].split("/")[0] for r in harmed)),
            margin_bins=bins,
        )
    return result


def timing_sensitivity(methods, recheck):
    """Keep every video and all raw timings; substitute only remeasured durations."""
    estimates = {}
    keys = [(r["path"], r["index"]) for r in recheck["frames"]]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate timing correction.")
    for budget in BUDGETS:
        method = methods[str(budget)]
        by_path = {r["path"]: r for r in method["rows"]}
        adjustment = 0.0
        for frame in recheck["frames"]:
            if budget not in frame["affected_budgets"]:
                continue
            if frame["index"] not in by_path[frame["path"]]["selected_indices"]:
                raise ValueError("Timing correction escaped its selected frame set.")
            adjustment += (
                frame["raw_processing_seconds"] - frame["median_remeasured_seconds"]
            )
        total = method["attributed_seconds"] - adjustment
        if total <= 0:
            raise ValueError("Timing correction exceeds measured duration.")
        estimates[str(budget)] = dict(
            raw_mean_seconds=method["mean_seconds"],
            raw_median_video_seconds=float(
                np.median([r["attributed_seconds"] for r in method["rows"]])
            ),
            remeasured_mean_seconds_estimate=total / len(method["rows"]),
            substituted_seconds=adjustment,
        )
    return estimates


def plot_results(methods, timing):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for axis, key, title in zip(
        axes[:2],
        ("roc_auc", "macro_f1"),
        ("ROC-AUC", "Macro-F1 (eşik 0.5)"),
        strict=True,
    ):
        values = [methods[str(n)]["metrics"][key] for n in BUDGETS]
        axis.plot(BUDGETS, values, marker="o", color="#146b91", linewidth=2)
        for x, y in zip(BUDGETS, values, strict=True):
            axis.annotate(
                f"{y:.5f}",
                (x, y),
                xytext=(0, 9),
                textcoords="offset points",
                ha="center",
                fontsize=9,
            )
        axis.margins(y=0.4)
        axis.set_title(title)
    raw_times = [methods[str(n)]["mean_seconds"] for n in BUDGETS]
    times = [timing[str(n)]["remeasured_mean_seconds_estimate"] for n in BUDGETS]
    excess = [raw - corrected for raw, corrected in zip(raw_times, times, strict=True)]
    positions = np.arange(len(BUDGETS))
    axes[2].bar(
        positions, times, color="#d48434", label="Yeniden ölçümle maliyet tahmini"
    )
    axes[2].bar(
        positions,
        excess,
        bottom=times,
        color="#eeeeee",
        edgecolor="#777777",
        hatch="///",
        label="Ham kayıttaki bekleme farkı",
    )
    axes[2].set_xticks(positions, [str(n) for n in BUDGETS])
    axes[2].set_title("Atfedilen süre / video (saniye)")
    for index, value in enumerate(times):
        if excess[index] > 0.01:
            axes[2].text(
                index,
                value - 0.15,
                f"{value:.2f}",
                ha="center",
                va="top",
                color="white",
            )
            axes[2].text(
                index,
                raw_times[index],
                f"{raw_times[index]:.2f} ham",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        else:
            axes[2].text(index, value, f"{value:.2f}", ha="center", va="bottom")
    axes[2].set_ylim(0, max(raw_times) * 1.3)
    axes[2].legend(loc="upper left", fontsize=7.5, frameon=False)
    for axis in axes:
        axis.set_xlabel("Kare sayısı")
        axis.grid(axis="y", alpha=0.2)
    for axis in axes[:2]:
        axis.set_xticks(BUDGETS)
        axis.set_xlim(2, 36)
    fig.suptitle(
        "SBI FF-raw · aynı 602 validation videosu · iç içe örnekleme", fontsize=13
    )
    fig.savefig(STUDY / "comparison.png", dpi=180)
    plt.close(fig)


def render_report(
    full, baseline, pilot, uncertainty, changes, preservation, checks, timing, recheck
):
    methods = full["methods"]
    best_auc = max(BUDGETS, key=lambda n: methods[str(n)]["metrics"]["roc_auc"])
    best_macro = max(BUDGETS, key=lambda n: methods[str(n)]["metrics"]["macro_f1"])
    near = changes["12"]["margin_bins"][:2]
    near_videos = sum(b["videos"] for b in near)
    near_corrected = sum(b["corrected"] for b in near)
    near_harmed = sum(b["harmed"] for b in near)
    lines = [
        "# SBI kare sayısı deneyi — 14 Ekim 2026",
        "",
        "602/602 sabit validation videosu; resmî FF-raw checkpoint, FP32, RetinaFace, 380×380 RGB/255, karede maksimum yüz skoru ve videoda basit ortalama. Eşik her koşulda 0.5. Eğitim adımı: **0**. Resmî test ve quarantine video içeriğine erişim: **0**.",
        "",
        "## Ölçülen karşılaştırma",
        "",
        "| Kare | ROC-AUC | Macro-F1 | REAL Recall | FAKE Recall | Celeb-real AUC | Video başına ham süre* |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for n in BUDGETS:
        r = methods[str(n)]
        m = r["metrics"]
        lines.append(
            f"| {n} | {m['roc_auc']:.5f} | {m['macro_f1']:.5f} | {m['per_class']['REAL']['recall']:.5f} | {m['per_class']['FAKE']['recall']:.5f} | {r['sources']['Celeb-real']['roc_auc_vs_celeb_synthesis']:.5f} | {r['mean_seconds']:.3f} s |"
        )
    lines.extend(
        [
            "",
            "*Süre, ölçülen ortak decode süresi + seçilen karelerin ölçülen yüz bulma/kırpma/skorlama süreleri toplamıdır. Dört ayrı bağımsız uçtan uca benchmark değildir. Model yükleme, dosya hash kontrolü, eski crop doğrulaması ve JSON yazımı bu tabloda yoktur. Tüm koşullar aynı son kareye ulaşır. Altı kare önce doğrulandığı için ek karelerde ikinci sıralı decode geçişi gerekir; eski altı kare yeniden retrieve/detect/classify edilmez. Bu ikinci decode deneyin gerçek toplam süresine dahildir, koşul maliyetine iki kez eklenmez. GPU ısınması, işletim sistemi ve ardışık fazlar küçük zaman farkları yaratabilir.",
            "",
            f"32-kare ham ortalaması ayrıca tek karedeki yaklaşık 757 saniyelik olağandışı beklemeyi içerir. O karenin ayrı yeniden ölçümüyle hesaplanan ortalama maliyet tahmini **{timing['32']['remeasured_mean_seconds_estimate']:.3f} s/video**; ham değer ve tüm skorlar korunmuştur. Ayrıntılar süre duyarlılığı bölümündedir.",
            "",
            f"En yüksek ROC-AUC: **{best_auc} kare**. En yüksek Macro-F1: **{best_macro} kare**. Bu ayrım yalnızca bu validation üzerinde gözlenen sonuçtur; istatistiksel üstünlük iddiası değildir.",
            "",
            "![Karşılaştırma](../outputs/video_frames_v1/comparison.png)",
            "",
            "## Altı kare yeniden üretimi ve kontrol edilen değişkenler",
            "",
            f"602 videonun altı karesi MP4 dosyalarından yeniden çıkarıldı. Eski bbox koordinatları ve yeniden boyutlandırılmış crop piksel SHA-256 değerleri birebir eşleşti. En büyük yüz skoru farkı: **{baseline['max_baseline_face_score_delta']:.10g}**. Yeniden üretilen altı kare metrikleri eski sonuç sözlüğüyle birebir eşit: **{baseline['baseline_verified']}**.",
            "",
            "Tarihsel seçim `np.linspace(0, frame_count-1, 6).round().astype(int)` idi. Örnek: 315 kareli `Celeb-real/id39_0001.mp4` → `[0, 63, 126, 188, 251, 314]`. Ek kareler en geniş aralığın aşağı yuvarlanan orta noktasından eklenir; eşit aralıklarda erken zaman seçilir. 6 ⊂ 12 ⊂ 24 ⊂ 32. Resmî SBI'nin bağımsız 32 noktalı linspace seçimiyle aynı örnekleme değildir: burada amaç tarihsel altı kareyi koruyan kontrollü karşılaştırmadır.",
            "",
            "Çoklu yüzde en büyük yüz alanının en az yarısı kadar olan yüzler korunur; bunların en yüksek FAKE softmax olasılığı kare skoru olur. Resmî crop fonksiyonu, 12.5% bbox marjları, resize, checkpoint ve sınıf sırası değişmedi. İndeksler, bbox'lar, crop hash'leri, yüz skorları ve hata nedenleri ayrı frame cache'inde saklanır. Skorla ağırlıklandırma, threshold tuning veya video eleme yapılmadı.",
            "",
            "## Tam metrikler, kaynaklar ve kapsam",
            "",
            "| Kare | Accuracy | Balanced accuracy | FAKE precision | REAL precision | Confusion [[TN,FP],[FN,TP]] |",
            "|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for n in BUDGETS:
        m = methods[str(n)]["metrics"]
        lines.append(
            f"| {n} | {m['accuracy']:.5f} | {m['balanced_accuracy']:.5f} | {m['precision']:.5f} | {m['per_class']['REAL']['precision']:.5f} | {m['confusion_matrix']} |"
        )
    lines.extend(
        [
            "",
            "| Kare | Celeb-real recall (63) | YouTube-real recall (43) | YouTube-real AUC | Değerlendirilen | Geçerli yüz karesi | Yüz sayısı | Hatalı / yetersiz kare |",
            "|---:|---:|---:|---:|---:|---:|---:|:---|",
        ]
    )
    for n in BUDGETS:
        r = methods[str(n)]
        lines.append(
            f"| {n} | {r['sources']['Celeb-real']['recall']:.5f} | {r['sources']['YouTube-real']['recall']:.5f} | {r['sources']['YouTube-real']['roc_auc_vs_celeb_synthesis']:.5f} | {r['evaluated']}/{r['attempted']} | {r['valid_face_frames']} | {r['retained_faces']} | {r['failed_frames']} / {r['unavailable_frames']} |"
        )
    lines.extend(
        [
            "",
            "Kaynak AUC'leri, ilgili REAL kaynak ile aynı 496 Celeb-synthesis videosunu karşılaştırır. Celeb-synthesis tek başına tek sınıftır; AUC hesaplanamaz. Bozuk veya yüzsüz kareler açık hata olarak tutulur; başka kare/video ile doldurulmaz. Tüm video tahminleri sonuç JSON'unda yer alır.",
            "",
            "## Kazanç, kayıp ve doygunluk",
            "",
            "| Değişim | ΔAUC | ΔMacro-F1 | Düzeltilen video | Bozulan video | REAL yanlış alarm | Süre oranı (6'ya göre) |",
            "|:---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for n in BUDGETS[1:]:
        r, b = methods[str(n)], methods["6"]
        c = changes[str(n)]
        lines.append(
            f"| 6→{n} | {r['metrics']['roc_auc'] - b['metrics']['roc_auc']:+.5f} | {r['metrics']['macro_f1'] - b['metrics']['macro_f1']:+.5f} | {len(c['corrected'])} | {len(c['harmed'])} | {r['metrics']['confusion_matrix'][0][1]} (önce {b['metrics']['confusion_matrix'][0][1]}) | {r['mean_seconds'] / b['mean_seconds']:.2f}× |"
        )
    lines.extend(
        [
            "",
            "| Ardışık adım | ΔAUC | ΔMacro-F1 | Ek saniye/video |",
            "|:---|---:|---:|---:|",
        ]
    )
    for a, b in pairwise(BUDGETS):
        x, y = methods[str(a)], methods[str(b)]
        lines.append(
            f"| {a}→{b} | {y['metrics']['roc_auc'] - x['metrics']['roc_auc']:+.5f} | {y['metrics']['macro_f1'] - x['metrics']['macro_f1']:+.5f} | {y['mean_seconds'] - x['mean_seconds']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Düzeltilen ve bozulan videoların tüm yolları, gerçek etiketleri ve eski/yeni skorları `analysis.json → transitions` alanındadır. Daha fazla kare her video için iyileşme garantilemez.",
            "",
            "Gözlenen doygunluk ölçüte bağlı: Macro-F1 12 kareden sonra hemen hemen sabit kalıyor (12→24 farkı yalnızca +0.00052); AUC 24 karede en yüksek noktaya ulaşıyor. 32 karede AUC ve Macro-F1 birlikte geriliyor. Bu noktalar tek iç içe örnekleme düzenindeki gözlemlerdir; evrensel optimum kanıtı değildir.",
            "",
            "REAL yanlış alarmlar 6/12/24/32 için 28/25/29/31: daha fazla kare bunları monoton azaltmıyor. Celeb-real AUC 24 karede en yüksekken Celeb-real recall 12 karede en yüksek. YouTube-real recall 12 karede 0.72093 iken 32 karede 0.60465'e düşüyor; sıralama başarısı (AUC) ile 0.5 eşiğindeki karar başarısı ayrı değerlendirilmelidir.",
            "",
            "## Eşleştirilmiş belirsizlik",
            "",
            f"{uncertainty['family_count']} bilinen dosya-adı ailesi üzerinden eşleştirilmiş cluster bootstrap; 42, 123, 2026 tohumlarında ayrı ayrı {uncertainty['repeats_per_seed']} çekiliş. Aşağıdaki tablo tohum 42'nin yüzdelik %95 aralıklarını gösterir; diğer iki tohum JSON'dadır.",
            "",
            "| Değişim | ΔAUC %95 aralığı | ΔMacro-F1 %95 aralığı |",
            "|:---|:---|:---|",
        ]
    )
    for name, r in uncertainty["seeds"]["42"].items():
        lines.append(
            f"| {name.replace('_to_', '→')} | [{r['roc_auc_delta_ci95'][0]:+.5f}, {r['roc_auc_delta_ci95'][1]:+.5f}] | [{r['macro_f1_delta_ci95'][0]:+.5f}, {r['macro_f1_delta_ci95'][1]:+.5f}] |"
        )
    lines.extend(
        [
            "",
            "Bunlar aynı validation'a koşullu, keşif amaçlı aralıklardır. Dosya ailesi gerçek kişi bağımsızlığını tam garanti etmez. Daha önce aynı validation üzerinde çok sayıda deney yapılması seçim yanlılığı yaratır; bootstrap bunu veya çoklu karşılaştırmayı düzeltmez. Sıfırı içeren küçük farklar kesin başarı olarak yorumlanmamalıdır. Resmî test açılarak seçim yapılmadı.",
            "",
            "Üç bootstrap tohumunda da 6→12 Macro-F1 farkının aralığı pozitiftir; AUC farkının aralığı sıfırı içerir. 6→24 AUC aralıkları pozitiftir; Macro-F1 aralığı sıfır sınırı çevresinde tohuma duyarlıdır. 12→24 ve 24→32 için iki ölçütün aralıkları da üç tohumda sıfırı içerir. Dolayısıyla 24'ün 12'ye küçük nokta avantajını veya 32'nin düşüşünü kesin genelleme sonucu olarak sunmuyoruz.",
            "",
            "## Pilot, gerçek çalışma süresi ve VRAM",
            "",
            f"Pilot: her kaynaktan SHA-256(path) sırasıyla 9 video, toplam **27**; seçim skor/hata bilgisine dayanmaz. Üç kaynağı kapsar, tam cohort sınıf oranını temsil etmez. Pilot ek-kare fazı duvar süresi **{pilot['phase_wall_seconds'] / 60:.2f} dk**. Bu 27 videonun altı kare + ek kare ölçülen inference geçişleri toplamı **{pilot['measured_inference_pass_seconds'] / 60:.2f} dk**. Pilot cache'i tam deneyde yeniden inference yapılmadan kullanıldı.",
            "",
            f"602 videonun baseline doğrulama fazı **{baseline['phase_wall_seconds'] / 60:.2f} dk**; tamamlama fazı **{full['phase_wall_seconds'] / 60:.2f} dk**. Üç fazın kayıtlı duvar süreleri toplamı **{(baseline['phase_wall_seconds'] + pilot['phase_wall_seconds'] + full['phase_wall_seconds']) / 60:.2f} dk** (süreç/model başlatma hariç). Benzersiz video inference geçişleri toplamı **{full['measured_inference_pass_seconds'] / 60:.2f} dk**; ikinci decode geçişlerinin maliyeti **{full['extra_decode_seconds']:.2f} s**. Fazlar arası rapor/test çalışmaları bu sürelere dahil değildir.",
            "",
            "| Kare | Atfedilen toplam süre (dk) | Peak allocated MiB | Peak reserved MiB |",
            "|---:|---:|---:|---:|",
        ]
    )
    for n in BUDGETS:
        r = methods[str(n)]
        lines.append(
            f"| {n} | {r['attributed_seconds'] / 60:.2f} | {r['peak_allocated_mib']:.1f} | {r['peak_reserved_mib']:.1f} |"
        )
    lines.extend(
        [
            "",
            "### Olağandışı bekleme ve süre duyarlılığı",
            "",
            "Tam çalışma sırasında izleme bağlantısı zaman aşımına uğradı; inference süreci ve cache korunarak devam etti. Windows güç günlüğünde 14:01:09 uyku (Button or Lid) ve 14:01:12 uyanma olayı var. Bu kayıtlar tek başına yaklaşık 757 saniyelik beklemenin tamamının nedenini kanıtlamaz. `power_events.json` ve ham `timing_incident.json` korunur.",
            "",
            "Aşağıdaki kareler için bir ısınma + üç ölçümlü tekrar yapıldı. Bbox, crop piksel hash'i ve yüz skorları özgün kayıtlarla karşılaştırıldı. Hiçbir başarı skoru, tahmin, video veya ham süre değiştirilmedi. Yeniden ölçüm sütunu yalnızca işaretli kare süresini üç tekrarın medyanıyla değiştiren ek bir maliyet tahminidir; bağımsız tam-cohort benchmark'ı değildir.",
            "",
            "| Video / kare | Ham kare süresi | Tekrar medyanı | Maksimum skor farkı | Etkilenen bütçe |",
            "|:---|---:|---:|---:|:---|",
        ]
    )
    for frame in recheck["frames"]:
        lines.append(
            f"| {frame['path']} / {frame['index']} | {frame['raw_processing_seconds']:.3f} s | {frame['median_remeasured_seconds']:.3f} s | {frame['max_face_score_absolute_delta']:.10g} | {frame['affected_budgets']} |"
        )
    lines.extend(
        [
            "",
            "| Kare | Ham ortalama s/video | Ham medyan s/video | Tek-kare yeniden ölçümüyle ortalama tahmini |",
            "|---:|---:|---:|---:|",
        ]
    )
    for n in BUDGETS:
        t = timing[str(n)]
        lines.append(
            f"| {n} | {t['raw_mean_seconds']:.3f} | {t['raw_median_video_seconds']:.3f} | {t['remeasured_mean_seconds_estimate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Bellek, ilgili seçili karelerin işlemleri sırasında CUDA allocator'dan ölçülen tepe değerdir; her fazda aynı model/detektör birlikte yüklenir ve koşullar ortak ölçümleri paylaşır. Reserved değeri allocator geçmişini içerir, dört izole koşulun bağımsız VRAM benchmark'ı değildir. Kareler sırayla, sınıflandırıcı batch=1 çalışır; 32 kare aynı anda GPU'ya yığılmaz.",
            "",
            "## Adaptif kare araştırması için tanı",
            "",
            "Bu deney temporal bir model öğrenmez; aynı görüntü sınıflandırıcısının zamana yayılmış skorlarını ortalar. İlk altı kare son kareyi de içerdiği için olası adaptif mekanizma tüm videoyu okuma maliyetini ortadan kaldırmaz. Tasarruf ek karelerin detector/sınıflandırıcı işlemlerinde olur; mevcut tasarımda genişletme için ikinci decode geçişi veya aday kareleri RAM'de tutma maliyeti ayrıca hesaba katılmalıdır.",
            "",
            "Altı kare skorunun 0.5'e uzaklığına göre önceden sabitlenmiş aralıklar; bunlar bir yönlendirme sistemi veya optimize edilmiş eşik değildir. Her hücre `video sayısı / düzeltilen / bozulan` biçimindedir.",
            "",
            "| Altı kare marjı | 12 kare | 24 kare | 32 kare |",
            "|:---|:---|:---|:---|",
        ]
    )
    for index, label in enumerate(
        ("[0, 0.05)", "[0.05, 0.10)", "[0.10, 0.20)", "[0.20, 0.50]")
    ):
        cells = []
        for n in BUDGETS[1:]:
            b = changes[str(n)]["margin_bins"][index]
            cells.append(f"{b['videos']} / {b['corrected']} / {b['harmed']}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "0.5'e yakınlık kalibre edilmiş belirsizlik değildir: model kendinden emin biçimde yanlış olabilir. Adaptif kare sayısı bilinen bir araştırma fikridir; yalnızca bu mekanizmayı eklemek dünya-ilki veya tek başına özgün bilimsel katkı oluşturmaz. Örneğin [AdaFrame (CVPR 2019)](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_AdaFrame_Adaptive_Frame_Selection_for_Fast_Video_Recognition_CVPR_2019_paper.html) girdi başına kare seçimini, [FrameExit (CVPR 2021)](https://openaccess.thecvf.com/content/CVPR2021/html/Ghodrati_FrameExit_Conditional_Early_Exiting_for_Efficient_Video_Recognition_CVPR_2021_paper.html) kolay videolarda az kareyle erken çıkışı araştırır. Bunlar genel video tanıma çalışmalarıdır; bu SBI/deepfake cohort'unda başarı garantisi vermez. Araştırma, hata/işlem maliyeti dengesini ayrı politika geliştirme verisinde belirleyip sabit bütçe ve rastgele ek-kare kontrolleriyle bağımsız değerlendirmede karşılaştırmalıdır. Bu görevde ana sisteme adaptif mekanizma eklenmedi.",
            "",
            f"Somut araştırma gerekçesi: altı kare marjı <0.10 olan **{near_videos}/602 video (%{100 * near_videos / 602:.2f})**, 6→12'nin düzelttiği **{near_corrected}/{len(changes['12']['corrected'])} videoyu** içeriyor; aynı grupta **{near_harmed} tahmin bozuluyor**. Bu yoğunlaşma önce 6, gerekirse 12 ve daha sonra 24 kareyi araştırmayı destekliyor. Ancak 24 karenin bazı düzeltmeleri bu dar grubun dışında; yalnızca 0.5'e uzaklık yeterli olmayabilir. Bu tablo bir adaptif sistemin ölçülmüş başarısı veya doğrulanmış durma eşiği değildir.",
            "",
            "## Karar: sabit bütçe, fine-tuning ve özgün katkı",
            "",
            "**En yüksek gözlenen başarı: 24 kare.** ROC-AUC 0.95371 ve Macro-F1 0.86209 ile iki ölçütte de en yüksek nokta; yaklaşık 11.37 s/video. Özellikle FAKE recall öncelikliyse bir doğruluk referansı olarak tutulabilir. 32 kare, ek maliyetine karşı gözlenen genel başarı avantajı sağlamıyor.",
            "",
            "**En iyi hız/başarı dengesi ve bundan sonraki sabit bütçe önerisi: 12 kare.** Macro-F1 0.86157 ile 24 kareye çok yakın, yaklaşık 5.83 s/video ile neredeyse iki kat hızlı; REAL recall ve balanced accuracy de dört koşul içinde en yüksek. 6→12: AUC +0.00601, Macro-F1 +0.02413, 12 düzelme / 3 bozulma, REAL yanlış alarm 28→25. Ek maliyet yaklaşık 2.77 s/video. 12→24: AUC +0.00417 ve Macro-F1 yalnızca +0.00052; maliyet yaklaşık +5.54 s/video ve REAL yanlış alarm 25→29. Bu, hız ve yanlış alarm hedefleri için 12 kareyi daha makul yapıyor. Mevcut ana SBI inference dosyasının altı-kare varsayılanı bu araştırma görevinde değiştirilmedi.",
            "",
            "**SBI fine-tuning planı: başlangıçta 12 kare/video ve sabit 12-kare validation protokolü.** Bu seçim zero-shot deneyden gelen pratik başlangıç bütçesidir; fine-tuned model için kanıtlanmış optimum değildir. 24 kare ek maliyetli karşılaştırma olarak korunabilir. Eğitimden önce 12-kare train cache'i, veri sınırları ve CUDA smoke ayrı doğrulanmalı; inference VRAM değeri eğitim VRAM'i sayılmaz. Mevcut eğitim kodunun altı-kare davranışı değiştirilmedi ve fine-tuning başlatılmadı.",
            "",
            "**Adaptif kare sayısı araştırmaya değer; tek başına özgünlük iddiası yeterli değil.** İlk araştırma hattı 6→12→24 olabilir; 32'nin zorunlu son basamak olması bu sonuçlarla desteklenmiyor. Katkı adayı, deepfake tespitinde yanlış alarm/kayıp maliyeti ve hesaplama bütçesini birlikte kontrol eden, ayrı geliştirme verisinde sabitlenmiş bir durma kuralının kaynak değişimi altında değerlendirilmesidir. Özgünlük ayrıca ilgili literatürle karşılaştırılmalı; mevcut validation üzerinde yeni eşikler seçip aynı veride başarı ilan edilmemelidir.",
            "",
            "## Koruma ve yeniden çalıştırma",
            "",
            f"Korunan {len(preservation)} dosyanın SHA-256 kontrolü geçti: app.py, V3 ve eski video checkpoint'leri, FFraw/FFc23, RetinaFace, manifestler, önceki SBI/kalite/robustness sonuçları ve mevcut SBI inference modülü değişmedi. Eski cache'ler yalnızca okundu. Yeni dependency yok; mevcut izole SBI runtime kullanıldı. Legacy RetinaFace paket metadata sürüm uyuşmazlığı önceki kurulumdan sürüyor; gerçek inference bu ortamda doğrulandı.",
            "",
            "```powershell",
            r".\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_frame_study baseline",
            r".\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_frame_study pilot",
            r".\outputs\video_research_v1\env\Scripts\python.exe -u -m src.video_frame_study full",
            r".\outputs\video_research_v1\env\Scripts\python.exe -m scripts.recheck_video_frame_timing",
            r".\.venv\Scripts\python.exe -m scripts.report_video_frames",
            "```",
            "",
            "Tamamlanmış fazların sonuçlarını ezme girişimi engellenir. Kesilen bir faz aynı komutla tekrar başlatılabilir; hash/provenance kontrolünden geçen video cache'leri yeniden kullanılır. Cache video başına atomik kaydedilir; kesilme anında tamamlanmamış videonun kareleri tekrar hesaplanabilir. Fazlar mutlaka baseline → pilot → full sırasıyla çalışır.",
            "",
            "## Yazılım doğrulaması",
            "",
            f"Tam regresyon: **{checks['pytest']['tests']} test, {checks['pytest']['failures']} failure, {checks['pytest']['errors']} error, {checks['pytest']['skipped']} skip**. Yeni testler; iç içe/benzersiz örnekleme, tarihi indekslerin korunması, kısa video, decode hatası, sabit eşik, cache bütünlüğü, çoklu yüz maksimumu, crop/skor sapması, süre duyarlılığı ve eşleştirilmiş istatistik kontrollerini kapsar.",
            "",
            "`ruff check`, `ruff format --check`, `compileall` ve `git diff --check` geçti. Python projesi; TypeScript kontrolü yok. Testler için gereken yerel soket/atomik geçici dosya işlemleri Windows sandbox'ta engellendiğinden tam pytest normal yerel izinlerle çalıştırıldı. Bu test ortamı kısıtı için uygulama kodu değiştirilmedi. Son kayıtlar: `outputs/video_frames_v1/checks.json` ve `regression.xml`.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    full, baseline, pilot = [
        read_json(STUDY / name) for name in ("full.json", "baseline.json", "pilot.json")
    ]
    manifest, _, records = accepted_roster("val")
    if (
        not baseline["baseline_verified"]
        or not full["baseline_verified"]
        or full["provenance"] != baseline["provenance"]
        or pilot["provenance"] != baseline["provenance"]
    ):
        raise ValueError("Study phases/provenance do not match.")
    methods = full["methods"]
    if (
        full["provenance"]["manifest_fingerprint"] != manifest["fingerprint"]
        or full["provenance"]["roster_sha256"] != digest_json(records)
        or set(methods) != {str(n) for n in BUDGETS}
    ):
        raise ValueError("Study roster/manifest/budgets changed.")
    expected = [(r["path"], r["label"]) for r in records]
    for result in methods.values():
        rows = result["rows"]
        if (
            [(r["path"], r["target"]) for r in rows] != expected
            or any(r["status"] != "accepted" for r in rows)
            or any(r["prediction"] != int(r["fake_probability"] >= 0.5) for r in rows)
        ):
            raise ValueError("Study cohort/labels/threshold changed.")
        recomputed = summarize(rows, len(records))
        if any(result[k] != value for k, value in recomputed.items()):
            raise ValueError("Saved metrics do not reproduce from predictions.")
    if any(r["attempted"] != 602 or r["evaluated"] != 602 for r in methods.values()):
        raise ValueError(
            "Full 602-video coverage required; report failures before comparison."
        )
    protected = read_json(STUDY / "protected_before.json")
    preservation = {
        path: file_sha256(Path(path)) == sha for path, sha in protected.items()
    }
    if not all(preservation.values()):
        raise ValueError("Protected files changed.")
    environment = read_json(STUDY / "environment.json")
    locked_source = {
        path: file_sha256(Path(path)) == sha
        for path, sha in environment["source_sha256"].items()
    }
    if not all(locked_source.values()):
        raise ValueError("Locked study sampler/protocol changed during inference.")
    uncertainty = paired_uncertainty(methods, records)
    changes = transitions(methods)
    checks = read_json(STUDY / "checks.json")
    if checks["pytest"]["failures"] != "0" or checks["pytest"]["errors"] != "0":
        raise ValueError("Regression checks failed.")
    recheck = read_json(STUDY / "timing_recheck.json")
    if recheck["full_result_sha256"] != file_sha256(STUDY / "full.json"):
        raise ValueError("Timing recheck belongs to a different full result.")
    timing = timing_sensitivity(methods, recheck)
    write_json(
        STUDY / "analysis.json",
        dict(
            paired_family_bootstrap=uncertainty,
            transitions=changes,
            preserved_files=preservation,
            locked_study_source=locked_source,
            timing_sensitivity=timing,
        ),
    )
    plot_results(methods, timing)
    Path("docs/VIDEO_FRAME_STUDY.md").write_text(
        render_report(
            full,
            baseline,
            pilot,
            uncertainty,
            changes,
            preservation,
            checks,
            timing,
            recheck,
        ),
        encoding="utf-8",
    )
    print(
        "Wrote docs/VIDEO_FRAME_STUDY.md and analysis.json/comparison.png", flush=True
    )


if __name__ == "__main__":
    main()
