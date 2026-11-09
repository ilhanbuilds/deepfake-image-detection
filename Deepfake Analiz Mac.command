#!/bin/bash
set -u

fail() {
    printf '\n%s\nKurulum adımları: docs/MACOS.md\n' "$1"
    if [ -t 0 ]; then
        read -r -p 'Kapatmak için Enter tuşuna basın. ' _answer
    fi
    exit 1
}

cd -- "$(dirname -- "$0")" || exit 1

python='./.venv/bin/python'
if [ ! -x "$python" ]; then
    fail 'Mac Python ortamı bulunamadı. Windows sanal ortamı Mac üzerinde kullanılamaz.'
fi
if ! "$python" -B -c 'import streamlit, torch, torchvision, PIL' ; then
    fail 'Python bağımlılıkları eksik veya yüklenemiyor. Mac kurulumunu tamamlayın.'
fi

printf '\nDeepfake Analiz Studio tarayıcıda açılıyor. Oturum adresi aşağıda gösterilecek.\n'
printf 'Bu Terminal penceresini açık tutun. Durdurmak için Control+C kullanın.\n'
printf 'Mac: fotoğraf analizi CPU kullanır; mevcut video modeli CUDA gerektirir.\n'
if [ ! -f './outputs/checkpoints/best_model_v3_robust.pth' ]; then
    printf '\nFotoğraf modeli bulunamadı. Arayüz açılır; analiz için docs/MACOS.md dosyasındaki modeli yerleştirin.\n'
fi

"$python" -B './scripts/start_deepfake_app.py'
status=$?
if [ "$status" -ne 0 ]; then
    fail "Uygulama başlatılamadı (çıkış kodu: $status). Yukarıdaki hata mesajını kontrol edin."
fi
