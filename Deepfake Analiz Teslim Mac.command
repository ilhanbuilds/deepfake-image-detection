#!/bin/bash
set -u
cd -- "$(dirname -- "$0")" || exit 1
if [ -x './.venv/bin/python' ]; then
    python='./.venv/bin/python'
elif command -v python3.12 >/dev/null 2>&1; then
    python='python3.12'
else
    python='python3'
fi
"$python" -B './scripts/setup_delivery.py' --photo-only
status=$?
if [ "$status" -ne 0 ]; then
    printf '\nPython 3.12 ve ilk kurulum için internet gerekir. Yukarıdaki hata mesajını kontrol edin.\n'
    if [ -t 0 ]; then
        read -r -p 'Kapatmak için Enter tuşuna basın. ' _answer
    fi
fi
exit "$status"
