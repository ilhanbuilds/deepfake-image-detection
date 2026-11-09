@echo off
setlocal
chcp 65001 >nul
set PYTHONUTF8=1
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -B "scripts\setup_delivery.py" %*
) else (
    where py >nul 2>&1
    if errorlevel 1 (
        python -B "scripts\setup_delivery.py" %*
    ) else (
        py -3.12 -B "scripts\setup_delivery.py" %*
    )
)
if errorlevel 1 (
    echo.
    echo Baslatma tamamlanamadi. Python 3.12 kurulu olmali.
    echo Yukaridaki hata mesajini ve README.md dosyasini kontrol edin.
    pause
    exit /b 1
)
