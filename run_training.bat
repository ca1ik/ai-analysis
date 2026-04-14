@echo off
chcp 65001 >nul 2>&1
title Social Good Training - Auto Chunk

cd /d "c:\Users\user\Documents\GitHub\ai-analysis"

echo ============================================================
echo   SOCIAL GOOD CHATBOT - Micro-Chunk Auto Training
echo   Her 50 step'te process yeniden baslar (GPU driver safe)
echo ============================================================
echo.

REM GPU Power Limit (admin gerekir, hata verirse devam eder)
nvidia-smi -pl 175 2>nul
if %errorlevel%==0 (
    echo [OK] GPU Power Limit: 175W
) else (
    echo [WARN] GPU power limit ayarlanamadi - admin ile calistirin
)
echo.

set PYTHONUTF8=1
set CUDA_LAUNCH_BLOCKING=1
set "VENV=c:\Users\user\Documents\GitHub\ai-analysis\venv\Scripts\python.exe"

set CHUNK=0
set MAX_RETRIES=3
set RETRY=0

:loop
set /a CHUNK+=1
echo.
echo [%date% %time%] === CHUNK #%CHUNK% baslatiiliyor ===

"%VENV%" train_chunk.py
set EXIT_CODE=%errorlevel%

if %EXIT_CODE%==0 (
    echo.
    echo ============================================================
    echo   TRAINING TAMAMLANDI!
    echo   Model: data\models\social_good_v1\
    echo ============================================================
    echo.
    REM GPU power limiti geri al
    nvidia-smi -pl 250 2>nul
    echo [OK] GPU Power Limit: 250W (reset)
    pause
    exit /b 0
)

if %EXIT_CODE%==2 (
    echo [%date% %time%] Chunk basarili, sonraki chunk'a geciliyor...
    set RETRY=0
    REM 5 saniye GPU sogumasi
    timeout /t 5 /nobreak >nul
    goto loop
)

REM Exit code 1 = error
set /a RETRY+=1
echo [%date% %time%] HATA! Retry %RETRY%/%MAX_RETRIES%

if %RETRY% GEQ %MAX_RETRIES% (
    echo.
    echo ============================================================
    echo   %MAX_RETRIES% ardisik hata - training durduruluyor
    echo   Son checkpointi kontrol edin.
    echo ============================================================
    nvidia-smi -pl 250 2>nul
    pause
    exit /b 1
)

echo 15 saniye bekleniyor (GPU soguyor)...
timeout /t 15 /nobreak >nul
goto loop
