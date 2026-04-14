@echo off
REM =====================================================
REM  Auto-Restart Training Watchdog
REM  - VS Code'dan bagimsiz calisir
REM  - BSOD/crash sonrasi otomatik devam eder
REM  - Her checkpoint'tan kaldigi yerden devam eder
REM  - Bu dosyayi cift tiklayarak calistir (Admin olarak)
REM =====================================================

set PYTHONUTF8=1
set ROOT=C:\Users\user\Documents\GitHub\ai-analysis
set VENV=%ROOT%\venv\Scripts\python.exe
set SCRIPT=%ROOT%\train_detached.py
set LOGFILE=%ROOT%\training.log

echo =====================================================
echo   SOCIAL GOOD CHATBOT - AUTO TRAINING WATCHDOG
echo   GPU Throttled: 175W + 1500MHz clock lock
echo =====================================================
echo.

REM GPU Throttle (requires admin)
echo Setting GPU power limit to 175W...
nvidia-smi -pl 175
echo Setting GPU clock lock to max 1500MHz...
nvidia-smi -lgc 0,1500
echo.

REM Training loop - restarts if crashes
:loop
echo [%date% %time%] Starting training session...
echo [%date% %time%] Starting training session... >> "%LOGFILE%"

"%VENV%" "%SCRIPT%"

REM Check exit code
if %errorlevel% equ 0 (
    echo [%date% %time%] Training completed successfully!
    echo [%date% %time%] Training completed successfully! >> "%LOGFILE%"
    goto done
)

echo [%date% %time%] Training crashed (exit code: %errorlevel%). Restarting in 30s...
echo [%date% %time%] Training crashed (exit code: %errorlevel%). Restarting in 30s... >> "%LOGFILE%"
timeout /t 30 /nobreak
goto loop

:done
echo.
echo =====================================================
echo   TRAINING COMPLETE!
echo   Check: %ROOT%\data\models\social_good_v1\
echo   Logs:  %LOGFILE%
echo =====================================================
REM Reset GPU
echo Resetting GPU limits...
nvidia-smi -pl 250
nvidia-smi -rgc
echo GPU reset to defaults.
pause
