@echo off
echo ============================================
echo  Social Good Chatbot Platform - Setup
echo ============================================
echo.

echo [1/3] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/3] Upgrading pip...
python -m pip install --upgrade pip

echo [3/3] Installing dependencies...
pip install -r requirements.txt

echo.
echo ============================================
echo  Setup complete!
echo  To start the platform:
echo    venv\Scripts\activate
echo    python -m uvicorn src.main:app --reload --port 8000
echo  Then visit: http://localhost:8000/docs
echo ============================================
