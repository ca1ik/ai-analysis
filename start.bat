@echo off
echo Starting Social Good Chatbot Platform...
echo API Docs: http://localhost:8000/docs
echo.
call venv\Scripts\activate 2>nul
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
