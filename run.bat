@echo off
echo ==========================================
echo   MindScan - Mental Health Risk Predictor
echo ==========================================
echo.
call venv\Scripts\activate
if errorlevel 1 (
    echo [ERROR] Run setup.bat first!
    pause
    exit
)
echo [INFO] Starting MindScan...
echo [INFO] Browser opens at http://localhost:8501
echo [INFO] Press Ctrl+C to stop
echo.
streamlit run app.py
pause
