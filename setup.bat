@echo off
echo ==========================================
echo   MindScan - One Time Setup
echo ==========================================
echo.

echo [1/4] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo [2/4] Installing packages (5-10 mins)...
pip install -r requirements.txt

echo [3/4] Downloading BERT tokenizer files...
python -c "from transformers import BertTokenizer; BertTokenizer.from_pretrained('bert-base-uncased').save_pretrained('models/bert_mental_health')"

echo [4/4] Setup complete!
echo.
echo ==========================================
echo  NEXT STEPS:
echo  1. Copy model files into models\bert_mental_health\
echo  2. Rename .env.example to .env
echo  3. Add your GROQ_API_KEY in .env
echo  4. Double-click run.bat to start the app
echo ==========================================
pause
