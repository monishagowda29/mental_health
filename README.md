# 🧠 MindScan — Mental Health Risk Prediction

> BERT + Groq AI | English + Kannada Support  
> Mysore University School of Engineering — AI&DS Dept. | Project Phase-I

---

## 📁 Folder Structure

```
MindScan/
├── app.py                  ← Main Streamlit app
├── train.py                ← BERT training pipeline
├── requirements.txt        ← All dependencies
├── setup.bat               ← One-time setup (Windows)
├── run.bat                 ← Launch app (Windows)
├── .env.example            ← Rename to .env and add Groq key
├── models/
│   └── bert_mental_health/ ← Place model files here
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── data/                   ← Place dataset CSV here
└── outputs/                ← Training plots saved here
```

---

## 🚀 Setup & Run (VS Code / Windows)

### Step 1 — Get model files
- Download all files from Google Drive folder: `bert_mental_health_model`
- Place them in `models\bert_mental_health\`

### Step 2 — Get Groq API key (free)
- Go to https://console.groq.com
- Sign up → Create API Key → Copy it

### Step 3 — Configure .env
- Rename `.env.example` → `.env`
- Open `.env` and paste your Groq key

### Step 4 — Run setup (ONE TIME ONLY)
- Double-click `setup.bat`
- Wait 5-10 mins for packages to install

### Step 5 — Launch app
- Double-click `run.bat`
- Browser opens at http://localhost:8501

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 Text Analysis | BERT-based depression/anxiety/normal classification |
| 🌐 Kannada Support | Auto-translate Kannada → English → BERT predict |
| 🖼️ Image Analysis | Groq Llama 4 Scout vision — 3 analysis modes |
| 📋 Batch Analysis | Analyze multiple posts, download CSV |
| 🇮🇳 Bilingual UI | Full English + Kannada interface toggle |

---

## ⚠️ Disclaimer
Research prototype only. NOT a clinical diagnostic tool.  
Always consult a qualified mental health professional.
