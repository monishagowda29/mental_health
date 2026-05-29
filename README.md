<div align="center">

# 🧠 MindScan

### AI-Powered Mental Health Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![BERT](https://img.shields.io/badge/BERT-Mental_Health-6366F1?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-Llama_4_Scout-F97316?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-Research_Only-8B5CF6?style=for-the-badge)](#disclaimer)

> **BERT + Groq Vision AI · 8 Indian Languages · 100% Private & Offline-First**  
> Mysore University — School of Engineering · AI&DS Department · Project Phase-I

</div>

---

## 📖 Overview

**MindScan** is a research-grade mental health screening tool that uses fine-tuned **BERT** to classify text into three categories — `anxiety`, `depression`, and `normal` — with support for **8 Indian languages** via an offline translation pipeline. It also supports image/screenshot analysis using **Groq's Llama 4 Scout 17B** vision model.

> **⚠️ This is a research prototype — NOT a clinical diagnostic tool.**  
> Always consult a qualified mental health professional.

---

## ✨ Features

| Feature | Detail |
|---|---|
| 📝 **Text Analysis** | Fine-tuned `bert-base-uncased` classifies text as `anxiety` / `depression` / `normal` |
| 🌐 **8 Indian Languages** | English, Hindi, Kannada, Tamil, Telugu, Malayalam, Bengali, Marathi |
| 🔄 **Offline Translation** | Helsinki-NLP `opus-mt-mul-en` — no API calls, fully private |
| 🖼️ **Image Analysis** | Groq Llama 4 Scout 17B — General Tone / Social Media / Scientific Chart |
| 📋 **Batch Analysis** | Paste multiple posts, get a CSV + summary report |
| 🎨 **Premium Dark UI** | Glassmorphism, animated gradients, Noto Sans Indic fonts |
| 🔒 **Privacy-First** | Text analysis + translation runs 100% locally — zero data sent to cloud |

---

## 🏗️ Architecture

```
User Input (any of 8 languages)
        │
        ▼
┌───────────────────────────────────────┐
│  Translation Pipeline (offline)       │
│  Helsinki-NLP opus-mt-mul-en          │  ← Only active for non-English input
│  Indic Language → English             │
└────────────────┬──────────────────────┘
                 │
        ▼ English text
┌───────────────────────────────────────┐
│  BERT Classifier (CPU/GPU)            │
│  bert-base-uncased (fine-tuned)       │
│  → anxiety / depression / normal      │
└───────────────────────────────────────┘

Image Input → Groq API (Llama 4 Scout 17B) → Vision Analysis
```

---

## 📁 Project Structure

```
MindScan/
├── app.py                        ← Streamlit UI (main entry point)
├── train.py                      ← BERT fine-tuning pipeline
├── download_dataset.py           ← Dataset download helper
├── requirements.txt              ← All Python dependencies
├── setup.bat                     ← One-time Windows setup script
├── run.bat                       ← Launch app (Windows)
│
├── src/
│   ├── config.py                 ← Centralised config (paths, API keys, models)
│   └── services/
│       ├── translation.py        ← TranslationService (thread-safe singleton)
│       └── vision.py             ← GroqVisionService (thread-safe singleton)
│
├── tests/
│   ├── run_tests.py              ← Full test runner
│   ├── test_translation.py       ← Translation service unit tests
│   └── test_vision.py            ← Vision service unit tests
│
├── models/
│   └── bert_mental_health/       ← Place downloaded model weights here
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer.json
│       └── tokenizer_config.json
│
├── data/                         ← Training dataset (not tracked in git)
└── outputs/                      ← Training plots & metrics
```

---

## 🚀 Quick Start (Windows)

### Prerequisites
- **Python 3.12** — [Download here](https://www.python.org/downloads/release/python-3120/)
- **Git** — [Download here](https://git-scm.com/downloads)
- A free **Groq API key** — [console.groq.com](https://console.groq.com) *(for image analysis only)*

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/AryaPartha/MindScan.git
cd MindScan
```

### Step 2 — Get the BERT model weights

The trained model weights are not stored in this repo (too large for Git).  
Download the `bert_mental_health` folder and place it at:

```
MindScan/models/bert_mental_health/
```

> Contact the project team or check the project's shared Google Drive for the model files.

### Step 3 — Configure your API key

Create a `.env` file in the project root:

```bash
# .env
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free key at [console.groq.com](https://console.groq.com) → Sign up → API Keys → Create key

### Step 4 — One-time setup

```bash
# Double-click, or run in terminal:
setup.bat
```

This creates a Python 3.12 virtual environment and installs all dependencies (~5 minutes on first run).

### Step 5 — Launch MindScan

```bash
# Double-click, or run in terminal:
run.bat
```

Browser opens automatically at **http://localhost:8501**

---

## 🌐 Supported Languages

| Code | Language | Script | Translation |
|------|----------|--------|-------------|
| `en` | English | Latin | Native (no translation) |
| `hi` | हिन्दी — Hindi | Devanagari | Helsinki-NLP offline |
| `kn` | ಕನ್ನಡ — Kannada | Kannada | Helsinki-NLP offline |
| `ta` | தமிழ் — Tamil | Tamil | Helsinki-NLP offline |
| `te` | తెలుగు — Telugu | Telugu | Helsinki-NLP offline |
| `ml` | മലയാളം — Malayalam | Malayalam | Helsinki-NLP offline |
| `bn` | বাংলা — Bengali | Bengali | Helsinki-NLP offline |
| `mr` | मराठी — Marathi | Devanagari | Helsinki-NLP offline |

All non-English inputs are translated **locally** using the `Helsinki-NLP/opus-mt-mul-en` model — no internet required, no data leaves your machine.

---

## 🧪 Running Tests

```bash
# Activate venv first
venv\Scripts\activate

# Run full test suite (25 tests)
python -m pytest tests/ -v

# Or run the custom test runner
python tests/run_tests.py
```

---

## 🔧 Training Your Own Model

```bash
# 1. Download the dataset
python download_dataset.py

# 2. Train BERT (requires GPU for reasonable speed)
python train.py

# Output saved to: models/bert_mental_health/
```

Training configuration is in [`src/config.py`](src/config.py).

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **UI Framework** | Streamlit 1.57 |
| **ML Framework** | PyTorch 2.12 + HuggingFace Transformers |
| **Classification Model** | `bert-base-uncased` (fine-tuned) |
| **Translation Model** | `Helsinki-NLP/opus-mt-mul-en` |
| **Vision Model** | Groq API — `meta-llama/llama-4-scout-17b-16e-instruct` |
| **Runtime** | Python 3.12 |
| **Fonts** | Google Noto Sans (Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali) |

---

## ⚙️ Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (for image tab) | Groq API key for vision analysis |
| `MODEL_DIR` | Optional | Override default model path |
| `DEVICE` | Optional | `cpu` or `cuda` (auto-detected) |

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Task** | 3-class text classification |
| **Classes** | `anxiety` · `depression` · `normal` |
| **Base Model** | `bert-base-uncased` |
| **Language** | English (with offline translation for 7 Indian languages) |

> Detailed training metrics and confusion matrices are saved to `outputs/` after training.

---

## 📜 Disclaimer

MindScan is a **research prototype** developed as an academic project at the **School of Engineering, University of Mysore (AI&DS Department)**.

- ❌ **NOT** a certified medical device  
- ❌ **NOT** a replacement for professional mental health diagnosis  
- ✅ For research, academic, and educational purposes only  

**If you or someone you know is in crisis, please contact a mental health professional or a helpline immediately.**

> 🇮🇳 **iCall (India):** 9152987821 | **Vandrevala Foundation:** 1860-2662-345 (24/7)

---

## 👥 Authors

**AryaPartha** · Mysore University · AI&DS Department

---

<div align="center">
  <sub>Built with ❤️ for mental health awareness · Research use only</sub>
</div>
