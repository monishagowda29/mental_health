<div align="center">

# 🧠 MindScan

### AI-Powered Clinical Mental Health Screening & Multi-Patient Wellbeing Tracker

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![BERT](https://img.shields.io/badge/BERT-Mental_Health-6366F1?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-Llama_4_Scout-F97316?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-Academic_Research-8B5CF6?style=for-the-badge)](#disclaimer)

> **Fine-tuned BERT + Groq Llama 4 Scout Vision AI · Offline Seq2Seq Translation · 100% Private & Scoped Multi-Patient Wellbeing Workspace**  
> Mysore University — School of Engineering · AI&DS Department · Academic Research Project

</div>

---

## 📖 Overview

**MindScan** is a 10/10 clinical-grade mental health screening prototype designed for clinics, therapists, and personal wellbeing self-assessments. It combines fine-tuned deep learning **BERT** classifiers and **Groq Llama 4 Scout 17B** vision models with local, private **Seq2Seq offline translation** for 8 Indian languages. 

MindScan is 100% offline-first and private-by-default — text analysis and translation run locally on your device, ensuring patient data sovereignty.

---

## ✨ Key Features & Capabilities

### 🛡️ 1. Safety Crisis Care & Auto-Language Detection
*   **Pulsating Crisis Care Card (`.crisis-card`)**: Instantly overlays a premium rose-to-crimson gradient card above the screen if suicidal/self-harm keywords are detected (English or translated), or if the BERT model outputs a depression probability $\ge 80\%$. Connects users directly to clickable 24/7 helplines (988, AASRA, Vandrevala Foundation).
*   **Empathetic Label Softening**: Alarms are softened into supportive language (e.g. `🔴 DEPRESSION RISK` ➔ `🧠 DEPRESSIVE PATTERNS DETECTED`, `🔴 ANXIETY RISK` ➔ `⚠️ ANXIETY PATTERNS DETECTED`).
*   **Script Auto-Detection**: Fast Unicode block scanner automatically identifies native scripts (Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali) and triggers a background toast optimization.

### 📋 2. Interactive Clinical Screeners (PHQ-9 & GAD-7)
*   **Gold-Standard Forms**: Dedicated screeners tab hosting official **PHQ-9 (Depression)** and **GAD-7 (Anxiety)** clinical questionnaires with live scoring, severity meters, and therapeutic advice.
*   **AI vs. Self-Report Contrast Card**: Dynamic comparison panel highlighting spontaneous linguistic predictions (BERT) against structured clinical metrics for counseling preparation.
*   **Personalized PDF/Text Report Downloader**: Stamped with patient Name/ID, date, scores, severity, and clinical disclaimers.

### 👤 3. Scoped Multi-Patient Wellbeing Dashboard
*   **Private Persistent Logging**: History logs (`outputs/history.json`) are scoped using the **Patient Name / ID** as a primary key. Multiple patients can share the same local dashboard without database leaks.
*   **Wellbeing trend Line Charts**: Plots chronological risk levels (`0=Low Risk, 1=Anxiety, 2=Depressive`) using styled dark-mode Seaborn graphs.
*   **Data Sovereignty**: Clean CSV history export and single-click secure erase controls to wipe private patient logs.

### 🧬 4. Direct Seq2Seq Translation & Sentiment Calibration (Pillar 4)
*   **Futureproof Seq2Seq Translation**: Uses direct Marian Seq2Seq AutoModels (`Helsinki-NLP/opus-mt-mul-en`) rather than legacy pipeline wrappers, avoiding `"Unknown task translation"` crashes in `transformers` v5.0+.
*   **Clinical Sentiment Calibration**: Employs rule-based overrides for general-domain wellness sentences (e.g. `"i am doing good feeling nice"`) to bypass BERT domain bias, mapping them to `normal` at **90% confidence** and preventing false alarms.

---

## 🏗️ Architecture Flow

```
User Multilingual Input (Any script U+0900–U+0D7F)
         │
         ▼ (Unicode Auto-Detector)
┌──────────────────────────────────────────┐
│ Helsinki-NLP opus-mt-mul-en AutoModel    │ ← Inferences Seq2Seq locally offline
│ (Bypasses legacy v5.x pipeline wrappers) │
└──────────────────┬───────────────────────┘
                   │
                   ▼ Translated English Text
┌──────────────────────────────────────────┐
│ Clinical Sentiment Calibration Guard     │ ← Bypasses model for pure wellness text
│ (Wellness terms + 0 Clinical terms)      │
└──────────────────┬───────────────────────┘
                   │
                   ▼ Standard Clinical Statements
┌──────────────────────────────────────────┐
│ Fine-tuned BERT Classifier (CPU/GPU)     │
│ [Index 0: Depression, 1: Normal, 2: Anx] │
└──────────────────┬───────────────────────┘
                   │
                   ▼ Dynamic Frontend Rendering
┌──────────────────────────────────────────┐
│ 👤 Scoped Patient Profiles & Dashboard    │
│ 📋 GAD-7 & PHQ-9 Clinical Screeners       │
│ 🖼️ Groq Llama 4 Scout Vision Analysis     │
└──────────────────────────────────────────┘
```

---

## 🚀 Local Setup & Quick Start

Follow these step-by-step instructions to get your local environment configured and running:

### 📋 Prerequisites
1.  **Python 3.12**: [Download Python 3.12.x](https://www.python.org/downloads/release/python-3120/) (Add to system PATH).
2.  **Git**: [Download Git](https://git-scm.com/downloads).
3.  **Groq API Key** (Free): Get a key at [console.groq.com](https://console.groq.com) (for Llama 4 Vision analysis only).

---

### Step 1 — Clone the Repository
Open a terminal (PowerShell or Command Prompt on Windows) and run:
```bash
git clone https://github.com/AryaPartha/MindScan.git
cd MindScan
```

---

### Step 2 — Configure Environment Variables (`.env`)
Create a file named `.env` in the root folder of the project (`MindScan/.env`) and add your Groq key:
```ini
# MindScan Environment Configuration
GROQ_API_KEY=your_groq_api_key_here
```

---

### Step 3 — Download Fine-Tuned Model Weights
The BERT model weights are too large for standard Git tracking. 
1. Obtain the `bert_mental_health` weights directory from the shared drive or project team.
2. Place the folder exactly under:
```
MindScan/models/bert_mental_health/
```
Verify that the following files are present inside that directory:
- `config.json`
- `model.safetensors`
- `tokenizer_config.json`
- `tokenizer.json`

---

### Step 4 — Run One-Time Automated Setup
Double-click `setup.bat` in the root directory, or run it in your terminal:
```bash
setup.bat
```
*This script will:*
- Create a clean local Python virtual environment (`venv/`)
- Upgrade `pip` to the latest version
- Install all heavy dependencies (`torch`, `transformers`, `streamlit`, `pandas`, `sacremoses`, etc.)

---

### Step 5 — Launch the MindScan Application
Double-click `run.bat` in the root directory, or run it in your terminal:
```bash
run.bat
```
Streamlit will compile modules and automatically launch a new browser window at:
**[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Running Automated Unit Tests
To verify all models, cache engines, offline Seq2Seq pipelines, and calibration overrides operate successfully:

1.  Activate the virtual environment:
    ```bash
    venv\Scripts\activate
    ```
2.  Run the test suite:
    ```bash
    python tests/run_tests.py
    ```
    *All **17 tests** (BERT Singleton, Sentiment overrides, Helsinki AutoModel wrappers, cached translation LRU, vision completions) should run and return `[SUCCESS] All test suites completed successfully!`.*

---

## 📊 Model Performance Metrics
The BERT classifier was fine-tuned on the Kaggle Reddit Mental Health dataset (~20,000+ records) and is optimized for the following clinical categories:

| Class Label | Model Output Index | Training Target Alignment |
|---|---|---|
| **Depression** | Index 0 | High-arousal bipolar, major depressive symptoms |
| **Normal** | Index 1 | General-domain neutral thoughts & mental wellness |
| **Anxiety** | Index 2 | Panic attacks, social phobias, stress indicators |

*Detailed training reports, confusion matrices, and Seaborn curves are generated automatically inside the `outputs/` folder.*

---

## 📜 Disclaimer & Scientific Disclosures

MindScan is a **research prototype** developed for academic purposes at the School of Engineering, University of Mysore (AI&DS Department).

- ❌ **NOT** a certified diagnostic medical device.
- ❌ **NOT** a replacement for professional clinical counseling or psychiatric guidance.
- ✅ Developed solely for educational and research awareness.

**If you or someone you know is in crisis, please seek immediate help. Confidential support is available 24/7:**
*   🇮🇳 **India:** Call **9152987821** (iCall) or **1860-2662-345** (Vandrevala Foundation)
*   🇺🇸 **US/Canada:** Call or Text **988** (Suicide & Crisis Lifeline)

---

<div align="center">
  <sub>Mysore University — AI&DS Dept · Built with ❤️ for mental health awareness</sub>
</div>
