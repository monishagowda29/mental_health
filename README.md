<div align="center">

# 🧠 MindScan

### AI-Powered Clinical Mental Health Screening & Privacy-First Wellbeing Dashboard

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-19-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![Celery](https://img.shields.io/badge/Celery-5.x-37814A?style=for-the-badge&logo=celery&logoColor=white)](https://docs.celeryq.dev)
[![BERT](https://img.shields.io/badge/BERT-Mental_Health-6366F1?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co)
[![Groq](https://img.shields.io/badge/Groq-Llama_4_Scout-F97316?style=for-the-badge)](https://groq.com)
[![License](https://img.shields.io/badge/License-Academic_Research-8B5CF6?style=for-the-badge)](#disclaimer)

> **Fine-tuned BERT + Groq Llama 4 Scout Vision AI · Celery Async Task Queue · Zero-Knowledge AES-GCM-256 Encryption · Offline Seq2Seq Translation for 8 Indian Languages**
> Mysore University — School of Engineering · AI&DS Department · Academic Research Project

</div>

---

## 📖 Overview

**MindScan** is a clinical-grade mental health screening prototype designed for clinics, therapists, and personal wellbeing self-assessments. It is built on a modern, production-grade microservices stack:

- **FastAPI** backend with async task dispatch
- **Celery** + **Redis** worker queue for non-blocking BERT and OCR inference
- **React 19 + Zustand** glassmorphism frontend with a privacy-first design
- **ONNX INT8-quantized BERT** for fast, CPU-efficient local inference
- **Groq Llama 4 Scout 17B** multimodal vision for scanned document analysis
- **Helsinki-NLP MarianMT** for offline Seq2Seq translation across 8 Indian languages
- **AES-GCM-256 Zero-Knowledge** client-side encryption — the server never sees plaintext

MindScan is 100% **private-by-default** — text analysis and translations run locally on-device; raw journal text never touches the server disk.

---

## ✨ Feature Overview

### 🛡️ 1. Automatic Crisis Detection & Emergency Support
- **Real-time keyword scanning** across all journal input (English + all 8 Indian languages): detects `suicide`, `self-harm`, `die`, `end my life`, `cutting`, `hanging`, and similar indicators
- **PHQ-9 self-harm trigger**: crisis banner auto-activates if Question 9 is answered above zero
- **BERT score threshold trigger**: activates if the latest journal analysis returns depression probability ≥ 50%
- **Pulsating emergency banner** with 24/7 crisis helpline deep-links:
  - 📞 Tele-MANAS: `14416`
  - 📞 TISS iCall: `9152987821`
  - 📞 AASRA: `9820466726`
  - 📞 988 Suicide & Crisis Lifeline (US)

---

### 📓 2. Zero-Knowledge Encrypted Journal Analytics
- **Client-side AES-GCM-256 encryption** via Web Crypto API — passphrase-derived with PBKDF2 (100,000 iterations, SHA-256)
- **4-segment passphrase strength meter** (Weak / Fair / Strong / Excellent) with per-character entropy scoring; submission blocked if passphrase is too weak
- **Live character counter** on the journal textarea (turns amber at 2,000, red at 2,400, hard-capped at 2,500)
- **Ctrl+Enter** keyboard shortcut to submit
- **Optimistic UI updates** — entry appears instantly with a PENDING badge, showing a live elapsed timer (`Analyzing… 12s`) while the Celery worker processes the task
- **Animated score breakdown pills** — emerald/amber/rose progress bars per journal card showing Normal / Anxiety / Depression confidence scores
- **Wellbeing Risk Severity Timeline** — SVG polyline chart with interactive hover tooltips (date, prediction label, D/A/N scores) for every plotted data point
- **Distribution stats with trend arrows** — session-over-session `↑` / `↓` / `—` comparison for Wellness, Anxious, and Depressive pattern percentages
- **Wipe History two-step confirmation modal** — prevents accidental single-click data erasure; displays patient name and irreversibility warning
- **Encrypted PDF export** — full journal history report with timestamped entries, confidence scores, and clinical disclaimers

---

### 📋 3. Interactive Clinical Screeners (PHQ-9 & GAD-7)
- **Gold-standard questionnaires** — official **PHQ-9** (9-item depression screener) and **GAD-7** (7-item anxiety screener)
- **Screener progress bar** — gradient fill bar showing `X of N answered` and `% complete` in real time
- **Question completion state** — answered questions highlighted with indigo number badge; calculate button disabled and shows remaining count until all questions are answered
- **Reset button** — one-click to clear all answers and start over on either screener
- **Severity outcome card** — colour-coded result (Severe / Moderate / Mild / Minimal) with a clinical action recommendation
- **AI vs. Self-Report contrast panel** — compares PHQ-9/GAD-7 severity against the latest BERT spontaneous text prediction
- **Crisis escalation** — PHQ-9 Question 9 (self-harm thoughts) and total score ≥ 10 both trigger the emergency crisis banner
- **Encrypted PDF screener report** — exports a full summary including per-question answers, scores, severity, and both PHQ-9 + GAD-7 results in one document

---

### 📷 4. Scanned Document OCR & Vision Analysis
- **OpenCV perspective correction** — adaptive warp, binarization, and deskew pipeline for handwritten or printed scanned pages
- **Groq Llama 4 Scout 17B** multimodal vision inference for three analysis modes:
  - `🌐 General Visual Tone` — general document tone and emotional context
  - `📱 Social Media Sentiment` — social media screenshot emotional analysis
  - `📊 Scientific Chart Analysis` — interpretation of clinical charts or graphs
- **Rich empty/drop state** — illustrated drop zone with hover scale animation and descriptive copy; no bare upload box
- **Remove button** on uploaded images — clear and re-upload without page reload
- **Celery async OCR dispatch** — non-blocking task with live `Processing via Celery…` status badge
- **Toast success/failure notification** on scan completion
- **Encrypted PDF export** of scan results

---

### 🔐 5. Zero-Knowledge Security Architecture
- **Client-side only encryption**: `encryptText()` and `decryptText()` run entirely in the browser using the Web Crypto API
- **Salt + IV per-entry randomness**: each journal entry gets a unique 16-byte PBKDF2 salt and 12-byte AES-GCM IV
- **LocalStorage scoped per patient**: `mindscan_history_{patientId}` — multiple users on the same device have isolated histories
- **Decrypted text never persisted**: `saveHistory()` strips `decryptedText` before writing to localStorage
- **Server-side ephemeral only**: FastAPI processes text in-memory; no raw text, images, or encryption keys are written to disk

---

### 🌐 6. Multilingual Support (8 Indian Languages)
| Language | Script | ISO Code |
|----------|--------|----------|
| Hindi | देवनागरी | `hi` |
| Kannada | ಕನ್ನಡ | `kn` |
| Tamil | தமிழ் | `ta` |
| Telugu | తెలుగు | `te` |
| Malayalam | മലയാളം | `ml` |
| Marathi | मराठी | `mr` |
| Bengali | বাংলা | `bn` |
| English | Latin | `en` / `auto` |

Uses **Helsinki-NLP `opus-mt-mul-en`** MarianMT AutoModels directly (bypasses legacy pipeline wrappers; compatible with `transformers` v5.0+). Language can be manually selected or auto-detected.

---

### 📄 7. Encrypted PDF Clinical Report Generation
All three report types (Journal, Screener, Document Scan) are generated server-side via `reportlab` and returned as a binary blob download. Reports include:
- Patient ID, generation timestamp, assessment type
- **Colour-coded status box**: 🔴 Severe → 🟡 Moderate/Mild → 🟢 Minimal
- Full processed text, confidence scores, and clinical disclaimers
- Watermarked as "AI-Assisted Screening — Not a Medical Diagnosis"

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         React 19 + Zustand                         │
│  ┌──────────────┐  ┌──────────────────┐  ┌───────────────────────┐│
│  │Journal Tab   │  │Clinical Screeners│  │Document Scan (OCR)    ││
│  │AES-GCM-256   │  │PHQ-9 · GAD-7     │  │OpenCV + Llama 4 Scout ││
│  │ZK Encryption │  │PDF Export        │  │Vision Analysis        ││
│  └──────┬───────┘  └────────┬─────────┘  └──────────┬────────────┘│
└─────────┼───────────────────┼────────────────────────┼────────────┘
          │    HTTP / REST     │                        │
          ▼                   ▼                        ▼
┌────────────────────────────────────────────────────────────────────┐
│                    FastAPI (uvicorn, port 8000)                     │
│   /api/analyze  ·  /api/scan  ·  /api/tasks/{id}  ·  /api/report   │
└─────────────────────────────┬──────────────────────────────────────┘
                              │  Task Dispatch
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│             Celery Worker  ←→  Redis Broker (port 6379)             │
│  ┌─────────────────────┐    ┌──────────────────────────────────┐   │
│  │ analyze_text task   │    │ scan_document task               │   │
│  │  • Helsinki-NLP NMT │    │  • OpenCV deskew + binarize      │   │
│  │  • Calibration guard│    │  • Groq Llama 4 Scout vision     │   │
│  │  • BERT INT8 ONNX   │    │  • Ephemeral file shred          │   │
│  └─────────────────────┘    └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Local Setup & Quick Start

### 📋 Prerequisites
1. **Python 3.12** — [Download](https://www.python.org/downloads/release/python-3120/) (add to PATH)
2. **Node.js 20+** — [Download](https://nodejs.org/en/download) (for the React frontend)
3. **Redis** — [Download for Windows](https://github.com/tporadowski/redis/releases) or run via Docker: `docker run -d -p 6379:6379 redis`
4. **Groq API Key** (free) — [console.groq.com](https://console.groq.com) (required for OCR/Vision tab only)

---

### Step 1 — Clone the Repository
```bash
git clone https://github.com/AryaPartha/MindScan.git
cd MindScan/mental_health_bert
```

---

### Step 2 — Configure Environment Variables
Create `.env` in `mental_health_bert/`:
```ini
# MindScan Backend Environment
GROQ_API_KEY=your_groq_api_key_here
```

---

### Step 3 — Download Fine-Tuned Model Weights
The BERT model weights are stored on Google Drive (too large for Git).
1. Download `bert_mental_health/` from the [Google Drive Folder](https://drive.google.com/drive/folders/1PR_CEWSri132m3GWNtxDK2MnOZ8kHFSy?usp=drive_link)
2. Place it at:
```
mental_health_bert/models/bert_mental_health/
  ├── config.json
  ├── model.safetensors
  ├── tokenizer_config.json
  └── tokenizer.json
```

---

### Step 4 — Install Backend Dependencies
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

---

### Step 5 — Install Frontend Dependencies
```bash
cd client
npm install
cd ..
```

---

### Step 6 — Launch All Services

Open **three separate terminals** in `mental_health_bert/`:

**Terminal 1 — FastAPI Backend**
```bash
venv\Scripts\uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 — Celery Worker**
```bash
venv\Scripts\celery -A app.tasks.worker.celery_app worker --loglevel=info -P solo
```

**Terminal 3 — React Frontend**
```bash
cd client
npm run dev
```

Open **[http://localhost:5173](http://localhost:5173)** in your browser.

---

## 🧪 Running Automated Unit Tests

```bash
# Activate venv first
venv\Scripts\activate

# Run full test suite
$env:PYTHONPATH="."; venv\Scripts\pytest tests -v
```

**Expected output — 22/22 tests passing:**
```
tests/test_bert.py            ······   6 passed  (singleton, inference, calibration)
tests/test_phase3.py          ·····    5 passed  (PDF, OCR pipeline, ephemeral shred)
tests/test_translation.py     ······   6 passed  (Helsinki-NLP, LRU cache, fallback)
tests/test_vision.py          ·····    5 passed  (Groq vision, RGBA→RGB, API errors)

========================= 22 passed in ~11s =========================
```

> All heavy model loaders (BERT, MarianMT, Groq) are globally mocked in `tests/conftest.py` — tests run without GPU, network access, or model downloads.

---

## 🔒 CI/CD Safety: Three-Tier ID Generation

To ensure zero crashes in headless GitHub Actions runners and JSDOM test environments, all ephemeral frontend entry IDs are generated via `client/src/utils/generateId.js` — a crash-safe three-tier fallback:

| Tier | Method | Environment |
|------|--------|-------------|
| **1** | `window.crypto.randomUUID()` | Chrome 92+, Firefox 95+, Node 19+ |
| **2a** | `window.crypto.getRandomValues()` + RFC-4122 assembly | Chrome 11+, Safari 7+, IE 11+ |
| **2b** | `globalThis.crypto.randomUUID()` | Vitest Node mode |
| **3** | `Math.random()` UUID-shape | Headless JSDOM, legacy CI runners |

The function is fully wrapped in `try/catch` — it **never throws** and **always returns a string**.

---

## 📊 Model Performance

The BERT classifier was fine-tuned on the Kaggle Reddit Mental Health dataset (~20,000+ records):

| Class | Index | Description |
|-------|-------|-------------|
| **Depression** | 0 | Major depressive symptoms, bipolar indicators |
| **Normal** | 1 | Neutral thoughts, general mental wellness |
| **Anxiety** | 2 | Panic attacks, social phobias, stress indicators |

**Clinical Sentiment Calibration Guard**: Rule-based overrides map pure wellness sentences (e.g. `"feeling good today"`) to `normal` at 90% confidence, preventing false-positive depression alarms from BERT's domain bias.

---

## 🗂️ Project Structure

```
mental_health_bert/
├── app/
│   ├── api/endpoints/
│   │   └── analysis.py         # FastAPI route handlers
│   ├── services/
│   │   ├── bert_classifier.py  # ONNX INT8 BERT inference singleton
│   │   ├── translation.py      # Helsinki-NLP MarianMT service
│   │   ├── vision.py           # Groq Llama 4 Scout vision service
│   │   ├── image_processor.py  # OpenCV deskew + binarization
│   │   └── pdf_generator.py    # ReportLab encrypted PDF generator
│   ├── tasks/
│   │   └── worker.py           # Celery task definitions
│   └── main.py                 # FastAPI app + CORS configuration
├── client/                     # React 19 + Vite frontend
│   ├── src/
│   │   ├── App.jsx             # Main application (tabs, state, rendering)
│   │   ├── index.css           # Glassmorphism design system + animations
│   │   ├── services/
│   │   │   └── cryptoService.js    # AES-GCM-256 encrypt/decrypt
│   │   ├── store/
│   │   │   └── useJournalStore.js  # Zustand state + polling + persistence
│   │   └── utils/
│   │       └── generateId.js   # Three-tier CI-safe UUID generator
│   └── index.html
├── models/
│   └── bert_mental_health/     # Fine-tuned weights (download separately)
├── tests/
│   ├── conftest.py             # Global pytest mocks (BERT, NMT, Groq)
│   ├── test_bert.py
│   ├── test_phase3.py
│   ├── test_translation.py
│   └── test_vision.py
└── requirements.txt
```

---

## 📜 Disclaimer & Scientific Disclosures

MindScan is a **research prototype** developed for academic purposes at the School of Engineering, University of Mysore (AI&DS Department).

- ❌ **NOT** a certified diagnostic medical device
- ❌ **NOT** a replacement for professional clinical counseling or psychiatric guidance
- ✅ Developed solely for educational and research awareness

**If you or someone you know is in crisis, please seek immediate help. Confidential support is available 24/7:**
- 🇮🇳 **India:** **14416** (Tele-MANAS) · **9152987821** (iCall) · **9820466726** (AASRA)
- 🇺🇸 **US/Canada:** Call or Text **988** (Suicide & Crisis Lifeline)

---

<div align="center">
  <sub>Mysore University — AI&DS Dept · Built with ❤️ for mental health awareness</sub>
</div>
