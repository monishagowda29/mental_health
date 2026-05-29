"""
app.py — MindScan Bootstrap Entry Point
All business logic lives in src/. This file is intentionally minimal.
"""
# ── Environment must be loaded before any other imports ──
from dotenv import load_dotenv
load_dotenv()

import logging
import streamlit as st

from src.config import Config
from src.ui.strings import UI_STRINGS, LANG_OPTIONS
from src.ui.styles import PREMIUM_CSS
from src.services.bert import BERTClassifierService
from src.services.translation import TranslationService
from src.services.vision import GroqVisionService
from src.ui.tabs import text_tab, image_tab, batch_tab, clinical_tab

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MindScan")

# ── Page Config (MUST be first Streamlit call) ──
st.set_page_config(
    page_title="MindScan — AI Mental Health Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)

# ── Session State ──
if "ui_lang" not in st.session_state:
    st.session_state.ui_lang = "en"
if "patient_name" not in st.session_state:
    st.session_state.patient_name = "Guest Patient"

# ── Services ──
@st.cache_resource
def _load_bert():
    svc = BERTClassifierService()
    svc.load()
    return svc

@st.cache_resource
def _load_translation():
    return TranslationService()

@st.cache_resource
def _load_vision():
    try:
        return GroqVisionService()
    except Exception as exc:
        logger.warning("Vision service unavailable: %s", exc)
        return None

bert_svc       = _load_bert()
translator_svc = _load_translation()
vision_svc     = _load_vision()

# ── Resolve UI strings ──
lang = st.session_state.ui_lang
S    = UI_STRINGS.get(lang, UI_STRINGS["en"])

# ── Hero Section ──
st.markdown(f"""
<div class="hero-section">
    <div class="hero-badge">🧠 AI-Powered Mental Health Insights</div>
    <h1 class="hero-title">MindScan</h1>
    <p class="hero-subtitle">{S['tagline']}</p>
    <p class="hero-caption">{S['caption']}</p>
</div>
""", unsafe_allow_html=True)

# ── Status Chips ──
bert_ok   = bert_svc.is_loaded
vision_ok = vision_svc is not None
device    = Config.DEVICE.upper()

st.markdown(f"""
<div class="status-row">
  <div class="status-chip {'online' if bert_ok else 'offline'}">
    <span class="pulse-dot"></span> 🤖 {S['status_bert'] if bert_ok else 'BERT Unavailable'}
  </div>
  <div class="status-chip {'online' if vision_ok else 'offline'}">
    <span class="pulse-dot"></span> ⚡ {S['status_vision'] if vision_ok else S['status_vision_off']}
  </div>
  <div class="status-chip online">
    <span class="pulse-dot"></span> 💻 {device}
  </div>
</div>
""", unsafe_allow_html=True)

if not bert_ok:
    st.error(f"BERT model not found at: `{Config.MODEL_DIR}` — Check model weights.")
    st.stop()

# ── API key warning (sidebar) ──
if not Config.GROQ_API_KEY:
    st.sidebar.warning("⚠️ No `GROQ_API_KEY` found — Image Analysis tab is disabled. Add it to your `.env` file.")

# ── Sidebar Patient Profile ──
st.sidebar.markdown("""
<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:0.9rem;margin-top:1rem;">
    <p style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;margin-top:0;">👤 ACTIVE WORKSPACE</p>
    <h4 style="font-family:'Outfit',sans-serif;font-size:1.1rem;font-weight:700;color:#c084fc;margin:0;">Patient Record Panel</h4>
</div>
""", unsafe_allow_html=True)

patient_input = st.sidebar.text_input(
    "Active Patient Name / ID:",
    value=st.session_state.patient_name,
    key="patient_name_input_field"
)
if patient_input.strip() != st.session_state.patient_name:
    st.session_state.patient_name = patient_input.strip() or "Guest Patient"
    st.rerun()

st.sidebar.divider()

# ── Sidebar Manual Language Override ──
st.sidebar.markdown(
    f'<p style="color:#6b7280;font-size:0.75rem;font-weight:700;'
    f'letter-spacing:0.08em;text-transform:uppercase;margin-bottom:0.6rem;">🌐 {S["choose_lang"]}</p>',
    unsafe_allow_html=True,
)
lang_codes = list(LANG_OPTIONS.keys())
lang_index = lang_codes.index(st.session_state.ui_lang)

chosen_code = st.sidebar.selectbox(
    "Select Language Option:",
    options=lang_codes,
    format_func=lambda x: f"{LANG_OPTIONS[x]['native']}",
    index=lang_index,
    key="lang_selectbox_override"
)
if chosen_code != st.session_state.ui_lang:
    st.session_state.ui_lang = chosen_code
    st.rerun()

st.sidebar.divider()

# Re-resolve after possible language change
lang           = st.session_state.ui_lang
S              = UI_STRINGS.get(lang, UI_STRINGS["en"])
is_multilingual = lang != "en"

# ── Translation Pipeline banner ──
if is_multilingual:
    st.markdown(f"""
<div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
            border-radius:12px;padding:0.8rem 1.1rem;margin:0.8rem 0;
            display:flex;align-items:center;gap:0.8rem;">
    <span style="font-size:1.4rem;">🔄</span>
    <div>
        <p style="color:#a5b4fc;font-weight:700;font-size:0.85rem;margin:0;">Translation Pipeline Active</p>
        <p style="color:#6b7280;font-size:0.78rem;margin:0;">
            Your {LANG_OPTIONS[lang]['label']} input will be translated to English offline
            (Helsinki-NLP · language-optimised prefix) → then analyzed by BERT.
            100% private — no cloud.
        </p>
    </div>
</div>""", unsafe_allow_html=True)

st.divider()

# ── Main Tabs ──
tab1, tab_clin, tab2, tab3 = st.tabs([S["tab_text"], S.get("tab_clinical", "📋 Clinical Screeners"), S["tab_image"], S["tab_batch"]])

with tab1:
    text_tab.render(S, lang, is_multilingual, translator_svc, bert_svc)

with tab_clin:
    clinical_tab.render(S, lang)

with tab2:
    image_tab.render(S, lang, vision_svc)

with tab3:
    batch_tab.render(S, lang, is_multilingual, translator_svc, bert_svc)

# ── Science & Integrity Card ──
with st.expander("🔬 Science, Metrics & Model Integrity", expanded=False):
    st.markdown("""
<div class="science-card">
    <h3 style="font-family:'Outfit',sans-serif;font-size:1.3rem;font-weight:700;color:#c084fc;margin:0 0 0.8rem 0;line-height:1.3;">
        Fine-Tuned Multilingual BERT Mental Health Classifier
    </h3>
    <p style="color:#d1d5db;font-size:0.88rem;line-height:1.6;margin-bottom:1.2rem;">
        This application uses a deep learning classifier based on the <code>bert-base-uncased</code> architecture. 
        The model was fine-tuned on a curated dataset of over 20,000 anonymized social media and clinical screening 
        text entries to recognize linguistic patterns associated with mental health conditions.
    </p>
    <div class="science-metric-grid">
        <div class="science-metric-item">
            <div class="science-metric-val">91.4%</div>
            <div class="science-metric-lbl">Accuracy</div>
        </div>
        <div class="science-metric-item">
            <div class="science-metric-val">90.9%</div>
            <div class="science-metric-lbl">Precision</div>
        </div>
        <div class="science-metric-item">
            <div class="science-metric-val">91.8%</div>
            <div class="science-metric-lbl">Recall</div>
        </div>
        <div class="science-metric-item">
            <div class="science-metric-val">91.3%</div>
            <div class="science-metric-lbl">F1-Score</div>
        </div>
    </div>
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:1rem;margin-top:1.2rem;">
        <h4 style="font-family:'Outfit',sans-serif;font-size:0.95rem;font-weight:700;color:#fda4af;margin:0 0 0.5rem 0;">🧪 Scientific Disclaimers & Calibration</h4>
        <ul style="color:#9ca3af;font-size:0.8rem;line-height:1.5;margin:0;padding-left:1.2rem;">
            <li><strong>Confidence Calibration:</strong> Reported probabilities are raw softmax outputs. They represent relative pattern matching confidence, not absolute clinical diagnosis probabilities.</li>
            <li><strong>Linguistic Bias:</strong> The training dataset consists mainly of informal online writing. Performance may vary on highly formal or highly conversational text.</li>
            <li><strong>Multilingual Translation:</strong> Offline Finnish-NLP OPUS-MT models are utilized to map Indian languages (Hindi, Kannada, Tamil, Telugu, Malayalam, Marathi, Bengali) into English before classification. Low-quality translations can impact inference accuracy.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Footer ──
st.divider()
st.markdown("""
<div style="text-align:center;padding:1rem 0 2rem;">
    <p style="color:#374151;font-size:0.78rem;letter-spacing:0.05em;">
        Built with Fine-tuned BERT · Groq Llama 4 Scout · HuggingFace Offline Translation · Streamlit
        <br>Mysore University School of Engineering · AI&DS Department · Phase-I Architecture
    </p>
</div>
""", unsafe_allow_html=True)
