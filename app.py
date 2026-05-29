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
from src.ui.tabs import text_tab, image_tab, batch_tab

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

# ── Language Selector (2×4 grid) ──
st.markdown(
    f'<p style="text-align:center;color:#6b7280;font-size:0.72rem;font-weight:700;'
    f'letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.6rem;">🌐 {S["choose_lang"]}</p>',
    unsafe_allow_html=True,
)
lang_codes = list(LANG_OPTIONS.keys())
for row in [lang_codes[:4], lang_codes[4:]]:
    cols = st.columns(4)
    for idx, code in enumerate(row):
        info      = LANG_OPTIONS[code]
        is_active = (code == st.session_state.ui_lang)
        label     = f"✓  {info['native']}" if is_active else info["native"]
        if cols[idx].button(label, key=f"lang_btn_{code}",
                            use_container_width=True,
                            type="primary" if is_active else "secondary"):
            if code != st.session_state.ui_lang:
                st.session_state.ui_lang = code
                st.rerun()

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
tab1, tab2, tab3 = st.tabs([S["tab_text"], S["tab_image"], S["tab_batch"]])

with tab1:
    text_tab.render(S, lang, is_multilingual, translator_svc, bert_svc)

with tab2:
    image_tab.render(S, lang, vision_svc)

with tab3:
    batch_tab.render(S, lang, is_multilingual, translator_svc, bert_svc)

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
