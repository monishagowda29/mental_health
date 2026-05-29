"""
src/ui/styles.py
All premium CSS for MindScan in one place.
Import PREMIUM_CSS and inject with st.markdown(PREMIUM_CSS, unsafe_allow_html=True).
"""

PREMIUM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&family=Noto+Sans+Tamil:wght@400;600;700&family=Noto+Sans+Telugu:wght@400;600;700&family=Noto+Sans+Kannada:wght@400;600;700&family=Noto+Sans+Malayalam:wght@400;600;700&family=Noto+Sans+Bengali:wght@400;600;700&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"], .stMarkdown, p, span, label, input, button, select, textarea {
    font-family: 'Plus Jakarta Sans', 'Outfit', 'Noto Sans Devanagari', 'Noto Sans Tamil',
                 'Noto Sans Telugu', 'Noto Sans Kannada', 'Noto Sans Malayalam',
                 'Noto Sans Bengali', sans-serif !important;
}

/* ── Animated Background ── */
.stApp {
    background: #060810 !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99, 102, 241, 0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(236, 72, 153, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse 50% 50% at 50% 50%, rgba(16, 20, 40, 0.95) 0%, transparent 100%) !important;
}

/* ── Floating particles ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image:
        radial-gradient(circle 1px at 20% 30%, rgba(139,92,246,0.4) 0%, transparent 2px),
        radial-gradient(circle 1px at 70% 15%, rgba(236,72,153,0.3) 0%, transparent 2px),
        radial-gradient(circle 1px at 45% 70%, rgba(99,102,241,0.35) 0%, transparent 2px),
        radial-gradient(circle 1px at 85% 60%, rgba(139,92,246,0.25) 0%, transparent 2px),
        radial-gradient(circle 1px at 10% 80%, rgba(236,72,153,0.2) 0%, transparent 2px),
        radial-gradient(circle 2px at 60% 40%, rgba(99,102,241,0.2) 0%, transparent 3px),
        radial-gradient(circle 1px at 30% 55%, rgba(168,85,247,0.3) 0%, transparent 2px);
    pointer-events: none;
    z-index: 0;
    animation: drift 20s linear infinite;
}

@keyframes drift {
    0%   { transform: translate(0, 0); }
    50%  { transform: translate(10px, -15px); }
    100% { transform: translate(0, 0); }
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding-top: 2rem !important; max-width: 1200px; }

/* ── Hero ── */
.hero-section {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px;
    padding: 0.35rem 1.2rem;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    color: #a5b4fc;
    text-transform: uppercase;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}
.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 900 !important;
    font-size: 4rem !important;
    line-height: 1.1 !important;
    background: linear-gradient(135deg, #818cf8 0%, #a78bfa 40%, #f472b6 80%, #fb923c 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    letter-spacing: -0.04em !important;
    animation: titleGlow 4s ease-in-out infinite alternate;
}
@keyframes titleGlow {
    0%   { filter: brightness(1); }
    100% { filter: brightness(1.2) drop-shadow(0 0 20px rgba(168,85,247,0.4)); }
}
.hero-subtitle {
    font-size: 1.1rem !important;
    color: #9ca3af !important;
    font-weight: 400 !important;
    margin-top: 0.5rem !important;
}
.hero-caption {
    font-size: 0.8rem !important;
    color: #4b5563 !important;
    margin-top: 0.5rem;
    font-weight: 500;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* ── Status Chips ── */
.status-row {
    display: flex;
    gap: 0.75rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.5rem 0;
}
.status-chip {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(15, 17, 30, 0.8);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 50px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    font-weight: 600;
    backdrop-filter: blur(12px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.status-chip.online { color: #34d399; border-color: rgba(52,211,153,0.25); box-shadow: 0 0 15px rgba(52,211,153,0.1); }
.status-chip.offline { color: #f87171; border-color: rgba(248,113,113,0.25); }
.pulse-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #6b7280 !important;
    border-bottom: 2px solid transparent !important;
    padding: 0.9rem 1.8rem !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
button[data-baseweb="tab"]:hover {
    color: #c4b5fd !important;
    background: rgba(139,92,246,0.08) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #a78bfa !important;
    border-bottom-color: #a78bfa !important;
    background: rgba(139,92,246,0.1) !important;
}
div[data-testid="stTabs"] > div:first-child {
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 0.25rem !important;
}

/* ── Glass Cards ── */
.analysis-card {
    background: rgba(13, 15, 28, 0.75);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
}

/* ── CTA Gradient Buttons — excludes file uploader ── */
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #d946ef 100%) !important;
    color: white !important;
    border: none !important;
    padding: 0.8rem 2rem !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    position: relative !important;
    overflow: hidden !important;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 8px 30px rgba(139, 92, 246, 0.55), 0 0 40px rgba(139,92,246,0.2) !important;
}
div.stButton > button:active,
div[data-testid="stFormSubmitButton"] > button:active {
    transform: translateY(1px) scale(0.98) !important;
}

/* ── File Uploader — Streamlit 1.57 wraps Upload btn in div.stButton ── */
div[data-testid="stFileUploader"] div.stButton > button,
div[data-testid="stFileUploader"] div.stButton > button:hover,
div[data-testid="stFileUploader"] div.stButton > button:active,
div[data-testid="stFileUploaderDropzone"] button,
div[data-testid="stFileUploaderDropzone"] button:hover {
    all: unset !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(99,102,241,0.12) !important;
    border: 1px solid rgba(139,92,246,0.4) !important;
    border-radius: 8px !important;
    color: #a5b4fc !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 0.45rem 1.1rem !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: background 0.2s ease, border-color 0.2s ease !important;
    transform: none !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    line-height: 1.5 !important;
    overflow: hidden !important;
}
div[data-testid="stFileUploader"] div.stButton > button:hover,
div[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(99,102,241,0.22) !important;
    border-color: rgba(139,92,246,0.7) !important;
    box-shadow: 0 0 12px rgba(99,102,241,0.2) !important;
}
div[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploaderDropzone"] {
    background: rgba(12, 14, 24, 0.75) !important;
    border: 1.5px dashed rgba(139,92,246,0.3) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}
div[data-testid="stFileUploaderDropzone"]:hover,
section[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(139,92,246,0.65) !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ── Result Cards ── */
.result-normal {
    background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.08));
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    box-shadow: 0 0 30px rgba(16,185,129,0.08);
    animation: resultFade 0.5s ease-out;
}
.result-anxiety {
    background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.08));
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    box-shadow: 0 0 30px rgba(245,158,11,0.08);
    animation: resultFade 0.5s ease-out;
}
.result-depression {
    background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(185,28,28,0.08));
    border: 1px solid rgba(239,68,68,0.3);
    border-radius: 14px; padding: 1.2rem 1.5rem; margin: 1rem 0;
    box-shadow: 0 0 30px rgba(239,68,68,0.08);
    animation: resultFade 0.5s ease-out;
}
@keyframes resultFade {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
.result-label  { font-family: 'Outfit', sans-serif !important; font-size: 1.5rem !important; font-weight: 800 !important; margin: 0 !important; letter-spacing: -0.02em; }
.result-advice { font-size: 0.9rem !important; margin-top: 0.4rem !important; opacity: 0.85; }

/* ── Confidence Bars ── */
.conf-row   { display: flex; align-items: center; gap: 1rem; margin: 0.5rem 0; }
.conf-label { font-size: 0.85rem; font-weight: 600; min-width: 140px; color: #d1d5db; }
.conf-bar-bg   { flex: 1; height: 8px; background: rgba(255,255,255,0.07); border-radius: 10px; overflow: hidden; }
.conf-bar-fill { height: 100%; border-radius: 10px; transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); }
.conf-pct      { font-size: 0.85rem; font-weight: 700; min-width: 45px; text-align: right; color: #9ca3af; }

/* ── Text Areas ── */
textarea, input[type="text"] {
    background: rgba(12, 14, 24, 0.9) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    transition: border-color 0.3s, box-shadow 0.3s !important;
    font-size: 0.95rem !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: rgba(139,92,246,0.6) !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.15) !important;
}

/* ── Misc ── */
div[data-testid="stSelectbox"] > div > div { background: rgba(12,14,24,0.9) !important; border-color: rgba(255,255,255,0.08) !important; border-radius: 10px !important; color: #e5e7eb !important; }
section[data-testid="stSidebar"]           { background: #07080e !important; border-right: 1px solid rgba(255,255,255,0.04) !important; }
div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #6366f1, #a855f7) !important; border-radius: 10px !important; }
div[data-testid="metric-container"]        { background: rgba(15,17,30,0.7) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 14px !important; padding: 1.2rem !important; backdrop-filter: blur(10px) !important; transition: transform 0.2s ease !important; }
div[data-testid="metric-container"]:hover  { transform: translateY(-2px) !important; }
div[data-testid="stMetricValue"]           { font-family: 'Outfit', sans-serif !important; font-weight: 800 !important; font-size: 1.8rem !important; }
hr  { border-color: rgba(255,255,255,0.06) !important; margin: 1.5rem 0 !important; }
div[data-testid="stAlert"] { border-radius: 12px !important; backdrop-filter: blur(10px) !important; }
div[data-testid="stDataFrame"] { border-radius: 12px !important; overflow: hidden !important; border: 1px solid rgba(255,255,255,0.06) !important; }
img { border-radius: 12px !important; }

/* ── Offline badge ── */
.offline-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.3);
    border-radius: 50px; padding: 0.2rem 0.8rem;
    font-size: 0.78rem; font-weight: 600; color: #a5b4fc; margin-left: 0.5rem;
}

/* ── Analysis Mode Radio ── */
div[data-testid="stRadio"] > div { display: flex !important; flex-wrap: wrap !important; gap: 0.6rem !important; }
div[data-testid="stRadio"] label {
    background: rgba(15,17,30,0.8) !important; border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 50px !important; padding: 0.45rem 1.2rem !important;
    cursor: pointer !important; transition: all 0.25s ease !important;
    font-size: 0.88rem !important; font-weight: 500 !important; color: #9ca3af !important; white-space: nowrap !important;
}
div[data-testid="stRadio"] label:hover {
    background: rgba(139,92,246,0.15) !important; border-color: rgba(139,92,246,0.4) !important; color: #c4b5fd !important;
}
</style>
"""
