"""
src/ui/tabs/text_tab.py
Tab 1 — Text Analysis rendering logic.
Handles: input validation, length warning, translation (with explicit failure stop),
BERT inference, result cards, confidence bars, and session history.
"""
import streamlit as st

from src.services.bert import MAX_BERT_WORDS
from src.services.translation import TranslationService
from src.services.bert import BERTClassifierService


LABELS = ["anxiety", "depression", "normal"]

ICONS_MAP = {
    "normal":     ("✅", "NO SIGNIFICANT RISK INDICATORS DETECTED", "result-normal",     "#34d399"),
    "anxiety":    ("⚠️", "ANXIETY PATTERNS DETECTED",               "result-anxiety",    "#fbbf24"),
    "depression": ("🧠", "DEPRESSIVE PATTERNS DETECTED",            "result-depression", "#f87171"),
}

CONF_ICONS = {
    "anxiety":    ("⚠️", "#fbbf24", "linear-gradient(90deg, #f59e0b, #d97706)"),
    "depression": ("🧠", "#f87171", "linear-gradient(90deg, #ef4444, #b91c1c)"),
    "normal":     ("✅", "#34d399", "linear-gradient(90deg, #10b981, #059669)"),
}

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "want to die", "end my life", "ending it all", 
    "self harm", "commit suicide", "no reason to live", "better off dead", 
    "want to end it", "cutting myself", "overdose", "hopelessness", "hanging myself",
    "atmahatya", "mar jana"
]

def detect_language_from_text(text: str) -> str | None:
    """Detects Indian languages based on Unicode script blocks.
    Returns the language code ('kn', 'hi', 'ta', 'te', 'ml', 'bn', 'mr') or None if English.
    """
    for char in text:
        val = ord(char)
        # Devanagari (Hindi / Marathi): U+0900 - U+097F
        if 0x0900 <= val <= 0x097F:
            return "mr" if st.session_state.get("ui_lang") == "mr" else "hi"
        # Bengali: U+0980 - U+09FF
        elif 0x0980 <= val <= 0x09FF:
            return "bn"
        # Tamil: U+0B80 - U+0BFF
        elif 0x0B80 <= val <= 0x0BFF:
            return "ta"
        # Telugu: U+0C00 - U+0C7F
        elif 0x0C00 <= val <= 0x0C7F:
            return "te"
        # Kannada: U+0C80 - U+0CFF
        elif 0x0C80 <= val <= 0x0CFF:
            return "kn"
        # Malayalam: U+0D00 - U+0D7F
        elif 0x0D00 <= val <= 0x0D7F:
            return "ml"
    return "en"

def _render_crisis_card() -> None:
    st.markdown("""
<div class="crisis-card">
    <div class="crisis-header">
        <span style="font-size:1.8rem;">🚨</span>
        <h3 class="crisis-title">Please Know That You Are Not Alone</h3>
    </div>
    <p class="crisis-desc">
        It sounds like you might be going through a very difficult time. If you or someone you know is struggling 
        with thoughts of self-harm or suicide, please reach out for help immediately. Compassionate, professional, 
        and confidential support is available 24/7.
    </p>
    <div class="crisis-btn-container">
        <a href="tel:988" class="crisis-btn" target="_blank">📞 Call 988 (India / US Helpline)</a>
        <a href="https://www.aasra.info" class="crisis-btn-sec" target="_blank">🌐 AASRA Helpline (24x7)</a>
        <a href="https://www.vandrevalafoundation.com" class="crisis-btn-sec" target="_blank">🌐 Vandrevala Foundation</a>
        <a href="https://findahelpline.com" class="crisis-btn-sec" target="_blank">🌍 International Crisis Contacts</a>
    </div>
</div>
""", unsafe_allow_html=True)


def render(S: dict, lang: str, is_multilingual: bool,
           translator_svc: TranslationService,
           bert_svc: BERTClassifierService) -> None:
    """Render the full Text Analysis tab."""

    st.markdown(f"### {S['tab_text']}")
    st.caption(S["model_info"])

    text_input = st.text_area(
        S["input_label"],
        height=150,
        placeholder=S["input_placeholder"],
        key="text_analysis_input",
    )

    analyze_clicked = st.button(
        S["btn_analyze"], type="primary", use_container_width=True, key="btn_analyze_text"
    )

    if analyze_clicked:
        raw = text_input.strip()
        if not raw:
            st.warning("⚠️ Please enter some text to analyze." if lang == "en" else S["input_label"])
            return

        # ── Script Auto-Detection ──
        detected_lang = detect_language_from_text(raw)
        if detected_lang and detected_lang != st.session_state.ui_lang:
            st.session_state.ui_lang = detected_lang
            st.toast(f"🌐 Optimizing translation for native input...", icon="🔄")
            st.rerun()

        # ── Input length guard ──
        word_count = len(raw.split())
        if word_count > MAX_BERT_WORDS:
            st.warning(
                S["warn_long_input"].format(words=word_count, max=MAX_BERT_WORDS)
            )

        text_to_analyze = raw

        # ── Translation ──
        if is_multilingual:
            with st.spinner(S["spinner_translate"]):
                try:
                    translated = translator_svc.translate(raw, source_lang=lang)
                    if translated:
                        st.markdown(
                            f'<div style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);'
                            f'border-radius:10px;padding:0.8rem 1rem;margin:0.5rem 0;font-size:0.9rem;">'
                            f'<span class="offline-badge">{S["offline_badge"]}</span> '
                            f'<strong style="color:#e5e7eb;">{translated}</strong></div>',
                            unsafe_allow_html=True,
                        )
                        text_to_analyze = translated
                except RuntimeError as exc:
                    # ── FIX: Hard stop — do NOT pass untranslated text to BERT ──
                    st.error(S["err_translation"].format(error=str(exc)))
                    return

        # ── Crisis Keyword Scanning ──
        has_crisis_keyword = any(kw in raw.lower() or kw in text_to_analyze.lower() for kw in CRISIS_KEYWORDS)

        # ── BERT Inference ──
        with st.spinner(S["spinner_bert"]):
            try:
                label_pred, probs = bert_svc.predict(text_to_analyze)
            except RuntimeError as exc:
                st.error(f"❌ BERT inference failed: {exc}")
                return

        # Trigger crisis card if high-risk keywords OR depression probability >= 80%
        is_crisis = has_crisis_keyword or (label_pred == "depression" and probs[1] >= 0.80)

        _render_result(S, lang, raw, label_pred, probs, is_crisis=is_crisis)

        # ── Session History ──
        if "result_history" not in st.session_state:
            st.session_state.result_history = []
        st.session_state.result_history.insert(0, {
            "text":  raw[:60] + ("..." if len(raw) > 60 else ""),
            "label": label_pred,
            "conf":  f"{float(probs.max()) * 100:.1f}%",
            "lang":  lang.upper(),
        })
        # Keep only last 10
        st.session_state.result_history = st.session_state.result_history[:10]

    # ── History Panel ──
    _render_history(S)


# ──────────────────────────────────────────────────────────────────────────────

def _render_result(S: dict, lang: str, raw: str, label_pred: str, probs, is_crisis: bool = False) -> None:
    st.divider()
    if is_crisis:
        _render_crisis_card()
        
    st.markdown(f"#### {S['result_title']}")

    advice_map = {
        "normal":     S["advice_normal"],
        "anxiety":    S["advice_anxiety"],
        "depression": S["advice_depression"],
    }
    icon, label_text, css_class, color = ICONS_MAP[label_pred]

    st.markdown(f"""
<div class="{css_class}">
    <p class="result-label" style="color:{color};">{icon} {label_text}</p>
    <p class="result-advice" style="color:{color}80;">{advice_map[label_pred]}</p>
</div>""", unsafe_allow_html=True)

    # Confidence bars
    st.markdown(f"#### {S['confidence_title']}")
    conf_html = ""
    for lbl, prob in zip(LABELS, probs):
        pct = float(prob) * 100
        ci, cc, cg = CONF_ICONS[lbl]
        conf_html += f"""
<div class="conf-row">
    <span class="conf-label">{ci} {lbl.capitalize()}</span>
    <div class="conf-bar-bg">
        <div class="conf-bar-fill" style="width:{pct:.1f}%;background:{cg};"></div>
    </div>
    <span class="conf-pct" style="color:{cc};">{pct:.1f}%</span>
</div>"""
    st.markdown(conf_html, unsafe_allow_html=True)

    # Calibration note
    st.markdown(
        '<p style="color:#4b5563;font-size:0.76rem;margin-top:0.6rem;">'
        '💡 Confidence scores are relative model outputs, not clinical probabilities.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#4b5563;font-size:0.78rem;margin-top:0.4rem;">{S["disclaimer"]}</p>',
        unsafe_allow_html=True,
    )


def _render_history(S: dict) -> None:
    history = st.session_state.get("result_history", [])
    if not history:
        return

    with st.expander(S["history_title"], expanded=False):
        for i, entry in enumerate(history):
            icon = ICONS_MAP[entry["label"]][0]
            color = ICONS_MAP[entry["label"]][3]
            st.markdown(
                f'<div style="padding:0.4rem 0.8rem;border-left:3px solid {color};margin:0.3rem 0;">'
                f'<span style="color:{color};font-weight:700;">{icon} {entry["label"].upper()}</span> '
                f'<span style="color:#9ca3af;font-size:0.82rem;"> · {entry["conf"]} · [{entry["lang"]}]</span><br>'
                f'<span style="color:#6b7280;font-size:0.8rem;">{entry["text"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
