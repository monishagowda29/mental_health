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
from src.services.history_manager import HistoryManager


LABELS = ["depression", "normal", "anxiety"]

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
        <a href="tel:988" class="crisis-btn" target="_blank">📞 Call 988 (Crisis Lifeline)</a>
        <a href="tel:9152987821" class="crisis-btn" target="_blank">📞 Call TISS iCall: 9152987821</a>
        <a href="tel:9820466726" class="crisis-btn-sec" target="_blank">📞 Call AASRA: 9820466726</a>
        <a href="tel:18602662345" class="crisis-btn-sec" target="_blank">📞 Call Vandrevala: 1860-2662-345</a>
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
        max_chars=2500,
        key="text_analysis_input",
    )

    # ── Live Character Countdown ──
    char_count = len(text_input)
    char_pct = min(100.0, (char_count / 2500.0) * 100.0)
    char_color = "text-gray-400"
    if char_count > 2000:
        char_color = "text-yellow-500 font-medium"
    if char_count >= 2500:
        char_color = "text-rose-500 font-bold"
        
    st.markdown(f"""
<div class="flex justify-between items-center px-1 mt-1 text-[11px] {char_color}">
    <span>Max limit: 2,500 characters</span>
    <span>{char_count:,} / 2,500 characters ({char_pct:.1f}%)</span>
</div>
<div class="w-full bg-gray-800 bg-opacity-40 rounded-full h-1 mt-1 overflow-hidden">
    <div class="h-1 bg-gradient-to-r from-indigo-500 via-purple-500 to-rose-500" style="width: {char_pct}%;"></div>
</div>
<div style="margin-bottom: 1.2rem;"></div>
""", unsafe_allow_html=True)

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
        is_crisis = has_crisis_keyword or (label_pred == "depression" and probs[0] >= 0.80)

        _render_result(S, lang, raw, label_pred, probs, is_crisis=is_crisis)

        # ── Persistent Storage Scoping ──
        conf_str = f"{float(probs.max()) * 100:.1f}%"
        HistoryManager.save_record(
            patient_name=st.session_state.patient_name,
            text=raw,
            label=label_pred,
            confidence=conf_str,
            lang=lang.upper()
        )

        # ── Session Cache ──
        if "result_history" not in st.session_state:
            st.session_state.result_history = []
        st.session_state.result_history.insert(0, {
            "text":  raw[:60] + ("..." if len(raw) > 60 else ""),
            "label": label_pred,
            "conf":  conf_str,
            "lang":  lang.upper(),
        })
        # Keep only last 10
        st.session_state.result_history = st.session_state.result_history[:10]

    # ── History Panel ──
    _render_history(S, st.session_state.patient_name)


# ──────────────────────────────────────────────────────────────────────────────

def _render_result(S: dict, lang: str, raw: str, label_pred: str, probs, is_crisis: bool = False) -> None:
    st.divider()
    if is_crisis:
        _render_crisis_card()
        
    # ── Continuous Severity Spectrum Meter ──
    p_dep = float(probs[0])
    p_norm = float(probs[1])
    p_anx = float(probs[2])
    
    # Calculate position mapping: Normal -> 15%, Anxiety -> 50%, Depression -> 85%
    position = (p_norm * 15.0) + (p_anx * 50.0) + (p_dep * 85.0)
    
    if label_pred == "normal":
        color_class = "text-emerald-400"
        status_text = "🟢 Stable Spectrum / Low Distress"
        dot_color = "bg-emerald-400"
        advice_text = S["advice_normal"]
    elif label_pred == "anxiety":
        color_class = "text-amber-400"
        status_text = "🟡 Moderate Distress / Anxiety Spectrum"
        dot_color = "bg-amber-400"
        advice_text = S["advice_anxiety"]
    else:
        color_class = "text-rose-400"
        status_text = "🔴 High Distress / Depressive Spectrum"
        dot_color = "bg-rose-500"
        advice_text = S["advice_depression"]

    spectrum_html = f"""
<div class="bg-gray-900 bg-opacity-65 border border-gray-800 border-opacity-70 rounded-2xl p-5 shadow-2xl backdrop-filter backdrop-blur-xl my-6">
    <div class="flex items-center justify-between mb-4">
        <span class="text-gray-400 text-xs font-semibold uppercase tracking-wider">Clinical Severity Spectrum Meter</span>
        <span class="text-xs sm:text-sm font-extrabold {color_class} tracking-wide">{status_text}</span>
    </div>
    
    <!-- Custom HSL Spectrum Bar -->
    <div class="relative w-full h-3 bg-gradient-to-r from-emerald-500 via-amber-400 to-rose-600 rounded-full my-6">
        <!-- Glowing Sliding Pointer Pin -->
        <div class="absolute w-5 h-5 bg-white border-2 border-gray-900 rounded-full flex items-center justify-center transform transition-all duration-700 shadow-xl" 
             style="left: {position}%; top: -4px; margin-left: -10px; box-shadow: 0 0 12px rgba(255, 255, 255, 0.8), 0 4px 6px rgba(0,0,0,0.4);">
            <div class="w-1.5 h-1.5 {dot_color} rounded-full animate-ping opacity-75"></div>
            <div class="w-1.5 h-1.5 {dot_color} rounded-full absolute"></div>
        </div>
    </div>
    
    <!-- Legend Markers below -->
    <div class="flex justify-between text-[9px] text-gray-500 font-bold uppercase tracking-wider px-1">
        <span class="text-emerald-500/80">🟢 STABLE (LOW)</span>
        <span class="text-amber-500/80">🟡 ANXIOUS (MID)</span>
        <span class="text-rose-500/80">🔴 DEPRESSIVE (HIGH)</span>
    </div>
    
    <!-- Detailed advice and confidence metrics -->
    <div class="mt-4 leading-relaxed bg-gray-950 bg-opacity-40 border border-gray-800 border-opacity-60 rounded-xl p-4 space-y-3">
        <p class="text-xs text-gray-300 font-medium leading-relaxed">
            🔍 <strong>Clinical Context:</strong> {advice_text}
        </p>
        <div class="grid grid-cols-3 gap-2 text-center pt-2 border-t border-gray-800 border-opacity-50">
            <div class="p-1">
                <p class="text-[9px] text-gray-500 font-semibold uppercase">Normal</p>
                <p class="text-xs font-bold text-emerald-400">{p_norm * 100:.1f}%</p>
            </div>
            <div class="p-1 border-x border-gray-800 border-opacity-50">
                <p class="text-[9px] text-gray-500 font-semibold uppercase">Anxiety</p>
                <p class="text-xs font-bold text-amber-400">{p_anx * 100:.1f}%</p>
            </div>
            <div class="p-1">
                <p class="text-[9px] text-gray-500 font-semibold uppercase">Depression</p>
                <p class="text-xs font-bold text-rose-400">{p_dep * 100:.1f}%</p>
            </div>
        </div>
    </div>
</div>
"""
    st.markdown(spectrum_html, unsafe_allow_html=True)

    # Calibration & Disclaimer
    st.markdown(
        '<p style="color:#4b5563;font-size:0.76rem;margin-top:0.6rem;margin-bottom:0;">'
        '💡 Confidence scores are relative model outputs, not clinical probabilities.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#4b5563;font-size:0.78rem;margin-top:0.4rem;margin-bottom:0.8rem;">{S["disclaimer"]}</p>',
        unsafe_allow_html=True,
    )


def _render_history(S: dict, patient_name: str) -> None:
    from src.services.history_manager import HistoryManager
    import pandas as pd
    from datetime import datetime

    records = HistoryManager.get_history(patient_name, limit=20)
    
    st.divider()
    st.markdown(f"### 📈 Personal Wellbeing Tracker")
    st.caption(f"Scoped history logging panel for patient: **{patient_name}** (100% Private · Stored Offline)")

    if not records:
        st.info("No persistent history logs found for this patient ID yet. Run your first analysis to see trend logs.")
        return

    # Extrapolate statistics
    total = len(records)
    dep_count = sum(1 for r in records if r["label"] == "depression")
    anx_count = sum(1 for r in records if r["label"] == "anxiety")
    norm_count = sum(1 for r in records if r["label"] == "normal")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assessments", total)
    c2.metric("🟢 Low Risk Score", f"{(norm_count/total)*100:.0f}%")
    c3.metric("🟡 Anxiety Patterns", f"{(anx_count/total)*100:.0f}%")
    c4.metric("🔴 Depressive Patterns", f"{(dep_count/total)*100:.0f}%")

    # Draw Wellbeing Trend Line Chart using premium Matplotlib & Seaborn
    st.markdown("**Linguistic Risk Severity Trend (Last 10 Checks)**")
    
    chart_data = []
    # Reverse to show chronological order left-to-right
    for r in reversed(records[:10]):
        score = 2 if r["label"] == "depression" else 1 if r["label"] == "anxiety" else 0
        try:
            dt = datetime.fromisoformat(r["timestamp"]).strftime("%m/%d\n%H:%M")
        except Exception:
            dt = "Unknown"
        chart_data.append({"Time": dt, "Risk Severity": score})
    
    df = pd.DataFrame(chart_data)
    if not df.empty:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Style setup for dark mode matching PREMIUM_CSS
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Clear color maps
        # Normal (0) -> Green, Anxiety (1) -> Orange/Amber, Depression (2) -> Crimson
        severity_colors = {0: "#10b981", 1: "#fbbf24", 2: "#f43f5e"}
        
        # Plot continuous connect line with premium indigo/violet gradient theme
        sns.lineplot(
            data=df,
            x="Time",
            y="Risk Severity",
            color="#8b5cf6", # Soft purple connection line
            alpha=0.45,
            linewidth=3,
            ax=ax,
            marker="",
            legend=False
        )
        
        # Plot custom colored node markers at each check-point
        for i, row in df.iterrows():
            val = row["Risk Severity"]
            col = severity_colors.get(val, "#8b5cf6")
            ax.plot(
                row["Time"],
                val,
                marker="o",
                color=col,
                markersize=9,
                markeredgecolor='#060810',
                markeredgewidth=2.2,
                zorder=5
            )
            
        # Customize y-ticks to match categorical severity
        ax.set_ylim(-0.4, 2.4)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["🟢 Low Risk", "🟡 Anxiety", "🔴 Depressive"], fontsize=8.5, fontweight='bold', color="#d1d5db")
        
        # Custom grid line styling
        ax.grid(True, linestyle="--", alpha=0.08, color="#ffffff")
        
        # Spines / Borders styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#333333")
        ax.spines['bottom'].set_color("#333333")
        
        # Labeling styling
        ax.set_xlabel("", fontsize=8)
        ax.set_ylabel("", fontsize=8)
        ax.tick_params(colors="#9ca3af", labelsize=8.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig) # Avoid memory leakage

    # Scoped Action controls
    col_x, col_y = st.columns(2)
    
    with col_x:
        csv_data = HistoryManager.export_csv(patient_name)
        st.download_button(
            label="⬇&nbsp; Export Patient History (CSV)",
            data=csv_data,
            file_name=f"mindscan_history_{patient_name.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="btn_export_scoped_history"
        )

    with col_y:
        if st.button("🗑&nbsp; Erase Patient Data Logs", type="secondary", use_container_width=True, key="btn_clear_scoped_history"):
            HistoryManager.clear_history(patient_name)
            # Clear volatile session state history cache
            if "result_history" in st.session_state:
                st.session_state.result_history = []
            st.toast(f"🧹 Scoped history records erased for {patient_name}!", icon="🗑️")
            st.rerun()

    # Detailed history expander
    with st.expander(f"📋 View Raw Records List ({total})", expanded=False):
        for entry in records:
            lbl = entry["label"]
            icon = ICONS_MAP[lbl][0]
            color = ICONS_MAP[lbl][3]
            try:
                dt_str = datetime.fromisoformat(entry["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                dt_str = "Unknown"
                
            st.markdown(
                f'<div style="padding:0.5rem 0.8rem;border-left:3px solid {color};margin:0.4rem 0;background:rgba(255,255,255,0.01);border-radius:0 8px 8px 0;">'
                f'<span style="color:{color};font-weight:700;">{icon} {lbl.upper()}</span> '
                f'<span style="color:#9ca3af;font-size:0.8rem;"> · {entry["confidence"]} · [{entry["lang"]}] · {dt_str}</span><br>'
                f'<span style="color:#6b7280;font-size:0.82rem;">{entry["text"][:150] + ("..." if len(entry["text"]) > 150 else "")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
