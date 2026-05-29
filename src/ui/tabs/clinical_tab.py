"""
src/ui/tabs/clinical_tab.py
Tab 4 — Interactive Clinical Screeners (PHQ-9 & GAD-7).
Includes official diagnostic questions, automatic sum scoring, severity mapping,
AI contrast comparison card, and clinical screening report downloader.
"""
import streamlit as st
import datetime

# ── Question Data ──
PHQ9_QUESTIONS = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
]

GAD7_QUESTIONS = [
    "Feeling nervous, anxious, or on edge",
    "Not being able to stop or control worrying",
    "Worrying too much about different things",
    "Trouble relaxing",
    "Being so restless that it is hard to sit still",
    "Becoming easily annoyed or irritable",
    "Feeling afraid as if something awful might happen"
]

FREQUENCY_CHOICES = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day"
}


def render(S: dict, lang: str) -> None:
    """Render the official clinical screeners tab."""
    st.markdown("### 📋 Clinically Validated Screeners")
    st.caption("Official self-assessment questionnaires. Compare self-reported patterns directly with the AI text model.")

    screener_mode = st.radio(
        "Select Questionnaire:",
        options=["PHQ-9 (Depression Screening)", "GAD-7 (Anxiety Screening)"],
        horizontal=True,
        key="clinical_screener_selection"
    )

    if "phq9_scores" not in st.session_state:
        st.session_state.phq9_scores = [0] * 9
    if "gad7_scores" not in st.session_state:
        st.session_state.gad7_scores = [0] * 7

    if "depression" in screener_mode.lower():
        _render_phq9(S)
    else:
        _render_gad7(S)

    # ── AI vs. Self-Report Contrast Card ──
    _render_contrast_card()


def _render_phq9(S: dict) -> None:
    st.markdown("#### Patient Health Questionnaire (PHQ-9)")
    st.caption("Over the last 2 weeks, how often have you been bothered by any of the following problems?")

    scores = []
    with st.form("phq9_form"):
        for idx, q in enumerate(PHQ9_QUESTIONS):
            st.markdown(f"**Q{idx + 1}: {q}**")
            choice = st.radio(
                "Choose frequency:",
                options=list(FREQUENCY_CHOICES.keys()),
                format_func=lambda x: FREQUENCY_CHOICES[x],
                horizontal=True,
                key=f"phq9_q_{idx}",
                label_visibility="collapsed"
            )
            scores.append(choice)
            st.divider()

        submit = st.form_submit_button("📊 Calculate PHQ-9 Score", use_container_width=True)

    if submit:
        st.session_state.phq9_scores = scores
        st.rerun()

    current_scores = st.session_state.phq9_scores
    total_score = sum(current_scores)
    
    # Severity assessment mapping
    if total_score <= 4:
        severity, color, desc = "Minimal or No Depression", "#34d399", "Symptoms suggest no clinical intervention is currently indicated. Maintain your healthy habits."
    elif total_score <= 9:
        severity, color, desc = "Mild Depressive Symptoms", "#fbbf24", "Symptoms suggest mild patterns. Watchful waiting, active self-care, or speaking with a counselor may be helpful."
    elif total_score <= 14:
        severity, color, desc = "Moderate Depressive Symptoms", "#f59e0b", "Symptoms indicate moderate patterns. Consider scheduling a clinical consultation with a healthcare provider."
    elif total_score <= 19:
        severity, color, desc = "Moderately Severe Depressive Symptoms", "#fb7185", "Symptoms are highly prominent. Seeking support from a clinical psychologist or psychiatrist is strongly recommended."
    else:
        severity, color, desc = "Severe Depressive Symptoms", "#f43f5e", "Critical symptom severity detected. Please reach out to a clinical mental health professional for a complete diagnosis and guidance."

    # Render result card
    st.markdown(f"""
<div style="background:rgba(15,17,30,0.7);border:1px solid {color}40;border-radius:16px;padding:1.5rem;margin:1.5rem 0;">
    <p style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">📊 Official Self-Report Rating</p>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
        <span style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;color:{color};">{severity}</span>
        <span style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;color:{color};">{total_score} <span style="font-size:0.9rem;color:#6b7280;font-weight:500;">/ 27</span></span>
    </div>
    <p style="color:#e5e7eb;font-size:0.88rem;line-height:1.5;margin-bottom:0.8rem;">{desc}</p>
    <div class="conf-bar-bg" style="height:10px;">
        <div class="conf-bar-fill" style="width:{(total_score / 27) * 100}%;background:linear-gradient(90deg, #10b981, {color});"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Offer downloadable report
    _render_report_downloader("PHQ-9 Depression Screener", total_score, severity, current_scores, PHQ9_QUESTIONS)


def _render_gad7(S: dict) -> None:
    st.markdown("#### Generalized Anxiety Disorder Assessment (GAD-7)")
    st.caption("Over the last 2 weeks, how often have you been bothered by any of the following problems?")

    scores = []
    with st.form("gad7_form"):
        for idx, q in enumerate(GAD7_QUESTIONS):
            st.markdown(f"**Q{idx + 1}: {q}**")
            choice = st.radio(
                "Choose frequency:",
                options=list(FREQUENCY_CHOICES.keys()),
                format_func=lambda x: FREQUENCY_CHOICES[x],
                horizontal=True,
                key=f"gad7_q_{idx}",
                label_visibility="collapsed"
            )
            scores.append(choice)
            st.divider()

        submit = st.form_submit_button("📊 Calculate GAD-7 Score", use_container_width=True)

    if submit:
        st.session_state.gad7_scores = scores
        st.rerun()

    current_scores = st.session_state.gad7_scores
    total_score = sum(current_scores)
    
    # Severity assessment mapping
    if total_score <= 4:
        severity, color, desc = "Minimal or No Anxiety", "#34d399", "Symptoms suggest no significant clinical anxiety indicators. Keep practicing supportive relaxation habits."
    elif total_score <= 9:
        severity, color, desc = "Mild Anxiety Symptoms", "#fbbf24", "Symptoms show mild anxiety triggers. Mindfulness, breathing exercises, or speaking with a therapist can provide excellent support."
    elif total_score <= 14:
        severity, color, desc = "Moderate Anxiety Symptoms", "#f59e0b", "Symptoms represent moderate anxiety levels. Consulting a healthcare provider or professional therapist is advised."
    else:
        severity, color, desc = "Severe Anxiety Symptoms", "#f43f5e", "Significant anxiety levels detected. Seeking professional clinical diagnostic evaluation and structured care is strongly recommended."

    # Render result card
    st.markdown(f"""
<div style="background:rgba(15,17,30,0.7);border:1px solid {color}40;border-radius:16px;padding:1.5rem;margin:1.5rem 0;">
    <p style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.4rem;">📊 Official Self-Report Rating</p>
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
        <span style="font-family:'Outfit',sans-serif;font-size:1.6rem;font-weight:800;color:{color};">{severity}</span>
        <span style="font-family:'Outfit',sans-serif;font-size:1.8rem;font-weight:900;color:{color};">{total_score} <span style="font-size:0.9rem;color:#6b7280;font-weight:500;">/ 21</span></span>
    </div>
    <p style="color:#e5e7eb;font-size:0.88rem;line-height:1.5;margin-bottom:0.8rem;">{desc}</p>
    <div class="conf-bar-bg" style="height:10px;">
        <div class="conf-bar-fill" style="width:{(total_score / 21) * 100}%;background:linear-gradient(90deg, #10b981, {color});"></div>
    </div>
</div>
""", unsafe_allow_html=True)

    # Offer downloadable report
    _render_report_downloader("GAD-7 Anxiety Screener", total_score, severity, current_scores, GAD7_QUESTIONS)


def _render_contrast_card() -> None:
    history = st.session_state.get("result_history", [])
    if not history:
        return

    latest = history[0]
    ai_label = latest["label"]
    ai_conf = latest["conf"]
    ai_text = latest["text"]
    
    ai_label_softened = (
        "🧠 Depressive Patterns Detected" if ai_label == "depression" else
        "⚠️ Anxiety Patterns Detected"    if ai_label == "anxiety"    else
        "✅ No Major Risk Patterns Detected"
    )
    ai_color = (
        "#f87171" if ai_label == "depression" else
        "#fbbf24" if ai_label == "anxiety"    else
        "#34d399"
    )

    st.markdown("### 🔬 Cross-Analysis: AI vs. Self-Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
<div style="background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:12px;padding:1.1rem;height:140px;">
    <p style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">🤖 Text AI Linguistic Analysis</p>
    <span style="font-family:'Outfit',sans-serif;font-size:1.15rem;font-weight:700;color:{ai_color};">{ai_label_softened}</span>
    <p style="color:#9ca3af;font-size:0.8rem;margin-top:0.4rem;line-height:1.4;">
        Analyzed: <code>"{ai_text}"</code><br>
        Linguistic Model Confidence: <strong>{ai_conf}</strong>
    </p>
</div>
""", unsafe_allow_html=True)

    # Grab the current clinical scores
    phq_sum = sum(st.session_state.phq9_scores)
    gad_sum = sum(st.session_state.gad7_scores)
    
    if phq_sum > 0:
        self_report_lbl = f"PHQ-9 Depression Score: {phq_sum}/27"
        self_color = "#f43f5e" if phq_sum >= 10 else "#fbbf24" if phq_sum >= 5 else "#34d399"
    elif gad_sum > 0:
        self_report_lbl = f"GAD-7 Anxiety Score: {gad_sum}/21"
        self_color = "#f43f5e" if gad_sum >= 10 else "#fbbf24" if gad_sum >= 5 else "#34d399"
    else:
        self_report_lbl = "No clinical questionnaires submitted yet."
        self_color = "#6b7280"

    with col2:
        st.markdown(f"""
<div style="background:rgba(192,132,252,0.06);border:1px solid rgba(192,132,252,0.15);border-radius:12px;padding:1.1rem;height:140px;">
    <p style="color:#6b7280;font-size:0.75rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.5rem;">📋 Self-Report Clinical Screener</p>
    <span style="font-family:'Outfit',sans-serif;font-size:1.15rem;font-weight:700;color:{self_color};">{self_report_lbl}</span>
    <p style="color:#9ca3af;font-size:0.8rem;margin-top:0.4rem;line-height:1.4;">
        Integrates clinical frequency metrics with text sentiment analysis to present a complete diagnostic comparison report.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);border-radius:12px;padding:1rem;margin-top:0.8rem;font-size:0.8rem;color:#9ca3af;line-height:1.5;">
    💡 <strong>Correlation Note:</strong> In clinical settings, therapists compare self-report metrics (PHQ-9/GAD-7) alongside spontaneous language patterns (Text AI analysis). 
    Matching scores suggest highly consistent emotional patterns, while mismatched scores are excellent focal points to explore in professional therapy.
</div>
""", unsafe_allow_html=True)


def _render_report_downloader(name: str, score: int, severity: str, scores: list, questions: list) -> None:
    patient = st.session_state.get("patient_name", "Guest Patient")
    report_text = f"""============================================================
           MINDSCAN CLINICAL SCREENING REPORT
============================================================
Patient Name : {patient}
Generated On : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Assessment   : {name}
Total Score  : {score}
Severity     : {severity}
------------------------------------------------------------
QUESTIONNAIRE DETAIL:
"""
    for idx, q in enumerate(questions):
        ans = scores[idx]
        report_text += f"Q{idx + 1}: {q}\n   ➔ Frequency: {FREQUENCY_CHOICES[ans]} (Points: {ans})\n\n"

    # Fetch latest BERT AI alignment if exists
    history = st.session_state.get("result_history", [])
    if history:
        latest = history[0]
        report_text += f"""------------------------------------------------------------
AI TEXT ANALYSIS CONTRAST:
Spontaneous text analyzed: "{latest['text']}"
BERT Predicted Label     : {latest['label'].upper()}
Model Prediction Conf    : {latest['conf']}
------------------------------------------------------------
"""
    else:
        report_text += """------------------------------------------------------------
AI TEXT ANALYSIS CONTRAST:
No AI text analysis history found for this session.
------------------------------------------------------------
"""

    report_text += """DISCLAIMER & NOTICE:
This screening report is NOT a formal psychiatric diagnosis. 
MindScan utilizes machine learning and standard self-report 
screening indices for information purposes only. Always present 
this sheet to a qualified professional (Psychiatrist / Therapist) 
for proper clinical assessment, guidance, and medical treatment.
============================================================"""

    st.download_button(
        label="⬇️ Download Clinical Screening Sheet",
        data=report_text,
        file_name=f"mindscan_clinical_sheet_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.txt",
        mime="text/plain",
        use_container_width=True
    )
