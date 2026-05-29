"""
src/ui/tabs/image_tab.py
Tab 2 — Image Analysis rendering logic.
Groq Llama 4 Scout vision with 3 analysis modes.
"""
import streamlit as st
from PIL import Image

from src.services.vision import GroqVisionService

MODE_PROMPTS = {
    "general": (
        "Analyze this image thoroughly and provide:\n"
        "1. **Scene Description** - Key visual elements, setting, objects, colors.\n"
        "2. **Emotional Tone** - Mood and atmosphere the image conveys.\n"
        "3. **Mental Health Relevance** - Visual cues related to emotional wellbeing. Be empathetic.\n"
        "4. **Key Insights** - Most important takeaways.\n"
        "Use clear headings. Be thorough and empathetic."
    ),
    "social_media": (
        "Analyze this as a social media post and provide:\n"
        "1. **Visual Content** - People, environment, objects, filters, aesthetic.\n"
        "2. **Emotional Signals** - Emotions via expressions, body language, colors.\n"
        "3. **Mental Health Indicators** - Signs of distress, isolation, or positivity. "
        "Rate: Positive / Neutral / Concerning.\n"
        "4. **Context** - What the poster may be communicating.\n"
        "Be empathetic and non-judgmental."
    ),
    "chart": (
        "Analyze this chart or graph and provide:\n"
        "1. **Chart Type and Structure** - Visualization type, axes, scales.\n"
        "2. **Key Data Patterns** - Trends, peaks, dips, anomalies.\n"
        "3. **Mental Health Context** - Interpret mental health metrics if applicable.\n"
        "4. **Conclusions** - Precise summaries and next steps.\n"
        "Be precise and technical."
    ),
}


def render(S: dict, lang: str, vision_svc: GroqVisionService | None) -> None:
    """Render the full Image Analysis tab."""
    st.markdown(f"### {S['tab_image']}")
    st.caption("Powered by Groq Vision — Meta Llama 4 Scout 17B")

    if not vision_svc:
        st.error(
            "⚠️ Groq Vision Service is unavailable. "
            "Add `GROQ_API_KEY=your_key` to your `.env` file and restart the app."
        )
        return

    mode_opts = {
        "general":      "🌐 " + ("General Visual Tone" if lang == "en" else "General Analysis"),
        "social_media": "📱 " + ("Social Media Sentiment" if lang == "en" else "Social Media"),
        "chart":        "📊 " + ("Scientific Chart Analysis" if lang == "en" else "Chart / Graph"),
    }
    mode = st.radio(
        "Analysis Mode:",
        options=list(mode_opts.keys()),
        format_func=lambda x: mode_opts[x],
        horizontal=True,
        key="img_analysis_mode",
    )

    uploaded = st.file_uploader(
        S["img_upload"],
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        key="img_uploader",
    )

    if not uploaded:
        return

    try:
        image = Image.open(uploaded)
    except Exception as exc:
        st.error(f"Failed to load image: {exc}")
        return

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.image(image, caption=uploaded.name, use_column_width=True)
    with col_b:
        st.markdown(f"""
<div style="background:rgba(15,17,30,0.7);border:1px solid rgba(255,255,255,0.07);border-radius:12px;padding:1.2rem;">
    <p style="color:#6b7280;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:0.8rem;">📋 File Metadata</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Name:</strong> {uploaded.name}</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Resolution:</strong> {image.size[0]} × {image.size[1]} px</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Mode:</strong> {image.mode}</p>
    <p style="color:#d1d5db;margin:0.3rem 0;"><strong>Size:</strong> {uploaded.size / 1024:.1f} KB</p>
</div>""", unsafe_allow_html=True)

    if st.button(S["btn_vision"], type="primary", use_container_width=True, key="btn_vision_analyze"):
        with st.spinner(S["spinner_vision"]):
            try:
                result = vision_svc.analyze_image(image, MODE_PROMPTS[mode])
                st.divider()
                st.markdown("### 📋 Analysis Report")
                st.markdown(result)
                st.download_button(
                    label=S["download_txt"],
                    data=result,
                    file_name=f"mindscan_report_{uploaded.name}.txt",
                    mime="text/plain",
                )
            except Exception as exc:
                st.error(f"Image analysis failed: {exc}")
