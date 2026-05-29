"""
src/ui/tabs/batch_tab.py
Tab 3 — Batch Analysis rendering logic.
Improvements:
  - Per-row try/except (one bad row doesn't crash the batch)
  - 200-row cap with warning
  - Progress bar per row
  - Uses ui_lang session state for translation language hint
"""
import streamlit as st
import pandas as pd

from src.services.translation import TranslationService
from src.services.bert import BERTClassifierService

BATCH_MAX_ROWS = 200


def render(S: dict, lang: str, is_multilingual: bool,
           translator_svc: TranslationService,
           bert_svc: BERTClassifierService) -> None:
    """Render the full Batch Analysis tab."""

    st.markdown(f"### {S['tab_batch']}")
    st.caption(
        "Auto-detects Indian scripts via Unicode range (U+0900–U+0D7F). "
        f"Translates offline. Maximum {BATCH_MAX_ROWS} rows per batch."
    )

    batch_text = st.text_area(
        S["batch_label"],
        height=200,
        placeholder=S["batch_placeholder"],
        key="batch_text_input",
    )

    if not st.button(S["btn_batch"], type="primary", use_container_width=True, key="btn_batch_run"):
        return

    lines = [ln.strip() for ln in batch_text.strip().split("\n") if len(ln.strip()) > 5]
    if not lines:
        st.warning("No valid entries found (minimum 5 characters per line).")
        return

    if len(lines) > BATCH_MAX_ROWS:
        st.warning(
            f"⚠️ Batch capped at {BATCH_MAX_ROWS} rows. "
            f"{len(lines) - BATCH_MAX_ROWS} rows were trimmed."
        )
        lines = lines[:BATCH_MAX_ROWS]

    results = []
    bar = st.progress(0, text="Processing batch records...")

    for i, line in enumerate(lines):
        has_indian = any("\u0900" <= c <= "\u0D7F" for c in line)
        analyzed = line
        lang_tag = "🇬🇧 English"
        translation_note = ""

        if has_indian:
            lang_tag = "🇮🇳 Indian Script"
            try:
                # Use the currently selected UI language as a hint
                source_hint = lang if is_multilingual else "auto"
                translated = translator_svc.translate(line, source_lang=source_hint)
                if translated:
                    analyzed = translated
                    translation_note = f"→ {translated[:40]}..." if len(translated) > 40 else f"→ {translated}"
            except RuntimeError:
                # Per-row failure: mark as error, don't crash the batch
                results.append({
                    "#":           i + 1,
                    "Script":      lang_tag,
                    "Input Text":  line[:65] + "..." if len(line) > 65 else line,
                    "Prediction":  "ERROR",
                    "Confidence":  "—",
                    "Risk Level":  "⚪ Unknown",
                })
                bar.progress((i + 1) / len(lines), text=f"Record {i + 1} of {len(lines)} (translation error)")
                continue

        try:
            lbl, prb = bert_svc.predict(analyzed)
        except RuntimeError:
            results.append({
                "#":           i + 1,
                "Script":      lang_tag,
                "Input Text":  line[:65] + "..." if len(line) > 65 else line,
                "Prediction":  "ERROR",
                "Confidence":  "—",
                "Risk Level":  "⚪ Unknown",
            })
            bar.progress((i + 1) / len(lines), text=f"Record {i + 1} of {len(lines)} (inference error)")
            continue

        results.append({
            "#":           i + 1,
            "Script":      lang_tag,
            "Input Text":  line[:65] + "..." if len(line) > 65 else line,
            "Prediction":  lbl.upper(),
            "Confidence":  f"{float(prb.max()) * 100:.1f}%",
            "Risk Level":  (
                "🔴 High"   if lbl == "depression" else
                "🟡 Medium" if lbl == "anxiety"    else
                "🟢 Low"
            ),
        })
        bar.progress((i + 1) / len(lines), text=f"Record {i + 1} of {len(lines)}...")

    bar.empty()

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.divider()

    st.markdown(f"#### {S['batch_summary']}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(S["total"],                len(results))
    c2.metric(f"🔴 {S['depression']}",   sum(1 for r in results if r["Prediction"] == "DEPRESSION"))
    c3.metric(f"🟡 {S['anxiety']}",      sum(1 for r in results if r["Prediction"] == "ANXIETY"))
    c4.metric(f"🟢 {S['normal']}",       sum(1 for r in results if r["Prediction"] == "NORMAL"))
    c5.metric("⚪ Errors",               sum(1 for r in results if r["Prediction"] == "ERROR"))

    st.download_button(
        label=S["download_csv"],
        data=df.to_csv(index=False),
        file_name="mindscan_batch_report.csv",
        mime="text/csv",
    )
