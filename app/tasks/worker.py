"""
app/tasks/worker.py
Celery task execution pipeline for local model inference.
"""
import logging
import os
import sys
import re
from pathlib import Path
from typing import Optional
import threading
from celery import Celery
import numpy as np

# Adjust search path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from src.services.bert_onnx import BERTOnnxService
from src.services.translator_onnx import TranslationOnnxService
from app.utils.anonymization import deidentify_text

logger = logging.getLogger(__name__)

import redis
# Monkey-patch redis-py Connection class to force protocol=2 (RESP2) globally.
# This bypasses the RESP3 HELLO handshake command which is unsupported by the host's Redis service.
_orig_init = redis.Connection.__init__
def _patched_init(self, *args, **kwargs):
    kwargs['protocol'] = 2
    kwargs['maint_notifications_config'] = None
    _orig_init(self, *args, **kwargs)
redis.Connection.__init__ = _patched_init

# Redis Broker and backend configurations
redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery(
    "mindscan_tasks",
    broker=redis_url,
    backend=redis_url
)

# Configure Celery serialization/concurrency
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_concurrency=1  # Lock model inference thread to 1 to prevent CPU starvation
)

# Thread-safe lazy loading container inside Celery Worker
_bert_onnx: Optional[BERTOnnxService] = None
_translator_onnx: Optional[TranslationOnnxService] = None
_init_lock = threading.Lock()

def get_onnx_services():
    global _bert_onnx, _translator_onnx
    if _bert_onnx is None or _translator_onnx is None:
        with _init_lock:
            if _bert_onnx is None:
                _bert_onnx = BERTOnnxService()
            if _translator_onnx is None:
                _translator_onnx = TranslationOnnxService()
    return _bert_onnx, _translator_onnx


# ---------------------------------------------------------------------------
# Unicode Script Auto-Detector
# Maps Unicode character ranges to BCP-47 language codes so that
# "Auto-Detect" submissions are automatically translated rather than
# being passed raw to BERT (which was trained only on English text).
# ---------------------------------------------------------------------------
_SCRIPT_RANGES = [
    (0x0C80, 0x0CFF, "kn"),   # Kannada
    (0x0900, 0x097F, "hi"),   # Devanagari  (Hindi / Marathi)
    (0x0B80, 0x0BFF, "ta"),   # Tamil
    (0x0C00, 0x0C7F, "te"),   # Telugu
    (0x0D00, 0x0D7F, "ml"),   # Malayalam
    (0x0980, 0x09FF, "bn"),   # Bengali
    (0x0A80, 0x0AFF, "gu"),   # Gujarati
    (0x0A00, 0x0A7F, "pa"),   # Gurmukhi (Punjabi)
    (0x0D80, 0x0DFF, "si"),   # Sinhala
    (0x0E00, 0x0E7F, "th"),   # Thai
    (0x0600, 0x06FF, "ur"),   # Arabic / Urdu
]

def detect_script_language(text: str) -> Optional[str]:
    """
    Inspect the Unicode code points of each character in `text`.
    Return the BCP-47 code for the first matching script range found,
    or None if the text appears to be ASCII/Latin (English).
    """
    for ch in text:
        cp = ord(ch)
        for lo, hi, lang in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                return lang
    return None  # Treat as English / Latin script


def safe_translate(translator_onnx: TranslationOnnxService, text: str, source_lang: str) -> str:
    """
    Two-tier translation fallback:
      Tier 1 — ONNX quantized MarianMT (fast, offline)
      Tier 2 — Pure HuggingFace AutoModelForSeq2SeqLM (slower, same model)
    If both fail the original text is returned with a warning so that BERT
    still runs (it will produce poor results on non-English text, but the
    task will not crash).
    """
    # Tier 1: ONNX
    try:
        result = translator_onnx.translate(text, source_lang=source_lang)
        if result and result.strip():
            logger.info("Translation (ONNX) succeeded for lang=%s", source_lang)
            return result
    except Exception as exc_onnx:
        logger.warning("ONNX translation failed (%s), falling back to HuggingFace pipeline.", exc_onnx)

    # Tier 2: Pure HuggingFace AutoModel
    try:
        from src.services.translation import TranslationService
        hf_translator = TranslationService()
        result = hf_translator.translate(text, source_lang=source_lang)
        if result and result.strip():
            logger.info("Translation (HuggingFace fallback) succeeded for lang=%s", source_lang)
            return result
    except Exception as exc_hf:
        logger.error(
            "HuggingFace fallback translation also failed for lang=%s: %s",
            source_lang, exc_hf
        )

    # Last resort: pass through untranslated (BERT will likely misclassify non-English)
    logger.error(
        "All translation tiers failed for lang=%s. Passing raw text to BERT — results may be inaccurate.",
        source_lang
    )
    return text

@celery_app.task(name="tasks.queue_text_analysis")
def queue_text_analysis(patient_id: str, text: str, lang_hint: str) -> dict:
    """
    Celery task running offline translation and text classification.

    Translation logic (in order of priority):
      1. If the frontend sent an explicit language code (e.g. 'kn', 'hi') use it.
      2. If the frontend sent 'auto' (default), inspect the Unicode script ranges
         of the input text to auto-detect the language.  This prevents raw
         Kannada / Hindi / Tamil script from reaching BERT unchanged.
      3. If the text looks like Latin/English, skip translation.
      4. Translation uses a two-tier fallback: ONNX → HuggingFace AutoModel.
    """
    logger.info("Initializing Celery Text Analysis task for patient: %s", patient_id)
    bert, translator = get_onnx_services()

    # 1. Anonymize/De-identify input text
    cleaned_input = deidentify_text(text)

    # 2. Resolve effective language code
    effective_lang = lang_hint  # may be 'auto', 'en', or a BCP-47 code
    if lang_hint == "auto":
        detected = detect_script_language(cleaned_input)
        if detected:
            effective_lang = detected
            logger.info(
                "Auto-detected non-Latin script — resolved language: %s", effective_lang
            )
        else:
            effective_lang = "en"
            logger.info("Auto-detect: Latin/ASCII text — treating as English, skipping translation.")

    # 3. Translate if the text is not already in English
    processed_text = cleaned_input
    if effective_lang and effective_lang not in ("en", "auto"):
        logger.info("Translating input from language: %s ...", effective_lang)
        processed_text = safe_translate(translator, cleaned_input, source_lang=effective_lang)
        logger.info("Translated text: %r", processed_text[:120])

    # 4. Quantized ONNX BERT forward pass
    logger.info("Executing quantized ONNX BERT inference...")
    label, probs = bert.predict(processed_text)

    # Construct task response JSON
    return {
        "patient_id": patient_id,
        "original_lang": effective_lang,
        "processed_text": processed_text,
        "prediction": label,
        "probabilities": [float(p) for p in probs],
        "normal_score": float(probs[1]),
        "anxiety_score": float(probs[2]),
        "depression_score": float(probs[0])
    }

_vision_service = None

def get_vision_service():
    global _vision_service
    if _vision_service is None:
        with _init_lock:
            if _vision_service is None:
                from src.services.vision import GroqVisionService
                _vision_service = GroqVisionService()
    return _vision_service

@celery_app.task(name="tasks.queue_image_scan")
def queue_image_scan(patient_id: str, image_bytes_hex: str, mode: str) -> dict:
    """
    Celery task running OpenCV image warping and Groq Llama 4 Scout Vision analysis.
    """
    logger.info("Initializing Celery Image Scan task for patient: %s", patient_id)
    img_bytes = bytes.fromhex(image_bytes_hex)
    
    # 1. OpenCV Preprocessing
    from app.services.image_processor import ImageProcessor
    processor = ImageProcessor()
    processed_bytes = processor.process_image(img_bytes)
    
    # 2. Convert processed bytes to PIL Image
    import io
    from PIL import Image
    pil_image = Image.open(io.BytesIO(processed_bytes))
    
    # 3. Resolve prompt structure
    prompts = {
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
            "3. **Mental Health Indicators** - Signs of distress, isolation, or positivity. Rate: Positive / Neutral / Concerning.\n"
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
        )
    }
    prompt = prompts.get(mode, prompts["general"])
    
    # 4. Invoke Groq Vision model
    vision = get_vision_service()
    analysis_result = vision.analyze_image(pil_image, prompt)
    
    return {
        "patient_id": patient_id,
        "mode": mode,
        "analysis": analysis_result
    }



