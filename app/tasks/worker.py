"""
app/tasks/worker.py
Celery task execution pipeline for local model inference.
"""
import logging
import os
import sys
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

@celery_app.task(name="tasks.queue_text_analysis")
def queue_text_analysis(patient_id: str, text: str, lang_hint: str) -> dict:
    """
    Celery task running offline translation and text classification.
    """
    logger.info("Initializing Celery Text Analysis task for patient: %s", patient_id)
    bert, translator = get_onnx_services()

    # 1. Anonymize/De-identify input text
    cleaned_input = deidentify_text(text)

    # 2. Local translation if native input is provided
    processed_text = cleaned_input
    if lang_hint and lang_hint != "en" and lang_hint != "auto":
        logger.info("Translating input text from language: %s ...", lang_hint)
        processed_text = translator.translate(cleaned_input, source_lang=lang_hint)
    
    # 3. Quantized ONNX BERT forward pass
    logger.info("Executing quantized ONNX BERT inference...")
    label, probs = bert.predict(processed_text)
    
    # Construct task response JSON
    return {
        "patient_id": patient_id,
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



