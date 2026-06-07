"""
app/tasks/worker.py
Celery task execution pipeline for local model inference.
"""
import logging
import os
import sys
from pathlib import Path
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

from typing import Optional
import threading
