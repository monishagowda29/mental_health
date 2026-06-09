"""
src/services/translator_onnx.py
Offline translation service using Helsinki-NLP/opus-mt-mul-en ONNX.
"""
import logging
import os
import threading
from functools import lru_cache
from typing import Dict, Optional

from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM

from src.config import Config

logger = logging.getLogger(__name__)

# Helsinki-NLP prefixes
LANG_PREFIXES: Dict[str, str] = {
    "kn": ">>kn<< ",
    "hi": ">>hi<< ",
    "ta": ">>ta<< ",
    "te": ">>te<< ",
    "ml": ">>ml<< ",
    "mr": ">>mr<< ",
    "bn": ">>bn<< ",
    "gu": ">>gu<< ",
    "pa": ">>pa<< ",
    "ur": ">>ur<< ",
    "or": ">>or<< ",
}

class TranslationOnnxService:
    """
    Offline local translation service running quantized Marian ONNX models.
    """
    SUPPORTED_LANGUAGES: Dict[str, str] = {
        "kn": "Kannada (ಕನ್ನಡ)",
        "hi": "Hindi (हिन्दी)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "ml": "Malayalam (മലയാളം)",
        "mr": "Marathi (मराठी)",
        "bn": "Bengali (বাংলা)",
        "gu": "Gujarati (ગુજરાતી)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "ur": "Urdu (اردو)",
        "or": "Odia (ଓಡ଼ిଆ)",
    }

    _instance: Optional["TranslationOnnxService"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self._tokenizer = None
        self._model = None
        self._init_lock = threading.Lock()
        self._initialized = True
        logger.debug("TranslationOnnxService singleton created.")

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        return cls.SUPPORTED_LANGUAGES.copy()

    def _load_model(self):
        if self._model is None or self._tokenizer is None:
            with self._init_lock:
                if self._model is None or self._tokenizer is None:
                    model_dir = os.environ.get("TRANSLATOR_ONNX_DIR", str(Config.ROOT_PATH / "models" / "translator_onnx"))
                    logger.info("Initializing ONNX translation model from %s ...", model_dir)
                    try:
                        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
                        # Explicitly pass file names because the folder uses the legacy
                        # split-decoder layout (encoder_model.onnx + decoder_model.onnx +
                        # decoder_with_past_model.onnx) rather than the merged single-file
                        # layout (decoder_model_merged.onnx) that optimum >= 1.14 expects
                        # by default. Without this, optimum raises a FileNotFoundError and
                        # falls back to a CPU stub that outputs garbage dots.
                        self._model = ORTModelForSeq2SeqLM.from_pretrained(
                            model_dir,
                            provider="CPUExecutionProvider",
                            encoder_file_name="encoder_model.onnx",
                            decoder_file_name="decoder_model.onnx",
                            decoder_with_past_file_name="decoder_with_past_model.onnx",
                        )
                        logger.info("ONNX translation model loaded successfully.")
                    except Exception as exc:
                        logger.error("Failed to load translation ONNX: %s", exc, exc_info=True)
                        self._tokenizer = None
                        self._model = None
                        raise RuntimeError(f"Translation ONNX initialization failed: {exc}") from exc

    def translate(self, text: str, source_lang: str = "auto") -> str:
        if not isinstance(text, str):
            raise ValueError("Input text must be a valid string.")

        if not text or not text.strip():
            return ""

        # Prepend the Helsinki-NLP language prefix so that opus-mt-mul-en
        # knows which source language it is translating FROM.  Without this
        # prefix the model silently produces garbage / wrong-language output.
        prefix = LANG_PREFIXES.get(source_lang, "")
        prefixed_text = prefix + text if prefix else text

        # Use cached internal translation
        return self._cached_translate(prefixed_text)

    @lru_cache(maxsize=512)
    def _cached_translate(self, prefixed_text: str) -> str:
        try:
            self._load_model()
            logger.info("Translating text (ONNX) length: %d chars.", len(prefixed_text))

            inputs = self._tokenizer(prefixed_text, return_tensors="pt", padding=True, truncation=True)

            # Run model generation utilizing ORT
            generated_tokens = self._model.generate(**inputs, num_beams=6, max_length=512)
            translated = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

            logger.info("Translation done (ONNX). Output length: %d chars.", len(translated))

            # ── Output quality guard ──────────────────────────────────────────
            # If the model produces garbage (all dots / punctuation / spaces,
            # fewer than 2 alphabetic characters) it means the ONNX session
            # silently failed — raise so safe_translate falls back to HuggingFace.
            alpha_chars = sum(1 for c in translated if c.isalpha())
            if alpha_chars < 2:
                raise RuntimeError(
                    f"ONNX translation produced garbage output (alpha_chars={alpha_chars}): "
                    f"{translated!r:.80}"
                )

            return translated
        except Exception as exc:
            logger.exception("Translation ONNX forward pass failed: %s", exc)
            raise RuntimeError(f"Offline translation failed: {exc}") from exc
