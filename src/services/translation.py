"""
src/services/translation.py
Offline translation service using Helsinki-NLP/opus-mt-mul-en.
Improvements over v1:
  - Language-code prefixes for better accuracy on short texts (>>kn<<, >>hi<<, etc.)
  - LRU cache for repeated translations (up to 512 entries)
  - Thread-safe singleton with lazy-loading pipeline
"""
import logging
import threading
from functools import lru_cache
from typing import Dict, Optional

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import Config

logger = logging.getLogger(__name__)

# Helsinki-NLP opus-mt-mul-en source-language prefixes.
# Prepending these significantly improves accuracy on short texts.
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


class TranslationService:
    """
    Offline local translation service using HuggingFace AutoModels.
    Designed for thread-safe, cached execution to preserve data privacy.

    Supports major Indian languages by mapping them to English using
    the multilingual model 'Helsinki-NLP/opus-mt-mul-en'.
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
        "or": "Odia (ଓଡ଼ିଆ)",
    }

    _instance: Optional["TranslationService"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Thread-safe Singleton initialization pattern."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initializes class variables exactly once."""
        if getattr(self, "_initialized", False):
            return
        self._tokenizer = None
        self._model = None
        self._init_lock = threading.Lock()
        self._initialized = True
        logger.debug("TranslationService singleton instance created.")

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Returns a copy of the supported Indian languages dictionary."""
        return cls.SUPPORTED_LANGUAGES.copy()

    def _load_model(self):
        """
        Lazy-loads the translation model and tokenizer in a thread-safe manner.
        Caches them internally so they are not re-initialized on subsequent calls.
        """
        if self._model is None or self._tokenizer is None:
            with self._init_lock:
                if self._model is None or self._tokenizer is None:
                    model_name = getattr(
                        Config, "HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en"
                    )
                    logger.info("Initializing translation model: %s ...", model_name)
                    try:
                        logger.info("Loading tokenizer for %s ...", model_name)
                        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
                        logger.info("Loading model for %s ...", model_name)
                        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(Config.DEVICE)
                        logger.info(
                            "Translation model ready on device: %s", Config.DEVICE
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to init translation model: %s", exc, exc_info=True
                        )
                        self._tokenizer = None
                        self._model = None
                        raise RuntimeError(
                            f"Translation model initialization failed: {exc}"
                        ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str, source_lang: str = "auto") -> str:
        """
        Translate multilingual text to English.

        Args:
            text:        Raw input text (any supported Indian language).
            source_lang: BCP-47 language code (e.g. 'kn', 'hi').
                         'auto' = let the model detect the language.
                         When a known code is passed, a language prefix is prepended
                         to the text for better accuracy on short inputs.

        Returns:
            str: Translated English text.

        Raises:
            ValueError:  Input is not a string.
            RuntimeError: Pipeline failure.
        """
        if not isinstance(text, str):
            logger.error(
                "Invalid input type for translation. Expected str, got: %s", type(text)
            )
            raise ValueError("Input text must be a valid string.")

        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text — returning empty string.")
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
        """
        LRU-cached translation. Caches up to 512 unique inputs.
        The cache key is the prefixed text string.
        """
        try:
            self._load_model()
            logger.info(
                "Translating text (length: %d chars) ...", len(prefixed_text)
            )
            
            # Direct model inference bypassing the legacy pipeline wrapper
            inputs = self._tokenizer(prefixed_text, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(Config.DEVICE) for k, v in inputs.items()}
            
            import torch
            with torch.no_grad():
                generated_tokens = self._model.generate(**inputs)
                
            translated = self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
            logger.info("Translation done. Output length: %d chars.", len(translated))
            return translated

        except Exception as exc:
            logger.exception("Translation failed: %s", exc)
            raise RuntimeError(f"Offline translation failed: {exc}") from exc
