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

from transformers import pipeline

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
    Offline local translation service using HuggingFace pipelines.
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
        self._pipeline = None
        self._init_lock = threading.Lock()
        self._initialized = True
        logger.debug("TranslationService singleton instance created.")

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Returns a copy of the supported Indian languages dictionary."""
        return cls.SUPPORTED_LANGUAGES.copy()

    def _get_pipeline(self):
        """
        Lazy-loads the translation pipeline in a thread-safe manner.
        Caches the pipeline internally so it is not re-initialized on subsequent calls.
        """
        if self._pipeline is None:
            with self._init_lock:
                if self._pipeline is None:
                    model_name = getattr(
                        Config, "HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en"
                    )
                    logger.info("Initializing translation pipeline: %s ...", model_name)
                    try:
                        device = 0 if Config.DEVICE == "cuda" else -1
                        self._pipeline = pipeline(
                            task="translation",
                            model=model_name,
                            device=device,
                        )
                        logger.info(
                            "Translation pipeline ready on device: %s", Config.DEVICE
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed to init translation pipeline: %s", exc, exc_info=True
                        )
                        raise RuntimeError(
                            f"Translation pipeline initialization failed: {exc}"
                        ) from exc
        return self._pipeline

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

        # Use cached internal translation (keyed on text + prefix)
        prefix = LANG_PREFIXES.get(source_lang, "") if source_lang != "auto" else ""
        return self._cached_translate(prefix + text)

    @lru_cache(maxsize=512)
    def _cached_translate(self, prefixed_text: str) -> str:
        """
        LRU-cached translation. Caches up to 512 unique inputs.
        The cache key is the prefixed text string, so different language
        prefixes for the same raw text are cached separately.
        """
        try:
            translator = self._get_pipeline()
            logger.info(
                "Translating text (length: %d chars) ...", len(prefixed_text)
            )
            result = translator(prefixed_text)

            if not result or "translation_text" not in result[0]:
                raise RuntimeError("Pipeline returned an invalid result payload.")

            translated = result[0]["translation_text"]
            logger.info("Translation done. Output length: %d chars.", len(translated))
            return translated

        except Exception as exc:
            logger.exception("Translation failed: %s", exc)
            raise RuntimeError(f"Offline translation failed: {exc}") from exc
