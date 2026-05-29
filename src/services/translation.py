import logging
import threading
from typing import Dict, List, Optional, Union
from transformers import pipeline
from src.config import Config

# Configure logging for the translation service module
logger = logging.getLogger(__name__)

class TranslationService:
    """
    Offline local translation service using Hugging Face pipelines.
    Designed for high-performance, thread-safe execution to maintain data privacy 
    and satisfy strict HIPAA/GDPR compliance parameters.
    
    Supports major standard Indian (Indic) languages by mapping them to English 
    using the multilingual model 'Helsinki-NLP/opus-mt-mul-en'.
    """
    
    # Formally documented standard Indian languages supported by the offline pipeline
    SUPPORTED_LANGUAGES = {
        "kn": "Kannada (ಕನ್ನಡ)",
        "hi": "Hindi (हिन्दी)",
        "ta": "Tamil (தமிழ்)",
        "te": "Telugu (తెలుగు)",
        "ml": "Malayalam (മലയാളം)",
        "mr": "Marathi (ಮರಾठी)",
        "bn": "Bengali (বাংলা)",
        "gu": "Gujarati (ગુજરાતી)",
        "pa": "Punjabi (ਪੰਜਾਬੀ)",
        "ur": "Urdu (اردو)",
        "or": "Odia (ଓଡ଼ିଆ)"
    }
    
    _instance = None
    _singleton_lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Thread-safe Singleton initialization pattern."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super(TranslationService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initializes class variables exactly once."""
        if getattr(self, "_initialized", False):
            return
            
        self._pipeline = None
        self._init_lock = threading.Lock()
        self._initialized = True
        logger.debug("TranslationService singleton instance structured.")

    @classmethod
    def get_supported_languages(cls) -> Dict[str, str]:
        """Returns a copy of the supported Indian languages dictionary."""
        return cls.SUPPORTED_LANGUAGES.copy()

    def _get_pipeline(self):
        """
        Lazy-loads the translation pipeline in a thread-safe manner.
        Caches the model internally so it is not re-initialized on subsequent calls.
        """
        if self._pipeline is None:
            with self._init_lock:
                if self._pipeline is None:
                    model_name = getattr(Config, "HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en")
                    logger.info("Initializing local translation pipeline using model: %s...", model_name)
                    try:
                        # Load the translation pipeline. For Helsinki-NLP multilingual models,
                        # the default task is translation. We configure it to run on CUDA if available.
                        device = 0 if Config.DEVICE == "cuda" else -1
                        self._pipeline = pipeline(
                            task="translation", 
                            model=model_name,
                            device=device
                        )
                        logger.info("Local translation pipeline initialized successfully on device: %s.", Config.DEVICE)
                    except Exception as e:
                        logger.error("Failed to initialize offline Hugging Face pipeline for translation: %s", str(e), exc_info=True)
                        raise RuntimeError(f"Translation pipeline initialization failed: {e}") from e
        return self._pipeline

    def translate(self, text: str) -> str:
        """
        Translates multilingual text (targeting Kannada, Hindi, Tamil, Telugu, etc.) into English.
        
        Args:
            text (str): The raw input text to translate.
            
        Returns:
            str: The translated text in English.
            
        Raises:
            ValueError: If the input parameter is not a string.
            RuntimeError: If translation fails during pipeline execution.
        """
        if not isinstance(text, str):
            logger.error("Invalid input type provided for translation. Expected string, got: %s", type(text))
            raise ValueError("Input text must be a valid string.")
            
        # Graceful fallback for empty or whitespace-only inputs
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text passed to translator. Returning empty string.")
            return ""

        try:
            # Thread-safe pipeline retrieval
            translator = self._get_pipeline()
            
            logger.info("Executing local offline translation on text block (length: %d chars)...", len(text))
            
            # Execute pipeline.
            # Helsinki-NLP/opus-mt-mul-en is a multilingual-to-English translation model.
            # It maps Indian languages dynamically to English targets without explicit source code prefixes.
            result = translator(text)
            
            if not result or "translation_text" not in result[0]:
                logger.error("Pipeline returned an invalid translation result: %s", result)
                raise RuntimeError("Translation output payload format is invalid.")
                
            translated_text = result[0]["translation_text"]
            logger.info("Translation completed successfully. Resulting length: %d chars.", len(translated_text))
            return translated_text
            
        except Exception as e:
            logger.exception("A critical exception occurred during the translation process: %s", str(e))
            raise RuntimeError(f"Offline translation pipeline execution failed: {e}") from e
