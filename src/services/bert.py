"""
src/services/bert.py
Thread-safe singleton BERT classifier service.
Follows the same pattern as TranslationService and GroqVisionService.
"""
import logging
import threading
import warnings
from typing import Optional

import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

from src.config import Config

# Suppress only known noisy HuggingFace deprecation warnings — not everything
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

logger = logging.getLogger(__name__)

LABELS: list[str] = ["anxiety", "depression", "normal"]
MAX_BERT_WORDS: int = 350  # Warn user if input exceeds this (BERT hard-truncates at 512 tokens)


class BERTClassifierService:
    """
    Thread-safe singleton for loading and running BERT mental health classification.

    Usage:
        svc = BERTClassifierService()
        label, probs = svc.predict("I feel very anxious today")
    """

    _instance: Optional["BERTClassifierService"] = None
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
        self._tokenizer: Optional[BertTokenizer] = None
        self._model: Optional[BertForSequenceClassification] = None
        self._device = torch.device(Config.DEVICE)
        self._load_lock = threading.Lock()
        self._initialized = True
        logger.debug("BERTClassifierService singleton created.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True once the model and tokenizer have been loaded successfully."""
        return self._tokenizer is not None and self._model is not None

    def load(self) -> bool:
        """
        Explicitly load the BERT model and tokenizer from disk.
        Safe to call multiple times — only loads once.

        Returns:
            bool: True if loaded successfully, False if model files not found.
        """
        if self.is_loaded:
            return True

        import os
        if not os.path.exists(Config.MODEL_DIR):
            logger.error("BERT model directory not found: %s", Config.MODEL_DIR)
            return False

        with self._load_lock:
            if self.is_loaded:  # double-checked locking
                return True
            try:
                logger.info("Loading BERT tokenizer from %s ...", Config.MODEL_DIR)
                self._tokenizer = BertTokenizer.from_pretrained(Config.MODEL_DIR)
                logger.info("Loading BERT model from %s ...", Config.MODEL_DIR)
                self._model = (
                    BertForSequenceClassification.from_pretrained(Config.MODEL_DIR)
                    .to(self._device)
                )
                self._model.eval()
                logger.info("BERT model loaded successfully on device: %s", self._device)
                return True
            except Exception as exc:
                logger.exception("Failed to load BERT model: %s", exc)
                self._tokenizer = None
                self._model = None
                return False

    def predict(self, text: str) -> tuple[str, np.ndarray]:
        """
        Run classification on English text.

        Args:
            text: English text (after translation if needed).

        Returns:
            (label, probs) where label is one of 'anxiety'/'depression'/'normal'
            and probs is a float32 numpy array of shape (3,).

        Raises:
            ValueError:  Empty or non-string input.
            RuntimeError: Model not loaded or inference error.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        if not self.is_loaded:
            raise RuntimeError(
                "BERT model is not loaded. Call BERTClassifierService().load() first."
            )

        word_count = len(text.split())
        if word_count > MAX_BERT_WORDS:
            logger.warning(
                "Input has %d words (> %d). BERT will truncate at 512 tokens.",
                word_count, MAX_BERT_WORDS,
            )

        try:
            enc = self._tokenizer(   # type: ignore[misc]
                text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                logits = self._model(  # type: ignore[misc]
                    input_ids=enc["input_ids"].to(self._device),
                    attention_mask=enc["attention_mask"].to(self._device),
                ).logits
            probs: np.ndarray = torch.softmax(logits, dim=1).cpu().numpy()[0]
            label = LABELS[int(np.argmax(probs))]
            logger.debug("BERT prediction: %s (probs=%s)", label, probs.tolist())
            return label, probs
        except Exception as exc:
            logger.exception("BERT inference failed: %s", exc)
            raise RuntimeError(f"BERT inference error: {exc}") from exc
