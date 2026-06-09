"""
src/services/bert_onnx.py
Thread-safe singleton service for quantized BERT text classification.
"""
import logging
import os
import threading
from typing import Optional
import numpy as np
from transformers import AutoTokenizer

from src.config import Config
from src.services.onnx_base import ONNXBaseService

logger = logging.getLogger(__name__)

LABELS = ["depression", "normal", "anxiety"]
MAX_BERT_WORDS = 350

class BERTOnnxService(ONNXBaseService):
    """
    Quantized BERT Classifier Service executing dynamic dynamic INT8 inference via ONNX.
    """
    _instance: Optional["BERTOnnxService"] = None
    _singleton_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    # Resolve configuration paths
                    model_path = os.environ.get("BERT_ONNX_PATH", str(Config.ROOT_PATH / "models" / "bert_quantized.onnx"))
                    cls._instance = super(ONNXBaseService, cls).__new__(cls)
                    # Initialize the ONNX session explicitly inside __new__
                    super(BERTOnnxService, cls._instance).__init__(model_path)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, *args, **kwargs):
        if getattr(self, "_initialized", False):
            return
        tokenizer_dir = os.environ.get("BERT_TOKENIZER_DIR", Config.MODEL_DIR)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        self._initialized = True
        logger.debug("BERTOnnxService initialized.")

    def predict(self, text: str) -> tuple[str, np.ndarray]:
        """
        Runs text classification using quantized BERT.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        cleaned = text.lower().strip()

        # Wellness vs Clinical lexicon calibration override.
        # This guard catches common HuggingFace translation outputs for
        # wellness phrases in Indian languages (e.g. "I am well", "I'm alright").
        WELLNESS_WORDS = {
            # Direct positive-state words
            "good", "great", "happy", "nice", "excellent", "wonderful", "beautiful", "calm",
            "peaceful", "perfect", "amazing", "content", "cheerful", "joy", "fine", "healthy",
            "glad", "blessed", "well", "alright", "okay", "ok", "comfortable", "relaxed",
            "relieved", "joyful", "thankful", "grateful", "positive", "refreshed",
            "energetic", "lively", "optimistic", "pleased", "delighted", "satisfied",
            "brilliant", "fantastic", "superb", "splendid", "elated", "better",
            # Common phrase fragments from HuggingFace NMT output
            "doing well", "feeling good", "doing good", "feeling nice", "feeling well",
            "feeling great", "feeling happy", "feeling fine", "feeling better",
            "am well", "am good", "am fine", "am okay", "am alright", "am happy",
            "i'm well", "i'm good", "i'm fine", "i'm okay", "i'm alright", "i'm happy",
            "im well", "im good", "im fine", "im okay", "im alright",
            "doing great", "doing fine", "doing okay",
        }
        CLINICAL_KEYWORDS = {
            "anxious", "anxiety", "worry", "worried", "panic", "panicked", "fear", "scared",
            "depressed", "depression", "sad", "sadness", "empty", "hopeless", "hopelessness",
            "suicide", "suicidal", "kill", "die", "death", "hurt", "pain", "cutting", "harm",
            "lonely", "darkness", "dark", "cry", "crying", "hate", "scary", "shake", "shaking",
            "stress", "stressed", "bipolar", "mental health", "struggle", "struggling",
            "worthless", "failure", "overwhelmed", "numb", "exhausted", "trapped",
        }

        has_wellness = any(w in cleaned for w in WELLNESS_WORDS)
        has_clinical = any(c in cleaned for c in CLINICAL_KEYWORDS)

        if has_wellness and not has_clinical:
            probs = np.array([0.05, 0.90, 0.05], dtype=np.float32)
            label = "normal"
            logger.debug("Quantized BERT prediction (override): %s (probs=%s)", label, probs.tolist())
            return label, probs

        # Tokenize inputs matching ONNX expected format
        inputs = self.tokenizer(
            text,
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="np"
        )

        input_feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
            "token_type_ids": inputs["token_type_ids"].astype(np.int64)
        }

        # Run session
        outputs = self.run_inference(input_feed)
        logits = outputs[0]

        # Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        probs = probs[0]

        pred_label = LABELS[int(np.argmax(probs))]
        logger.debug("Quantized BERT prediction: %s (probs=%s)", pred_label, probs.tolist())
        return pred_label, probs

