import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from src.services.bert import BERTClassifierService, LABELS

class TestBERTClassifierService(unittest.TestCase):

    def setUp(self):
        # Reset singleton instance between tests if necessary for clean mocks
        BERTClassifierService._instance = None
        self.svc = BERTClassifierService()

    def test_singleton_behavior(self):
        """Verify that BERTClassifierService is a strict singleton."""
        another_svc = BERTClassifierService()
        self.assertIs(self.svc, another_svc)

    def test_invalid_input_raises_value_error(self):
        """Verify that empty or non-string inputs raise clean ValueError."""
        # Service loaded state doesn't block validation check, but let's mock it is_loaded = True
        self.svc._tokenizer = MagicMock()
        self.svc._model = MagicMock()

        with self.assertRaises(ValueError):
            self.svc.predict("")
        with self.assertRaises(ValueError):
            self.svc.predict("   ")
        with self.assertRaises(ValueError):
            self.svc.predict(None)  # type: ignore
        with self.assertRaises(ValueError):
            self.svc.predict(123)   # type: ignore

    def test_not_loaded_raises_runtime_error(self):
        """Verify that calling predict when model is not loaded raises RuntimeError."""
        self.svc._tokenizer = None
        self.svc._model = None
        
        with self.assertRaises(RuntimeError):
            self.svc.predict("Some valid text")

    def test_clinical_sentiment_calibration_override(self):
        """Verify that positive/wellness texts with NO clinical keywords bypass deep BERT model and return 'normal'."""
        # Ensure service is marked as loaded
        self.svc._tokenizer = MagicMock()
        self.svc._model = MagicMock()

        # Input: positive words only
        label, probs = self.svc.predict("i am doing good feeling nice")
        self.assertEqual(label, "normal")
        # probs order should be [depression, normal, anxiety] -> normal is index 1
        self.assertEqual(probs[1], 0.90)
        self.assertEqual(probs[0], 0.05)
        self.assertEqual(probs[2], 0.05)

        # Input: beautiful/peaceful
        label2, probs2 = self.svc.predict("What a beautiful, peaceful and wonderful day today!")
        self.assertEqual(label2, "normal")
        self.assertEqual(probs2[1], 0.90)

    @patch("src.services.bert.BertTokenizer.from_pretrained")
    @patch("src.services.bert.BertForSequenceClassification.from_pretrained")
    def test_successful_model_inference_anxiety(self, mock_model_class, mock_tokenizer_class):
        """Verify that standard clinical texts execute deep BERT model inference correctly (mocked)."""
        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Fix: chain .to(device) calls
        mock_model_class.return_value = mock_model
        
        # Load the mocked model
        self.svc.load()
        
        # Prepare mock outputs
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        
        # Mock classification outputs (logits)
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[-2.0, -1.0, 3.0]])
        mock_model.return_value = mock_outputs
        
        # Predict an anxious sentence containing clinical keyword 'anxious' (bypassing override)
        label, probs = self.svc.predict("I feel so anxious and worried today.")
        
        self.assertEqual(label, "anxiety")
        self.assertTrue(probs[2] > probs[0])
        self.assertTrue(probs[2] > probs[1])

    @patch("src.services.bert.BertTokenizer.from_pretrained")
    @patch("src.services.bert.BertForSequenceClassification.from_pretrained")
    def test_successful_model_inference_depression(self, mock_model_class, mock_tokenizer_class):
        """Verify that standard clinical texts execute deep BERT model inference correctly (mocked)."""
        # Mock tokenizer and model
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model  # Fix: chain .to(device) calls
        mock_model_class.return_value = mock_model
        
        # Load the mocked model
        self.svc.load()
        
        # Prepare mock outputs
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        
        # Mock classification outputs (logits)
        mock_outputs = MagicMock()
        mock_outputs.logits = torch.tensor([[4.0, -2.0, -3.0]])
        mock_model.return_value = mock_outputs
        
        # Predict a depressed sentence containing clinical keyword 'depressed' (bypassing override)
        label, probs = self.svc.predict("I am feeling extremely depressed and hopeless.")
        
        self.assertEqual(label, "depression")
        self.assertTrue(probs[0] > probs[1])
        self.assertTrue(probs[0] > probs[2])

if __name__ == "__main__":
    unittest.main()
