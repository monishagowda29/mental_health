import unittest
from unittest.mock import MagicMock, patch
import threading
import torch
from src.services.translation import TranslationService

class TestTranslationService(unittest.TestCase):
    
    def setUp(self):
        # Reset singleton instance between tests if necessary for clean mocks
        TranslationService._instance = None
        self.translator = TranslationService()

    def test_singleton_behavior(self):
        """Verify that TranslationService is a strict singleton."""
        another_translator = TranslationService()
        self.assertIs(self.translator, another_translator)

    def test_get_supported_languages(self):
        """Verify that standard Indian languages are documented and returned."""
        langs = TranslationService.get_supported_languages()
        self.assertIsInstance(langs, dict)
        self.assertIn("kn", langs) # Kannada
        self.assertIn("hi", langs) # Hindi
        self.assertIn("ta", langs) # Tamil
        self.assertIn("te", langs) # Telugu
        self.assertIn("ml", langs) # Malayalam
        self.assertEqual(langs["hi"], "Hindi (हिन्दी)")

    def test_empty_and_whitespace_fallback(self):
        """Verify that empty or whitespace-only inputs return empty strings gracefully."""
        self.assertEqual(self.translator.translate(""), "")
        self.assertEqual(self.translator.translate("   "), "")

    def test_invalid_type_raises_value_error(self):
        """Verify that passing non-string objects raises a ValueError."""
        with self.assertRaises(ValueError):
            self.translator.translate(None) # type: ignore
        with self.assertRaises(ValueError):
            self.translator.translate(123) # type: ignore

    @patch("src.services.translation.AutoTokenizer.from_pretrained")
    @patch("src.services.translation.AutoModelForSeq2SeqLM.from_pretrained")
    def test_lazy_loading_and_caching(self, mock_model_class, mock_tokenizer_class):
        """Verify that the HF Seq2Seq model is loaded only once and cached."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]])
        }
        mock_tokenizer.batch_decode.return_value = ["Hello"]

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.generate.return_value = torch.tensor([[1, 2]])
        mock_model_class.return_value = mock_model

        # Call translate twice
        res1 = self.translator.translate("ನಮಸ್ಕಾರ")
        res2 = self.translator.translate("ಹೇಗಿದ್ದೀರಾ")

        self.assertEqual(res1, "Hello")
        self.assertEqual(res2, "Hello")
        
        # Verify HF model and tokenizer initializations were only called once
        mock_model_class.assert_called_once()
        mock_tokenizer_class.assert_called_once()

    @patch("src.services.translation.AutoTokenizer.from_pretrained")
    @patch("src.services.translation.AutoModelForSeq2SeqLM.from_pretrained")
    def test_translation_execution_failure_propagation(self, mock_model_class, mock_tokenizer_class):
        """Verify that translation generation execution errors are propagated as RuntimeErrors."""
        mock_tokenizer = MagicMock()
        mock_tokenizer_class.return_value = mock_tokenizer
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2]]),
            "attention_mask": torch.tensor([[1, 1]])
        }

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.generate.side_effect = Exception("Seq2Seq generation crash")
        mock_model_class.return_value = mock_model

        with self.assertRaises(RuntimeError) as context:
            self.translator.translate("ನಮಸ್ಕಾರ")
        
        self.assertIn("Offline translation failed", str(context.exception))

if __name__ == "__main__":
    unittest.main()
