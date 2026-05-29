import unittest
from unittest.mock import MagicMock, patch
import threading
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

    @patch("src.services.translation.pipeline")
    def test_lazy_loading_and_caching(self, mock_pipeline):
        """Verify that the HF pipeline is loaded only once and cached."""
        mock_pipe_instance = MagicMock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_pipe_instance.return_value = [{"translation_text": "Hello"}]

        # Call translate twice
        res1 = self.translator.translate("ನಮಸ್ಕಾರ")
        res2 = self.translator.translate("ಹೇಗಿದ್ದೀರಾ")

        self.assertEqual(res1, "Hello")
        self.assertEqual(res2, "Hello")
        
        # Verify HF pipeline initialization was only called once
        mock_pipeline.assert_called_once()

    @patch("src.services.translation.pipeline")
    def test_translation_execution_failure_propagation(self, mock_pipeline):
        """Verify that pipeline execution errors are propagated as RuntimeErrors."""
        mock_pipe_instance = MagicMock()
        mock_pipeline.return_value = mock_pipe_instance
        mock_pipe_instance.side_effect = Exception("HF pipeline internal crash")

        with self.assertRaises(RuntimeError) as context:
            self.translator.translate("ನಮಸ್ಕಾರ")
        
        self.assertIn("Offline translation pipeline execution failed", str(context.exception))

if __name__ == "__main__":
    unittest.main()
