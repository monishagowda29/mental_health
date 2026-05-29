import unittest
from unittest.mock import MagicMock, patch
from PIL import Image
from src.services.vision import GroqVisionService

class TestGroqVisionService(unittest.TestCase):

    @patch("src.services.vision.Groq")
    @patch("src.services.vision.Config")
    def setUp(self, mock_config, mock_groq_class):
        # Configure Config mocks
        mock_config.GROQ_API_KEY = "gsk_test_key_1234567890"
        mock_config.GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
        
        # Instantiate GroqVisionService with mocked dependencies
        self.mock_groq_client = MagicMock()
        mock_groq_class.return_value = self.mock_groq_client
        self.service = GroqVisionService()

    def test_missing_api_key_raises_value_error(self):
        """Verify that an empty API key raises a ValueError during initialization."""
        with patch("src.services.vision.Config") as mock_config:
            mock_config.GROQ_API_KEY = ""
            with self.assertRaises(ValueError) as context:
                GroqVisionService()
            self.assertIn("GROQ_API_KEY is missing", str(context.exception))

    def test_invalid_arguments_raise_value_error(self):
        """Verify that invalid arguments raise a ValueError in analyze_image."""
        # Test None image
        with self.assertRaises(ValueError) as context:
            self.service.analyze_image(None, "analyze this") # type: ignore
        self.assertIn("valid PIL Image object", str(context.exception))

        # Test empty prompt
        img = Image.new("RGB", (100, 100), color="red")
        with self.assertRaises(ValueError) as context:
            self.service.analyze_image(img, "  ")
        self.assertIn("Prompt must be a non-empty string", str(context.exception))

    def test_analyze_image_successful(self):
        """Verify standard successful image conversion and Groq SDK payload structure."""
        # Create a simple dummy red image
        img = Image.new("RGB", (100, 100), color="red")
        
        # Configure API mock responses
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Analysis result text"
        self.mock_groq_client.chat.completions.create.return_value = mock_completion

        # Execute analysis
        result = self.service.analyze_image(img, "Describe this color")

        self.assertEqual(result, "Analysis result text")
        
        # Verify Groq chat completions was called with the correct model and format
        self.mock_groq_client.chat.completions.create.assert_called_once()
        call_kwargs = self.mock_groq_client.chat.completions.create.call_args[1]
        
        self.assertEqual(call_kwargs["model"], "meta-llama/llama-4-scout-17b-16e-instruct")
        self.assertEqual(len(call_kwargs["messages"]), 1)
        
        message = call_kwargs["messages"][0]
        self.assertEqual(message["role"], "user")
        self.assertEqual(len(message["content"]), 2)
        
        self.assertEqual(message["content"][0]["type"], "text")
        self.assertEqual(message["content"][0]["text"], "Describe this color")
        
        self.assertEqual(message["content"][1]["type"], "image_url")
        self.assertTrue(message["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,"))

    def test_image_mode_conversion_rgba_to_rgb(self):
        """Verify that RGBA images are converted to RGB prior to base64 JPEG encoding."""
        # Create a 4-channel dummy image (RGBA)
        rgba_img = Image.new("RGBA", (100, 100), color=(255, 0, 0, 128))
        
        # Configure API mocks
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Conversion succeeded"
        self.mock_groq_client.chat.completions.create.return_value = mock_completion

        # This should execute cleanly without throwing a "cannot write mode RGBA as JPEG" PIL error
        result = self.service.analyze_image(rgba_img, "Verify mode conversion")
        
        self.assertEqual(result, "Conversion succeeded")

    def test_groq_api_failure_propagation(self):
        """Verify that Groq API completions errors are cleanly propagated."""
        img = Image.new("RGB", (100, 100), color="blue")
        
        # Simulate an API error (e.g. rate limit, auth crash)
        self.mock_groq_client.chat.completions.create.side_effect = Exception("Groq API rate limit reached")

        with self.assertRaises(RuntimeError) as context:
            self.service.analyze_image(img, "Verify exception safety")
            
        self.assertIn("Groq Vision analysis failed", str(context.exception))

if __name__ == "__main__":
    unittest.main()
