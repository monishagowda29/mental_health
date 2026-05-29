import base64
import io
import logging
import threading
from typing import Optional
from PIL import Image
from groq import Groq
from src.config import Config

# Configure logging for the vision service module
logger = logging.getLogger(__name__)

class GroqVisionService:
    """
    Multimodal analysis service using the official Groq Python SDK.
    Designed for high throughput, absolute thread safety, and robust error management.
    """
    
    def __init__(self):
        """
        Initializes the GroqVisionService and verifies dependencies.
        Pulls configuration parameters cleanly from src.config.Config.
        """
        logger.info("Initializing GroqVisionService instance...")
        
        # Access the API key cleanly from the centralized configuration class
        api_key = getattr(Config, "GROQ_API_KEY", "")
        if not api_key:
            logger.critical("GROQ_API_KEY is not defined in src.config.Config. Service cannot start.")
            raise ValueError("GROQ_API_KEY is missing from the configuration environment.")
            
        # Retrieve the vision model identifier
        self.model = getattr(Config, "GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
        
        try:
            # The Groq client is thread-safe as it coordinates requests via httpx.Client,
            # which maintains an internal thread-safe connection pool.
            self.client = Groq(api_key=api_key)
            logger.info("Groq SDK client successfully instantiated. Model set to: %s", self.model)
        except Exception as e:
            logger.error("Failed to initialize the Groq client: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to instantiate Groq client: {e}") from e

    def analyze_image(self, pil_image: Image.Image, prompt: str) -> str:
        """
        Processes a PIL Image object, encodes it to an in-memory base64 JPEG string,
        and executes a multimodal chat completion request to the Groq API.
        
        Args:
            pil_image (Image.Image): A valid PIL Image object (e.g. medical chart, log sheet).
            prompt (str): The textual prompt guiding the model's analytical focus.
            
        Returns:
            str: The textual response containing the model's analysis.
            
        Raises:
            ValueError: If the inputs are invalid or missing.
            RuntimeError: If image encoding or Groq API connection fails.
        """
        if pil_image is None:
            logger.error("NoneType image passed to analyze_image.")
            raise ValueError("A valid PIL Image object must be provided.")
            
        if not prompt or not prompt.strip():
            logger.error("Empty or invalid prompt passed to analyze_image.")
            raise ValueError("Prompt must be a non-empty string.")

        logger.info("Starting image analysis with prompt (length: %d chars)...", len(prompt))

        # ── 1. In-Memory Image Format Conversion & Base64 Encoding ──
        try:
            buffered = io.BytesIO()
            
            # Check and convert color modes. JPEG does not support alpha channels (RGBA)
            # or palettized images (P). Convert these to standard 3-channel RGB.
            if pil_image.mode in ("RGBA", "LA", "P"):
                logger.info("Converting PIL Image mode from %s to RGB for JPEG compatibility.", pil_image.mode)
                pil_image = pil_image.convert("RGB")
            
            # Save the image in-memory as a JPEG
            pil_image.save(buffered, format="JPEG")
            img_bytes = buffered.getvalue()
            
            if not img_bytes:
                logger.error("Encoded JPEG byte stream is empty.")
                raise RuntimeError("Failed to generate JPEG byte stream from PIL Image.")
                
            base64_string = base64.b64encode(img_bytes).decode("utf-8")
            logger.info("PIL image successfully converted to base64 JPEG string (encoded size: %d chars).", len(base64_string))
            
        except Exception as e:
            logger.exception("Failed to encode PIL Image to base64 JPEG byte stream: %s", str(e))
            raise RuntimeError(f"Failed to process and encode PIL image: {e}") from e

        # ── 2. Construct API Payload and Query Groq SDK ──
        try:
            logger.info("Sending chat completion request to Groq API using model: %s...", self.model)
            
            # Call Groq completions endpoint
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_string}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1024,
            )
            
            # Validate response payload structure
            if not response or not response.choices or not response.choices[0].message:
                logger.error("Groq API returned an empty or invalid response structure: %s", response)
                raise RuntimeError("Invalid API response format received from Groq.")
                
            result_text = response.choices[0].message.content
            if not result_text:
                logger.warning("Groq vision analysis completed, but returned an empty completion string.")
                return ""
                
            logger.info("Groq vision analysis completed successfully.")
            return result_text
            
        except Exception as e:
            # Log the specific exception with traceback details and propagate it
            logger.exception("A critical exception occurred during the Groq API call: %s", str(e))
            raise RuntimeError(f"Groq Vision analysis failed: {e}") from e
