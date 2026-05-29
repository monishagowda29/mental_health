import os
from pathlib import Path
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from dotenv import load_dotenv

# Locate the root path of the project (parent of src)
ROOT_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT_DIR / ".env"

# Load the environment variables from the .env file if it exists
if ENV_PATH.exists():
    load_dotenv(dotenv_path=str(ENV_PATH))
else:
    load_dotenv()

class Config:
    """Centralized configuration manager for the MindScan platform."""
    
    # ── General Settings ──
    DEBUG = os.environ.get("DEBUG", "False").lower() in ("true", "1", "t")
    
    # ── Path Configurations ──
    ROOT_PATH = ROOT_DIR
    MODEL_DIR = os.environ.get("MODEL_DIR", str(ROOT_DIR / "models" / "bert_mental_health"))
    
    # ── Hardware Settings ──
    DEVICE = "cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu"
    
    # ── External APIs & Models ──
    # Groq API Configuration
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
    
    # Hugging Face Settings
    HF_TRANSLATION_MODEL = os.environ.get("HF_TRANSLATION_MODEL", "Helsinki-NLP/opus-mt-mul-en")
