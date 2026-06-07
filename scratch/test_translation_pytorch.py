"""
scratch/test_translation_pytorch.py
Verification script for standard PyTorch translation service.
"""
import sys
from pathlib import Path

# Adjust path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.services.translation import TranslationService

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing TranslationService (PyTorch)...")
    svc = TranslationService()
    
    test_text = "ನನಗೆ ತುಂಬಾ ಬೇಜಾರಾಗಿದೆ"
    print(f"\nTranslating Kannada text: '{test_text}'")
    
    translated = svc.translate(test_text, source_lang="kn")
    print(f"Translated English Output: '{translated}'")

if __name__ == "__main__":
    main()
