"""
scratch/test_translation_onnx.py
Verification script for translation ONNX service.
"""
import sys
from pathlib import Path

# Adjust path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.services.translator_onnx import TranslationOnnxService

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("Initializing TranslationOnnxService...")
    svc = TranslationOnnxService()
    
    # "ನನಗೆ ತುಂಬಾ ಬೇಜಾರಾಗಿದೆ" is Kannada for "I am very sad / disappointed"
    test_text = "ನನಗೆ ತುಂಬಾ ಬೇಜಾರಾಗಿದೆ"
    print(f"\nTranslating Kannada text: '{test_text}'")
    
    translated = svc.translate(test_text, source_lang="kn")
    print(f"Translated English Output: '{translated}'")

if __name__ == "__main__":
    main()
