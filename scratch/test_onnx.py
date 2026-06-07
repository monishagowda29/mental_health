"""
scratch/test_onnx.py
Verification script for quantized BERT ONNX service.
"""
import sys
from pathlib import Path

# Adjust path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.services.bert_onnx import BERTOnnxService

def main():
    print("Initializing BERTOnnxService...")
    svc = BERTOnnxService()
    
    test_text = "I feel incredibly anxious and worried that I might fail this project."
    print(f"\nAnalyzing Text: '{test_text}'")
    
    label, probs = svc.predict(test_text)
    print(f"Prediction Label: {label}")
    print(f"Probabilities: {probs.tolist()}")
    print("Normal (stable) score:", probs[1])

if __name__ == "__main__":
    main()
