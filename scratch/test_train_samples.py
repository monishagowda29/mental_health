import sys
import os
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_DIR = "models/bert_mental_health"

def test_samples():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    samples = [
        "I have been feeling really happy lately, everything is going well in my life!",
        "I can't stop worrying about everything. My heart races and I feel like something bad is going to happen.",
        "I don't see the point of anything anymore. I feel empty and hopeless every single day.",
    ]

    print("\n" + "="*80)
    for text in samples:
        enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
        print(f"Text: '{text}'")
        print(f"Logits: {logits.numpy()[0].tolist()}")
        print(f"Probs:  {probs.tolist()}")
        print(f"Max Prob Index: {np.argmax(probs)}")
        print("="*80)

if __name__ == "__main__":
    test_samples()
