import sys
import os
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_DIR = "models/bert_mental_health"

def test_neutral():
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    samples = [
        # General mental health / self improvement / motivation
        "I am working on self improvement and trying to stay motivated.",
        "Just wanted to post a quick update on my mental health journey.",
        "How do you stay motivated when you have a lot of work to do?",
        "I want to improve my daily habits and be more productive.",
        "Sharing some positive thoughts about my personal growth.",
        
        # Simple neutral sentences
        "The sky is blue and the grass is green.",
        "I am going to the grocery store to buy some milk.",
        "What is the weather like today?",
        "I am writing some code on my computer."
    ]

    print("\n" + "="*80)
    for text in samples:
        enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
        print(f"Text: '{text}'")
        print(f"Probs:  Index 0 (Dep): {probs[0]:.4f} | Index 1 (Norm): {probs[1]:.4f} | Index 2 (Anx): {probs[2]:.4f}")
        print("="*80)

if __name__ == "__main__":
    test_neutral()
