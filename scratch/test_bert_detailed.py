import sys
import os
import torch
import numpy as np
from transformers import BertForSequenceClassification, BertTokenizer

# Adjust search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = "models/bert_mental_health"

def run_detailed_test():
    print(f"Loading model from: {MODEL_DIR}")
    if not os.path.exists(MODEL_DIR):
        print(f"Error: model dir {MODEL_DIR} does not exist!")
        return

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    test_sentences = [
        # Normal sentences
        "i am doing good feeling nice",
        "Today is such a beautiful day, I went to the park, read a nice book, and had a wonderful dinner.",
        "I feel happy, calm, content, and peaceful today.",
        
        # Anxious sentences
        "I feel so anxious and my heart is beating incredibly fast I am having a major panic attack",
        "I am so worried and nervous about my exam, my hands are shaking and I can't sleep.",
        "I constantly feel tense, uneasy, and panicked about the future.",
        
        # Depressed sentences
        "I want to kill myself, I have no reason to live, everything is hopeless and dark",
        "I feel so depressed, empty, sad, and hopeless every single day, nothing makes me happy.",
        "I have lost all interest in life, I just want to lie in bed and cry all day."
    ]

    print("\n" + "="*80)
    print(f"{'Text':<50} | {'Idx 0':<8} | {'Idx 1':<8} | {'Idx 2':<8}")
    print("="*80)
    for text in test_sentences:
        enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        
        # truncated text
        trunc_text = text[:47] + "..." if len(text) > 47 else text
        print(f"{trunc_text:<50} | {probs[0]:.4f}   | {probs[1]:.4f}   | {probs[2]:.4f}")
    print("="*80)

if __name__ == "__main__":
    run_detailed_test()
