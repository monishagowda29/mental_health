import sys
import os
import torch
import numpy as np
from itertools import permutations
from transformers import BertForSequenceClassification, BertTokenizer

MODEL_DIR = "models/bert_mental_health"

def test_permutations():
    if not os.path.exists(MODEL_DIR):
        print(f"Error: model dir {MODEL_DIR} does not exist!")
        return

    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()

    # Define test sentences with their true gold labels
    test_data = [
        # Normal
        ("I am feeling great today, very happy.", "normal"),
        ("What a nice day to go for a run in the park.", "normal"),
        ("I am doing normal things, everything is fine.", "normal"),
        ("Had a great lunch with my friends.", "normal"),
        ("i am doing good feeling nice", "normal"),
        ("Today is such a beautiful day, I went to the park, read a nice book, and had a wonderful dinner.", "normal"),
        ("I feel happy, calm, content, and peaceful today.", "normal"),
        
        # Anxiety
        ("I am so anxious and scared of what will happen next.", "anxiety"),
        ("I feel extremely worried and panicked, my heart is racing.", "anxiety"),
        ("Constant panic attacks and anxiety are ruining my life.", "anxiety"),
        ("I can't stop shaking and worrying about everything.", "anxiety"),
        ("I feel so anxious and my heart is beating incredibly fast I am having a major panic attack", "anxiety"),
        ("I am so worried and nervous about my exam, my hands are shaking and I can't sleep.", "anxiety"),
        
        # Depression
        ("I feel so depressed and hopeless, I want to die.", "depression"),
        ("There is no point in living anymore, everything is dark.", "depression"),
        ("Severe depression and sadness have taken over my life.", "depression"),
        ("I want to end my life, I have no motivation to do anything.", "depression"),
        ("I want to kill myself, I have no reason to live, everything is hopeless and dark", "depression"),
        ("I feel so depressed, empty, sad, and hopeless every single day, nothing makes me happy.", "depression"),
        ("I have lost all interest in life, I just want to lie in bed and cry all day.", "depression")
    ]

    # Predict index for each test sentence
    predictions = []
    for text, gold_label in test_data:
        enc = tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        pred_idx = int(np.argmax(probs))
        predictions.append((text, gold_label, pred_idx, probs.tolist()))

    # Try all 6 permutations of labels mapping: [label_at_idx_0, label_at_idx_1, label_at_idx_2]
    all_classes = ["anxiety", "depression", "normal"]
    best_acc = -1
    best_perm = None

    print("\nEvaluating all 6 possible label mappings:")
    print("="*60)
    for perm in permutations(all_classes):
        correct = 0
        for text, gold_label, pred_idx, probs in predictions:
            pred_label = perm[pred_idx]
            if pred_label == gold_label:
                correct += 1
        acc = correct / len(test_data)
        print(f"Mapping [0={perm[0]:<10}, 1={perm[1]:<10}, 2={perm[2]:<10}] | Accuracy: {acc*100:.2f}% ({correct}/{len(test_data)})")
        if acc > best_acc:
            best_acc = acc
            best_perm = perm
    print("="*60)
    print(f"BEST MAPPING: [0={best_perm[0]}, 1={best_perm[1]}, 2={best_perm[2]}] with Accuracy {best_acc*100:.2f}%")
    print("="*60)

    # Print detailed classification under the best mapping
    print("\nDetailed predictions under the BEST mapping:")
    print("="*100)
    print(f"{'Text':<55} | {'Gold':<10} | {'Idx':<4} | {'Pred':<10} | {'Match':<5}")
    print("="*100)
    for text, gold_label, pred_idx, probs in predictions:
        pred_label = best_perm[pred_idx]
        trunc_text = text[:52] + "..." if len(text) > 52 else text
        match_str = "YES" if pred_label == gold_label else "NO"
        print(f"{trunc_text:<55} | {gold_label:<10} | {pred_idx:<4} | {pred_label:<10} | {match_str:<5}")
    print("="*100)

if __name__ == "__main__":
    test_permutations()
