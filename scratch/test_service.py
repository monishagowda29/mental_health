import sys
import os
import numpy as np

# Adjust search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.bert import BERTClassifierService

def test_service_predictions():
    svc = BERTClassifierService()
    svc.load()

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

    print("\nDetailed predictions under the updated BERT Classifier Service:")
    print("="*120)
    print(f"{'Text':<55} | {'Gold':<10} | {'Pred Label':<10} | {'Match':<5} | {'Probs (Dep, Norm, Anx)':<25}")
    print("="*120)
    correct = 0
    for text, gold_label in test_data:
        label, probs = svc.predict(text)
        match_str = "YES" if label == gold_label else "NO"
        if label == gold_label:
            correct += 1
        trunc_text = text[:52] + "..." if len(text) > 52 else text
        probs_str = f"[{probs[0]:.4f}, {probs[1]:.4f}, {probs[2]:.4f}]"
        print(f"{trunc_text:<55} | {gold_label:<10} | {label:<10} | {match_str:<5} | {probs_str:<25}")
    print("="*120)
    acc = correct / len(test_data)
    print(f"OVERALL ACCURACY: {acc*100:.2f}% ({correct}/{len(test_data)})")
    print("="*120)

if __name__ == "__main__":
    test_service_predictions()
