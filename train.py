"""
=============================================================================
Mental Health Risk Prediction using BERT-based Depression/Anxiety Detection
=============================================================================
Project  : Mysore University School of Engineering — AI&DS Dept.
Authors  : Mokshitha P, Monisha M, Mubara Fathima
Dataset  : Reddit Mental Health Dataset (Kaggle)
           https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data
Model    : bert-base-uncased (fine-tuned for 3-class classification)
Classes  : 0 → Normal | 1 → Anxiety Risk | 2 → Depression Risk
=============================================================================
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────
CFG = {
    "model_name"    : "bert-base-uncased",
    "max_len"       : 128,
    "batch_size"    : 16,
    "epochs"        : 4,
    "lr"            : 2e-5,
    "warmup_ratio"  : 0.1,
    "weight_decay"  : 0.01,
    "num_classes"   : 3,
    "data_path"     : "data/reddit_mental_health.csv",   # after Kaggle download
    "output_dir"    : "outputs/",
    "model_dir"     : "models/bert_mental_health/",
    "label_col"     : "label",   # column name in dataset
    "text_col"      : "text",    # column name in dataset
}

os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs(CFG["model_dir"], exist_ok=True)

# ─────────────────────────────────────────────
# 2. DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(path: str) -> pd.DataFrame:
    """
    Loads the Kaggle Reddit Mental Health dataset.
    Actual columns: 'text' (post body), 'target' (numeric label)
    target values: 0 = normal, 1 = anxiety, 2 = depression
    """
    print(f"[DATA] Loading dataset from: {path}")
    df = pd.read_csv(path)
    print(f"[DATA] Raw shape: {df.shape}")
    print(f"[DATA] Columns: {df.columns.tolist()}")

    # ── Identify text column ──
    for candidate in ["text", "post_text", "selftext", "content", "body"]:
        if candidate in df.columns:
            df.rename(columns={candidate: "text"}, inplace=True)
            break

    # ── Identify label column ──
    # This dataset uses 'target': 0=normal, 1=anxiety, 2=depression
    TARGET_NAME_MAP = {0: "normal", 1: "anxiety", 2: "depression"}

    if "target" in df.columns:
        df["label"] = df["target"].map(TARGET_NAME_MAP)
    elif "label" in df.columns:
        pass   # already correct
    elif "subreddit" in df.columns:
        SUBREDDIT_MAP = {
            "mentalhealth": "normal", "happy": "normal",
            "selfimprovement": "normal", "motivation": "normal",
            "Anxiety": "anxiety", "PanicAttack": "anxiety",
            "socialanxiety": "anxiety", "OCD": "anxiety", "ptsd": "anxiety",
            "depression": "depression", "SuicideWatch": "depression",
            "lonely": "depression", "offmychest": "depression", "bipolar": "depression",
        }
        df["label"] = df["subreddit"].map(SUBREDDIT_MAP)
    else:
        raise ValueError(f"Cannot find a label column. Available: {df.columns.tolist()}")

    df.dropna(subset=["label", "text"], inplace=True)

    # ── Text cleaning ──
    df["text"] = (
        df["text"]
        .astype(str)
        .str.replace(r"http\S+|www\S+", "", regex=True)    # remove URLs
        .str.replace(r"[^a-zA-Z\s!?.,']", " ", regex=True) # keep alpha + basic punct
        .str.replace(r"\s+", " ", regex=True)               # collapse whitespace
        .str.strip()
        .str[:512]                                           # hard cap
    )
    df = df[df["text"].str.len() > 20]
    df.drop_duplicates(subset=["text"], inplace=True)

    # ── Encode labels ──
    le = LabelEncoder()
    df["label_id"] = le.fit_transform(df["label"])
    print(f"[DATA] Classes : {list(le.classes_)}")
    print(f"[DATA] Distribution:\n{df['label'].value_counts()}\n")

    return df, le


# ─────────────────────────────────────────────
# 3. PYTORCH DATASET
# ─────────────────────────────────────────────
class MentalHealthDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts.reset_index(drop=True)
        self.labels    = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length      = self.max_len,
            padding         = "max_length",
            truncation      = True,
            return_tensors  = "pt",
        )
        return {
            "input_ids"      : encoding["input_ids"].squeeze(0),
            "attention_mask" : encoding["attention_mask"].squeeze(0),
            "labels"         : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 4. TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []

    for batch in loader:
        optimizer.zero_grad()
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss    = outputs.loss
        logits  = outputs.logits

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        preds        = torch.argmax(logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def eval_epoch(model, loader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss
            logits  = outputs.logits

            total_loss  += loss.item()
            preds        = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_preds, all_labels


# ─────────────────────────────────────────────
# 5. EVALUATION & VISUALISATION
# ─────────────────────────────────────────────
def plot_training_curves(history: dict, save_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("BERT Training Curves — Mental Health Risk Prediction", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="#e74c3c")
    axes[0].plot(epochs, history["val_loss"],   "s-", label="Val Loss",   color="#3498db")
    axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc", color="#e74c3c")
    axes[1].plot(epochs, history["val_acc"],   "s-", label="Val Acc",   color="#3498db")
    axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved training curves → {path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names: list, save_dir: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax)
    ax.set_title("Confusion Matrix — Test Set", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
    plt.tight_layout()
    path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved confusion matrix → {path}")
    plt.close()


def plot_class_distribution(df: pd.DataFrame, save_dir: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["label"].value_counts()
    colors = ["#2ecc71", "#e67e22", "#e74c3c"]
    bars = ax.bar(counts.index, counts.values, color=colors[:len(counts)], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_title("Dataset Class Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mental Health Class"); ax.set_ylabel("Post Count")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[PLOT] Saved class distribution → {path}")
    plt.close()


def save_metrics_report(y_true, y_pred, class_names, history, save_dir):
    report = classification_report(y_true, y_pred, target_names=class_names)
    metrics = {
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro"), 4),
        "Recall"   : round(recall_score(y_true, y_pred, average="macro"), 4),
        "F1-Score" : round(f1_score(y_true, y_pred, average="macro"), 4),
    }
    path = os.path.join(save_dir, "metrics_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  MENTAL HEALTH RISK PREDICTION — BERT EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Best Validation Accuracy : {max(history['val_acc']):.4f}\n")
        f.write(f"Final Test Accuracy      : {metrics['Accuracy']}\n")
        f.write(f"Macro Precision          : {metrics['Precision']}\n")
        f.write(f"Macro Recall             : {metrics['Recall']}\n")
        f.write(f"Macro F1-Score           : {metrics['F1-Score']}\n\n")
        f.write("─" * 60 + "\n")
        f.write("PER-CLASS CLASSIFICATION REPORT\n")
        f.write("─" * 60 + "\n")
        f.write(report)
    print(f"[REPORT] Saved metrics report → {path}")
    print("\n" + "="*55)
    print("  FINAL TEST METRICS")
    print("="*55)
    for k, v in metrics.items():
        print(f"  {k:<12}: {v}")
    print("="*55)
    print(report)


# ─────────────────────────────────────────────
# 6. INFERENCE UTILITY
# ─────────────────────────────────────────────
def predict_text(text: str, model, tokenizer, le, device, max_len=128):
    """Predict mental health risk for a single text string."""
    model.eval()
    enc = tokenizer(
        text, max_length=max_len, padding="max_length",
        truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        logits = model(
            input_ids      = enc["input_ids"].to(device),
            attention_mask = enc["attention_mask"].to(device),
        ).logits

    probs     = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred_id   = int(np.argmax(probs))
    pred_label = le.inverse_transform([pred_id])[0]
    confidence = float(probs[pred_id])

    result = {
        "text"         : text[:100] + "..." if len(text) > 100 else text,
        "prediction"   : pred_label,
        "confidence"   : round(confidence * 100, 2),
        "probabilities": {cls: round(float(p) * 100, 2)
                         for cls, p in zip(le.classes_, probs)},
    }
    return result


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  MENTAL HEALTH RISK PREDICTION — BERT FINE-TUNING PIPELINE")
    print("="*60 + "\n")

    # ── Load data ──
    df, le = load_and_preprocess(CFG["data_path"])
    plot_class_distribution(df, CFG["output_dir"])

    # ── Train / Val / Test split  (70 / 15 / 15) ──
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["label_id"], random_state=SEED)
    val_df,   test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["label_id"], random_state=SEED)

    print(f"[SPLIT] Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

    # ── Tokenizer ──
    print(f"\n[MODEL] Loading tokenizer: {CFG['model_name']}")
    tokenizer = BertTokenizer.from_pretrained(CFG["model_name"])

    # ── Datasets & Loaders ──
    train_ds = MentalHealthDataset(train_df["text"], train_df["label_id"], tokenizer, CFG["max_len"])
    val_ds   = MentalHealthDataset(val_df["text"],   val_df["label_id"],   tokenizer, CFG["max_len"])
    test_ds  = MentalHealthDataset(test_df["text"],  test_df["label_id"],  tokenizer, CFG["max_len"])

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"], shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"], shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"], shuffle=False, num_workers=2)

    # ── Model ──
    print(f"[MODEL] Loading BERT: {CFG['model_name']} with {CFG['num_classes']} output classes")
    model = BertForSequenceClassification.from_pretrained(
        CFG["model_name"],
        num_labels          = CFG["num_classes"],
        ignore_mismatched_sizes = True,
    ).to(DEVICE)

    # ── Optimizer & Scheduler ──
    optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    total_steps   = len(train_loader) * CFG["epochs"]
    warmup_steps  = int(total_steps * CFG["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── Training ──
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\n[TRAIN] Starting training for {CFG['epochs']} epochs ...\n")
    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        vl_loss, vl_acc, _, _ = eval_epoch(model, val_loader, DEVICE)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"  Epoch {epoch}/{CFG['epochs']}  "
              f"| Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f}  "
              f"| Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            model.save_pretrained(CFG["model_dir"])
            tokenizer.save_pretrained(CFG["model_dir"])
            print(f"  ✓ Best model saved (val_acc={best_val_acc:.4f})")

    # ── Evaluation on test set ──
    print("\n[EVAL] Evaluating on test set ...")
    _, _, test_preds, test_labels = eval_epoch(model, test_loader, DEVICE)
    class_names = list(le.classes_)

    plot_training_curves(history, CFG["output_dir"])
    plot_confusion_matrix(test_labels, test_preds, class_names, CFG["output_dir"])
    save_metrics_report(test_labels, test_preds, class_names, history, CFG["output_dir"])

    # ── Demo inference ──
    print("\n[DEMO] Running sample predictions ...")
    samples = [
        "I have been feeling really happy lately, everything is going well in my life!",
        "I can't stop worrying about everything. My heart races and I feel like something bad is going to happen.",
        "I don't see the point of anything anymore. I feel empty and hopeless every single day.",
    ]
    for text in samples:
        result = predict_text(text, model, tokenizer, le, DEVICE)
        print(f"\n  Text       : {result['text']}")
        print(f"  Prediction : {result['prediction'].upper()}")
        print(f"  Confidence : {result['confidence']}%")
        print(f"  Probs      : {result['probabilities']}")

    print("\n[DONE] Pipeline complete! Check outputs/ folder for all artifacts.\n")


if __name__ == "__main__":
    main()
