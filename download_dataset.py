"""
=============================================================================
Kaggle Dataset Downloader
=============================================================================
Downloads the Reddit Mental Health dataset from Kaggle.

SETUP (one-time):
  1. Go to https://www.kaggle.com/settings/account
  2. Click "Create New Token" → downloads kaggle.json
  3. Place kaggle.json at ~/.kaggle/kaggle.json  (Linux/Mac)
                          or C:\\Users\\<user>\\.kaggle\\kaggle.json (Windows)
  4. Run: pip install kaggle
  5. Run: python download_dataset.py
=============================================================================
"""

import os
import subprocess
import sys

DATASET_ID  = "neelghoshal/reddit-mental-health-data"
OUTPUT_DIR  = "data/"

def download_via_kaggle_cli():
    """Download using the official Kaggle CLI."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"[KAGGLE] Downloading dataset: {DATASET_ID}")
    cmd = [
        "kaggle", "datasets", "download",
        "-d", DATASET_ID,
        "-p", OUTPUT_DIR,
        "--unzip"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[ERROR] Kaggle CLI failed:\n{result.stderr}")
        print("\nTrying alternative method (kagglehub)...")
        download_via_kagglehub()
    else:
        print(f"[OK] Downloaded to: {OUTPUT_DIR}")
        list_files()


def download_via_kagglehub():
    """Alternative: download via kagglehub library."""
    try:
        import kagglehub
        path = kagglehub.dataset_download(DATASET_ID)
        print(f"[OK] Downloaded to: {path}")
        print("     Move the CSV files to the data/ folder.")
    except ImportError:
        print("[ERROR] kagglehub not installed. Run: pip install kagglehub")
    except Exception as e:
        print(f"[ERROR] {e}")
        print_manual_instructions()


def print_manual_instructions():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              MANUAL DOWNLOAD INSTRUCTIONS                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Visit: https://www.kaggle.com/datasets/               ║
║            neelghoshal/reddit-mental-health-data             ║
║                                                              ║
║  2. Click "Download" (you need a free Kaggle account)        ║
║                                                              ║
║  3. Unzip and place the CSV file at:                         ║
║     mental_health_bert/data/reddit_mental_health.csv         ║
║                                                              ║
║  4. The CSV should have columns:                             ║
║     - 'text' or 'post_text'  (post content)                  ║
║     - 'subreddit'             (r/depression, r/Anxiety etc.) ║
║                                                              ║
║  ALTERNATIVE DATASET (if above is unavailable):             ║
║  https://www.kaggle.com/datasets/reihanenamdari/             ║
║  mental-health-corpus                                        ║
╚══════════════════════════════════════════════════════════════╝
""")


def list_files():
    print("\n[DATA] Files in data/ directory:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"  {f}  ({size/1024:.1f} KB)")


if __name__ == "__main__":
    download_via_kaggle_cli()
