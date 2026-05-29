# Data Directory

This folder holds the training dataset used to fine-tune the BERT model.
It is **not tracked in git** (see `.gitignore`).

## Required File

| Filename | Description |
|---|---|
| `reddit_mental_health.csv` | Reddit Mental Health Dataset from Kaggle |

## How to Download

```bash
python download_dataset.py
```

This script downloads and places the file in `data/reddit_mental_health.csv` automatically.

Or download manually from Kaggle:
- **Dataset**: [Reddit Mental Health Data](https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data)
- Place the CSV file in this `data/` folder

## Schema

See `data/schema.json` for the expected column format.
