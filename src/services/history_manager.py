"""
src/services/history_manager.py
Thread-safe, offline patient history logging manager.
Persists records locally on disk in outputs/history.json, scoped by patient_name as a primary key.
"""
import os
import json
import logging
import threading
from datetime import datetime

logger = logging.getLogger("MindScan.History")

HISTORY_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "outputs",
    "history.json"
)

class HistoryManager:
    """Offline, thread-safe manager for saving and loading scoped patient history records."""
    
    _lock = threading.Lock()

    @classmethod
    def _load_raw_data(cls) -> list[dict]:
        """Loads all raw records from outputs/history.json."""
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                return json.loads(content)
        except Exception as exc:
            logger.error("Failed to load history file: %s", exc)
            return []

    @classmethod
    def _save_raw_data(cls, data: list[dict]) -> None:
        """Saves all raw records to outputs/history.json."""
        try:
            os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as exc:
            logger.error("Failed to save history file: %s", exc)

    @classmethod
    def save_record(cls, patient_name: str, text: str, label: str, confidence: str, lang: str) -> None:
        """Saves a new screening record scoped by patient_name."""
        with cls._lock:
            data = cls._load_raw_data()
            new_record = {
                "patient_name": patient_name.strip() or "Guest Patient",
                "timestamp": datetime.now().isoformat(),
                "text": text.strip(),
                "label": label.strip(),
                "confidence": confidence.strip(),
                "lang": lang.strip().upper()
            }
            data.append(new_record)
            cls._save_raw_data(data)

    @classmethod
    def get_history(cls, patient_name: str, limit: int = 20) -> list[dict]:
        """Loads screening logs matching the active patient name, sorted by descending timestamp."""
        with cls._lock:
            data = cls._load_raw_data()
            target = patient_name.strip() or "Guest Patient"
            filtered = [r for r in data if r.get("patient_name") == target]
            # Sort by timestamp descending
            filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            return filtered[:limit]

    @classmethod
    def clear_history(cls, patient_name: str) -> None:
        """Removes only log entries matching the specified patient name."""
        with cls._lock:
            data = cls._load_raw_data()
            target = patient_name.strip() or "Guest Patient"
            cleaned = [r for r in data if r.get("patient_name") != target]
            cls._save_raw_data(cleaned)

    @classmethod
    def export_csv(cls, patient_name: str) -> str:
        """Generates a CSV string containing all logs for the specified patient."""
        with cls._lock:
            data = cls._load_raw_data()
            target = patient_name.strip() or "Guest Patient"
            filtered = [r for r in data if r.get("patient_name") == target]
            filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            csv_lines = ["Timestamp,Linguistic AI Result,Confidence,Language,Analyzed Text Snippet"]
            for r in filtered:
                # Escape commas and quotes for CSV safety
                clean_text = r['text'].replace('"', '""')
                csv_lines.append(
                    f"{r['timestamp']},{r['label'].upper()},{r['confidence']},{r['lang']},\"{clean_text}\""
                )
            return "\n".join(csv_lines)
