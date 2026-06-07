"""
app/utils/anonymization.py
Utility script to redact personal identifiers from clinical notes.
"""
import re

# Regular expressions for Indian identity documents and contact coordinates
ADHAAR_REGEX = r"\b[2-9]\d{3}\s\d{4}\s\d{4}\b"
EMAIL_REGEX = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
PHONE_REGEX = r"\b(?:\+91|91)?[6-9]\d{9}\b"

def deidentify_text(text: str) -> str:
    """
    Scans text and redacts sensitive PII to guarantee data sovereignty.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    cleaned = re.sub(ADHAAR_REGEX, "[REDACTED_AADHAAR]", cleaned)
    cleaned = re.sub(EMAIL_REGEX, "[REDACTED_EMAIL]", cleaned)
    cleaned = re.sub(PHONE_REGEX, "[REDACTED_PHONE]", cleaned)
    return cleaned
