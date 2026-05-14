import re
import pandas as pd
from urllib.parse import urlparse

SUSPICIOUS_WORDS = [
    'verify', 'account', 'urgent', 'click', 'login', 'password',
    'bank', 'suspended', 'confirm', 'security', 'update', 'limited',
    'winner', 'free', 'prize', 'claim'
]

SHORTENERS = ['bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'buff.ly']


def safe_parse_url(url_text: str):
    if not url_text:
        return None

    candidate = str(url_text).strip()
    candidate = candidate.strip("\"'<>[](){}.,;")

    if not candidate:
        return None

    if not candidate.startswith(("http://", "https://")):
        candidate = f"http://{candidate}"

    try:
        return urlparse(candidate)
    except ValueError:
        return None


def extract_features(text: str):
    text = str(text) if text is not None else ""
    lower_text = text.lower()

    urls = re.findall(r'(https?://[^\s]+|www\.[^\s]+)', text, flags=re.IGNORECASE)
    first_url = urls[0] if urls else None
    parsed = safe_parse_url(first_url)

    suspicious_word_count = sum(1 for w in SUSPICIOUS_WORDS if w in lower_text)
    url_count = len(urls)
    exclamation_count = text.count('!')
    digit_count = sum(ch.isdigit() for ch in text)

    has_ip_url = 0
    uses_shortener = 0

    if parsed:
        netloc = (parsed.netloc or "").lower()
        has_ip_url = int(bool(re.search(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', netloc)))
        uses_shortener = int(any(shortener in netloc for shortener in SHORTENERS))

    return {
        'text': text,
        'char_count': len(text),
        'word_count': len(text.split()),
        'digit_count': digit_count,
        'url_count': url_count,
        'exclamation_count': exclamation_count,
        'suspicious_word_count': suspicious_word_count,
        'has_ip_url': has_ip_url,
        'uses_shortener': uses_shortener,
    }


def build_feature_frame(texts):
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows)