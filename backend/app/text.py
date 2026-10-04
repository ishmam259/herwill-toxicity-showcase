"""Text helpers, copied from the competition code so the back end stays self-contained.

`normalize` matches UAP-Datathon/Scripts/common.py (what every model was trained on);
`script_bucket` matches kernel/text_utils.py (the 3-way script badge).
"""
import re
import unicodedata

BANGLA = re.compile("[ঀ-৿]")
LATIN = re.compile(r"[A-Za-z]")
MAX_CHARS = 2000


def normalize(text: str) -> str:
    """NFC, collapse whitespace, cap length. Keeps case, emoji and punctuation."""
    t = unicodedata.normalize("NFC", str(text))
    t = re.sub(r"\s+", " ", t).strip()
    return t[:MAX_CHARS]


def script_bucket(text: str) -> str:
    """'bangla' (>=80% Bangla letters), 'latin' (<=20%; English or Banglish), 'mixed', or 'none'."""
    n_bn, n_lat = len(BANGLA.findall(text)), len(LATIN.findall(text))
    if n_bn + n_lat == 0:
        return "none"
    share = n_bn / (n_bn + n_lat)
    return "bangla" if share >= 0.8 else "latin" if share <= 0.2 else "mixed"


def word_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each whitespace-separated token, so the UI can highlight the original string."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
