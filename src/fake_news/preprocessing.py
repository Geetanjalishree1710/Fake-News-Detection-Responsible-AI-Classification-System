import re
from typing import Iterable

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
NON_WORD_PATTERN = re.compile(r"[^a-z0-9\s]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NON_WORD_PATTERN.sub(" ", text)
    text = WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def preprocess_series(values: pd.Series) -> pd.Series:
    return values.fillna("").map(clean_text)


def normalize_texts(values: Iterable[object]) -> list[str]:
    return [clean_text(value) for value in values]
