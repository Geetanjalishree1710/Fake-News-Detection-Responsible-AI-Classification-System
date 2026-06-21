from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    DEFAULT_INPUT_PATH,
    LABEL_COLUMN,
    KNOWN_LABELS,
    RANDOM_STATE,
    SAMPLE_DATA_PATH,
    SOURCE_COLUMN,
    TEXT_COLUMN,
)
from .preprocessing import preprocess_series


@dataclass
class LeakageReport:
    shared_text_count: int
    shared_text_examples: list[str]


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = Path(path) if path is not None else DEFAULT_INPUT_PATH
    if not dataset_path.exists():
        dataset_path = SAMPLE_DATA_PATH
    frame = pd.read_csv(dataset_path)
    required_columns = {TEXT_COLUMN, LABEL_COLUMN}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return frame


def normalize_labels(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[LABEL_COLUMN] = normalized[LABEL_COLUMN].astype(str).str.upper().str.strip()
    return normalized[normalized[LABEL_COLUMN].isin(KNOWN_LABELS)].copy()


def deduplicate_news(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    cleaned[TEXT_COLUMN] = preprocess_series(cleaned[TEXT_COLUMN])
    cleaned = cleaned.drop_duplicates(subset=[TEXT_COLUMN]).reset_index(drop=True)
    return cleaned


def split_dataset(frame: pd.DataFrame, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, test_frame = train_test_split(
        frame,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=frame[LABEL_COLUMN],
    )
    return train_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def detect_leakage(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> LeakageReport:
    train_texts = set(train_frame[TEXT_COLUMN])
    test_texts = set(test_frame[TEXT_COLUMN])
    shared_texts = sorted(train_texts & test_texts)
    return LeakageReport(
        shared_text_count=len(shared_texts),
        shared_text_examples=shared_texts[:5],
    )


def source_confounding_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    if SOURCE_COLUMN not in frame.columns:
        return pd.DataFrame(columns=[SOURCE_COLUMN, "article_count", "fake_rate", "real_rate"])

    grouped = (
        frame.groupby(SOURCE_COLUMN)[LABEL_COLUMN]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=KNOWN_LABELS, fill_value=0)
        .reset_index()
    )
    grouped.columns = [SOURCE_COLUMN, "real_rate", "fake_rate"]
    grouped["article_count"] = frame.groupby(SOURCE_COLUMN).size().values
    grouped = grouped[[SOURCE_COLUMN, "article_count", "fake_rate", "real_rate"]]
    return grouped.sort_values(["fake_rate", "article_count"], ascending=[False, False]).reset_index(drop=True)
