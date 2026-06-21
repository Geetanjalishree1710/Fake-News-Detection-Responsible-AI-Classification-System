from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news.data import deduplicate_news, detect_leakage, source_confounding_analysis
from fake_news.model import classify_probabilities
from fake_news.preprocessing import clean_text


def test_clean_text_removes_noise() -> None:
    assert clean_text(" Visit https://example.com NOW!!! ") == "visit now"


def test_deduplicate_news_removes_duplicate_texts() -> None:
    frame = pd.DataFrame(
        {
            "text": ["Breaking news", "Breaking news", "Another story"],
            "label": ["REAL", "REAL", "FAKE"],
        }
    )
    cleaned = deduplicate_news(frame)
    assert len(cleaned) == 2


def test_detect_leakage_finds_shared_texts() -> None:
    train_frame = pd.DataFrame({"text": ["same story", "different"], "label": ["REAL", "FAKE"]})
    test_frame = pd.DataFrame({"text": ["same story"], "label": ["REAL"]})
    report = detect_leakage(train_frame, test_frame)
    assert report.shared_text_count == 1


def test_source_confounding_analysis_returns_source_summary() -> None:
    frame = pd.DataFrame(
        {
            "text": ["a", "b", "c"],
            "label": ["REAL", "FAKE", "FAKE"],
            "source": ["site-a", "site-a", "site-b"],
        }
    )
    summary = source_confounding_analysis(frame)
    assert list(summary.columns) == ["source", "article_count", "fake_rate", "real_rate"]


def test_uncertain_prediction_threshold() -> None:
    predictions = classify_probabilities(
        probabilities=np.array([[0.5, 0.5]]),
        class_labels=["FAKE", "REAL"],
        threshold=0.75,
    )
    assert predictions[0]["label"] == "UNCERTAIN"
