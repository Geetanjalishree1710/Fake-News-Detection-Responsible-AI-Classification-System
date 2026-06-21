from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news.config import ARTIFACTS_DIR, LABEL_COLUMN, TEXT_COLUMN
from fake_news.data import (
    detect_leakage,
    deduplicate_news,
    load_dataset,
    normalize_labels,
    source_confounding_analysis,
    split_dataset,
)
from fake_news.evaluation import save_dataframe_report, save_json_report
from fake_news.model import ArtifactMetadata, evaluate_predictions, save_artifacts, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fake news classifier")
    parser.add_argument("--input", type=str, default=None, help="Path to a CSV dataset")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.input)
    frame = normalize_labels(frame)
    frame = deduplicate_news(frame)

    training_frame, test_frame = split_dataset(frame, test_size=args.test_size)
    leakage = detect_leakage(training_frame, test_frame)
    confounding = source_confounding_analysis(frame)

    model = train_model(training_frame[TEXT_COLUMN], training_frame[LABEL_COLUMN])
    probabilities = model.predict_proba(test_frame[TEXT_COLUMN])
    evaluation = evaluate_predictions(test_frame[LABEL_COLUMN], probabilities, list(model.classes_))

    metadata = ArtifactMetadata(
        threshold=0.55,
        classes=list(model.classes_),
        training_rows=len(training_frame),
        test_rows=len(test_frame),
        accuracy=float(evaluation["overall_accuracy"]),
        coverage=float(evaluation["coverage"]),
    )

    save_artifacts(model, metadata, evaluation, ARTIFACTS_DIR)
    save_json_report(
        {
            "shared_text_count": leakage.shared_text_count,
            "shared_text_examples": leakage.shared_text_examples,
        },
        "leakage_report.json",
    )
    save_dataframe_report(confounding, "source_confounding.csv")

    print(f"Artifacts written to {ARTIFACTS_DIR}")


if __name__ == "__main__":
    main()
