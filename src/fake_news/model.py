from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ARTIFACTS_DIR, DISPLAY_LABELS, UNCERTAINTY_THRESHOLD
from .preprocessing import clean_text


@dataclass
class ArtifactMetadata:
    threshold: float
    classes: list[str]
    training_rows: int
    test_rows: int
    accuracy: float
    coverage: float


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=clean_text,
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=15000,
                    stop_words="english",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def classify_probabilities(
    probabilities: np.ndarray,
    class_labels: list[str],
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> list[dict[str, float | str]]:
    predictions: list[dict[str, float | str]] = []
    for row in probabilities:
        index = int(np.argmax(row))
        confidence = float(row[index])
        predicted_label = class_labels[index]
        if confidence < threshold:
            predicted_label = "UNCERTAIN"
        predictions.append(
            {
                "label": predicted_label,
                "confidence": confidence,
            }
        )
    return predictions


def evaluate_predictions(
    y_true: pd.Series,
    probabilities: np.ndarray,
    class_labels: list[str],
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> dict[str, object]:
    predictions = classify_probabilities(probabilities, class_labels, threshold)
    predicted_labels = [entry["label"] for entry in predictions]
    covered_mask = [label != "UNCERTAIN" for label in predicted_labels]

    total = len(y_true)
    covered = sum(covered_mask)
    correct = sum(
        true == predicted
        for true, predicted in zip(y_true.tolist(), predicted_labels)
        if predicted != "UNCERTAIN"
    )
    overall_accuracy = float(sum(true == predicted for true, predicted in zip(y_true.tolist(), predicted_labels)) / total) if total else 0.0
    covered_accuracy = float(correct / covered) if covered else 0.0
    coverage = float(covered / total) if total else 0.0
    return {
        "overall_accuracy": overall_accuracy,
        "covered_accuracy": covered_accuracy,
        "coverage": coverage,
        "predictions": predictions,
    }


def train_model(texts: pd.Series, labels: pd.Series) -> Pipeline:
    pipeline = build_pipeline()
    pipeline.fit(texts, labels)
    return pipeline


def save_artifacts(
    model: Pipeline,
    metadata: ArtifactMetadata,
    evaluation: dict[str, object],
    artifact_dir: Path = ARTIFACTS_DIR,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    evaluation_path = artifact_dir / "evaluation.json"
    checksums_path = artifact_dir / "checksums.json"

    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    evaluation_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")

    checksums = {
        "model.joblib": sha256_file(model_path),
        "metadata.json": sha256_file(metadata_path),
        "evaluation.json": sha256_file(evaluation_path),
    }
    checksums_path.write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    return checksums


def load_artifacts(artifact_dir: Path = ARTIFACTS_DIR) -> tuple[Pipeline, ArtifactMetadata, dict[str, object], dict[str, str]]:
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    evaluation_path = artifact_dir / "evaluation.json"
    checksums_path = artifact_dir / "checksums.json"

    checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    verify_file_hash(model_path, checksums["model.joblib"])
    verify_file_hash(metadata_path, checksums["metadata.json"])
    verify_file_hash(evaluation_path, checksums["evaluation.json"])

    model = joblib.load(model_path)
    metadata = ArtifactMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return model, metadata, evaluation, checksums


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_hash(path: Path, expected_hash: str) -> None:
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"Checksum verification failed for {path.name}")


def predict_texts(model: Pipeline, texts: list[str], threshold: float = UNCERTAINTY_THRESHOLD) -> list[dict[str, float | str]]:
    probabilities = model.predict_proba(texts)
    class_labels = list(model.classes_)
    return classify_probabilities(probabilities, class_labels, threshold)


def ensure_display_labels(labels: list[str]) -> list[str]:
    ordered = [label for label in DISPLAY_LABELS if label in labels or label == "UNCERTAIN"]
    return ordered
