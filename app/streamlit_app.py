from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news.config import ARTIFACTS_DIR, UNCERTAINTY_THRESHOLD
from fake_news.model import load_artifacts, predict_texts


st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

st.title("Fake News Detection & Responsible AI Dashboard")
st.caption("TF-IDF + Logistic Regression with confidence-based uncertainty handling")

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Analyze an article")
    input_text = st.text_area(
        "Paste a news article or claim",
        height=240,
        placeholder="Enter article text here...",
    )

    analyze_clicked = st.button("Analyze", type="primary")

with right_col:
    st.subheader("Model status")
    if (ARTIFACTS_DIR / "model.joblib").exists():
        try:
            model, metadata, evaluation, checksums = load_artifacts()
            st.success("Artifacts verified successfully")
            metric_cols = st.columns(2)
            metric_cols[0].metric("Training rows", metadata.training_rows)
            metric_cols[1].metric("Test rows", metadata.test_rows)
            metric_cols = st.columns(2)
            metric_cols[0].metric("Coverage", f"{metadata.coverage:.1%}")
            metric_cols[1].metric("Accuracy", f"{metadata.accuracy:.1%}")
            st.caption(f"Checksum entries: {len(checksums)}")
        except Exception as exc:  # pragma: no cover - surfaced in UI
            st.error(f"Artifact verification failed: {exc}")
            model = None
            metadata = None
            evaluation = {}
    else:
        st.info("Train the model first with `python scripts/train.py`.")
        model = None
        metadata = None
        evaluation = {}

if analyze_clicked:
    if not input_text.strip():
        st.warning("Enter article text before analyzing.")
    elif model is None:
        st.warning("Train and verify the model artifacts first.")
    else:
        result = predict_texts(model, [input_text], threshold=UNCERTAINTY_THRESHOLD)[0]
        probabilities = model.predict_proba([input_text])[0]
        classes = list(model.classes_)

        st.subheader("Prediction")
        display = pd.DataFrame(
            {
                "label": classes,
                "probability": probabilities,
            }
        ).sort_values("probability", ascending=False)
        label = str(result["label"])
        confidence = float(result["confidence"])

        if label == "UNCERTAIN":
            st.warning(f"UNCERTAIN with confidence {confidence:.1%}")
        elif label == "REAL":
            st.success(f"REAL with confidence {confidence:.1%}")
        else:
            st.error(f"FAKE with confidence {confidence:.1%}")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(display["label"], display["probability"], color=["#0f766e", "#b91c1c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Class probabilities")
        st.pyplot(fig, clear_figure=True)

st.subheader("Evaluation summary")
if (ARTIFACTS_DIR / "evaluation.json").exists():
    _, _, evaluation, _ = load_artifacts()
    summary_cols = st.columns(3)
    summary_cols[0].metric("Overall accuracy", f"{float(evaluation['overall_accuracy']):.1%}")
    summary_cols[1].metric("Covered accuracy", f"{float(evaluation['covered_accuracy']):.1%}")
    summary_cols[2].metric("Coverage", f"{float(evaluation['coverage']):.1%}")
else:
    st.info("Run training to generate evaluation metrics.")
