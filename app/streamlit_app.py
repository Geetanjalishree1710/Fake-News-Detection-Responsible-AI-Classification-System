from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fake_news.config import ARTIFACTS_DIR, UNCERTAINTY_THRESHOLD
from fake_news.insights import (
    assess_source_credibility,
    build_fact_check_links,
    build_pdf_report,
    explain_prediction,
    extract_summary,
)
from fake_news.model import load_artifacts, predict_texts


st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")

st.title("Fake News Detection & Responsible AI Dashboard")
st.caption("TF-IDF + Logistic Regression with confidence-based uncertainty handling")


def initialise_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def apply_theme(dark_mode: bool) -> None:
    if not dark_mode:
        return
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0b1220 0%, #111827 100%);
                color: #f9fafb;
            }
            .stMarkdown, .stText, .stCaption, label, p, h1, h2, h3, h4, h5, h6 {
                color: #f9fafb !important;
            }
            section[data-testid="stSidebar"] {
                background: #0f172a;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_probability_map(classes: list[str], probabilities: object) -> dict[str, float]:
    return {label: float(probabilities[index]) for index, label in enumerate(classes)}


def format_terms(items: list[dict[str, float | str]]) -> str:
    if not items:
        return "No strong terms found."
    return "\n".join(f"- {item['term']} ({float(item['score']):.3f})" for item in items)


def add_history_entry(entry: dict[str, object]) -> None:
    st.session_state.prediction_history.insert(0, entry)
    st.session_state.prediction_history = st.session_state.prediction_history[:100]


def render_verdict(label: str, confidence: float) -> None:
    if label == "REAL":
        headline = "Prediction for this article is REAL (TRUE)."
        banner_color = "#0f766e"
        background = "#ecfdf5"
    elif label == "FAKE":
        headline = "Prediction for this article is FAKE (FALSE)."
        banner_color = "#b91c1c"
        background = "#fef2f2"
    elif label == "UNCERTAIN":
        headline = "Prediction for this article is UNCERTAIN."
        banner_color = "#b45309"
        background = "#fffbeb"
    else:
        headline = f"Prediction for this article is {label}."
        banner_color = "#1f2937"
        background = "#f3f4f6"

    st.markdown(
        f"""
        <div style="
            padding: 1rem 1.25rem;
            border-radius: 14px;
            border: 2px solid {banner_color};
            background: {background};
            margin: 0.5rem 0 1rem 0;
        ">
            <div style="font-size: 1.4rem; font-weight: 800; color: {banner_color};">
                {headline}
            </div>
            <div style="font-size: 1rem; margin-top: 0.35rem; color: #111827;">
                Confidence: {confidence:.1%}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


initialise_state()

with st.sidebar:
    st.header("Display")
    dark_mode = st.toggle("Dark mode", value=False)
    apply_theme(dark_mode)

    st.header("Prediction")
    uncertainty_threshold = st.slider(
        "Uncertainty threshold",
        min_value=0.50,
        max_value=0.95,
        value=float(UNCERTAINTY_THRESHOLD),
        step=0.01,
        help="Predictions below this confidence become UNCERTAIN.",
    )

    st.header("Fact-check")
    st.caption("Use the links below to verify claims in external fact-check databases.")

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.subheader("Analyze an article")
    input_text = st.text_area(
        "Paste a news article or claim",
        height=240,
        placeholder="Enter article text here...",
    )
    source_url = st.text_input("Source URL (optional)", placeholder="https://example.com/article")

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
        result = predict_texts(model, [input_text], threshold=uncertainty_threshold)[0]
        probabilities = model.predict_proba([input_text])[0]
        classes = list(model.classes_)
        summary = extract_summary(input_text)
        explanation = explain_prediction(model, input_text, top_n=5)
        source_report = assess_source_credibility(source_url)
        fact_check_links = build_fact_check_links(input_text)

        st.subheader("Prediction")
        display = pd.DataFrame(
            {
                "label": classes,
                "probability": probabilities,
            }
        ).sort_values("probability", ascending=False)
        label = str(classes[int(probabilities.argmax())])
        confidence = float(result["confidence"])
        verdict_label = str(result["label"])
        probability_map = build_probability_map(classes, probabilities)
        truth_probability = float(probability_map.get("REAL", confidence if label == "REAL" else 0.0))
        false_probability = float(probability_map.get("FAKE", confidence if label == "FAKE" else 0.0))

        render_verdict(verdict_label, confidence)

        summary_cols = st.columns(3)
        if verdict_label == "UNCERTAIN":
            summary_cols[0].metric("Verdict", "UNCERTAIN")
        elif label == "REAL":
            summary_cols[0].metric("Verdict", "Likely TRUE")
        else:
            summary_cols[0].metric("Verdict", "Likely FALSE")
        summary_cols[1].metric("Probability of truth", f"{truth_probability:.1%}")
        summary_cols[2].metric("Probability of false", f"{false_probability:.1%}")

        st.progress(int(round(truth_probability * 100)))
        st.caption("Probability of truth is the model's estimated chance that the article is REAL.")

        if verdict_label == "UNCERTAIN":
            st.warning(
                f"Prediction is UNCERTAIN because confidence is below {uncertainty_threshold:.0%}. Top class is {label} at {confidence:.1%}."
            )
        elif label == "REAL":
            st.success(f"Likely TRUE with confidence {confidence:.1%}")
        else:
            st.error(f"Likely FALSE with confidence {confidence:.1%}")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(display["label"], display["probability"], color=["#0f766e", "#b91c1c"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Probability")
        ax.set_title("Class probabilities")
        st.pyplot(fig, clear_figure=True)

        st.subheader("AI Summary")
        st.info(f"TL;DR: {summary}")

        st.subheader("Explainable AI")
        explain_cols = st.columns(2)
        explain_cols[0].markdown("**Words supporting the prediction**")
        explain_cols[0].write(format_terms(explanation["supporting_terms"]))
        explain_cols[1].markdown("**Words pushing against the prediction**")
        explain_cols[1].write(format_terms(explanation["opposing_terms"]))

        st.subheader("Source Credibility Checker")
        if source_report["valid"]:
            source_cols = st.columns(3)
            source_cols[0].metric("Domain", str(source_report["domain"]))
            source_cols[1].metric("HTTPS", "Yes" if source_report["https"] else "No")
            source_cols[2].metric("Trust score", f"{int(source_report['trust_score'])}/100")
            st.caption(f"Domain age: {source_report['domain_age']}")
            for note in source_report["notes"]:
                st.write(f"- {note}")
        else:
            st.info("Enter a source URL to check domain trust and HTTPS status.")

        st.subheader("Fact-Check Suggestions")
        for item in fact_check_links:
            st.markdown(f"- [{item['name']}]({item['url']})")

        report_record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "article": input_text,
            "summary": summary,
            "verdict": verdict_label,
            "confidence_display": f"{confidence:.1%}",
            "truth_probability_display": f"{truth_probability:.1%}",
            "false_probability_display": f"{false_probability:.1%}",
            "supporting_terms": explanation["supporting_terms"],
            "opposing_terms": explanation["opposing_terms"],
            "source_domain": source_report.get("domain", ""),
            "source_trust_score": source_report.get("trust_score", "N/A"),
            "fact_check_text": "Use the links in the dashboard to verify the claim externally.",
        }
        add_history_entry(report_record)

        pdf_bytes = build_pdf_report(report_record)
        st.download_button(
            label="Download PDF report",
            data=pdf_bytes,
            file_name="fake_news_report.pdf",
            mime="application/pdf",
        )

st.subheader("Evaluation summary")
if (ARTIFACTS_DIR / "evaluation.json").exists():
    _, _, evaluation, _ = load_artifacts()
    summary_cols = st.columns(3)
    summary_cols[0].metric("Overall accuracy", f"{float(evaluation['overall_accuracy']):.1%}")
    summary_cols[1].metric("Covered accuracy", f"{float(evaluation['covered_accuracy']):.1%}")
    summary_cols[2].metric("Coverage", f"{float(evaluation['coverage']):.1%}")
else:
    st.info("Run training to generate evaluation metrics.")

st.divider()
st.subheader("Prediction History")
if st.session_state.prediction_history:
    history_frame = pd.DataFrame(st.session_state.prediction_history)
    st.dataframe(
        history_frame[["timestamp", "verdict", "confidence_display", "truth_probability_display", "source_domain"]],
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download prediction history as CSV",
        data=history_frame.to_csv(index=False).encode("utf-8"),
        file_name="prediction_history.csv",
        mime="text/csv",
    )

    st.subheader("Analytics Dashboard")
    analytics_cols = st.columns(4)
    total_articles = len(history_frame)
    fake_articles = int((history_frame["verdict"] == "FAKE").sum() if "verdict" in history_frame else 0)
    real_articles = int((history_frame["verdict"] == "REAL").sum() if "verdict" in history_frame else 0)
    uncertain_cases = int((history_frame["verdict"] == "UNCERTAIN").sum() if "verdict" in history_frame else 0)
    analytics_cols[0].metric("Total Articles Checked", total_articles)
    analytics_cols[1].metric("Real Articles Found", real_articles)
    analytics_cols[2].metric("Fake Articles Found", fake_articles)
    analytics_cols[3].metric("Uncertain Cases", uncertain_cases)

    chart_frame = history_frame.groupby("verdict").size().reindex(["REAL", "FAKE", "UNCERTAIN"]).fillna(0)
    chart_frame = chart_frame.rename(index={"REAL": "REAL", "FAKE": "FAKE", "UNCERTAIN": "UNCERTAIN"})
    st.bar_chart(chart_frame)

    pie_values = chart_frame.tolist()
    if sum(pie_values) > 0:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.pie(pie_values, labels=chart_frame.index.tolist(), autopct="%1.0f%%", colors=["#0f766e", "#b91c1c", "#b45309"])
        ax2.set_title("Prediction distribution")
        st.pyplot(fig2, clear_figure=True)

    history_frame["timestamp"] = pd.to_datetime(history_frame["timestamp"], errors="coerce")
    trend_frame = history_frame.dropna(subset=["timestamp"]).copy()
    if not trend_frame.empty:
        trend_frame["date"] = trend_frame["timestamp"].dt.date
        trend_counts = trend_frame.groupby(["date", "verdict"]).size().unstack(fill_value=0)
        st.line_chart(trend_counts)
else:
    st.info("Prediction history will appear here after you analyze articles.")
