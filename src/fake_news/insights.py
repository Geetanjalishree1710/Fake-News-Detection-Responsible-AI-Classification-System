from __future__ import annotations

import io
import re
from collections import Counter
from urllib.parse import quote_plus, urlparse

import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sklearn.pipeline import Pipeline


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "he",
    "in",
    "is",
    "it",
    "its",
    "not",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
    "you",
}


def extract_summary(text: str, max_sentences: int = 2) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", text.strip()) if sentence.strip()]
    if not sentences:
        return "No summary available."
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = re.findall(r"[a-zA-Z']+", text.lower())
    word_frequencies = Counter(word for word in words if word not in STOPWORDS and len(word) > 2)
    if not word_frequencies:
        return " ".join(sentences[:max_sentences])

    sentence_scores: list[tuple[float, str]] = []
    for sentence in sentences:
        sentence_words = re.findall(r"[a-zA-Z']+", sentence.lower())
        if not sentence_words:
            continue
        score = sum(word_frequencies.get(word, 0) for word in sentence_words) / len(sentence_words)
        sentence_scores.append((score, sentence))

    top_sentences = [sentence for _, sentence in sorted(sentence_scores, key=lambda item: item[0], reverse=True)[:max_sentences]]
    return " ".join(top_sentences) if top_sentences else " ".join(sentences[:max_sentences])


def explain_prediction(model: Pipeline, text: str, top_n: int = 5) -> dict[str, object]:
    vectorizer = model.named_steps["tfidf"]
    classifier = model.named_steps["classifier"]
    classes = list(model.classes_)
    probabilities = model.predict_proba([text])[0]
    predicted_label = str(classes[int(np.argmax(probabilities))])

    vector = vectorizer.transform([text])
    feature_names = vectorizer.get_feature_names_out()
    active_indices = vector.nonzero()[1]

    if len(classes) == 2:
        coefficient_row = classifier.coef_[0]
        aligned_coefficients = coefficient_row if predicted_label == classes[-1] else -coefficient_row
    else:
        aligned_coefficients = classifier.coef_[classes.index(predicted_label)]

    contributions = vector.toarray()[0] * aligned_coefficients
    supporting_terms: list[dict[str, float | str]] = []
    opposing_terms: list[dict[str, float | str]] = []

    for index in active_indices:
        term = feature_names[index]
        score = float(contributions[index])
        if score >= 0:
            supporting_terms.append({"term": term, "score": score})
        else:
            opposing_terms.append({"term": term, "score": score})

    supporting_terms = sorted(supporting_terms, key=lambda item: float(item["score"]), reverse=True)[:top_n]
    opposing_terms = sorted(opposing_terms, key=lambda item: float(item["score"]))[:top_n]

    return {
        "predicted_label": predicted_label,
        "supporting_terms": supporting_terms,
        "opposing_terms": opposing_terms,
    }


def assess_source_credibility(source_url: str) -> dict[str, object]:
    source_url = source_url.strip()
    if not source_url:
        return {
            "valid": False,
            "domain": "",
            "https": False,
            "domain_age": "Not checked",
            "trust_score": 0,
            "notes": ["No source URL provided."],
        }

    parsed = urlparse(source_url if "://" in source_url else f"https://{source_url}")
    domain = (parsed.netloc or parsed.path).lower().strip()
    https = parsed.scheme == "https"
    subdomain_count = domain.count(".")
    suspicious_tlds = {".xyz", ".top", ".info", ".buzz", ".click", ".site"}

    trust_score = 55
    notes = ["Heuristic score only; domain age requires an external lookup."]

    if https:
        trust_score += 15
    else:
        trust_score -= 10
        notes.append("HTTPS is not enabled.")

    if subdomain_count > 2:
        trust_score -= 10
        notes.append("Many subdomains can be a risk signal.")

    if any(domain.endswith(tld) for tld in suspicious_tlds):
        trust_score -= 10
        notes.append("The top-level domain is often used by low-trust sites.")

    if "-" in domain:
        trust_score -= 5
        notes.append("Hyphenated domains can be a weak risk signal.")

    if len(domain) > 30:
        trust_score -= 5
        notes.append("Very long domain names can be suspicious.")

    if "news" in domain:
        trust_score += 5

    trust_score = max(0, min(100, trust_score))

    if trust_score >= 75:
        domain_age = "Likely established, but not verified locally"
    elif trust_score >= 50:
        domain_age = "Unknown / needs external lookup"
    else:
        domain_age = "Potentially risky / needs verification"

    return {
        "valid": True,
        "domain": domain,
        "https": https,
        "domain_age": domain_age,
        "trust_score": trust_score,
        "notes": notes,
    }


def build_fact_check_links(text: str) -> list[dict[str, str]]:
    query = quote_plus(extract_summary(text, max_sentences=1)) if text.strip() else ""
    return [
        {"name": "Snopes", "url": f"https://www.snopes.com/search/?q={query}"},
        {"name": "PolitiFact", "url": f"https://www.politifact.com/search/?q={query}"},
        {"name": "FactCheck.org", "url": f"https://www.factcheck.org/?s={query}"},
        {"name": "Google Fact Check Tools", "url": f"https://toolbox.google.com/factcheck/explorer/search/{query}"},
    ]


def build_pdf_report(record: dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    story: list[object] = []

    story.append(Paragraph("Fake News Detection Report", styles["Title"]))
    story.append(Spacer(1, 12))

    rows = [
        ["Timestamp", str(record.get("timestamp", ""))],
        ["Verdict", str(record.get("verdict", ""))],
        ["Confidence", str(record.get("confidence_display", ""))],
        ["Probability of truth", str(record.get("truth_probability_display", ""))],
        ["Probability of false", str(record.get("false_probability_display", ""))],
        ["Source trust score", str(record.get("source_trust_score", "N/A"))],
    ]

    table = Table(rows, colWidths=[150, 360])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("AI Summary", styles["Heading2"]))
    story.append(Paragraph(str(record.get("summary", "No summary available.")), styles["BodyText"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Explainability", styles["Heading2"]))
    support_terms = record.get("supporting_terms", [])
    oppose_terms = record.get("opposing_terms", [])
    explain_rows = [["Support terms", "Contribution"]]
    for item in support_terms if isinstance(support_terms, list) else []:
        explain_rows.append([str(item.get("term", "")), f"{float(item.get('score', 0.0)):.3f}"])
    if not support_terms:
        explain_rows.append(["No strong support terms", ""])

    explain_table = Table(explain_rows, colWidths=[300, 210])
    explain_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    story.append(explain_table)
    story.append(Spacer(1, 8))

    oppose_rows = [["Opposition terms", "Contribution"]]
    if isinstance(oppose_terms, list) and oppose_terms:
        for item in oppose_terms:
            oppose_rows.append([str(item.get("term", "")), f"{float(item.get('score', 0.0)):.3f}"])
    else:
        oppose_rows.append(["No strong opposition terms", ""])

    oppose_table = Table(oppose_rows, colWidths=[300, 210])
    oppose_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke)]))
    story.append(oppose_table)
    story.append(Spacer(1, 12))

    story.append(Paragraph("Fact-Check Suggestions", styles["Heading2"]))
    story.append(Paragraph(str(record.get("fact_check_text", "")), styles["BodyText"]))

    document.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes