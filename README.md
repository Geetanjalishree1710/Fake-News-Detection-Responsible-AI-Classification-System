# Fake News Detection & Responsible AI Classification System

This project classifies news articles as `REAL`, `FAKE`, or `UNCERTAIN` using a TF-IDF + Logistic Regression pipeline.

It includes:

- text preprocessing and normalization
- dataset deduplication and leakage detection
- source-confounding analysis for reliability checks
- confidence-based uncertainty handling
- checksum-verified model artifacts
- Streamlit dashboard for interactive analysis
- automated tests and GitHub Actions CI

## Project layout

- `src/fake_news/` contains the reusable pipeline and utilities
- `scripts/train.py` trains the model and exports artifacts
- `app/streamlit_app.py` runs the dashboard
- `tests/` contains automated checks

## Data format

Provide a CSV file with at least these columns:

- `text`: the article content
- `label`: `REAL` or `FAKE`

Optional columns:

- `source`: publication or domain name for confounding analysis

## Quick start

1. Create and activate the virtual environment.
2. Install dependencies.
3. Train the model.
4. Launch the Streamlit app.

Example:

```powershell
python -m pip install -r requirements.txt
python scripts/train.py --input data/sample/news_sample.csv
streamlit run app/streamlit_app.py
```

## Outputs

Training writes artifacts to `artifacts/latest/`:

- `model.joblib`
- `metadata.json`
- `checksums.json`
- `evaluation.json`
- `leakage_report.json`
- `source_confounding.csv`

The dashboard reads those artifacts and shows prediction confidence, probability bars, and evaluation summaries.
