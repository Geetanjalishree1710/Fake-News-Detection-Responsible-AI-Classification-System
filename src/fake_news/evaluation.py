from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import REPORTS_DIR


def save_json_report(report: dict[str, object], filename: str, output_dir: Path = REPORTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return output_path


def save_dataframe_report(frame: pd.DataFrame, filename: str, output_dir: Path = REPORTS_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    frame.to_csv(output_path, index=False)
    return output_path
