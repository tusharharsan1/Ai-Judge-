"""
data_loader.py
==============
Data loading and preparation utilities for the AI Judge pipeline.

Responsibilities:
  - Load and validate rubric.json
  - Load sample_student_answers.csv
  - Merge judge results with human scores for metric computation
  - Validate the CSV schema against expected columns

All paths are configurable so the notebook can override defaults.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DEFAULT PATHS  (relative to the project root — override as needed)
# ---------------------------------------------------------------------------
_DEFAULT_ROOT = Path(__file__).parent.parent  # ai_judge_implementation/../

RUBRIC_PATH = _DEFAULT_ROOT / "rubric_1 (1).json"
CSV_PATH    = _DEFAULT_ROOT / "sample_student_answers_2 (1).csv"

# Required columns in the student answers CSV
_REQUIRED_CSV_COLUMNS = {"id", "student_answer"}
_HUMAN_SCORE_COLUMNS = {
    "human_content_accuracy",
    "human_example_evidence",
    "human_clarity_organization",
    "human_scientific_vocabulary",
    "human_total",
}


# ---------------------------------------------------------------------------
# RUBRIC LOADER
# ---------------------------------------------------------------------------
def load_rubric(path: Optional[Path | str] = None) -> dict[str, Any]:
    """
    Load and validate the rubric JSON file.

    Args:
        path: Path to rubric.json.  Defaults to RUBRIC_PATH.

    Returns:
        Parsed rubric dict.

    Raises:
        FileNotFoundError:  If the rubric file does not exist.
        ValueError:         If the rubric is missing required keys.
    """
    rubric_path = Path(path) if path else RUBRIC_PATH

    if not rubric_path.exists():
        raise FileNotFoundError(
            f"Rubric file not found: {rubric_path}\n"
            f"Make sure 'rubric_1 (1).json' is in the project root directory."
        )

    with open(rubric_path, "r", encoding="utf-8") as f:
        rubric = json.load(f)

    # Validate required keys
    required_keys = {"question", "dimensions"}
    missing = required_keys - set(rubric.keys())
    if missing:
        raise ValueError(f"Rubric is missing required keys: {missing}")

    if not isinstance(rubric.get("dimensions"), list) or len(rubric["dimensions"]) == 0:
        raise ValueError("Rubric 'dimensions' must be a non-empty list.")

    for dim in rubric["dimensions"]:
        dim_required = {"id", "name", "scale", "description"}
        dim_missing = dim_required - set(dim.keys())
        if dim_missing:
            raise ValueError(
                f"Dimension '{dim.get('id', '?')}' is missing keys: {dim_missing}"
            )

    logger.info(
        "Loaded rubric: %d dimensions, subject=%s, grade=%s",
        len(rubric["dimensions"]),
        rubric.get("subject", "N/A"),
        rubric.get("grade_level", "N/A"),
    )
    return rubric


# ---------------------------------------------------------------------------
# CSV LOADER
# ---------------------------------------------------------------------------
def load_student_answers(
    path: Optional[Path | str] = None,
    rows: Optional[list[int]] = None,
) -> pd.DataFrame:
    """
    Load the student answers CSV into a DataFrame.

    Normalises column names to match the pipeline's expectations:
      - The CSV uses columns like 'human_content_accuracy'; we rename to match
        the HUMAN_SCORE_COLUMNS convention.

    Args:
        path:  Path to the CSV.  Defaults to CSV_PATH.
        rows:  Optional list of student IDs to filter to (e.g. [1..15] for
               the required evaluation set, or None for all rows).

    Returns:
        DataFrame with at least: id, student_answer, human_*, human_total, notes.

    Raises:
        FileNotFoundError:  If the CSV does not exist.
        ValueError:         If required columns are missing.
    """
    csv_path = Path(path) if path else CSV_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found: {csv_path}\n"
            f"Expected: sample_student_answers_2 (1).csv in the project root."
        )

    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Validate required columns
    missing = _REQUIRED_CSV_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    # Rename score columns to human_* prefix if they're not already
    rename_map = {
        "content_accuracy": "human_content_accuracy",
        "example_evidence": "human_example_evidence",
        "clarity_organization": "human_clarity_organization",
        "scientific_vocabulary": "human_scientific_vocabulary",
        "total": "human_total",
    }
    # Only rename columns that exist in the CSV and haven't already been prefixed
    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Ensure student_answer is string (handle NaN for empty rows)
    df["student_answer"] = df["student_answer"].fillna("").astype(str)

    # Filter by row IDs if requested
    if rows is not None:
        df = df[df["id"].isin(rows)].copy()
        logger.info("Filtered CSV to %d rows (IDs: %s).", len(df), rows)

    logger.info("Loaded CSV: %d rows total.", len(df))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# MERGE RESULTS WITH HUMAN SCORES
# ---------------------------------------------------------------------------
def merge_results_with_human_scores(
    results_df: pd.DataFrame,
    answers_df: pd.DataFrame,
    on: str = "student_id",
) -> pd.DataFrame:
    """
    Join the judge results DataFrame with the human scores from the CSV.

    Args:
        results_df:  DataFrame of judge results (from JudgeResult.to_flat_dict()).
                     Must have a 'student_id' column.
        answers_df:  The full student answers DataFrame (from load_student_answers).
                     Must have 'id' and human_* columns.
        on:          Join key in results_df.  Defaults to 'student_id'.

    Returns:
        Merged DataFrame ready for metric computation.
    """
    # Rename 'id' in answers_df to match results_df's join key
    answers_slim = answers_df.rename(columns={"id": "student_id"}).copy()

    # Select only the columns we need from the answers side
    keep_cols = ["student_id", "student_answer"] + [
        c for c in answers_slim.columns
        if c.startswith("human_") or c == "notes"
    ]
    answers_slim = answers_slim[[c for c in keep_cols if c in answers_slim.columns]]

    merged = results_df.merge(answers_slim, on="student_id", how="left")
    logger.info("Merged results: %d rows.", len(merged))
    return merged


# ---------------------------------------------------------------------------
# CONVENIENCE: LOAD EVERYTHING
# ---------------------------------------------------------------------------
def load_all(
    rubric_path: Optional[Path | str] = None,
    csv_path: Optional[Path | str] = None,
    required_rows: Optional[list[int]] = None,
) -> tuple[dict, str, pd.DataFrame]:
    """
    One-call loader for the full pipeline.

    Returns:
        (rubric, question, df)
        - rubric:   Parsed rubric dict
        - question: The exam question string extracted from the rubric
        - df:       Student answers DataFrame (optionally filtered)
    """
    rubric = load_rubric(rubric_path)
    question = rubric["question"]
    df = load_student_answers(csv_path, rows=required_rows)
    return rubric, question, df
