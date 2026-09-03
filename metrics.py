"""
metrics.py
==========
Evaluation metrics for the AI Judge.

Implements exactly the metrics specified in the PRD (§9) and TRD (§7):

    1. MAE  — mean absolute error between judge_total and human_total (rows 1–15)
    2. Tolerance accuracy  — % of rows where |judge_total - human_total| ≤ 1
    3. Per-dimension signed bias  — mean(judge_score - human_score) per dimension

Why SIGNED (not absolute) per-dimension error:
    Absolute error hides direction.  A judge that is systematically +1 on
    content_accuracy and a perfectly calibrated judge have the same absolute
    error magnitude but opposite implications.  Signed bias reveals whether
    the system consistently over- or under-scores specific dimensions — which
    is exactly what a teacher auditing the system needs to know.

Why NOT Cohen's / Quadratic Weighted Kappa (per PRD §10):
    1. With only 15 rows, marginal distributions are sparse → unstable kappa.
    2. The score distribution is intentionally skewed (many highs and lows),
       creating the well-documented "kappa paradox" (artificially high/low kappa
       unrelated to actual agreement).
    3. Kappa collapses everything to a single number per dimension, hiding the
       directional bias that signed mean error surfaces directly.
    This is a deliberate design decision, not an omission.
"""

import pandas as pd
import numpy as np
from typing import Optional


# Dimension IDs in the order they appear in the rubric
DIMENSION_IDS = [
    "content_accuracy",
    "example_evidence",
    "clarity_organization",
    "scientific_vocabulary",
]


# ---------------------------------------------------------------------------
# MAIN METRICS FUNCTION
# ---------------------------------------------------------------------------
def compute_metrics(
    df: pd.DataFrame,
    judge_total_col: str = "judge_total",
    human_total_col: str = "human_total",
    dim_ids: Optional[list[str]] = None,
    tolerance: int = 1,
) -> dict:
    """
    Compute all required evaluation metrics on the provided DataFrame.

    Expects the DataFrame to have:
        - {judge_total_col}        : int (system-computed judge total 0–20)
        - {human_total_col}        : int (human grader total 0–20)
        - judge_{dim_id}           : int for each dimension in dim_ids
        - human_{dim_id}           : int for each dimension in dim_ids

    Args:
        df:               Results DataFrame (rows 1–15 only).
        judge_total_col:  Column name for judge total scores.
        human_total_col:  Column name for human total scores.
        dim_ids:          Dimension IDs to compute per-dim bias for.
                          Defaults to DIMENSION_IDS.
        tolerance:        Tolerance for tolerance-accuracy metric (default 1).

    Returns:
        Dict with keys:
            mae, tolerance_accuracy_pct, per_dim_signed_bias,
            n_rows, n_flagged, n_no_llm_call
    """
    if dim_ids is None:
        dim_ids = DIMENSION_IDS

    n_rows = len(df)
    if n_rows == 0:
        return {"error": "Empty DataFrame — no rows to evaluate."}

    # --- MAE ---
    errors = (df[judge_total_col] - df[human_total_col]).abs()
    mae = float(errors.mean())

    # --- Tolerance Accuracy ---
    within_tolerance = (df[judge_total_col] - df[human_total_col]).abs() <= tolerance
    tolerance_accuracy_pct = float(within_tolerance.mean() * 100)

    # --- Per-Dimension Signed Bias ---
    # 4 dimensions × 15 rows = 60 paired points (per PRD TRD §7)
    per_dim_signed_bias: dict[str, float] = {}
    for dim_id in dim_ids:
        judge_col = f"judge_{dim_id}"
        human_col = f"human_{dim_id}"
        if judge_col in df.columns and human_col in df.columns:
            signed_errors = df[judge_col] - df[human_col]
            per_dim_signed_bias[dim_id] = float(signed_errors.mean())
        else:
            per_dim_signed_bias[dim_id] = float("nan")

    # --- Bonus stats for the written summary ---
    n_flagged = int(df.get("flagged_for_review", pd.Series([False] * n_rows)).sum())
    n_no_llm = int((~df.get("llm_called", pd.Series([True] * n_rows))).sum())

    return {
        "n_rows": n_rows,
        "mae": mae,
        "tolerance_accuracy_pct": tolerance_accuracy_pct,
        "tolerance": tolerance,
        "per_dim_signed_bias": per_dim_signed_bias,
        "n_flagged": n_flagged,
        "n_no_llm_call": n_no_llm,
    }


# ---------------------------------------------------------------------------
# PRETTY PRINTER
# ---------------------------------------------------------------------------
def print_metrics_report(metrics: dict) -> None:
    """
    Print a human-readable metrics report to stdout.

    Suitable for inclusion in a Jupyter notebook or a Colab output cell.
    """
    if "error" in metrics:
        print(f"[ERROR] {metrics['error']}")
        return

    print("=" * 60)
    print("  AI JUDGE — EVALUATION METRICS REPORT")
    print(f"  Evaluated on {metrics['n_rows']} rows")
    print("=" * 60)
    print()
    print(f"  MAE (Mean Absolute Error, 0–20 scale):  {metrics['mae']:.3f}")
    print()
    tol = metrics["tolerance"]
    ta = metrics["tolerance_accuracy_pct"]
    print(f"  Tolerance Accuracy (|Δ| ≤ {tol}):        {ta:.1f}%")
    print()
    print("  Per-Dimension Signed Bias (mean judge − human):")
    bias = metrics["per_dim_signed_bias"]
    for dim, val in bias.items():
        direction = "↑ over-scores" if val > 0.1 else ("↓ under-scores" if val < -0.1 else "≈ calibrated")
        print(f"    {dim:<30s}  {val:+.3f}  ({direction})")
    print()
    print(f"  Rows flagged for review:  {metrics['n_flagged']}")
    print(f"  Rows without LLM call:    {metrics['n_no_llm_call']}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# RATIONALE CONSISTENCY REVIEW
# ---------------------------------------------------------------------------
REVIEW_ROWS = [2, 5, 6, 8, 9]
"""
The 5 rows selected for mandatory manual rationale review (FR-7), chosen from
the core 1-15 dataset:

  Row  2  — Lamarckian definition without mechanism.
  Row  5  — Survival of the fittest misconception (strength instead of reproduction).
  Row  6  — Solid answer with beetle example, testing fine-grained distinction from perfect.
  Row  8  — Nuanced answer about antibiotic resistance pre-existing mutation.
  Row  9  — Lamarckian example with giraffes.
"""


def manual_review_table(df: pd.DataFrame, review_ids: Optional[list[int]] = None) -> pd.DataFrame:
    """
    Extract the rows selected for manual rationale consistency review.

    Returns a DataFrame with student_id, all judge_* scores, all rationale_*
    columns, and the human_* scores for comparison — formatted for easy reading.

    Args:
        df:          Full results DataFrame.
        review_ids:  List of student IDs to review.  Defaults to REVIEW_ROWS.

    Returns:
        Filtered and column-ordered DataFrame for manual review.
    """
    if review_ids is None:
        review_ids = REVIEW_ROWS

    review_df = df[df["student_id"].isin(review_ids)].copy()

    # Preferred column order for readability
    priority_cols = ["student_id"]
    score_cols = [c for c in df.columns if c.startswith("judge_") or c.startswith("human_")]
    rationale_cols = [c for c in df.columns if c.startswith("rationale_")]
    meta_cols = ["flagged_for_review", "flag_reason", "llm_called", "retries_used", "error"]

    ordered = (
        priority_cols
        + [c for c in score_cols if c in df.columns]
        + [c for c in rationale_cols if c in df.columns]
        + [c for c in meta_cols if c in df.columns]
    )
    # Keep only columns that actually exist
    ordered = [c for c in ordered if c in review_df.columns]

    return review_df[ordered].reset_index(drop=True)


# ---------------------------------------------------------------------------
# STRETCH: VARIANCE / CONSISTENCY CHECK  (FR-9 optional)
# ---------------------------------------------------------------------------
def compute_run_variance(
    scores_per_run: list[dict[str, int]],
) -> dict[str, float]:
    """
    Compute variance across multiple scoring runs for the same answer.

    Args:
        scores_per_run: List of dicts, each mapping dimension_id → score,
                        from N independent judge runs on the same answer.

    Returns:
        Dict mapping dimension_id → standard deviation across the N runs,
        plus 'total_score' → std of the total score.
    """
    if not scores_per_run:
        return {}

    keys = list(scores_per_run[0].keys())
    result: dict[str, float] = {}
    for key in keys:
        vals = [run[key] for run in scores_per_run if key in run]
        result[key] = float(np.std(vals))

    return result
