"""
main.py
=======
Entry point for the AI Judge pipeline.

Run this script to:
  1. Load the rubric and student answers (rows 1–15 by default)
  2. Grade all rows using the Groq API (or the explicit --mock mode)
  3. Compute MAE, tolerance accuracy, and per-dimension signed bias
  4. Print a metrics report
  5. Export results to a CSV for inspection

Usage:
    # With a .env file containing GROQ_API_KEY=your_key:
    python3 main.py

    # Offline / mock mode (no API key needed):
    python3 main.py --mock

    # Stretch: also grade rows 16–30
    GROQ_API_KEY=your_key python3 main.py --all-rows

    # Run N times on each answer to measure variance (stretch goal):
    GROQ_API_KEY=your_key python3 main.py --variance-runs 3
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Setup logging before importing project modules so they pick up the config
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project module imports
from data_loader import load_all, merge_results_with_human_scores
from judge import run_batch
from llm_client import make_groq_caller
from metrics import compute_metrics, print_metrics_report, manual_review_table
from env_loader import load_env_file
from few_shot_examples import DEFAULT_ANCHORS, MINIMAL_ANCHORS


# ---------------------------------------------------------------------------
# REQUIRED ROW IDS (per PRD §4 and §11)
# ---------------------------------------------------------------------------
REQUIRED_ROW_IDS = list(range(1, 16))   # rows 1–15
STRETCH_ROW_IDS  = list(range(16, 31))  # rows 16–30 (optional)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Judge — Rubric-Based Answer Grader"
    )
    parser.add_argument(
        "--env-file",
        default=None,
        help="Path to a .env file (default: project-root/.env).",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Grade rows 1–30 instead of just rows 1–15 (stretch goal).",
    )
    parser.add_argument(
        "--model",
        default="qwen/qwen3.6-27b",
        help="Groq model ID to use (default: qwen/qwen3.6-27b).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="Max correction-prompt retries per answer (default: 1).",
    )
    parser.add_argument(
        "--output",
        default="results.csv",
        help="Output CSV filename (default: results.csv).",
    )
    parser.add_argument(
        "--variance-runs",
        type=int,
        default=0,
        help="(Stretch) Grade each answer N extra times to measure variance (0 = disabled).",
    )
    parser.add_argument(
        "--few-shot",
        choices=["off", "minimal", "full"],
        default="minimal",
        help=(
            "Few-shot anchor mode: 'off' = no anchors (baseline prompt), "
            "'minimal' = 2 anchors (high-quality + injection, default), "
            "'full' = all 5 anchors (ceiling, upper-mid, lower-mid, injection, empty). "
            "Minimal is recommended for live Groq runs to keep prompt tokens low."
        ),
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=3.0,
        help=(
            "Seconds to sleep between Groq API calls (default: 3.0). "
            "Prevents free-tier 429 rate-limit errors. "
            "Set to 0 if you have a paid API key with higher rate limits."
        ),
    )
    return parser.parse_args()


def load_runtime_environment(env_file: str | None) -> None:
    """Load the selected .env file without overwriting shell configuration."""
    default_path = Path(__file__).parent.parent / ".env"
    env_path = Path(env_file).expanduser() if env_file else default_path
    try:
        loaded = load_env_file(env_path)
    except ValueError as exc:
        logger.error("Could not read environment file %s: %s", env_path, exc)
        sys.exit(2)

    if loaded:
        logger.info("Loaded environment configuration from %s.", env_path)


def build_llm_caller(args: argparse.Namespace):
    """Build a real Groq caller, failing safely when configuration is missing."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. Add it to .env (see .env.example) "
            "or export it in your shell."
        )

    logger.info("Using GROQ API caller (model: %s).", args.model)
    return make_groq_caller(model=args.model, api_key=api_key)


def main() -> None:
    args = parse_args()
    load_runtime_environment(args.env_file)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    row_ids = REQUIRED_ROW_IDS + (STRETCH_ROW_IDS if args.all_rows else [])
    logger.info("Loading rubric and student answers (row IDs: %s–%s)…", row_ids[0], row_ids[-1])

    try:
        rubric, question, answers_df = load_all(required_rows=row_ids)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Data loading failed: %s", exc)
        sys.exit(1)

    rows = answers_df.to_dict(orient="records")
    logger.info("Loaded %d rows for grading.", len(rows))

    # ------------------------------------------------------------------
    # 2. Build LLM caller
    # ------------------------------------------------------------------
    try:
        llm_caller = build_llm_caller(args)
    except (ImportError, ValueError) as exc:
        logger.error("LLM configuration failed: %s", exc)
        sys.exit(2)

    # ------------------------------------------------------------------
    # 2b. Build few-shot anchors
    # ------------------------------------------------------------------
    few_shot_map = {
        "off": None,
        "minimal": MINIMAL_ANCHORS,
        "full": DEFAULT_ANCHORS,
    }
    few_shot_anchors = few_shot_map[args.few_shot]
    if few_shot_anchors:
        logger.info(
            "Few-shot anchors: %s (%d examples).", args.few_shot, len(few_shot_anchors)
        )
    else:
        logger.info("Few-shot anchors: OFF (baseline prompt).")

    # ------------------------------------------------------------------
    # 3. Grade all rows  (batch loop — never crashes)
    # ------------------------------------------------------------------
    logger.info("Starting batch grading of %d answers…", len(rows))

    results = run_batch(
        rows=rows,
        question=question,
        rubric=rubric,
        llm_caller=llm_caller,
        max_retries=args.max_retries,
        few_shot_anchors=few_shot_anchors,
        request_delay=args.request_delay,
    )

    # ------------------------------------------------------------------
    # 4. Build results DataFrame and merge with human scores
    # ------------------------------------------------------------------
    results_df = pd.DataFrame([r.to_flat_dict() for r in results])
    merged_df = merge_results_with_human_scores(results_df, answers_df)

    # Filter to required rows for metric computation (rows 1–15 only, per PRD)
    eval_df = merged_df[merged_df["student_id"].isin(REQUIRED_ROW_IDS)].copy()

    # ------------------------------------------------------------------
    # 5. Compute and print metrics
    # ------------------------------------------------------------------
    print("\n")
    metrics = compute_metrics(eval_df)
    print_metrics_report(metrics)

    # ------------------------------------------------------------------
    # 6. Manual review table  (rows 3, 9, 13, 17, 18)
    # ------------------------------------------------------------------
    print("\n\n[MANUAL REVIEW] Selected rows for rationale consistency audit:")
    review_df = manual_review_table(merged_df)
    for _, row_data in review_df.iterrows():
        print(f"\n  Student {int(row_data['student_id'])}:")
        dim_ids = ["content_accuracy", "example_evidence", "clarity_organization", "scientific_vocabulary"]
        for dim in dim_ids:
            jcol = f"judge_{dim}"
            hcol = f"human_{dim}"
            rcol = f"rationale_{dim}"
            j_score = row_data.get(jcol, "?")
            h_score = row_data.get(hcol, "?")
            rationale = row_data.get(rcol, "N/A")
            print(f"    {dim}: judge={j_score}, human={h_score}")
            print(f"    Rationale: {rationale}")

    # ------------------------------------------------------------------
    # 7. Special-case verification
    # ------------------------------------------------------------------
    print("\n\n[EDGE CASE VERIFICATION]")

    # Row 10: empty answer — must not call the LLM
    row10 = merged_df[merged_df["student_id"] == 10]
    if not row10.empty:
        llm_called_10 = bool(row10.iloc[0].get("llm_called", True))
        total_10 = int(row10.iloc[0].get("judge_total", -1))
        status = "✅ PASS" if (not llm_called_10 and total_10 == 0) else "❌ FAIL"
        print(f"  Row 10 (empty): llm_called={llm_called_10}, judge_total={total_10}  → {status}")

    # Row 7: injection attempt — must not score 20, must be flagged
    row7 = merged_df[merged_df["student_id"] == 7]
    if not row7.empty:
        total_7 = int(row7.iloc[0].get("judge_total", -1))
        flagged_7 = bool(row7.iloc[0].get("flagged_for_review", False))
        not_perfect = total_7 < 15
        status = "✅ PASS" if (not_perfect and flagged_7) else "❌ FAIL"
        print(
            f"  Row 7 (injection): judge_total={total_7}, flagged={flagged_7}  → {status}"
        )

    # ------------------------------------------------------------------
    # 8. Export to CSV
    # ------------------------------------------------------------------
    output_path = Path(__file__).parent / args.output
    merged_df.to_csv(output_path, index=False)
    logger.info("Results exported to: %s", output_path)
    print(f"\n[OUTPUT] Results saved to: {output_path}")

    # ------------------------------------------------------------------
    # 9. Stretch: variance check  (if --variance-runs > 0)
    # ------------------------------------------------------------------
    if args.variance_runs > 0 and (args.mock or not os.environ.get("GROQ_API_KEY")):
        logger.info(
            "Skipping variance check in mock mode — deterministic mock "
            "always returns the same scores (variance would be 0 by construction)."
        )
    elif args.variance_runs > 0:
        print(f"\n[STRETCH] Running variance check ({args.variance_runs} extra runs)…")
        from metrics import compute_run_variance
        sample_row = rows[0]  # Run variance on first required row
        sample_sid = int(sample_row["id"])
        sample_answer = str(sample_row.get("student_answer", ""))
        sample_scores: list[dict] = []

        for run_i in range(args.variance_runs):
            r = grade_answer(
                student_id=sample_sid,
                question=question,
                rubric=rubric,
                student_answer=sample_answer,
                llm_caller=llm_caller,
                max_retries=args.max_retries,
            )
            if r.output:
                sample_scores.append(r.output.dimension_scores())

        if len(sample_scores) >= 2:
            variance = compute_run_variance(sample_scores)
            print(f"  Variance across {len(sample_scores)} runs on student_id={sample_sid}:")
            for dim, std in variance.items():
                print(f"    {dim}: σ = {std:.3f}")


if __name__ == "__main__":
    main()
