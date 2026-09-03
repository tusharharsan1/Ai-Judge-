"""
judge.py
========
Core grading pipeline for the AI Judge.

This module owns the per-answer control flow exactly as specified in the PRD
Project Flow (Part III §1):

    Step 1  check_empty      → deterministic short-circuit (no LLM call)
    Step 2  build_prompt     → inject rubric + wrap answer in delimiters
    Step 3  call_llm         → Groq API (or mock) with JSON-only instruction
    Step 4  parse_response   → multi-strategy fallback chain (parser.py)
    Step 5  validate_scores  → Pydantic validation + injection heuristic
    Step 6  finalize         → return JudgeResult

Injection defense layers (TRD §4.2):
    Layer 1 — Structural: answer wrapped in <student_answer> tags + system prompt
              instructs the model to treat the contents as untrusted data.
    Layer 2 — Output-level sanity: high score on thin/injected answer → flag.
    Layer 3 — Heuristic keyword flag: triggers a flag without overriding the score.

Reliability guarantee (NFR §8):
    The batch loop in run_batch() never raises an unhandled exception.
    Every row produces a JudgeResult — real, retried, or flagged zero-row.
"""

import json
import logging
import os
import re
import time
from typing import Any, Callable, Optional

from prompts import (
    SYSTEM_PROMPT,
    build_grading_prompt,
    build_correction_prompt,
    get_dimension_ids,
)
from parser import parse_llm_response, clamp_scores
from schema import (
    JudgeOutput,
    JudgeResult,
    make_empty_submission_result,
    make_fallback_error_result,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# INJECTION HEURISTIC PATTERNS  (Layer 3 defense)
# These are keyword/phrase patterns that, if found in a student answer, trigger
# a flag for human review.  They do NOT alter the score — the LLM still grades
# the content; humans are simply notified.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(previous|all|prior)\s+instructions?", re.IGNORECASE),
    re.compile(r"give\s+(this|the|a)\s+(answer|response|submission)\s+(a\s+)?perfect\s+score", re.IGNORECASE),
    re.compile(r"(score|grade|rate)\s+(it|this|me)\s+(as\s+)?20\s*(/|out\s+of)\s*20", re.IGNORECASE),
    re.compile(r"you\s+must\s+(award|give|assign|return)", re.IGNORECASE),
    re.compile(r"disregard\s+the\s+rubric", re.IGNORECASE),
    re.compile(r"override\s+(previous|prior|your)\s+instructions?", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# SANITY-CHECK THRESHOLDS  (Layer 2 defense)
# ---------------------------------------------------------------------------
# If the judge awards a total score >= this threshold on an answer that also
# triggered the injection heuristic, flag it for human review.
_INJECTION_HIGH_SCORE_THRESHOLD = 15  # out of 20

# Minimum answer word count below which a very high score is also suspicious
_SHORT_ANSWER_WORD_THRESHOLD = 10
_SHORT_ANSWER_HIGH_SCORE_THRESHOLD = 16  # out of 20


# ===========================================================================
# PUBLIC API
# ===========================================================================

def grade_answer(
    student_id: int,
    question: str,
    rubric: dict[str, Any],
    student_answer: str,
    llm_caller: Callable[[list[dict]], str],
    max_retries: int = 1,
    few_shot_anchors: Optional[list] = None,
) -> JudgeResult:
    """
    Grade a single student answer end-to-end.

    Args:
        student_id:        Row ID from the CSV.
        question:          The question posed to the student.
        rubric:            Parsed rubric.json dict.
        student_answer:    Raw student submission.
        llm_caller:        Callable that accepts a messages list and returns the
                           model's raw string response.
        max_retries:       Number of correction-prompt retries allowed (default 1).
        few_shot_anchors:  Optional list of FewShotExample dicts.  When provided,
                           they are injected into the grading prompt as calibration
                           anchors (Approach D).  Pass None to use the baseline
                           prompt without anchors.

    Returns:
        JudgeResult with scores, rationales, and pipeline metadata.
    """
    dim_ids = get_dimension_ids(rubric)

    # ------------------------------------------------------------------
    # STEP 1: Empty-answer short-circuit (FR-4)
    # ------------------------------------------------------------------
    if not student_answer or not student_answer.strip():
        logger.info("Student %d: empty submission — short-circuiting.", student_id)
        return make_empty_submission_result(student_id)

    # ------------------------------------------------------------------
    # STEP 2: Build prompt
    # ------------------------------------------------------------------
    injection_triggered = _check_injection_heuristic(student_answer)
    if injection_triggered:
        logger.warning(
            "Student %d: injection-pattern heuristic triggered.  "
            "Proceeding to grade content; row will be flagged.",
            student_id,
        )

    grading_prompt = build_grading_prompt(
        question, rubric, student_answer, few_shot_anchors=few_shot_anchors
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": grading_prompt},
    ]

    # ------------------------------------------------------------------
    # STEP 3 + 4 + 5: LLM call → parse → validate, with retry loop
    # ------------------------------------------------------------------
    raw_response: Optional[str] = None
    parsed: Optional[dict] = None
    retries_used = 0

    for attempt in range(max_retries + 1):
        try:
            raw_response = llm_caller(messages)
        except Exception as exc:
            logger.error(
                "Student %d, attempt %d: LLM call failed — %s",
                student_id, attempt, exc,
            )
            if attempt < max_retries:
                continue
            return make_fallback_error_result(
                student_id, f"LLM call exception after {attempt + 1} attempts: {exc}"
            )

        parsed = parse_llm_response(raw_response)

        if parsed is not None:
            # Clamp scores as defense-in-depth before Pydantic validation
            parsed = clamp_scores(parsed, dim_ids)
            break

        # Parse failed — prepare correction prompt for the next attempt
        if attempt < max_retries:
            logger.warning(
                "Student %d: parse failed on attempt %d, sending correction prompt.",
                student_id, attempt,
            )
            correction = build_correction_prompt(grading_prompt, raw_response or "")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": correction},
            ]
            retries_used += 1

    if parsed is None:
        return make_fallback_error_result(
            student_id,
            f"JSON parsing failed after {max_retries + 1} attempt(s). "
            f"Last raw response: {(raw_response or '')[:300]}",
        )

    # ------------------------------------------------------------------
    # STEP 5: Pydantic validation
    # ------------------------------------------------------------------
    try:
        judge_output = JudgeOutput(**parsed)
    except Exception as exc:
        logger.error(
            "Student %d: Pydantic validation failed — %s\nParsed dict: %s",
            student_id, exc, parsed,
        )
        return make_fallback_error_result(
            student_id, f"Pydantic validation error: {exc}"
        )

    # ------------------------------------------------------------------
    # STEP 5 (continued): Sanity checks + injection flags
    # ------------------------------------------------------------------
    flagged = False
    flag_reasons: list[str] = []

    # Layer 3: Keyword heuristic
    if injection_triggered:
        flagged = True
        flag_reasons.append("Injection-pattern keyword detected in student answer.")

    # Layer 2: High score on injected answer
    if injection_triggered and judge_output.total_score >= _INJECTION_HIGH_SCORE_THRESHOLD:
        flag_reasons.append(
            f"High score ({judge_output.total_score}/20) on answer with detected "
            "injection pattern — requires human review."
        )

    # Layer 2: Suspiciously high score on very short answer
    word_count = len(student_answer.split())
    if (
        word_count < _SHORT_ANSWER_WORD_THRESHOLD
        and judge_output.total_score >= _SHORT_ANSWER_HIGH_SCORE_THRESHOLD
    ):
        flagged = True
        flag_reasons.append(
            f"High score ({judge_output.total_score}/20) on very short answer "
            f"({word_count} words) — sanity check."
        )

    # ------------------------------------------------------------------
    # STEP 6: Finalize and return
    # ------------------------------------------------------------------
    return JudgeResult(
        student_id=student_id,
        output=judge_output,
        llm_called=True,
        flagged_for_review=flagged,
        flag_reason="; ".join(flag_reasons) if flag_reasons else None,
        retries_used=retries_used,
    )


def run_batch(
    rows: list[dict[str, Any]],
    question: str,
    rubric: dict[str, Any],
    llm_caller: Callable[[list[dict]], str],
    max_retries: int = 1,
    few_shot_anchors: Optional[list] = None,
    request_delay: float = 3.0,
) -> list[JudgeResult]:
    """
    Grade a list of student answer rows.

    The batch loop NEVER raises an unhandled exception.  Every row produces a
    JudgeResult, even if it is a zero-score fallback.

    Args:
        rows:              List of dicts with at least 'id' and 'student_answer' keys.
        question:          The exam question.
        rubric:            Parsed rubric.json.
        llm_caller:        LLM abstraction callable.
        max_retries:       Passed through to grade_answer.
        few_shot_anchors:  Optional list of FewShotExample dicts (Approach D).
                           Passed through to grade_answer for every row.
        request_delay:     Seconds to sleep between API calls (default 3.0).
                           Prevents Groq free-tier 429 rate-limit errors without
                           relying on reactive backoff.  Set to 0 for mock runs
                           (auto-detected) or when using a paid-tier API key.

    Returns:
        List of JudgeResult, one per input row, in the same order.
    """
    results: list[JudgeResult] = []
    total = len(rows)

    for i, row in enumerate(rows, start=1):
        student_id = int(row["id"])
        answer = str(row.get("student_answer", ""))
        logger.info("Grading row %d/%d (student_id=%d)…", i, total, student_id)

        # The deterministic test double uses the row ID to select its fixture.
        set_student_id = getattr(llm_caller, "set_student_id", None)
        if callable(set_student_id):
            set_student_id(student_id)

        try:
            result = grade_answer(
                student_id=student_id,
                question=question,
                rubric=rubric,
                student_answer=answer,
                llm_caller=llm_caller,
                max_retries=max_retries,
                few_shot_anchors=few_shot_anchors,
            )
        except Exception as exc:
            # Absolute last-resort catch — should never be reached because
            # grade_answer itself handles all exceptions internally.
            logger.critical(
                "Unexpected exception for student_id=%d: %s", student_id, exc
            )
            result = make_fallback_error_result(
                student_id, f"Unexpected outer exception: {exc}"
            )

        results.append(result)

        # Proactive rate-limit courtesy delay between API calls.
        # Skipped for: last row (no next call to protect),
        # and when request_delay is 0.
        is_last_row = (i == total)
        if request_delay > 0 and not is_last_row and result.llm_called:
            logger.info("Rate-limit delay: sleeping %.1fs before next request.", request_delay)
            time.sleep(request_delay)
        _log_result_summary(result)

    return results


# ===========================================================================
# PRIVATE HELPERS
# ===========================================================================

def _check_injection_heuristic(student_answer: str) -> bool:
    """
    Layer 3 injection defense: lightweight keyword/pattern scan of the answer.

    Returns True if any injection-pattern is detected.  Does NOT alter scoring.
    """
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(student_answer):
            return True
    return False


def _log_result_summary(result: JudgeResult) -> None:
    """Log a one-line summary of the result for monitoring."""
    total = result.output.total_score if result.output else "N/A"
    flag_str = f" [FLAGGED: {result.flag_reason}]" if result.flagged_for_review else ""
    err_str = f" [ERROR: {result.error}]" if result.error else ""
    llm_str = "LLM" if result.llm_called else "NO-LLM"
    logger.info(
        "  → student_id=%d | total=%s | %s | retries=%d%s%s",
        result.student_id,
        total,
        llm_str,
        result.retries_used,
        flag_str,
        err_str,
    )
