"""
schema.py
=========
Pydantic v2 data models for the AI Judge output.

Design decisions (all defensible on the live review call):

1. total_score is ALWAYS recomputed by the system from the dimension scores.
   The model's arithmetic is never trusted — a model_validator enforces this.
   This prevents a prompt-injection attack from inflating the total even if it
   somehow changed one dimension score.

2. score fields use Field(ge=0, le=5) for strict range validation at parse time.
   A separate post-parse clamp in the pipeline (judge.py) provides defense-in-
   depth in case the Pydantic layer is bypassed (e.g. in mocked/test paths).

3. rationale is a required non-empty string — empty rationales would pass the
   functional test but fail the qualitative audit requirement.

4. The schema is driven by the rubric's dimension IDs.  However, because Pydantic
   requires field names to be known at class-definition time, the four concrete
   fields below match the four dimensions in rubric.json exactly.
   If the rubric gains new dimensions the schema must be updated — this is
   called out explicitly as a known coupling (and acceptable at this scale).

5. llm_called and flagged_for_review are output metadata, not grading data;
   they live in JudgeResult (the wrapper), not JudgeOutput (the model response).
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional


# ---------------------------------------------------------------------------
# PER-DIMENSION SCORE MODEL
# ---------------------------------------------------------------------------
class DimensionScore(BaseModel):
    """Score and rationale for a single rubric dimension."""

    score: int = Field(
        ...,
        ge=0,
        le=5,
        description="Integer score for this dimension (0–5 inclusive).",
    )
    rationale: str = Field(
        ...,
        min_length=1,
        description="1–3 sentence justification referencing specific answer content.",
    )

    class Config:
        # Allow construction from dict (important for the JSON → Pydantic path)
        from_attributes = True


# ---------------------------------------------------------------------------
# JUDGE OUTPUT MODEL  (mirrors the LLM JSON contract exactly)
# ---------------------------------------------------------------------------
class JudgeOutput(BaseModel):
    """
    Structured output produced by the LLM for a single student answer.

    Field names MUST match the dimension IDs in rubric.json so the prompt's
    JSON schema and this model stay in sync.
    """

    content_accuracy: DimensionScore
    example_evidence: DimensionScore
    clarity_organization: DimensionScore
    scientific_vocabulary: DimensionScore

    total_score: int = Field(
        ...,
        ge=0,
        le=20,
        description=(
            "Overall total (0–20).  "
            "The system ALWAYS recomputes this as the sum of the four dimension "
            "scores and overwrites whatever the model returned here."
        ),
    )

    @model_validator(mode="after")
    def enforce_computed_total(self) -> "JudgeOutput":
        """
        System-computed total always wins over the model's claimed total_score.

        This is a deliberate design statement: LLM arithmetic is unreliable and
        can be manipulated by prompt-injection (e.g. "give me 20/20").
        The validator silently corrects any discrepancy rather than raising an
        error, so the pipeline never crashes due to a math mistake in the response.
        """
        computed = (
            self.content_accuracy.score
            + self.example_evidence.score
            + self.clarity_organization.score
            + self.scientific_vocabulary.score
        )
        if self.total_score != computed:
            self.total_score = computed
        return self

    def dimension_scores(self) -> dict[str, int]:
        """Return {dimension_id: score} dict for easy metric computation."""
        return {
            "content_accuracy": self.content_accuracy.score,
            "example_evidence": self.example_evidence.score,
            "clarity_organization": self.clarity_organization.score,
            "scientific_vocabulary": self.scientific_vocabulary.score,
        }

    def dimension_rationales(self) -> dict[str, str]:
        """Return {dimension_id: rationale} dict for review/logging."""
        return {
            "content_accuracy": self.content_accuracy.rationale,
            "example_evidence": self.example_evidence.rationale,
            "clarity_organization": self.clarity_organization.rationale,
            "scientific_vocabulary": self.scientific_vocabulary.rationale,
        }


# ---------------------------------------------------------------------------
# JUDGE RESULT  (full pipeline output for one student answer)
# ---------------------------------------------------------------------------
class JudgeResult(BaseModel):
    """
    Complete pipeline output for a single student-answer row.

    Combines the grading scores (JudgeOutput) with pipeline metadata
    (whether the LLM was called, any flags raised, errors encountered).
    """

    student_id: int = Field(..., description="Row ID from the CSV.")

    # Grading output — None only when the fallback zero-row path is taken
    output: Optional[JudgeOutput] = Field(
        default=None,
        description="Parsed and validated LLM output.  None on hard failure.",
    )

    # Pipeline metadata
    llm_called: bool = Field(
        default=False,
        description=(
            "False when the answer was short-circuited (e.g. empty submission) "
            "or when the result came from the fallback zero-row path."
        ),
    )
    flagged_for_review: bool = Field(
        default=False,
        description=(
            "True if the answer triggered an injection heuristic, a sanity-check "
            "failure, or a parse/retry failure.  Does not alter the score — it "
            "surfaces the row for human inspection."
        ),
    )
    flag_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason for the flag, if flagged.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if parsing failed after exhausting all retries.",
    )
    retries_used: int = Field(
        default=0,
        description="Number of correction-prompt retries consumed (max 1 by design).",
    )

    # Convenience accessors that flatten the nested structure for DataFrame export
    def to_flat_dict(self) -> dict:
        """
        Flatten to a dict suitable for a pandas row.

        Produces columns:
            student_id, judge_content_accuracy, judge_example_evidence,
            judge_clarity_organization, judge_scientific_vocabulary,
            judge_total, rationale_*, llm_called, flagged_for_review,
            flag_reason, error, retries_used
        """
        if self.output is None:
            # Fallback: zero everything out
            dim_scores = {
                "judge_content_accuracy": 0,
                "judge_example_evidence": 0,
                "judge_clarity_organization": 0,
                "judge_scientific_vocabulary": 0,
            }
            dim_rationales = {
                "rationale_content_accuracy": "Pipeline error — no output produced.",
                "rationale_example_evidence": "Pipeline error — no output produced.",
                "rationale_clarity_organization": "Pipeline error — no output produced.",
                "rationale_scientific_vocabulary": "Pipeline error — no output produced.",
            }
            judge_total = 0
        else:
            dim_scores = {
                f"judge_{k}": v for k, v in self.output.dimension_scores().items()
            }
            dim_rationales = {
                f"rationale_{k}": v
                for k, v in self.output.dimension_rationales().items()
            }
            judge_total = self.output.total_score

        return {
            "student_id": self.student_id,
            **dim_scores,
            "judge_total": judge_total,
            **dim_rationales,
            "llm_called": self.llm_called,
            "flagged_for_review": self.flagged_for_review,
            "flag_reason": self.flag_reason,
            "error": self.error,
            "retries_used": self.retries_used,
        }


# ---------------------------------------------------------------------------
# FALLBACK ZERO-ROW FACTORY
# ---------------------------------------------------------------------------
def make_empty_submission_result(student_id: int) -> JudgeResult:
    """
    Create a deterministic zero-score result for empty/whitespace-only answers.

    No LLM call is ever made for this path — cheaper and more reliable than
    trusting the model to recognise emptiness.  (FR-4 in the PRD.)
    """
    zero_output = JudgeOutput(
        content_accuracy=DimensionScore(
            score=0, rationale="No content submitted."
        ),
        example_evidence=DimensionScore(
            score=0, rationale="No content submitted."
        ),
        clarity_organization=DimensionScore(
            score=0, rationale="No content submitted."
        ),
        scientific_vocabulary=DimensionScore(
            score=0, rationale="No content submitted."
        ),
        total_score=0,
    )
    return JudgeResult(
        student_id=student_id,
        output=zero_output,
        llm_called=False,
        flagged_for_review=False,
    )


def make_fallback_error_result(
    student_id: int, error_message: str
) -> JudgeResult:
    """
    Create a zero-score flagged result when all parse/retry attempts are exhausted.

    The batch loop never raises an unhandled exception — this is the safe landing
    spot for any row that completely fails.  (NFR: Reliability, PRD §8.)
    """
    return JudgeResult(
        student_id=student_id,
        output=None,
        llm_called=True,
        flagged_for_review=True,
        flag_reason="Parse failure after exhausting retry budget.",
        error=error_message,
    )
