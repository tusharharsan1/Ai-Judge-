"""
parser.py
=========
Robust JSON parsing for LLM responses.

Why this module exists:
  LLMs sometimes return Markdown-fenced code blocks, partial JSON, or invalid JSON
  even when instructed not to.  A single bad response must never crash a batch run
  (NFR: Reliability).  This module implements the ordered fallback chain described
  in the PRD (TRD §5):

    1. json.loads() on the raw response   ← best case, fast
    2. Strip markdown fences, retry       ← handles ```json ... ``` wrapping
    3. Regex-extract first {...} block    ← handles leading/trailing prose
    4. Signal failure to the caller       ← triggers the correction-prompt retry

  Callers (judge.py) handle the correction-prompt retry and the final fallback
  zero-row, keeping this module focused purely on parsing.

Design notes:
  - All functions are pure / side-effect-free — easy to unit-test.
  - Logging is used instead of print() so callers can control verbosity.
  - No LLM calls happen here; no schema imports needed — raw dict output only.
"""

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINT
# ---------------------------------------------------------------------------
def parse_llm_response(raw: str) -> Optional[dict]:
    """
    Attempt to parse a raw LLM string into a Python dict via a multi-step fallback.

    Returns:
        Parsed dict on success, or None if all attempts fail.

    Steps attempted in order:
        1. json.loads on the raw string (trimmed)
        2. json.loads after stripping markdown fences (```json ... ```)
        3. json.loads on the first {...} block extracted by regex
    """
    if not raw or not raw.strip():
        logger.warning("parse_llm_response: received empty or whitespace-only response.")
        return None

    # --- Step 1: Direct parse ---
    result = _try_json_loads(raw.strip(), label="direct")
    if result is not None:
        return result

    # --- Step 1b: Strip <think>...</think> block (Qwen extended-thinking mode) ---
    stripped_think = _strip_think_block(raw)
    if stripped_think != raw:
        result = _try_json_loads(stripped_think.strip(), label="think-stripped")
        if result is not None:
            return result
        # Try further strategies on the think-stripped text
        raw_for_further = stripped_think
    else:
        raw_for_further = raw

    # --- Step 2: Strip markdown fences ---
    stripped = _strip_markdown_fences(raw_for_further)
    if stripped != raw_for_further.strip():
        result = _try_json_loads(stripped, label="fence-stripped")
        if result is not None:
            return result

    # --- Step 3: Regex extract first {...} block ---
    extracted = _extract_first_json_object(raw_for_further)
    if extracted is not None:
        result = _try_json_loads(extracted, label="regex-extracted")
        if result is not None:
            return result

    logger.error(
        "parse_llm_response: all parsing strategies failed.\n"
        "Raw response (first 500 chars): %s",
        raw[:500],
    )
    return None


# ---------------------------------------------------------------------------
# PRIVATE HELPERS
# ---------------------------------------------------------------------------
def _try_json_loads(text: str, label: str = "") -> Optional[dict]:
    """
    Attempt json.loads; return the parsed dict or None on failure.
    Only returns dicts — if the JSON root is a list or scalar, returns None.
    """
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
        logger.warning(
            "_try_json_loads [%s]: parsed JSON is not a dict (got %s).",
            label,
            type(parsed).__name__,
        )
        return None
    except json.JSONDecodeError as exc:
        logger.debug("_try_json_loads [%s]: JSONDecodeError — %s", label, exc)
        return None


def _strip_markdown_fences(text: str) -> str:
    """
    Remove leading/trailing markdown code fences.

    Handles:
        ```json\\n...\\n```
        ```\\n...\\n```
        ~~~json\\n...\\n~~~
    """
    text = text.strip()
    # Match optional language specifier after the fence opener
    fence_pattern = re.compile(
        r"^(?:```|~~~)[a-zA-Z]*\s*\n(.*?)\n(?:```|~~~)\s*$",
        re.DOTALL,
    )
    match = fence_pattern.match(text)
    if match:
        return match.group(1).strip()
    return text


def _strip_think_block(text: str) -> str:
    """
    Remove <think>...</think> blocks from model output.

    Qwen and other reasoning models emit an extended-thinking chain inside
    <think> tags before outputting their final response.  These blocks must
    be stripped before JSON parsing — otherwise the parser fails on the
    reasoning prose and wastes a correction-prompt retry.

    Handles:
        <think>\\n...reasoning...\\n</think>\\n{...json...}
        <think>...inline reasoning...</think>{...json...}

    Returns the text after the closing </think> tag (stripped), or the
    original text unchanged if no <think> block is found.
    """
    think_pattern = re.compile(
        r"<think>.*?</think>",
        re.DOTALL | re.IGNORECASE,
    )
    cleaned = think_pattern.sub("", text).strip()
    if cleaned != text.strip():
        logger.debug("_strip_think_block: removed <think> block, %d chars stripped.", len(text) - len(cleaned))
    return cleaned


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Find and extract the first top-level {...} block from a string using a
    bracket-counting approach.  More reliable than a greedy regex for nested JSON.

    Returns:
        The first {...} substring (including braces) or None if not found.
    """
    start = text.find("{")
    if start == -1:
        logger.debug("_extract_first_json_object: no '{' found in response.")
        return None

    depth = 0
    in_string = False
    escape_next = False

    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    logger.debug(
        "_extract_first_json_object: found opening '{' but no matching '}'."
    )
    return None


# ---------------------------------------------------------------------------
# SCORE CLAMPING  (defense-in-depth after Pydantic validation)
# ---------------------------------------------------------------------------
def clamp_scores(data: dict, dim_ids: list[str], min_val: int = 0, max_val: int = 5) -> dict:
    """
    Clamp dimension scores to [min_val, max_val] in-place on the parsed dict.

    This runs AFTER Pydantic validation as a second safety layer, primarily
    guarding against:
      - Mock / test paths that bypass the Pydantic model
      - Future rubric changes where the scale might differ per dimension

    Args:
        data:     Parsed response dict (mutated in-place and returned).
        dim_ids:  List of dimension ID strings from the rubric.
        min_val:  Minimum allowed score (default 0).
        max_val:  Maximum allowed score (default 5).

    Returns:
        The (mutated) data dict.
    """
    for dim_id in dim_ids:
        if dim_id in data and isinstance(data[dim_id], dict):
            raw_score = data[dim_id].get("score")
            if isinstance(raw_score, (int, float)):
                clamped = max(min_val, min(max_val, int(raw_score)))
                if clamped != raw_score:
                    logger.warning(
                        "clamp_scores: '%s' score %s clamped to %s.",
                        dim_id, raw_score, clamped,
                    )
                data[dim_id]["score"] = clamped
    return data
