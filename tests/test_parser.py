"""
tests/test_parser.py
====================
Unit tests for parser.py — the JSON parsing fallback chain.

These tests are fully self-contained (no API key, no rubric.json, no pydantic).
They test the pure parsing logic in isolation.

Run:
    cd ai_judge_implementation
    python3 -m pytest tests/test_parser.py -v

    # or standalone:
    python3 tests/test_parser.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from parser import (
    parse_llm_response,
    _try_json_loads,
    _strip_markdown_fences,
    _extract_first_json_object,
    clamp_scores,
)


# ---------------------------------------------------------------------------
# _try_json_loads
# ---------------------------------------------------------------------------
def test_json_loads_valid_dict():
    result = _try_json_loads('{"a": 1, "b": "hello"}')
    assert result == {"a": 1, "b": "hello"}

def test_json_loads_invalid():
    result = _try_json_loads("not json at all")
    assert result is None

def test_json_loads_list_returns_none():
    # We only accept dict roots — list JSON should return None
    result = _try_json_loads('[1, 2, 3]')
    assert result is None

def test_json_loads_number_returns_none():
    result = _try_json_loads('42')
    assert result is None

def test_json_loads_empty_dict():
    result = _try_json_loads('{}')
    assert result == {}

def test_json_loads_nested():
    payload = json.dumps({"score": {"value": 3, "rationale": "good"}})
    result = _try_json_loads(payload)
    assert result["score"]["value"] == 3


# ---------------------------------------------------------------------------
# _strip_markdown_fences
# ---------------------------------------------------------------------------
def test_strip_backtick_json_fence():
    fenced = "```json\n{\"a\": 1}\n```"
    assert _strip_markdown_fences(fenced) == '{"a": 1}'

def test_strip_plain_backtick_fence():
    fenced = "```\n{\"a\": 1}\n```"
    assert _strip_markdown_fences(fenced) == '{"a": 1}'

def test_strip_tilde_fence():
    fenced = "~~~json\n{\"a\": 1}\n~~~"
    assert _strip_markdown_fences(fenced) == '{"a": 1}'

def test_no_fence_unchanged():
    plain = '{"a": 1}'
    assert _strip_markdown_fences(plain) == plain

def test_strip_with_whitespace():
    fenced = "  ```json\n{\"a\": 1}\n```  "
    # After strip() in the function, should still match
    assert _strip_markdown_fences(fenced.strip()) == '{"a": 1}'


# ---------------------------------------------------------------------------
# _extract_first_json_object
# ---------------------------------------------------------------------------
def test_extract_simple():
    text = 'Here is the result: {"score": 3} Hope that helps.'
    assert _extract_first_json_object(text) == '{"score": 3}'

def test_extract_nested():
    text = 'Result: {"outer": {"inner": 1}} done.'
    extracted = _extract_first_json_object(text)
    assert extracted == '{"outer": {"inner": 1}}'

def test_extract_no_brace():
    assert _extract_first_json_object("no json here") is None

def test_extract_unclosed():
    # No closing brace — should return None
    assert _extract_first_json_object('{"unclosed": 1') is None

def test_extract_with_string_containing_braces():
    # Braces inside a JSON string should not confuse the counter
    text = 'Before {"key": "value with {braces} inside"} after'
    extracted = _extract_first_json_object(text)
    parsed = json.loads(extracted)
    assert parsed["key"] == "value with {braces} inside"

def test_extract_first_of_multiple():
    # Should extract only the FIRST complete {...} block
    text = '{"first": 1} and then {"second": 2}'
    extracted = _extract_first_json_object(text)
    assert extracted == '{"first": 1}'


# ---------------------------------------------------------------------------
# parse_llm_response (end-to-end fallback chain)
# ---------------------------------------------------------------------------
def test_parse_direct():
    payload = json.dumps({"content_accuracy": {"score": 4, "rationale": "good"}})
    result = parse_llm_response(payload)
    assert result is not None
    assert result["content_accuracy"]["score"] == 4

def test_parse_fenced():
    payload = "```json\n" + json.dumps({"score": 3}) + "\n```"
    result = parse_llm_response(payload)
    assert result is not None
    assert result["score"] == 3

def test_parse_with_leading_prose():
    payload = "Here is my grading result:\n" + json.dumps({"a": 1}) + "\nHope this helps!"
    result = parse_llm_response(payload)
    assert result is not None
    assert result["a"] == 1

def test_parse_garbage_returns_none():
    assert parse_llm_response("this is complete garbage!!") is None

def test_parse_empty_returns_none():
    assert parse_llm_response("") is None
    assert parse_llm_response("   ") is None
    assert parse_llm_response(None) is None  # type: ignore

def test_parse_fenced_with_prose():
    payload = (
        "I analyzed the answer carefully. Here is my assessment:\n"
        "```json\n"
        '{"content_accuracy": {"score": 5, "rationale": "excellent"}}\n'
        "```\n"
        "Please let me know if you need clarification."
    )
    result = parse_llm_response(payload)
    assert result is not None
    assert result["content_accuracy"]["score"] == 5

def test_parse_realistic_judge_response():
    """A realistic full judge response — tests end-to-end happy path."""
    payload = json.dumps({
        "content_accuracy": {"score": 4, "rationale": "Mechanism partially explained."},
        "example_evidence": {"score": 3, "rationale": "Example present but underdeveloped."},
        "clarity_organization": {"score": 4, "rationale": "Clear structure."},
        "scientific_vocabulary": {"score": 3, "rationale": "Some technical terms used."},
        "total_score": 14,
    })
    result = parse_llm_response(payload)
    assert result is not None
    assert result["total_score"] == 14
    assert len(result) == 5  # 4 dims + total_score


# ---------------------------------------------------------------------------
# clamp_scores
# ---------------------------------------------------------------------------
def test_clamp_in_range():
    data = {"content_accuracy": {"score": 3}, "example_evidence": {"score": 5}}
    result = clamp_scores(data, ["content_accuracy", "example_evidence"])
    assert result["content_accuracy"]["score"] == 3
    assert result["example_evidence"]["score"] == 5

def test_clamp_above_max():
    data = {"content_accuracy": {"score": 7}}
    result = clamp_scores(data, ["content_accuracy"], max_val=5)
    assert result["content_accuracy"]["score"] == 5

def test_clamp_below_min():
    data = {"content_accuracy": {"score": -2}}
    result = clamp_scores(data, ["content_accuracy"], min_val=0)
    assert result["content_accuracy"]["score"] == 0

def test_clamp_ignores_missing_dim():
    # If a dim_id is not in the data, clamp_scores should not crash
    data = {"content_accuracy": {"score": 3}}
    result = clamp_scores(data, ["content_accuracy", "nonexistent_dim"])
    assert result["content_accuracy"]["score"] == 3

def test_clamp_float_rounded():
    # Float scores should be cast to int and clamped
    data = {"content_accuracy": {"score": 4.9}}
    result = clamp_scores(data, ["content_accuracy"])
    assert isinstance(result["content_accuracy"]["score"], int)
    assert result["content_accuracy"]["score"] == 4


# ---------------------------------------------------------------------------
# STANDALONE RUNNER
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_functions = [
        # _try_json_loads
        test_json_loads_valid_dict,
        test_json_loads_invalid,
        test_json_loads_list_returns_none,
        test_json_loads_number_returns_none,
        test_json_loads_empty_dict,
        test_json_loads_nested,
        # _strip_markdown_fences
        test_strip_backtick_json_fence,
        test_strip_plain_backtick_fence,
        test_strip_tilde_fence,
        test_no_fence_unchanged,
        test_strip_with_whitespace,
        # _extract_first_json_object
        test_extract_simple,
        test_extract_nested,
        test_extract_no_brace,
        test_extract_unclosed,
        test_extract_with_string_containing_braces,
        test_extract_first_of_multiple,
        # parse_llm_response
        test_parse_direct,
        test_parse_fenced,
        test_parse_with_leading_prose,
        test_parse_garbage_returns_none,
        test_parse_empty_returns_none,
        test_parse_fenced_with_prose,
        test_parse_realistic_judge_response,
        # clamp_scores
        test_clamp_in_range,
        test_clamp_above_max,
        test_clamp_below_min,
        test_clamp_ignores_missing_dim,
        test_clamp_float_rounded,
    ]

    print("=" * 60)
    print("  PARSER UNIT TESTS")
    print(f"  Total: {len(test_functions)}")
    print("=" * 60)

    passed = 0
    failed = 0
    for fn in test_functions:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {fn.__name__}: AssertionError — {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {fn.__name__}: {type(e).__name__} — {e}")
            failed += 1

    print()
    print(f"  {passed}/{len(test_functions)} passed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
