"""
llm_client.py
=============
LLM provider abstraction for the AI Judge.

This module provides:
  1. A real Groq API caller (groq_caller) — production path.

Design decision:
  The LLM caller is a plain callable (messages: list[dict]) -> str.
  This means judge.py has zero knowledge of which provider is in use —
  swapping Groq for OpenAI, Anthropic, or a local model requires changing
  only this file, not the pipeline.

Usage:
  # Production
  from llm_client import make_groq_caller
  caller = make_groq_caller(model="llama-3.3-70b-versatile")
"""

import json
import logging
import os
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GROQ REAL CALLER
# ---------------------------------------------------------------------------

def make_groq_caller(
    model: str = "qwen/qwen3.6-27b",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    request_timeout: float = 60.0,
) -> Callable[[list[dict]], str]:
    """
    Factory that returns a Groq API caller.

    Why Groq:
      - Free tier is sufficient for the 15 required rows.
      - Very fast inference (low latency per call).
      - response_format json_object is enabled only for models that reliably
        support it; others use plain-text mode with parser fallback.

    Why temperature=0.0:
      Determinism is preferred for a grading system — the same answer should
      produce the same score.  (Variance as a production risk is documented
      separately in the written summary / stretch goal.)

    Args:
        model:           Groq model ID.  qwen/qwen3.6-27b is the recommended default;
                         openai/gpt-oss-120b is a heavier alternative.
        api_key:         Groq API key.  Falls back to GROQ_API_KEY env var.
        temperature:     Sampling temperature (0.0 = fully deterministic).
        max_tokens:      Maximum response length (raised to 2048 for long few-shot prompts).
        request_timeout: Per-call timeout in seconds.

    Returns:
        Callable[[list[dict]], str] — pass messages, receive raw model string.

    Raises:
        ImportError:  If the `groq` package is not installed.
        ValueError:   If no API key is found.
    """
    try:
        from groq import Groq  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "The 'groq' package is required for the real LLM caller. "
            "Install it with:  pip install groq"
        ) from exc

    resolved_key = api_key or os.environ.get("GROQ_API_KEY")
    if not resolved_key:
        raise ValueError(
            "No Groq API key found.  Set the GROQ_API_KEY environment variable "
            "or pass api_key= explicitly."
        )

    client = Groq(api_key=resolved_key, timeout=request_timeout)

    # Models that reliably support response_format=json_object via Groq.
    # Qwen models return 400 json_validate_failed when json_object mode is
    # forced with complex/long prompts — use plain-text mode for them and
    # let the parser's 3-strategy fallback chain handle extraction.
    _JSON_MODE_MODELS = {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "groq/compound",
        "groq/compound-mini",
    }
    use_json_mode = model in _JSON_MODE_MODELS
    if use_json_mode:
        logger.info("Groq caller: json_object mode ENABLED for model '%s'.", model)
    else:
        logger.info(
            "Groq caller: plain-text mode for model '%s' — parser fallback active.", model
        )

    def caller(messages: list[dict]) -> str:
        kwargs: dict = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        logger.debug("Groq raw response: %s", content[:300])
        return content or ""

    return caller


