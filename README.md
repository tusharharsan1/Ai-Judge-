# AI Judge Implementation

A robust, production-ready AI grading pipeline built for the AI Engineer Assessment. This pipeline evaluates student answers against a rubric using Large Language Models (LLMs).

## Features
- **Robust Parsing**: Advanced JSON extraction with a 3-strategy fallback chain (handles Markdown fences, thinking blocks, and regex extraction).
- **Few-Shot Anchoring (Approach D)**: Calibrates the LLM using gold-standard and edge-case examples to heavily reduce bias without extra API calls.
- **Adversarial Red-Teaming (Approach F)**: Includes a full suite of prompt injection tests covering social engineering, Unicode homoglyphs, and nested tags.
- **Resilience**: Gracefully handles rate limits (`429 Too Many Requests`) and empty submissions (short-circuits to save tokens).

## Quickstart

**1. Setup Environment**
Ensure you have Python 3.10+ installed. Install the requirements:
```bash
pip install pandas pydantic groq
```

**2. Configure API Key**
Create a `.env` file in the root directory (or export it in your shell):
```
GROQ_API_KEY=your_groq_api_key_here
```

**3. Run the Pipeline**
Run the grading pipeline on the 15 required rows:
```bash
# Run with live Groq API
python3 main.py


# Adjust API rate limit delay if needed (default is 3.0s)
python3 main.py --request-delay 0
```

## Testing
The project includes a comprehensive test suite for the parser and adversarial injection cases.

```bash
# Run parser unit tests
python3 tests/test_parser.py

# Run adversarial injection test suite
python3 tests/test_adversarial.py
```

## Architecture

- `main.py`: CLI entrypoint and pipeline orchestrator.
- `judge.py`: Core grading logic, LLM calling, and retry handling.
- `parser.py`: Resilient JSON extraction and schema validation.
- `prompts.py`: Dynamic assembly of system instructions, rubric, and few-shot examples.
- `few_shot_examples.py`: Curated high/low quality anchors for prompt calibration.
- `llm_client.py`: LLM provider abstraction (Groq).
- `metrics.py`: Computes MAE, tolerance accuracy, and dimension biases.
- `schema.py`: Pydantic models defining the expected LLM output structure.
