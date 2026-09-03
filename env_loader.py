"""Minimal .env loader for local development without an extra dependency."""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: Path, *, override: bool = False) -> bool:
    """Load simple KEY=VALUE entries from *path* into ``os.environ``.

    Existing environment variables win by default, so deployment-provided
    configuration is never replaced by a local `.env` file.
    """
    if not path.is_file():
        return False

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"Invalid .env entry on line {line_number}: expected KEY=VALUE.")

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid .env entry on line {line_number}: key is empty.")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value

    return True
