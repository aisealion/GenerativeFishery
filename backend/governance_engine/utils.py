from __future__ import annotations

import re


def strip_json_fences(text: str) -> str:
    """Remove markdown code fences that LLMs often wrap JSON responses in.

    Handles:
        ```json { ... } ```
        ```     { ... } ```
        ` { ... } `        (single backtick, rare)
    Returns the inner content stripped of surrounding whitespace.
    """
    text = text.strip()
    # Triple-backtick fence (with optional language tag)
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Single-backtick wrapping
    match = re.match(r"^`([\s\S]*?)`$", text)
    if match:
        return match.group(1).strip()
    return text
