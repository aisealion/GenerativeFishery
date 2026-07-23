from __future__ import annotations

import json

from loguru import logger

from governance_engine.utils import strip_json_fences

_CATEGORIZATION_PROMPT = """\
You are analysing governance rules proposed by members of a fishing commons after a resource collapse.

For each rule, do TWO things per governance dimension:
  a) Write a brief description of what the rule says about that dimension (or "none stated").
  b) Write a clean, formal one-sentence policy clause extracted or paraphrased from the rule.
     This clause must stand alone as a self-contained policy statement.
     If the rule says nothing about that dimension, use an empty string "".

Dimensions to analyse:
1. LIMITS    — what extraction is capped or constrained
2. MONITORS  — how compliance is tracked (peer, ledger, rotating role, etc.)
3. SANCTIONS — what penalty applies for violations
4. RESPONSIBLE — who carries out enforcement
5. CONDITIONAL — whether the rule adapts based on pool state (true/false)

Do not use any external framework. Only categorise what is explicitly stated in the rule text.

Rules to categorise:
{rules_block}

Return a JSON array with exactly one object per rule, in the same order as the rules above:
[
  {{
    "rule_key": "R1",
    "limits": "<brief description or 'none stated'>",
    "limits_clause": "<formal policy sentence about the extraction limit, or ''>",
    "monitors": "<brief description or 'none stated'>",
    "monitors_clause": "<formal policy sentence about the monitoring mechanism, or ''>",
    "sanctions": "<brief description or 'none stated'>",
    "sanctions_clause": "<formal policy sentence about the penalty for violations, or ''>",
    "responsible": "<brief description or 'unclear'>",
    "responsible_clause": "<formal policy sentence about who enforces compliance, or ''>",
    "conditional": <true if the rule adapts to pool state, false if fixed>,
    "conditional_clause": "<formal policy sentence stating the condition trigger, or ''>"
  }},
  ...
]

Return ONLY the JSON array. No markdown fences, no explanation, no preamble."""

_CLAUSE_KEYS = ("limits", "monitors", "sanctions", "responsible", "conditional")


def _parse_cat(item: dict, index: int) -> dict:
    """Normalise a raw categorisation dict, filling in defaults for missing keys."""
    if not isinstance(item, dict):
        item = {}
    result: dict = {"rule_key": item.get("rule_key", f"R{index + 1}")}
    for key in _CLAUSE_KEYS:
        result[key] = item.get(key, "none stated" if key != "conditional" else False)
        result[f"{key}_clause"] = item.get(f"{key}_clause", "")
    result["conditional"] = bool(result["conditional"])
    return result


class RuleCategorizer:
    def __init__(self, llm_client, model_name: str):
        self._client = llm_client
        self._model = model_name

    def _call_llm(self, messages: list[dict], attempt: str = "") -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        content = response.choices[0].message.content.strip()
        logger.bind(log_type="governance").info(
            f"[rule_categorizer | {attempt}]\n"
            f"{'─'*38} PROMPT {'─'*38}\n"
            f"{messages[-1].get('content', '')}\n"
            f"{'─'*37} RESPONSE {'─'*37}\n"
            f"{content}\n"
            f"{'═'*84}"
        )
        return content

    def categorize(self, proposals: list[tuple[str, str]]) -> list[dict]:
        """Categorize (agent_id, rule_text) proposals.

        Returns one categorisation dict per proposal (same order).
        Each dict includes a ``{dim}_clause`` field — a formal policy sentence
        covering only what that rule says about that dimension.
        """
        if not proposals:
            return []

        rules_block = "\n".join(
            f"R{i + 1} (proposed by {agent_id}): {rule_text}"
            for i, (agent_id, rule_text) in enumerate(proposals)
        )
        prompt = _CATEGORIZATION_PROMPT.replace("{rules_block}", rules_block)
        messages = [{"role": "user", "content": prompt}]

        raw = ""
        try:
            raw = self._call_llm(messages, attempt="attempt=1")
            result = json.loads(strip_json_fences(raw))
            if not isinstance(result, list):
                raise ValueError("Expected a JSON array")
            return [_parse_cat(item, i) for i, item in enumerate(result)]

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"RuleCategorizer: first attempt failed ({e}), retrying")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your response failed to parse as JSON: {e}. "
                    "Return ONLY the raw JSON array with no surrounding text."
                ),
            })
            try:
                raw = self._call_llm(messages, attempt="attempt=2(retry)")
                result = json.loads(strip_json_fences(raw))
                if not isinstance(result, list):
                    raise ValueError("Expected a JSON array")
                return [_parse_cat(item, i) for i, item in enumerate(result)]

            except Exception as retry_err:
                logger.error(
                    f"RuleCategorizer: retry also failed ({retry_err}). "
                    "Returning minimal fallback."
                )
                return [
                    _parse_cat({}, i)
                    for i in range(len(proposals))
                ]
