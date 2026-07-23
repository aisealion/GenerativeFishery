from __future__ import annotations

import json

from loguru import logger

from governance_engine.interfaces import Primitive
from governance_engine.primitives import PRIMITIVE_REGISTRY
from governance_engine.utils import strip_json_fences

_EXTRACTION_PROMPT = """You are a governance rule parser. Extract structured institutional primitives from the rule text below.

Available primitives and their required parameter schemas:

1. cap — {"basis": "fixed_units|pct_of_stock|pct_of_total_catch|sustainable_yield", "value": float, "scope": "per_agent_per_round|cumulative_over_N_rounds|total_pool_per_round", "dynamic": bool, "N": int|null}
2. declare — {"timing": "pre_round|post_round|within_N_hours", "content": "intended_quota|actual_take|violation_observed", "visibility": "public|anonymous|ledger_only"}
3. monitor — {"method": "peer|rotating_role|central_board|automated|random_audit", "frequency": "every_round|random|triggered", "target": "individual|total_pool", "ledger": bool}
4. penalise — {"trigger": "exceed_cap|fail_declare|fail_monitor_duty", "type": "forfeit|fine_units|quota_reduction|temporary_ban|proportional_fine", "value": float, "duration": int, "destination": "pool|communal_fund|redistribute_equal"}
5. adjust — {"trigger": "collective_threshold_breached|end_of_round|cumulative_over_N_rounds", "target": "individual_quota|pool_cap|group_cap", "direction": "reduce|restore|recalculate", "value": float, "restore_condition": str|null}
6. redistribute — {"source": "violator|communal_fund|pool", "destination": "pool|all_agents_equal|communal_fund|all_agents_proportional", "trigger": "violation|end_of_round|threshold", "amount": "excess_units|fixed|proportional"}
7. assign_role — {"role": "monitor|auditor|ledger_keeper|whistleblower", "selection": "rotating|random|elected", "duration": int, "obligation": str}

Example rule: "Each fisher may harvest at most 20 units per round. A rotating monitor checks compliance each round. Anyone who exceeds the limit forfeits the excess to the communal pool."

Example output:
[
  {"primitive": "cap", "parameters": {"basis": "fixed_units", "value": 20, "scope": "per_agent_per_round", "dynamic": false, "N": null}},
  {"primitive": "monitor", "parameters": {"method": "rotating_role", "frequency": "every_round", "target": "individual", "ledger": true}},
  {"primitive": "penalise", "parameters": {"trigger": "exceed_cap", "type": "forfeit", "value": 1.0, "duration": 1, "destination": "communal_fund"}}
]

Rule to extract:
{rule_text}

Respond with ONLY a valid JSON array. No markdown fences, no explanation, no preamble. Omit any primitive you cannot confidently extract parameters for rather than guessing."""


class RuleExtractor:
    def __init__(self, llm_client, model_name: str, primitive_whitelist: list[str]):
        self._client = llm_client
        self._model = model_name
        self._whitelist = set(primitive_whitelist)

    def _call_llm(self, messages: list[dict], context: str = "", attempt: str = "") -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        content = response.choices[0].message.content.strip()
        prompt_text = messages[-1].get("content", "")
        label = f"{context} {attempt}".strip()
        logger.bind(log_type="extraction").info(
            f"{label}\n"
            f"{'─'*38} PROMPT {'─'*38}\n"
            f"{prompt_text}\n"
            f"{'─'*37} RESPONSE {'─'*37}\n"
            f"{content}\n"
            f"{'═'*84}"
        )
        return content

    def extract(self, rule_text: str, context: str = "") -> list[dict]:
        prompt = _EXTRACTION_PROMPT.replace("{rule_text}", rule_text)
        messages = [{"role": "user", "content": prompt}]

        raw = ""
        try:
            raw = self._call_llm(messages, context=context, attempt="attempt=1")
            result = json.loads(strip_json_fences(raw))
            if not isinstance(result, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"RuleExtractor: first parse attempt failed ({e}), retrying")
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": (
                    f"Your response failed to parse as JSON: {e}. "
                    "Return ONLY the raw JSON array with no surrounding text."
                ),
            })
            try:
                raw = self._call_llm(messages, context=context, attempt="attempt=2(retry)")
                result = json.loads(strip_json_fences(raw))
                if not isinstance(result, list):
                    raise ValueError("Expected a JSON array")
            except Exception as retry_err:
                logger.error(f"RuleExtractor: retry also failed ({retry_err}). Returning empty list.")
                return []

        filtered = [
            item for item in result
            if isinstance(item, dict) and item.get("primitive") in self._whitelist
        ]
        logger.bind(log_type="governance").info(
            f"[extractor] EXTRACTED {len(filtered)} primitives:"
            f" {[p.get('primitive') for p in filtered]}\n"
            f"  rule: '{rule_text[:120]}{'…' if len(rule_text) > 120 else ''}'"
        )
        return filtered

    def instantiate_primitives(self, primitive_dicts: list[dict]) -> list[Primitive]:
        instances: list[Primitive] = []
        for pdata in primitive_dicts:
            primitive_name = pdata.get("primitive", "")
            cls = PRIMITIVE_REGISTRY.get(primitive_name)
            if cls is None:
                logger.warning(f"RuleExtractor: unknown primitive '{primitive_name}', skipping")
                continue
            try:
                instance = cls(pdata.get("parameters", {}))
                instances.append(instance)
            except Exception as e:
                logger.warning(f"RuleExtractor: failed to instantiate {primitive_name} ({e}), skipping")
        return instances
