"""Per-call-type model selection (tunable without touching code).

The simulation makes many high-volume, low-complexity structured-output calls
(one effort decision per agent per round) alongside far fewer, higher-stakes
calls (reflection synthesis, NormCompiler). `models.yaml` maps each call type to
a model id + sampling params; defaults below tier cheap/high-volume calls to
Haiku and quality-sensitive/low-volume calls to Sonnet, but every entry can be
overridden per deployment without a code change.
"""

from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel


class LLMCallType(StrEnum):
    EFFORT_DECISION = "effort_decision"
    PROPOSAL = "proposal"
    VOTE = "vote"
    COUNCILLOR_REPLY = "councillor_reply"
    ELECTION_DECISION = "election_decision"
    MONITOR_REVIEW = "monitor_review"
    DISCLOSURE = "disclosure"
    IMPORTANCE_RATING = "importance_rating"
    REFLECTION = "reflection"
    NORM_COMPILER = "norm_compiler"


class ModelParams(BaseModel):
    model: str
    max_tokens: int = 1024
    temperature: float = 1.0


HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-5"

_DEFAULTS: dict[LLMCallType, ModelParams] = {
    LLMCallType.EFFORT_DECISION: ModelParams(model=HAIKU, max_tokens=256, temperature=1.0),
    # The vote itself is just a short ballot-number id (`chosen_id`), but the
    # prompt now shows candidates as full (policy + operationalization) text
    # -- kept above the old 256 for models that "think" before emitting the
    # tool call (see the Ollama config for why this matters more there).
    LLMCallType.VOTE: ModelParams(model=HAIKU, max_tokens=1024, temperature=1.0),
    LLMCallType.ELECTION_DECISION: ModelParams(model=HAIKU, max_tokens=256, temperature=1.0),
    LLMCallType.IMPORTANCE_RATING: ModelParams(model=HAIKU, max_tokens=64, temperature=0.0),
    LLMCallType.DISCLOSURE: ModelParams(model=HAIKU, max_tokens=256, temperature=1.0),
    # Fills personal_norm + community_proposal in one call -- how to
    # operationalize the proposal is no longer asked here; it's worked out
    # afterward in a back-and-forth with the fishery councillor (see
    # sim.engine.run_operationalization_discussion_phase).
    LLMCallType.PROPOSAL: ModelParams(model=SONNET, max_tokens=2048, temperature=1.0),
    # One free-text reply per councillor turn -- similar shape/length to a
    # VOTE call's headroom, not a load-bearing structured extraction.
    LLMCallType.COUNCILLOR_REPLY: ModelParams(model=HAIKU, max_tokens=1024, temperature=1.0),
    LLMCallType.MONITOR_REVIEW: ModelParams(model=SONNET, max_tokens=512, temperature=1.0),
    LLMCallType.REFLECTION: ModelParams(model=SONNET, max_tokens=1024, temperature=1.0),
    LLMCallType.NORM_COMPILER: ModelParams(model=SONNET, max_tokens=1024, temperature=0.0),
}


class ModelConfig(BaseModel):
    calls: dict[LLMCallType, ModelParams]

    @classmethod
    def default(cls) -> "ModelConfig":
        return cls(calls=dict(_DEFAULTS))

    def for_call(self, call_type: LLMCallType) -> ModelParams:
        return self.calls[call_type]


def load_model_config(path: str | Path) -> ModelConfig:
    """Load models.yaml, falling back to defaults for any call type left unspecified."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    calls = dict(_DEFAULTS)
    for call_name, override in raw.get("calls", {}).items():
        call_type = LLMCallType(call_name)
        base = calls[call_type].model_dump()
        base.update(override)
        calls[call_type] = ModelParams(**base)
    return ModelConfig(calls=calls)
