"""Councillor client selection, mirroring `llm/provider.py`'s env-var pattern.

`OPENCODE_MODEL_ID` defaults to whatever model the fishing agents themselves
use for `LLMCallType.PROPOSAL` in the active `ModelConfig` -- so the councillor
runs on the same backend LLM by default, without needing a second config kept
in sync by hand. Override it only to deliberately run the councillor on a
different model.
"""

import os

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.councillor.client import HttpCouncillorClient

DEFAULT_OPENCODE_SERVER_URL = "http://127.0.0.1:4096"
DEFAULT_OPENCODE_AGENT = "fishery-councillor"
DEFAULT_OPENCODE_PROVIDER_ID = "ollama"


def build_default_councillor_client(model_config: ModelConfig) -> HttpCouncillorClient:
    base_url = os.environ.get("OPENCODE_SERVER_URL", DEFAULT_OPENCODE_SERVER_URL)
    agent = os.environ.get("OPENCODE_AGENT", DEFAULT_OPENCODE_AGENT)
    provider_id = os.environ.get("OPENCODE_PROVIDER_ID", DEFAULT_OPENCODE_PROVIDER_ID)
    model_id = os.environ.get("OPENCODE_MODEL_ID") or model_config.for_call(LLMCallType.PROPOSAL).model
    return HttpCouncillorClient(base_url, agent=agent, provider_id=provider_id, model_id=model_id)
