"""SE agent client selection, mirroring `llm/provider.py`'s env-var pattern.

`OPENCODE_MODEL_ID` defaults to whatever model the fishing agents themselves
use for `LLMCallType.PROPOSAL` in the active `ModelConfig` -- so the SE agent
runs on the same backend LLM by default, without needing a second config kept
in sync by hand. Override it only to deliberately run it on a different model.
"""

import os
from pathlib import Path

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.se_agent.client import HttpSEAgentClient

DEFAULT_OPENCODE_SERVER_URL = "http://127.0.0.1:4096"
DEFAULT_OPENCODE_AGENT = "fishery-se-agent"
DEFAULT_OPENCODE_PROVIDER_ID = "ollama"

# The repo root (not just `backend/`) -- `implement_norm` verifies success by
# checking whether git's HEAD actually moved, and this is a single git repo
# with `backend/` as a subdirectory of it, not its own repo.
REPO_DIR = Path(__file__).resolve().parents[4]


def build_default_se_agent_client(model_config: ModelConfig) -> HttpSEAgentClient:
    base_url = os.environ.get("OPENCODE_SERVER_URL", DEFAULT_OPENCODE_SERVER_URL)
    agent = os.environ.get("OPENCODE_AGENT", DEFAULT_OPENCODE_AGENT)
    provider_id = os.environ.get("OPENCODE_PROVIDER_ID", DEFAULT_OPENCODE_PROVIDER_ID)
    model_id = os.environ.get("OPENCODE_MODEL_ID") or model_config.for_call(LLMCallType.PROPOSAL).model
    return HttpSEAgentClient(base_url, agent=agent, provider_id=provider_id, model_id=model_id, repo_dir=REPO_DIR)
