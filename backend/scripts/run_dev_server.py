"""Runs the FastAPI app locally with a scripted FakeLLMClient instead of real
Claude calls -- for local frontend development / manual verification without
an Anthropic API key. Real deployments run `uvicorn genfishery.api.app:app`
directly, which uses AnthropicLLMClient.

Usage: uv run python scripts/run_dev_server.py
"""

import uvicorn

from genfishery.api.app import create_app
from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import LLMCallType
from genfishery.llm.fake_client import FakeLLMClient
from genfishery.memory.importance import ImportanceRating
from genfishery.memory.reflection import ReflectionInsight
from genfishery.sim.decisions import EffortDecision, ProposalDecision
from genfishery.sim.norm_compiler import NormCompilerOutput

# Deliberately harsher than live_demo_a.yaml: with every agent scripted to
# behave identically (fixed effort, no punishment), all agents in a fishery
# always have identical trajectories and starve in lockstep -- so a
# consumption-driven collapse always wipes the whole roster out at once,
# leaving no survivor to migrate. Zeroing consumption and setting r_min high
# relative to K instead makes collapse purely stock-triggered, which doesn't
# require anyone to starve -- so there are always survivors left to migrate,
# which is the point of this dev demo.
DEV_FISHERY_A = FisheryConfig(
    fishery_id="fishery_a",
    alpha=0.5,
    r=0.5,
    k=10.0,
    initial_stock=5.0,
    consumption=0.0,
    initial_agent_ids=["a1", "a2", "a3", "a4", "a5"],
    r_min=4.0,
    n_min=1,
    max_rounds=None,
)
DEV_FISHERY_B = FisheryConfig(
    fishery_id="fishery_b",
    alpha=0.05,
    r=0.5,
    k=200.0,
    initial_stock=200.0,
    consumption=0.0,
    initial_agent_ids=["b1", "b2", "b3", "b4", "b5"],
    r_min=0.0,
    n_min=1,
    max_rounds=None,
)


def make_dev_llm() -> FakeLLMClient:
    return FakeLLMClient(
        {
            LLMCallType.EFFORT_DECISION: EffortDecision(effort=0.35),
            LLMCallType.PUNISHMENT_DECISION: lambda response_model, system, prompt: response_model(
                target_id=None
            ),
            LLMCallType.IMPORTANCE_RATING: ImportanceRating(score=2.0),
            LLMCallType.REFLECTION: ReflectionInsight(
                insight="Fishing moderately keeps the lake healthy.", importance=3.0
            ),
            LLMCallType.PROPOSAL: ProposalDecision(
                personal_norm="fish moderately", community_proposal="Fish moderately to preserve the lake."
            ),
            LLMCallType.VOTE: lambda response_model, system, prompt: response_model(
                chosen_text="Fish moderately to preserve the lake."
            ),
            LLMCallType.NORM_COMPILER: NormCompilerOutput(primitives=[]),
        }
    )


app = create_app(
    llm_client=make_dev_llm(),
    fishery_configs=[DEV_FISHERY_A, DEV_FISHERY_B],
    round_interval_seconds=3.0,
)

if __name__ == "__main__":
    print("Prompt/response logs: logs/fishery_a.log, logs/fishery_b.log")
    uvicorn.run(app, host="127.0.0.1", port=8000)
