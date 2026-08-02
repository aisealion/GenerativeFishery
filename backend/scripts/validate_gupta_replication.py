"""Build-order step 2 exploratory script: baseline survival-time sweep using
rule-based (non-LLM) agents, no governance.

KNOWN LIMITATION -- this does NOT reproduce the paper's qualitative pattern.
`RuleBasedDecisionSource` uses a simplified "nudge effort toward my own
belief" heuristic instead of the paper's actual mechanism: payoff-biased
social learning, where agents periodically adopt a peer's entire strategy
tuple via a logit rule weighted by payoff comparison. Fixing this requires
implementing that imitation step -- deliberately deferred (see project
decision in build session), not done here.

The punishment on/off (Fig 2/3) comparison this script used to run has been
removed: it depended on a closed primitive governance layer
(`CapPrimitive`/`PenalisePrimitive`, `state.active_norms`) that no longer
exists -- a winning norm is now implemented as an actual code change by an
autonomous SE agent (see `sim/engine.py`'s module docstring), not a
scriptable primitive, so there's no generic "punishment_on" toggle left to
sweep against. What remains here is the baseline no-governance survival-time
sweep across environments, useful for eyeballing regrowth-rate sensitivity.

The engine mechanics themselves (harvest/regrowth arithmetic, collapse
conditions) are validated instead by `tests/test_engine.py`, which is the
real source of truth for step 2.

Usage: uv run python scripts/validate_gupta_replication.py
"""

import asyncio
import statistics
from pathlib import Path

import yaml

from genfishery.config.fishery_config import FisheryConfig
from genfishery.sim.engine import run_simulation
from genfishery.sim.events_sink import InMemoryEventSink
from genfishery.sim.policies import RuleBasedDecisionSource
from genfishery.sim.state import FisheryState

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "fisheries"
TRIALS = 100  # matches paper's I=100 iterations per condition


def load_replication_config(**overrides) -> FisheryConfig:
    raw = yaml.safe_load((CONFIGS_DIR / "replication.yaml").read_text())
    raw.update(overrides)
    return FisheryConfig(**raw)


async def run_trial(config: FisheryConfig, *, seed: int) -> int:
    state = FisheryState.initial(config)
    decisions = RuleBasedDecisionSource(config.initial_agent_ids, seed=seed)
    events = InMemoryEventSink()
    return await run_simulation(state, decisions, events)


async def sweep(config: FisheryConfig, *, trials: int) -> list[int]:
    return [await run_trial(config, seed=seed) for seed in range(trials)]


async def main() -> None:
    print(f"Running {TRIALS} trials per environment (rule-based agents, no LLM calls, no governance)...")
    print(
        "KNOWN LIMITATION: the rule-based proxy here omits payoff-biased social\n"
        "learning, so it is NOT expected to reproduce the paper's pattern -- see\n"
        "module docstring. Engine mechanics are validated separately by\n"
        "tests/test_engine.py.\n"
    )

    print("Baseline survival time across growth rates (no governance):")
    for r in (0.2, 0.6):
        env_config = load_replication_config(r=r)
        times = await sweep(env_config, trials=TRIALS)
        print(f"    r={r}: mean survival = {statistics.mean(times):.1f} rounds")


if __name__ == "__main__":
    asyncio.run(main())
