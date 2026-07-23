"""Build-order step 2 exploratory script: attempts to reproduce Gupta et al.'s
qualitative findings (Fig 2/3) using rule-based (non-LLM) agents.

KNOWN LIMITATION -- this does NOT currently reproduce the paper's pattern.
`RuleBasedDecisionSource` uses a simplified "nudge effort toward my own
belief" heuristic instead of the paper's actual mechanism: payoff-biased
social learning, where agents periodically adopt a peer's entire strategy
tuple (effort, monitoring, belief, punish-propensity) via a logit rule
weighted by payoff comparison. That selection dynamic is what the paper
credits with making punishment sustain cooperation; without it, punishment
here just adds penalty on top of already-unsustainable harvesting and
collapses the fishery *faster*, not slower. Fixing this requires implementing
that imitation step -- deliberately deferred (see project decision in build
session), not done here.

Governance-primitive note: the old peer-chosen `DecentralizedPunishment`
("should I punish someone?", asked of every agent every round) has been
replaced project-wide by automatic, rule-triggered `PenalisePrimitive`
enforcement (see `models/norms.py`). There is no more per-agent punishment
*decision* to compare on/off -- `RuleBasedDecisionSource` no longer has one
at all. "punishment_on" here means "a CapPrimitive + PenalisePrimitive pair
is active" (anyone who individually overharvests past a fixed per-round cap
is automatically fined `beta`), the closest equivalent under the new design;
"punisher_cost" (a cost to whoever caught the violator) has no equivalent
since nobody catches anyone anymore -- it's automatic.

The engine mechanics themselves (harvest/regrowth arithmetic, penalties
resolving before starvation, collapse conditions) are validated instead by
`tests/test_engine.py`, which is the real source of truth for step 2.

Usage: uv run python scripts/validate_gupta_replication.py
"""

import asyncio
import statistics
from pathlib import Path

import yaml

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.norms import CapPrimitive, PenalisePrimitive
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


async def run_trial(config: FisheryConfig, *, punishment_on: bool, beta: float, seed: int) -> int:
    state = FisheryState.initial(config)
    if punishment_on:
        # Illustrative only (not paper-calibrated, same caveat as beta
        # itself): a per-agent fair-share cap, half of K split evenly --
        # scaled to *this* config rather than a fixed constant, so the cap
        # stays meaningful (neither trivially unreachable nor immediately
        # violated by everyone) whichever environment (`r`, `k`) is passed in.
        cap_value = 0.5 * config.k / len(config.initial_agent_ids)
        state.active_norms = [
            CapPrimitive(id="cap", scope="individual", basis="fixed_units", value=cap_value),
            PenalisePrimitive(
                id="pen", scope="collective", trigger="exceed_cap", penalty_type="fine_units",
                value=beta, destination="pool",
            ),
        ]
    else:
        state.active_norms = []
    decisions = RuleBasedDecisionSource(config.initial_agent_ids, seed=seed)
    events = InMemoryEventSink()
    return await run_simulation(state, decisions, events)


async def sweep(config: FisheryConfig, *, punishment_on: bool, beta: float, trials: int) -> list[int]:
    return [
        await run_trial(config, punishment_on=punishment_on, beta=beta, seed=seed)
        for seed in range(trials)
    ]


async def main() -> None:
    print(f"Running {TRIALS} trials per condition (rule-based agents, no LLM calls)...")
    print(
        "KNOWN LIMITATION: the rule-based proxy here omits payoff-biased social\n"
        "learning, so it is NOT expected to reproduce the paper's pattern -- see\n"
        "module docstring. Engine mechanics are validated separately by\n"
        "tests/test_engine.py.\n"
    )

    # (a) Punishment on vs off, at the paper's default beta=10, rich environment r=0.6.
    base_config = load_replication_config()
    on_times = await sweep(base_config, punishment_on=True, beta=10.0, trials=TRIALS)
    off_times = await sweep(base_config, punishment_on=False, beta=10.0, trials=TRIALS)

    print("(a) Punishment sustains cooperation (paper Fig 2)")
    print(f"    punishment ON  (beta=10): mean survival = {statistics.mean(on_times):.1f} rounds")
    print(f"    punishment OFF          : mean survival = {statistics.mean(off_times):.1f} rounds")
    verdict_a = "PASS" if statistics.mean(on_times) > statistics.mean(off_times) else "FAIL"
    print(f"    -> {verdict_a}: punishment-on survives longer than punishment-off\n")

    # (b) Beta x r sweep (paper Fig 3): beta in {10, 14}, r in {0.2 (harsh), 0.6 (rich)}.
    print("(b) Punishment strength x growth rate interaction (paper Fig 3)")
    results: dict[tuple[float, float], float] = {}
    for r in (0.2, 0.6):
        env_config = load_replication_config(r=r)
        for beta in (10.0, 14.0):
            times = await sweep(env_config, punishment_on=True, beta=beta, trials=TRIALS)
            mean_t = statistics.mean(times)
            results[(beta, r)] = mean_t
            print(f"    beta={beta:>4}, r={r}: mean survival = {mean_t:.1f} rounds")

    stronger_helps_harsh = results[(14.0, 0.2)] >= results[(10.0, 0.2)]
    stronger_helps_rich = results[(14.0, 0.6)] >= results[(10.0, 0.6)]
    verdict_b = "PASS" if (stronger_helps_harsh and stronger_helps_rich) else "FAIL (non-monotonic, consistent with paper's reported non-linearity)"
    print(f"    -> {verdict_b}: stronger punishment (beta=14) >= weaker (beta=10) within each environment\n")


if __name__ == "__main__":
    asyncio.run(main())
