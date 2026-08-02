"""Per-fishery physical/economic parameters (build spec §2, §6).

Nothing here is hardcoded in engine logic -- a faithful single-fishery
replication of Gupta et al. (N_min == starting population, R_min == 0) and a
looser multi-fishery research run are both expressible purely through config.
"""

from pydantic import BaseModel, Field


class FisheryConfig(BaseModel):
    fishery_id: str

    # Gupta et al. effort/harvest/regrowth parameters. `alpha`/`r`/`consumption`
    # allow 0 for degenerate edge-case configs (no fishing effect, no regrowth,
    # no survival cost); `k` must stay strictly positive to avoid dividing by
    # zero in the logistic regrowth term.
    alpha: float = Field(ge=0, description="Fishing efficiency")
    r: float = Field(ge=0, description="Logistic growth rate")
    k: float = Field(gt=0, description="Carrying capacity")

    initial_stock: float = Field(gt=0)
    consumption: float = Field(ge=0, description="Units of fish an agent needs per round to survive")

    initial_agent_ids: list[str]

    # Collapse thresholds (§6) -- explicit config, not implicit constants.
    r_min: float = Field(ge=0, description="Stock level at/below which the fishery collapses")
    n_min: int = Field(ge=0, description="Population below which the fishery collapses")

    max_rounds: int | None = Field(default=None, description="Optional hard stop for a run")

    # Gupta et al. Table 2 persona initialization: fraction of initial_agent_ids
    # assigned an altruistic starting persona (the rest selfish) -- see
    # sim.personas.assign_personas. 0.5 (the paper's default mixed condition)
    # unless a config overrides it.
    altruism_ratio: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Fraction of agents starting with an altruistic persona"
    )

    # After each agent proposes a norm, the fishery SE agent and that agent
    # go back and forth this many turn-pairs before the agent's norm is
    # finalized for voting (see sim.engine.run_operationalization_discussion_phase).
    operationalization_discussion_rounds: int = Field(
        default=5, ge=1, description="SE agent Q / fishing agent A turn-pairs before a proposal is finalized"
    )
