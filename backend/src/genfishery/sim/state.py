"""In-memory fishery state (build spec §2, §6).

This is the mutable "ground truth for the current tick" that phase resolution
functions read and write. It is not itself persisted -- the event log is; this
is reconstructible from replaying events, and today is held in-process per the
build spec's phase-registry model.
"""

import random
from dataclasses import dataclass, field

from genfishery.config.fishery_config import FisheryConfig
from genfishery.models.norms import NormPrimitive
from genfishery.sim.personas import assign_personas

NO_NORM_YET = "(no norm has been established yet)"


@dataclass
class AgentState:
    agent_id: str
    payoff: float = 0.0
    alive: bool = True
    last_effort: float | None = None
    last_harvest: float | None = None
    # Payoff immediately after this round's harvest-consumption update, before
    # any punishment/dispute adjustment -- lets `run_starvation_check`
    # distinguish "already negative from harvest alone" (underharvest) from
    # "harvest covered consumption, punishment tipped them over" (punished).
    # Overwritten every round in `run_harvest_phase`, so always fresh by the
    # time starvation is checked for a currently-alive agent.
    payoff_after_harvest: float | None = None


@dataclass
class FisheryState:
    config: FisheryConfig
    stock: float
    round: int = 0
    agents: dict[str, AgentState] = field(default_factory=dict)
    # No norms active by default (project decision). Every primitive
    # (cap/declare/monitor/penalise/adjust/redistribute/assign_role/
    # peer_observability) is inert until a community actually votes one in;
    # the phase that would enforce it already no-ops cleanly when its
    # primitive type isn't present in this list.
    active_norms: list[NormPrimitive] = field(default_factory=list)
    collapsed: bool = False
    # Cause(s) of the most recent collapse ("resource_depletion",
    # "population_loss", "underharvest_death") -- set alongside `collapsed`
    # in `run_round`. `handle_collapse_and_migrate` reads this to decide
    # whether to un-pause after migrating: an underharvest_death collapse is
    # terminal (exactly one migrant, then stays paused for good), while the
    # other causes remain "trigger, not necessarily terminal" per build spec
    # §7 point 6.
    collapse_reasons: list[str] = field(default_factory=list)
    # Natural-language norm text (build spec §4/§5): tracked separately from
    # `active_norms` (the compiled, mechanically-enforced primitives) because
    # a norm that fails to compile is still believed/descriptive-only -- the
    # text updates regardless of whether compilation succeeded.
    agent_norms: dict[str, str] = field(default_factory=dict)
    group_norm_text: str = NO_NORM_YET
    # Role holders (assigned via an active AssignRolePrimitive): role_name ->
    # agent_id. Separate from `active_norms` (which holds the AssignRole
    # *definition*) since who currently holds a role changes over time
    # (elections, rotation, random reassignment) without the primitive
    # itself being recompiled.
    roles: dict[str, str] = field(default_factory=dict)
    # role_name -> round its rotating/random selection schedule started, so
    # the current holder is a pure function of elapsed rounds (no separate
    # "tick" state to drift).
    role_rotation_start: dict[str, int] = field(default_factory=dict)
    # Pooled balance an active RedistributePrimitive can route penalties
    # into (destination="communal_fund") and later redistribute out of.
    communal_fund: float = 0.0
    # An active AdjustPrimitive's current effective per-agent harvest
    # ceiling, applied alongside (the tighter of the two wins) any active
    # CapPrimitive during harvest enforcement. None until an adjust trigger
    # first fires.
    adjusted_quota: float | None = None
    # NormPrimitive.id -> the raw natural-language text it was compiled from
    # (build spec §9: UI renders active norms "from NormSpec.raw_text, not the
    # raw JSON"). A primitive never voted on (there are none by default now)
    # would have no entry here.
    norm_source_text: dict[str, str] = field(default_factory=dict)
    # agent_id -> "altruistic" | "selfish", the persona type each agent was
    # initialized with (Gupta et al. Table 2) -- reference/UI data; the actual
    # behavioral effect is through `persona_descriptions`' text, stated in
    # every decision prompt.
    persona_types: dict[str, str] = field(default_factory=dict)
    # agent_id -> the Table 2 template text for that agent's persona_type.
    # Fixed at initialization and never overwritten -- unlike `agent_norms`
    # (the agent's own evolving personal strategy), this stays constant for
    # the agent's whole lifetime, so their underlying disposition remains
    # visible in the prompt even after they've revised their stated strategy.
    persona_descriptions: dict[str, str] = field(default_factory=dict)
    # agent_id -> "underharvest" | "punished" for every agent who has
    # starved, set once by `run_starvation_check` and never cleared -- lets
    # the roster block in every prompt state why a removed villager is gone.
    starvation_reasons: dict[str, str] = field(default_factory=dict)
    # agent_id -> the round (in *this* fishery's own clock) a migrated-in
    # agent arrived, set once by `migrate_random_survivor`. Lets the roster
    # block tag them as a newcomer for a few rounds before folding back into
    # a plain "active" entry, same as any other villager.
    migration_arrival_round: dict[str, int] = field(default_factory=dict)

    @classmethod
    def initial(cls, config: FisheryConfig, *, seed: int | None = None) -> "FisheryState":
        rng = random.Random(seed)
        persona_descriptions, persona_types = assign_personas(
            config.initial_agent_ids, config.altruism_ratio, rng=rng
        )
        return cls(
            config=config,
            stock=config.initial_stock,
            agents={aid: AgentState(agent_id=aid) for aid in config.initial_agent_ids},
            persona_descriptions=persona_descriptions,
            persona_types=persona_types,
        )

    @property
    def alive_agents(self) -> list[AgentState]:
        return [a for a in self.agents.values() if a.alive]
