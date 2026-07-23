"""Norm primitive grammar.

A voted natural-language norm is extracted by `NormCompiler` into one or more
instances of this small, closed set of primitives. Nothing outside this set is
ever mechanically enforced -- text that doesn't map onto a primitive is logged
as `could_not_compile` and treated as descriptive-only.

This set (`cap`, `declare`, `monitor`, `penalise`, `adjust`, `redistribute`,
`assign_role`) replaced an earlier, more abstract 8-primitive grammar
(`threshold_rule`, `decentralized_punishment`, `role_grant`,
`pre_action_disclosure`, `review_schedule`, `boundary_rule`,
`dispute_phase`, plus two never-wired primitives) -- ported in from a
separate, more fine-grained primitive design (see `backend/governance_engine/`
for the original domain-agnostic reference implementation this was adapted
from). `peer_observability` is this project's own addition (not part of that
design) and is unaffected by the swap. Punishment is no longer a per-round
agent decision (there is no more "should I punish someone?" LLM call) --
`penalise` triggers automatically off violations recorded during
enforcement, by project decision.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class NormPrimitive(BaseModel):
    id: str
    scope: Literal["collective", "individual"]


class CapPrimitive(NormPrimitive):
    """A hard ceiling on harvest, enforced at harvest time -- exceeding it
    doesn't just get logged (as the old `ThresholdRule`/cap-check did), the
    excess is actually clipped off the agent's catch.
    """

    type: Literal["cap"] = "cap"
    basis: Literal["fixed_units", "pct_of_stock", "pct_of_total_catch", "sustainable_yield"] = "fixed_units"
    value: float
    cap_scope: Literal["per_agent_per_round", "cumulative_over_N_rounds", "total_pool_per_round"] = (
        "per_agent_per_round"
    )
    dynamic: bool = False
    n_rounds: int | None = None


class DeclarePrimitive(NormPrimitive):
    """Villagers must publicly disclose something before/after a round --
    the mechanical descendant of the old `PreActionDisclosure`.
    """

    type: Literal["declare"] = "declare"
    timing: Literal["pre_round", "post_round", "within_N_hours"] = "pre_round"
    content: Literal["intended_quota", "actual_take", "violation_observed"] = "intended_quota"
    disclosure_visibility: Literal["public", "anonymous", "ledger_only"] = "public"


class MonitorPrimitive(NormPrimitive):
    type: Literal["monitor"] = "monitor"
    method: Literal["peer", "rotating_role", "central_board", "automated", "random_audit"] = "peer"
    frequency: Literal["every_round", "random", "triggered"] = "every_round"
    monitor_target: Literal["individual", "total_pool"] = "individual"
    ledger: bool = True


class PenalisePrimitive(NormPrimitive):
    """Automatic sanction triggered by a recorded violation -- no agent is
    asked whether to punish; this fires on its own once `trigger` matches a
    violation recorded that round (by project decision, replacing the old
    `DecentralizedPunishment`'s per-round "should I punish someone?" choice).
    """

    type: Literal["penalise"] = "penalise"
    trigger: Literal["exceed_cap", "fail_declare", "fail_monitor_duty"] = "exceed_cap"
    penalty_type: Literal["forfeit", "fine_units", "quota_reduction", "temporary_ban", "proportional_fine"] = (
        "forfeit"
    )
    value: float = 0.0
    duration: int = 1
    destination: Literal["pool", "communal_fund", "redistribute_equal"] = "pool"


class AdjustPrimitive(NormPrimitive):
    type: Literal["adjust"] = "adjust"
    trigger: Literal["collective_threshold_breached", "end_of_round", "cumulative_over_N_rounds"] = "end_of_round"
    adjust_target: Literal["individual_quota", "pool_cap", "group_cap"] = "individual_quota"
    direction: Literal["reduce", "restore", "recalculate"] = "reduce"
    value: float = 0.0
    restore_condition: str | None = None


class RedistributePrimitive(NormPrimitive):
    type: Literal["redistribute"] = "redistribute"
    source: Literal["violator", "communal_fund", "pool"] = "violator"
    destination: Literal["pool", "all_agents_equal", "communal_fund", "all_agents_proportional"] = (
        "all_agents_equal"
    )
    trigger: Literal["violation", "end_of_round", "threshold"] = "violation"
    amount_basis: Literal["excess_units", "fixed", "proportional"] = "excess_units"
    # Only meaningful when amount_basis="fixed" -- the ported reference
    # implementation had no field for this at all (a "fixed" amount_basis was
    # permanently a no-op there), which this closes.
    fixed_amount: float = 0.0


class AssignRolePrimitive(NormPrimitive):
    type: Literal["assign_role"] = "assign_role"
    role_name: Literal["monitor", "auditor", "ledger_keeper", "whistleblower"] = "monitor"
    selection: Literal["rotating", "random", "elected"] = "elected"
    duration: int = 5
    obligation: str = ""


class PeerObservability(NormPrimitive):
    """Makes every villager's fishing effort and payoff visible to the whole
    community in decision prompts. Off by default -- absent this primitive,
    a villager's prompt only shows their own effort and payoff. This
    project's own addition, not part of the ported primitive set above.
    """

    id: str = "peer_observability_default"
    scope: Literal["collective", "individual"] = "collective"
    type: Literal["peer_observability"] = "peer_observability"


# Discriminated union over the primitives, keyed on `type`.
AnyNormPrimitive = Annotated[
    Union[
        CapPrimitive,
        DeclarePrimitive,
        MonitorPrimitive,
        PenalisePrimitive,
        AdjustPrimitive,
        RedistributePrimitive,
        AssignRolePrimitive,
        PeerObservability,
    ],
    Field(discriminator="type"),
]


class NormSpec(BaseModel):
    id: str
    raw_text: str
    primitives: list[AnyNormPrimitive]
    adopted_round: int
