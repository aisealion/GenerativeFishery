"""NormCompiler: maps voted natural-language norm text onto instances of the
closed primitive set via one structured-output call.

Primitive set (`cap`, `declare`, `monitor`, `penalise`, `adjust`,
`redistribute`, `assign_role`, `peer_observability`) and extraction prompt
content ported from `backend/governance_engine/` (a separate, more
fine-grained reference implementation) -- adapted here to this project's
async structured-tool-call pattern instead of that folder's raw-JSON/sync
chat-completion + manual retry approach, so a malformed response is a single
`LLMStructuredCallError` like every other call in this codebase, not a
hand-rolled parse-and-retry loop.

Text that doesn't map to any primitive is `could_not_compile` -- logged as
data, never improvised into new enforcement logic. Compiled by hash of raw
text so identical text seen again doesn't re-invoke the LLM.
"""

import hashlib
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from genfishery.config.model_config import LLMCallType
from genfishery.llm.client import LLMClient
from genfishery.models.norms import (
    AdjustPrimitive,
    AssignRolePrimitive,
    CapPrimitive,
    DeclarePrimitive,
    MonitorPrimitive,
    NormSpec,
    PeerObservability,
    PenalisePrimitive,
    RedistributePrimitive,
)

NORM_COMPILER_SYSTEM_PROMPT = """You extract structured institutional primitives from a villager community's
voted natural-language policy. Only these primitive types exist right now:

- cap: a numeric ceiling on how much one villager (or the whole community)
  may harvest, e.g. "no one should take more than 20 units a round" or "cap
  total catch at 40% of the stock". Fields: scope ("collective" or
  "individual"), basis ("fixed_units", "pct_of_stock", "pct_of_total_catch",
  or "sustainable_yield"), value (a number -- units for fixed_units, a
  fraction like 0.4 for the pct_of_* / sustainable_yield bases), cap_scope
  ("per_agent_per_round", "cumulative_over_N_rounds", or
  "total_pool_per_round"), dynamic (true if the cap should move with the
  stock), n_rounds (only if cap_scope is cumulative_over_N_rounds,
  otherwise null).
- declare: villagers must publicly state something before or after fishing,
  e.g. "everyone must announce how much they intend to catch before the
  round" or "publish everyone's actual catch afterward". Fields: scope,
  timing ("pre_round", "post_round", or "within_N_hours"), content
  ("intended_quota", "actual_take", or "violation_observed"),
  disclosure_visibility ("public", "anonymous", or "ledger_only").
- monitor: how compliance gets checked, e.g. "a rotating villager checks
  everyone's catch each round" or "randomly audit villagers". Fields: scope,
  method ("peer", "rotating_role", "central_board", "automated", or
  "random_audit"), frequency ("every_round", "random", or "triggered"),
  monitor_target ("individual" or "total_pool"), ledger (true if a written
  record should be kept).
- penalise: an automatic sanction for a violation -- this fires on its own
  once the trigger condition is met, no villager is asked whether to punish.
  E.g. "anyone who exceeds their cap forfeits the excess" or "fine violators
  5 units". Fields: scope, trigger ("exceed_cap", "fail_declare", or
  "fail_monitor_duty"), penalty_type ("forfeit", "fine_units",
  "quota_reduction", "temporary_ban", or "proportional_fine"), value (a
  number -- ignored for "forfeit"), duration (rounds the penalty lasts,
  default 1), destination ("pool", "communal_fund", or
  "redistribute_equal" -- where the forfeited/fined amount goes).
- adjust: automatically changes a quota/cap based on the state of the lake,
  e.g. "reduce everyone's quota if the stock drops too low" or "recalculate
  the cap every round". Fields: scope, trigger
  ("collective_threshold_breached", "end_of_round", or
  "cumulative_over_N_rounds"), adjust_target ("individual_quota",
  "pool_cap", or "group_cap"), direction ("reduce", "restore", or
  "recalculate"), value (a number), restore_condition (free text, or null).
- redistribute: moves units from one place to villagers, e.g. "split
  forfeited catch equally among everyone" or "share the communal fund at the
  end of the round". Fields: scope, source ("violator", "communal_fund", or
  "pool"), destination ("pool", "all_agents_equal", "communal_fund", or
  "all_agents_proportional"), trigger ("violation", "end_of_round", or
  "threshold"), amount_basis ("excess_units", "fixed", or "proportional"),
  fixed_amount (only meaningful when amount_basis is "fixed" -- the number
  of units to redistribute, otherwise 0).
- assign_role: establishes a named role in the community, e.g. "the
  community should elect a monitor" or "rotate an auditor every 5 rounds".
  Fields: scope, role_name ("monitor", "auditor", "ledger_keeper", or
  "whistleblower"), selection ("rotating", "random", or "elected"),
  duration (rounds the role lasts), obligation (free text describing what
  the role holder must do, or "" if unstated).
- peer_observability: villagers can see each other's fishing effort and
  payoff, not just their own, e.g. "let's make everyone's effort and
  earnings visible to the whole community" or "we should be able to see
  what everyone else is catching". Fields: scope (always "collective").

If the policy text does not clearly map to any of these, return an empty list
of primitives. Do not invent a new kind of primitive and do not guess numeric
values that aren't implied by the text."""


class CompiledCap(BaseModel):
    type: Literal["cap"] = "cap"
    scope: Literal["collective", "individual"] = "collective"
    basis: Literal["fixed_units", "pct_of_stock", "pct_of_total_catch", "sustainable_yield"]
    value: float
    cap_scope: Literal["per_agent_per_round", "cumulative_over_N_rounds", "total_pool_per_round"] = (
        "per_agent_per_round"
    )
    dynamic: bool = False
    n_rounds: int | None = None


class CompiledDeclare(BaseModel):
    type: Literal["declare"] = "declare"
    scope: Literal["collective", "individual"] = "collective"
    timing: Literal["pre_round", "post_round", "within_N_hours"]
    content: Literal["intended_quota", "actual_take", "violation_observed"]
    disclosure_visibility: Literal["public", "anonymous", "ledger_only"]


class CompiledMonitor(BaseModel):
    type: Literal["monitor"] = "monitor"
    scope: Literal["collective", "individual"] = "collective"
    method: Literal["peer", "rotating_role", "central_board", "automated", "random_audit"]
    frequency: Literal["every_round", "random", "triggered"]
    monitor_target: Literal["individual", "total_pool"]
    ledger: bool = True


class CompiledPenalise(BaseModel):
    type: Literal["penalise"] = "penalise"
    scope: Literal["collective", "individual"] = "collective"
    trigger: Literal["exceed_cap", "fail_declare", "fail_monitor_duty"]
    penalty_type: Literal["forfeit", "fine_units", "quota_reduction", "temporary_ban", "proportional_fine"]
    value: float
    duration: int = 1
    destination: Literal["pool", "communal_fund", "redistribute_equal"]


class CompiledAdjust(BaseModel):
    type: Literal["adjust"] = "adjust"
    scope: Literal["collective", "individual"] = "collective"
    trigger: Literal["collective_threshold_breached", "end_of_round", "cumulative_over_N_rounds"]
    adjust_target: Literal["individual_quota", "pool_cap", "group_cap"]
    direction: Literal["reduce", "restore", "recalculate"]
    value: float
    restore_condition: str | None = None


class CompiledRedistribute(BaseModel):
    type: Literal["redistribute"] = "redistribute"
    scope: Literal["collective", "individual"] = "collective"
    source: Literal["violator", "communal_fund", "pool"]
    destination: Literal["pool", "all_agents_equal", "communal_fund", "all_agents_proportional"]
    trigger: Literal["violation", "end_of_round", "threshold"]
    amount_basis: Literal["excess_units", "fixed", "proportional"]
    fixed_amount: float = 0.0


class CompiledAssignRole(BaseModel):
    type: Literal["assign_role"] = "assign_role"
    scope: Literal["collective", "individual"] = "collective"
    role_name: Literal["monitor", "auditor", "ledger_keeper", "whistleblower"]
    selection: Literal["rotating", "random", "elected"]
    duration: int
    obligation: str = ""


class CompiledPeerObservability(BaseModel):
    type: Literal["peer_observability"] = "peer_observability"
    scope: Literal["collective", "individual"] = "collective"


CompiledPrimitive = Annotated[
    Union[
        CompiledCap,
        CompiledDeclare,
        CompiledMonitor,
        CompiledPenalise,
        CompiledAdjust,
        CompiledRedistribute,
        CompiledAssignRole,
        CompiledPeerObservability,
    ],
    Field(discriminator="type"),
]


class NormCompilerOutput(BaseModel):
    primitives: list[CompiledPrimitive] = Field(default_factory=list)


class NormCompiler:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._cache: dict[str, NormSpec] = {}

    async def compile(self, raw_text: str, *, norm_id: str, adopted_round: int) -> NormSpec:
        cache_key = hashlib.sha256(raw_text.encode()).hexdigest()
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        output = await self._llm.structured_call(
            call_type=LLMCallType.NORM_COMPILER,
            system=NORM_COMPILER_SYSTEM_PROMPT,
            prompt=f'Compile this community policy:\n\n"{raw_text}"',
            response_model=NormCompilerOutput,
        )

        primitives = []
        for i, compiled in enumerate(output.primitives):
            primitive_id = f"{norm_id}_p{i}"
            if isinstance(compiled, CompiledCap):
                primitives.append(
                    CapPrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        basis=compiled.basis,
                        value=compiled.value,
                        cap_scope=compiled.cap_scope,
                        dynamic=compiled.dynamic,
                        n_rounds=compiled.n_rounds,
                    )
                )
            elif isinstance(compiled, CompiledDeclare):
                primitives.append(
                    DeclarePrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        timing=compiled.timing,
                        content=compiled.content,
                        disclosure_visibility=compiled.disclosure_visibility,
                    )
                )
            elif isinstance(compiled, CompiledMonitor):
                primitives.append(
                    MonitorPrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        method=compiled.method,
                        frequency=compiled.frequency,
                        monitor_target=compiled.monitor_target,
                        ledger=compiled.ledger,
                    )
                )
            elif isinstance(compiled, CompiledPenalise):
                primitives.append(
                    PenalisePrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        trigger=compiled.trigger,
                        penalty_type=compiled.penalty_type,
                        value=compiled.value,
                        duration=compiled.duration,
                        destination=compiled.destination,
                    )
                )
            elif isinstance(compiled, CompiledAdjust):
                primitives.append(
                    AdjustPrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        trigger=compiled.trigger,
                        adjust_target=compiled.adjust_target,
                        direction=compiled.direction,
                        value=compiled.value,
                        restore_condition=compiled.restore_condition,
                    )
                )
            elif isinstance(compiled, CompiledRedistribute):
                primitives.append(
                    RedistributePrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        source=compiled.source,
                        destination=compiled.destination,
                        trigger=compiled.trigger,
                        amount_basis=compiled.amount_basis,
                        fixed_amount=compiled.fixed_amount,
                    )
                )
            elif isinstance(compiled, CompiledAssignRole):
                primitives.append(
                    AssignRolePrimitive(
                        id=primitive_id,
                        scope=compiled.scope,
                        role_name=compiled.role_name,
                        selection=compiled.selection,
                        duration=compiled.duration,
                        obligation=compiled.obligation,
                    )
                )
            elif isinstance(compiled, CompiledPeerObservability):
                primitives.append(PeerObservability(id=primitive_id, scope=compiled.scope))

        spec = NormSpec(
            id=norm_id, raw_text=raw_text, primitives=primitives, adopted_round=adopted_round
        )
        self._cache[cache_key] = spec
        return spec
