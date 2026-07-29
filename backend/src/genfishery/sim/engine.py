"""Round engine: Gupta et al.'s effort/harvest/regrowth equations + resolution
order fixed by build spec §5/§6, plus the primitive-based governance layer
ported from `backend/governance_engine/` (see `models/norms.py`'s module
docstring for the full rationale of that swap).

Resolution order: Strategy -> Declare (pre_round) -> Harvest (cap enforcement
happens inline here) -> Declare (post_round) -> Monitor -> Penalise (automatic,
off violations Harvest recorded) -> Redistribute -> Adjust -> Starvation check
-> Propose -> Vote -> NormCompiler (only if a new policy won) -> collapse
check. Propose/Vote/NormCompiler only run if a `NormCompiler` is supplied to
`run_round`/`run_simulation` -- omitting it (as the step-2/3 tests and the
rule-based validation script do) preserves the earlier, governance-free round
shape exactly. Declare/Monitor/Penalise/Redistribute/Adjust are called
unconditionally regardless, but no-op when their primitive type isn't active
(same pattern cap-check originally used) -- they only need `DecisionSource`,
not the governance-specific protocol; punishment is no longer a per-agent
decision, so none of them need `GovernanceDecisionSource` either.

Every proposal now bundles both what the policy should be AND how to
operationalize/enforce it in one structured call
(`ProposalDecision.community_proposal` + `.operationalization`), carried
through voting and into the compiler together as one `PolicyProposal`
candidate (`sim.decisions.proposal_candidate_key` is its display text on the
ballot, both fields folded into it; the vote itself picks a short ballot id
rather than reproducing this text). The winning proposal's operationalization
detail is appended directly to the text
`NormCompiler.compile` receives, so the compiled primitives reflect the
community's practical detail, not just the policy's headline text.
"""

import logging
import random
from typing import Literal

from genfishery.llm.client import LLMClient, LLMStructuredCallError
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.memory.writer import write_observation
from genfishery.models.events import Event, EventType, visible_events_for
from genfishery.models.norms import (
    AdjustPrimitive,
    AssignRolePrimitive,
    CapPrimitive,
    DeclarePrimitive,
    MonitorPrimitive,
    NormPrimitive,
    PeerObservability,
    PenalisePrimitive,
    RedistributePrimitive,
)
from genfishery.sim.decisions import PolicyProposal
from genfishery.sim.events_sink import EventSink, RoundEventCollector
from genfishery.sim.norm_compiler import NormCompiler
from genfishery.sim.observations import render_event_as_observation
from genfishery.sim.policies import DecisionSource, GovernanceDecisionSource
from genfishery.sim.state import FisheryState

logger = logging.getLogger(__name__)

# A violation recorded during enforcement: {"trigger": "exceed_cap", "excess": float}.
# Keyed by agent_id, except the reserved "__pool__" key used for a
# total_pool_per_round cap violation, which isn't attributable to one agent
# and is skipped by Penalise/Redistribute (both act on real agents only).
Violations = dict[str, list[dict]]


async def run_strategy_phase(
    state: FisheryState, decisions: DecisionSource, events: EventSink
) -> None:
    for agent in state.alive_agents:
        effort = await decisions.decide_effort(agent.agent_id, state)
        effort = min(1.0, max(0.0, effort))
        agent.last_effort = effort
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="strategy",
                type=EventType.STRATEGY_DECLARED,
                actor_id=agent.agent_id,
                payload={"effort": effort},
            )
        )


async def run_declare_phase(
    state: FisheryState, events: EventSink, *, timing: Literal["pre_round", "post_round"]
) -> None:
    """Publicizes a value tied to this round -- effort/intended-quota before
    harvest, actual harvest afterward -- per an active `DeclarePrimitive`. No
    LLM call: the value already exists, this just makes it public. Only
    `disclosure_visibility="public"` currently changes behavior (an
    "anonymous"/"ledger_only" declare compiles but has no separate mechanism
    yet, so nothing is disclosed for those -- a scoping simplification, not a
    silent no-op bug).
    """
    declare = next((p for p in state.active_norms if isinstance(p, DeclarePrimitive)), None)
    if declare is None or declare.disclosure_visibility != "public":
        return
    # "within_N_hours" doesn't map to a round-boundary in this turn-based sim
    # -- treated as post_round (closest real moment: right after the round's
    # action is known).
    effective_timing = "post_round" if declare.timing == "within_N_hours" else declare.timing
    if effective_timing != timing:
        return

    attr = "last_effort" if declare.content == "intended_quota" else "last_harvest"
    for agent in state.alive_agents:
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="declare",
                type=EventType.DISCLOSURE_MADE,
                actor_id=agent.agent_id,
                payload={"field": declare.content, "value": getattr(agent, attr, None)},
            )
        )


def _sustainable_threshold(state: FisheryState) -> float:
    """Reference point for `sustainable_yield`-basis caps and
    `collective_threshold_breached`-triggered adjust/redistribute -- 40% of
    carrying capacity, the same fallback default the ported reference
    implementation used when a domain doesn't supply its own.
    """
    return 0.4 * state.config.k


def _cap_value(cap: CapPrimitive, state: FisheryState, *, total_requested: float) -> float:
    if cap.basis == "fixed_units":
        value = cap.value
    elif cap.basis == "pct_of_stock":
        value = state.stock * cap.value
    elif cap.basis == "pct_of_total_catch":
        value = total_requested * cap.value
    elif cap.basis == "sustainable_yield":
        value = _sustainable_threshold(state) * cap.value
    else:
        value = cap.value
    # An active AdjustPrimitive's quota is a further ceiling on top of
    # whatever the cap itself computes -- the tighter of the two wins.
    if state.adjusted_quota is not None:
        value = min(value, state.adjusted_quota)
    return value


async def run_harvest_phase(state: FisheryState, events: EventSink) -> Violations:
    """Computes each agent's requested harvest, then applies an active
    `CapPrimitive` (if any) to clip it down -- unlike the old cap-check (which
    only logged violations), this actually changes what agents catch and get
    paid for. Returns this round's violations for `run_penalise_phase`/
    `run_redistribute_phase` to act on.
    """
    cfg = state.config
    requested: dict[str, float] = {
        agent.agent_id: cfg.alpha * (agent.last_effort or 0.0) * state.stock
        for agent in state.alive_agents
    }
    total_requested = sum(requested.values())

    cap = next((p for p in state.active_norms if isinstance(p, CapPrimitive)), None)
    harvests: dict[str, float] = {}
    violations: Violations = {}

    if cap is not None and cap.cap_scope == "total_pool_per_round":
        cap_value = _cap_value(cap, state, total_requested=total_requested)
        if total_requested > cap_value > 0:
            scale = cap_value / total_requested
            harvests = {aid: amount * scale for aid, amount in requested.items()}
            violations["__pool__"] = [{"trigger": "exceed_cap", "excess": total_requested - cap_value}]
            await events.record(
                Event.create(
                    fishery_id=state.config.fishery_id,
                    round=state.round,
                    phase="harvest",
                    type=EventType.CAP_EXCEEDED,
                    payload={"cap_id": cap.id, "observed": total_requested, "cap_value": cap_value},
                )
            )
        else:
            harvests = dict(requested)
    else:
        for agent_id, amount in requested.items():
            cap_value = _cap_value(cap, state, total_requested=total_requested) if cap is not None else None
            if cap_value is not None and amount > cap_value:
                excess = amount - cap_value
                harvests[agent_id] = cap_value
                violations[agent_id] = [{"trigger": "exceed_cap", "excess": excess}]
                await events.record(
                    Event.create(
                        fishery_id=state.config.fishery_id,
                        round=state.round,
                        phase="harvest",
                        type=EventType.CAP_EXCEEDED,
                        actor_id=agent_id,
                        payload={"cap_id": cap.id, "observed": amount, "cap_value": cap_value},
                    )
                )
            else:
                harvests[agent_id] = amount

    for agent in state.alive_agents:
        agent.last_harvest = harvests[agent.agent_id]

    total_harvest = sum(harvests.values())
    post_harvest_stock = max(0.0, state.stock - total_harvest)
    regrown_stock = post_harvest_stock + cfg.r * post_harvest_stock * (
        1 - post_harvest_stock / cfg.k
    )

    for agent in state.alive_agents:
        agent.payoff += harvests[agent.agent_id] - cfg.consumption
        agent.payoff_after_harvest = agent.payoff

    await events.record(
        Event.create(
            fishery_id=cfg.fishery_id,
            round=state.round,
            phase="harvest",
            type=EventType.HARVEST_RESOLVED,
            payload={
                "pre_harvest_stock": state.stock,
                "harvests": harvests,
                "post_harvest_stock": post_harvest_stock,
                "regrown_stock": regrown_stock,
            },
        )
    )
    state.stock = regrown_stock
    return violations


async def run_monitor_phase(
    state: FisheryState, events: EventSink, *, rng: random.Random | None = None
) -> None:
    """Only `method="random_audit"` produces a per-round mechanical effect
    (a 30% chance per agent, matching the ported reference implementation) --
    the other methods (peer/rotating_role/central_board/automated) compile
    and stay active but are otherwise passive/informational for now, a
    scoping simplification (no "monitor report" obligation-tracking state
    exists yet for `assign_role`'s `fail_monitor_duty` trigger to consume).
    """
    monitor = next((p for p in state.active_norms if isinstance(p, MonitorPrimitive)), None)
    if monitor is None or monitor.method != "random_audit":
        return
    rng = rng or random.Random()
    for agent in state.alive_agents:
        if rng.random() < 0.3:
            await events.record(
                Event.create(
                    fishery_id=state.config.fishery_id,
                    round=state.round,
                    phase="monitor",
                    type=EventType.MONITOR_REVIEW,
                    actor_id=agent.agent_id,
                    payload={"method": monitor.method},
                )
            )


async def run_penalise_phase(state: FisheryState, events: EventSink, violations: Violations) -> None:
    """Automatic sanction -- unlike the old `DecentralizedPunishment`, no
    agent is asked whether to punish; this fires on its own the moment
    `trigger` matches a violation Harvest recorded this round (project
    decision, see `models/norms.py`'s `PenalisePrimitive` docstring).
    """
    penalise = next((p for p in state.active_norms if isinstance(p, PenalisePrimitive)), None)
    if penalise is None:
        return

    for agent_id, agent_violations in violations.items():
        if agent_id not in state.agents:  # e.g. the reserved "__pool__" key
            continue
        matching = next((v for v in agent_violations if v.get("trigger") == penalise.trigger), None)
        if matching is None:
            continue

        excess = matching.get("excess", 0.0)
        if penalise.penalty_type == "proportional_fine":
            amount = excess * penalise.value
        elif penalise.penalty_type == "forfeit":
            amount = excess
        else:
            amount = penalise.value

        state.agents[agent_id].payoff -= amount
        if penalise.destination == "pool":
            state.stock += amount
        elif penalise.destination == "communal_fund":
            state.communal_fund += amount
        # "redistribute_equal" is realized by a separately-active
        # RedistributePrimitive (source="violator"), not here.

        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="penalise",
                type=EventType.PENALTY_APPLIED,
                target_id=agent_id,
                payload={
                    "penalty_type": penalise.penalty_type,
                    "amount": amount,
                    "destination": penalise.destination,
                    "trigger": penalise.trigger,
                },
            )
        )


async def run_redistribute_phase(state: FisheryState, events: EventSink, violations: Violations) -> None:
    redistribute = next((p for p in state.active_norms if isinstance(p, RedistributePrimitive)), None)
    if redistribute is None:
        return

    total_excess = sum(v.get("excess", 0.0) for vs in violations.values() for v in vs)
    if redistribute.trigger == "violation":
        triggered = bool(violations)
    elif redistribute.trigger == "end_of_round":
        triggered = True
    else:  # "threshold"
        triggered = state.stock < _sustainable_threshold(state)
    if not triggered:
        return

    agent_ids = [a.agent_id for a in state.alive_agents]
    n = max(len(agent_ids), 1)
    if redistribute.amount_basis == "excess_units":
        total = total_excess
    elif redistribute.amount_basis == "fixed":
        total = redistribute.fixed_amount
    else:  # "proportional"
        total = state.communal_fund / n

    if total <= 0:
        return

    if redistribute.source == "communal_fund" and redistribute.destination != "communal_fund":
        state.communal_fund = max(0.0, state.communal_fund - total)
    elif redistribute.source == "pool" and redistribute.destination != "pool":
        state.stock = max(0.0, state.stock - total)
    # source == "violator": nothing further to deduct -- the excess was
    # already clipped off their harvest by cap enforcement.

    deltas: dict[str, float] = {}
    if redistribute.destination == "all_agents_equal":
        per_agent = total / n
        deltas = dict.fromkeys(agent_ids, per_agent)
    elif redistribute.destination == "all_agents_proportional":
        weights = {aid: max(state.agents[aid].payoff, 0.0) for aid in agent_ids}
        total_w = sum(weights.values()) or 1.0
        deltas = {aid: total * w / total_w for aid, w in weights.items()}
    elif redistribute.destination == "communal_fund":
        state.communal_fund += total
    elif redistribute.destination == "pool":
        state.stock += total

    for aid, delta in deltas.items():
        state.agents[aid].payoff += delta

    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="redistribute",
            type=EventType.REDISTRIBUTION_APPLIED,
            payload={
                "source": redistribute.source,
                "destination": redistribute.destination,
                "total_amount": total,
                "per_agent_deltas": deltas,
            },
        )
    )


async def run_adjust_phase(state: FisheryState, events: EventSink) -> None:
    adjust = next((p for p in state.active_norms if isinstance(p, AdjustPrimitive)), None)
    if adjust is None:
        return

    if adjust.trigger == "end_of_round":
        triggered = True
    elif adjust.trigger == "collective_threshold_breached":
        triggered = state.stock < _sustainable_threshold(state)
    else:  # "cumulative_over_N_rounds" -- not tracked, a scoping simplification
        triggered = False
    if not triggered:
        return

    if adjust.direction == "reduce":
        current = state.adjusted_quota if state.adjusted_quota is not None else float("inf")
        state.adjusted_quota = max(0.0, current - adjust.value)
    elif adjust.direction == "restore":
        current = state.adjusted_quota if state.adjusted_quota is not None else 0.0
        state.adjusted_quota = current + adjust.value
    else:  # "recalculate"
        state.adjusted_quota = adjust.value

    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="adjust",
            type=EventType.QUOTA_ADJUSTED,
            payload={
                "adjust_target": adjust.adjust_target,
                "direction": adjust.direction,
                "value": adjust.value,
                "new_quota": state.adjusted_quota,
            },
        )
    )


async def run_starvation_check(state: FisheryState, events: EventSink) -> bool:
    """Returns True if any agent starved from underharvest this round (as
    opposed to being penalised into starvation) -- `run_round` feeds this
    into `check_fishery_collapse`, since underharvest collapse is triggered
    by the event itself, not a population/stock floor.
    """
    underharvest_death = False
    for agent in state.alive_agents:
        if agent.payoff < 0:
            agent.alive = False
            # Already negative right after this round's harvest-consumption
            # update (before penalise/redistribute) -> underharvest is the
            # cause. Otherwise harvest covered consumption and a penalty is
            # what tipped them into starvation.
            reason = "underharvest" if (agent.payoff_after_harvest or 0.0) < 0 else "punished"
            state.starvation_reasons[agent.agent_id] = reason
            if reason == "underharvest":
                underharvest_death = True
            await events.record(
                Event.create(
                    fishery_id=state.config.fishery_id,
                    round=state.round,
                    phase="starvation",
                    type=EventType.AGENT_STARVED,
                    target_id=agent.agent_id,
                    payload={"payoff": agent.payoff, "reason": reason},
                )
            )
    return underharvest_death


async def run_propose_phase(
    state: FisheryState, decisions: GovernanceDecisionSource, events: EventSink
) -> dict[str, PolicyProposal]:
    """Each agent updates its own personal norm (private) and proposes a
    bundled community policy + how to operationalize/enforce it (public
    candidate for this round's vote). Returns agent_id -> proposed
    `PolicyProposal`.
    """
    proposals: dict[str, PolicyProposal] = {}
    for agent in state.alive_agents:
        decision = await decisions.decide_proposal(agent.agent_id, state)
        state.agent_norms[agent.agent_id] = decision.personal_norm
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="propose",
                type=EventType.PERSONAL_NORM_UPDATED,
                actor_id=agent.agent_id,
                payload={"personal_norm": decision.personal_norm},
            )
        )
        proposal = PolicyProposal(
            community_proposal=decision.community_proposal, operationalization=decision.operationalization
        )
        proposals[agent.agent_id] = proposal
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="propose",
                type=EventType.PROPOSAL_MADE,
                actor_id=agent.agent_id,
                payload={
                    "proposal": proposal.community_proposal,
                    "operationalization": proposal.operationalization,
                },
            )
        )
    return proposals


async def run_vote_phase(
    state: FisheryState,
    decisions: GovernanceDecisionSource,
    events: EventSink,
    proposals: dict[str, PolicyProposal],
) -> PolicyProposal | None:
    """Tallies one vote per alive agent among this round's distinct
    (policy, operationalization) candidates. Returns the winning
    `PolicyProposal`, or None if there was nothing to vote on.
    """
    candidates = list(dict.fromkeys(proposals.values()))  # dedupe, preserve order
    if not candidates:
        return None

    tally: dict[PolicyProposal, int] = dict.fromkeys(candidates, 0)
    for agent in state.alive_agents:
        choice = await decisions.decide_vote(agent.agent_id, state, candidates=candidates)
        tally[choice] += 1
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="vote",
                type=EventType.VOTE_CAST,
                actor_id=agent.agent_id,
                payload={"choice": choice.community_proposal, "operationalization": choice.operationalization},
            )
        )

    winner = max(candidates, key=lambda c: tally[c])  # ties broken by proposal order (stable)
    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="vote",
            type=EventType.VOTE_RESULT,
            payload={
                "tally": {c.community_proposal: n for c, n in tally.items()},
                "winner": winner.community_proposal,
                "operationalization": winner.operationalization,
            },
        )
    )
    return winner


def _merge_compiled_primitives(
    active_norms: list[NormPrimitive], new_primitives: list[NormPrimitive]
) -> list[NormPrimitive]:
    """Each primitive type is a singleton (at most one active at a time,
    replaced by a newly-compiled one), except `cap`/`penalise`/`adjust`
    (keyed by the field that makes two instances genuinely different rather
    than competing versions of the same rule) and `assign_role` (keyed by
    role_name, so granting a new role doesn't clobber an existing one).
    """
    merged = list(active_norms)
    for new_primitive in new_primitives:
        if isinstance(new_primitive, PeerObservability):
            merged = [p for p in merged if not isinstance(p, PeerObservability)]
        elif isinstance(new_primitive, CapPrimitive):
            merged = [
                p
                for p in merged
                if not (isinstance(p, CapPrimitive) and p.cap_scope == new_primitive.cap_scope)
            ]
        elif isinstance(new_primitive, DeclarePrimitive):
            merged = [p for p in merged if not isinstance(p, DeclarePrimitive)]
        elif isinstance(new_primitive, MonitorPrimitive):
            merged = [p for p in merged if not isinstance(p, MonitorPrimitive)]
        elif isinstance(new_primitive, PenalisePrimitive):
            merged = [
                p
                for p in merged
                if not (isinstance(p, PenalisePrimitive) and p.trigger == new_primitive.trigger)
            ]
        elif isinstance(new_primitive, AdjustPrimitive):
            merged = [
                p
                for p in merged
                if not (isinstance(p, AdjustPrimitive) and p.adjust_target == new_primitive.adjust_target)
            ]
        elif isinstance(new_primitive, RedistributePrimitive):
            merged = [p for p in merged if not isinstance(p, RedistributePrimitive)]
        elif isinstance(new_primitive, AssignRolePrimitive):
            merged = [
                p
                for p in merged
                if not (isinstance(p, AssignRolePrimitive) and p.role_name == new_primitive.role_name)
            ]
        merged.append(new_primitive)
    return merged


async def run_election_phase(
    state: FisheryState, decisions: GovernanceDecisionSource, events: EventSink, *, role_name: str
) -> str | None:
    """Self-nomination followed by a vote among nominees, appended only for
    the round an `AssignRolePrimitive(selection="elected")` is adopted -- it
    is not repeated automatically in later rounds.
    """
    nominees = []
    for agent in state.alive_agents:
        if await decisions.decide_nomination(agent.agent_id, state, role_name=role_name):
            nominees.append(agent.agent_id)

    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="election",
            type=EventType.ROLE_ELECTION_CALLED,
            payload={"role_name": role_name, "nominees": nominees},
        )
    )
    if not nominees:
        return None

    if len(nominees) == 1:
        winner = nominees[0]
    else:
        tally = {n: 0 for n in nominees}
        for agent in state.alive_agents:
            choice = await decisions.decide_election_vote(
                agent.agent_id, state, role_name=role_name, candidates=nominees
            )
            tally[choice] += 1
        winner = max(nominees, key=lambda n: tally[n])  # ties broken by nomination order (stable)

    state.roles[role_name] = winner
    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="election",
            type=EventType.ROLE_ELECTED,
            actor_id=winner,
            payload={"role_name": role_name, "agent_id": winner, "selection": "elected"},
        )
    )
    return winner


async def update_role_rotations(state: FisheryState, events: EventSink) -> None:
    """Recomputes the current holder of every rotating/random-selected
    `AssignRolePrimitive` as a pure function of elapsed rounds -- no separate
    "tick" state to drift, and it naturally re-derives the right holder even
    across a gap in calls. "rotating" walks the roster deterministically;
    "random" re-rolls a new holder each window, seeded by
    (role_name, window_index) so it's reproducible if replayed.
    """
    for grant in (
        p
        for p in state.active_norms
        if isinstance(p, AssignRolePrimitive) and p.selection in ("rotating", "random")
    ):
        # Live roster (dict insertion order), not the fixed initial config
        # list -- migrated arrivals (build spec §7) join at the end of this
        # order and must be eligible like anyone else.
        roster = [aid for aid, agent in state.agents.items() if agent.alive]
        if not roster:
            continue
        term = grant.duration or 1
        start = state.role_rotation_start.setdefault(grant.role_name, state.round)
        window_index = (state.round - start) // term
        if grant.selection == "rotating":
            holder = roster[window_index % len(roster)]
        else:  # "random"
            holder = random.Random(f"{grant.role_name}:{window_index}").choice(roster)
        if state.roles.get(grant.role_name) != holder:
            state.roles[grant.role_name] = holder
            await events.record(
                Event.create(
                    fishery_id=state.config.fishery_id,
                    round=state.round,
                    phase="role_rotation",
                    type=EventType.ROLE_ELECTED,
                    actor_id=holder,
                    payload={"role_name": grant.role_name, "agent_id": holder, "selection": grant.selection},
                )
            )


async def _resolve_assign_role(
    state: FisheryState, decisions: GovernanceDecisionSource, grant: AssignRolePrimitive, events: EventSink
) -> None:
    if grant.selection == "elected":
        await run_election_phase(state, decisions, events, role_name=grant.role_name)
    elif grant.selection in ("rotating", "random"):
        state.role_rotation_start[grant.role_name] = state.round
        # Actual holder assignment happens uniformly in `update_role_rotations`.


async def run_norm_adoption(
    state: FisheryState,
    decisions: GovernanceDecisionSource,
    norm_compiler: NormCompiler,
    events: EventSink,
) -> None:
    proposals = await run_propose_phase(state, decisions, events)
    winner = await run_vote_phase(state, decisions, events, proposals)
    if winner is None or winner.community_proposal == state.group_norm_text:
        await update_role_rotations(state, events)
        return

    state.group_norm_text = winner.community_proposal

    # Every proposal bundles its own operationalization suggestion
    # (`ProposalDecision.operationalization`), carried through voting as part
    # of the winning `PolicyProposal` -- so the compiler gets both pieces
    # straight from the winner.
    compiled_text = f"{winner.community_proposal}\n\nHow to operationalize this: {winner.operationalization}"

    spec = await norm_compiler.compile(
        compiled_text, norm_id=f"norm_r{state.round}", adopted_round=state.round
    )
    if spec.primitives:
        primitives_to_merge = list(spec.primitives)
        state.active_norms = _merge_compiled_primitives(state.active_norms, primitives_to_merge)
        for primitive in primitives_to_merge:
            state.norm_source_text[primitive.id] = compiled_text

        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="norm_compile",
                type=EventType.NORM_ADOPTED,
                payload={
                    "raw_text": compiled_text,
                    "primitive_ids": [p.id for p in primitives_to_merge],
                },
            )
        )

        for grant in (p for p in spec.primitives if isinstance(p, AssignRolePrimitive)):
            await _resolve_assign_role(state, decisions, grant, events)
    else:
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="norm_compile",
                type=EventType.NORM_COULD_NOT_COMPILE,
                payload={"raw_text": compiled_text},
            )
        )

    await update_role_rotations(state, events)


async def run_memory_write_phase(
    state: FisheryState,
    llm: LLMClient,
    memory_registry: MemoryBankRegistry,
    round_events: list[Event],
) -> None:
    """Turns this round's events into memory-stream writes, per agent, per
    build spec §3 ("every observation ... stored"). Visibility-filtered via
    `Event.is_visible_to` -- an agent only ever remembers what they were
    actually entitled to see, same rule the frontend/prompt assembly uses.

    Memory-writing is best-effort: this round's game state (effort, harvest,
    payoffs, norms) is already finalized by the time this runs, so a single
    failed importance-rating/reflection call (a real-world example: an
    external moderation filter false-positiving on benign "X starved"
    narrative text) should cost that one memory, not crash the whole round.
    """
    for agent in state.alive_agents:
        bank = memory_registry.get_or_create(agent.agent_id)
        for event in visible_events_for(agent.agent_id, round_events):
            content = render_event_as_observation(event, agent.agent_id)
            if content is None:
                continue
            try:
                await write_observation(llm, bank, content, round=state.round)
            except LLMStructuredCallError:
                logger.warning(
                    "memory write failed for agent %s, round %s (event type %s) -- skipping",
                    agent.agent_id,
                    state.round,
                    event.type,
                    exc_info=True,
                )


def check_fishery_collapse(state: FisheryState, *, underharvest_death_this_round: bool = False) -> bool:
    return (
        state.stock <= state.config.r_min
        or len(state.alive_agents) < state.config.n_min
        or underharvest_death_this_round
    )


def _collapse_reasons(state: FisheryState, *, underharvest_death_this_round: bool = False) -> list[str]:
    """All conditions are independent and can fire in the same round --
    returns every cause that actually applied, not just the first one found.
    """
    reasons = []
    if state.stock <= state.config.r_min:
        reasons.append("resource_depletion")
    if len(state.alive_agents) < state.config.n_min:
        reasons.append("population_loss")
    if underharvest_death_this_round:
        reasons.append("underharvest_death")
    return reasons


async def run_round(
    state: FisheryState,
    decisions: DecisionSource,
    events: EventSink,
    *,
    norm_compiler: NormCompiler | None = None,
    llm: LLMClient | None = None,
    memory_registry: MemoryBankRegistry | None = None,
) -> None:
    collector = RoundEventCollector(events)
    state.round += 1
    await run_strategy_phase(state, decisions, collector)
    await run_declare_phase(state, collector, timing="pre_round")
    violations = await run_harvest_phase(state, collector)
    await run_declare_phase(state, collector, timing="post_round")
    await run_monitor_phase(state, collector)
    await run_penalise_phase(state, collector, violations)
    await run_redistribute_phase(state, collector, violations)
    await run_adjust_phase(state, collector)

    had_underharvest_death = await run_starvation_check(state, collector)

    if norm_compiler is not None:
        await run_norm_adoption(state, decisions, norm_compiler, collector)

    if check_fishery_collapse(state, underharvest_death_this_round=had_underharvest_death):
        state.collapsed = True
        state.collapse_reasons = _collapse_reasons(state, underharvest_death_this_round=had_underharvest_death)
        await collector.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="collapse",
                type=EventType.FISHERY_COLLAPSED,
                payload={
                    "stock": state.stock,
                    "n_alive": len(state.alive_agents),
                    "reasons": state.collapse_reasons,
                },
            )
        )

    if llm is not None and memory_registry is not None:
        await run_memory_write_phase(state, llm, memory_registry, collector.this_round)


async def run_simulation(
    state: FisheryState,
    decisions: DecisionSource,
    events: EventSink,
    *,
    norm_compiler: NormCompiler | None = None,
    llm: LLMClient | None = None,
    memory_registry: MemoryBankRegistry | None = None,
) -> int:
    """Run rounds until collapse or `max_rounds`; returns the survival time T_s."""
    while not state.collapsed:
        if state.config.max_rounds is not None and state.round >= state.config.max_rounds:
            break
        await run_round(
            state,
            decisions,
            events,
            norm_compiler=norm_compiler,
            llm=llm,
            memory_registry=memory_registry,
        )
    return state.round
