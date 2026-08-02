"""Round engine: Gupta et al.'s effort/harvest/regrowth equations, plus norm
proposal/discussion/voting.

Resolution order: Strategy -> Harvest -> Starvation check -> [if governance
enabled] Propose -> Operationalization discussion -> Vote -> collapse check.
Propose/Vote only run if `enable_governance=True` is passed to
`run_round`/`run_simulation` -- omitting it (as the baseline Gupta et al.
validation tests and `scripts/validate_gupta_replication.py` do) preserves
the earlier, governance-free round shape exactly.

There used to be a closed, 8-primitive governance layer here (cap/declare/
monitor/penalise/adjust/redistribute/assign_role/peer_observability),
mechanically enforced by generic phase functions, with a winning norm
compiled into one of those primitives by a separate `NormCompiler` LLM call.
That whole layer has been removed by project decision: instead, a winning
norm is implemented as an actual code change to this simulation, made by an
autonomous SE agent (see `se_agent/`), then committed with git. There is
deliberately no generic
enforcement layer anymore -- what enforces a given round's norm is whatever
code that agent has, over time, actually written into this engine. Which
means: **whenever you (the SE agent) add a mechanic here that changes
`FisheryState` in a new way, also update `sim/state_replay.py` so a process
restart can still correctly reconstruct state from the event log** -- state
persistence only understands event types it's been taught to replay.

Each agent proposes only a norm (`ProposalDecision.community_proposal`, no
operationalization question attached). Right after proposing, that agent
discusses how to operationalize it with the fishery SE agent --
`run_operationalization_discussion_phase` -- for
`state.config.operationalization_discussion_rounds` turn-pairs; the final
turn's reply is the finalized operationalization text. Voting carries the
(norm, operationalization) pair through together as one `PolicyProposal`
candidate (`sim.decisions.proposal_candidate_key` is its display text on the
ballot, both fields folded into it; the vote itself picks a short ballot id
rather than reproducing this text).
"""

import logging

from genfishery.se_agent.client import SEAgentClient
from genfishery.llm.client import LLMClient, LLMStructuredCallError
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.memory.writer import write_observation
from genfishery.models.events import Event, EventType, visible_events_for
from genfishery.sim.decisions import PolicyProposal, active_norms_summary
from genfishery.sim.events_sink import EventSink, RoundEventCollector
from genfishery.sim.observations import render_event_as_observation
from genfishery.sim.policies import DecisionSource, GovernanceDecisionSource
from genfishery.sim.state import FisheryState

logger = logging.getLogger(__name__)


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


async def run_harvest_phase(state: FisheryState, events: EventSink) -> None:
    """Bare Gupta et al. effort/harvest/regrowth equations -- no cap
    enforcement or any other governance mechanic. Anything a winning norm
    needs to actually enforce is implemented directly here (or in new
    functions called from `run_round`) by the SE agent, per that norm.
    """
    cfg = state.config
    harvests: dict[str, float] = {
        agent.agent_id: cfg.alpha * (agent.last_effort or 0.0) * state.stock
        for agent in state.alive_agents
    }

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


async def run_starvation_check(state: FisheryState, events: EventSink) -> bool:
    """Returns True if any agent starved from underharvest this round --
    `run_round` feeds this into `check_fishery_collapse`, since underharvest
    collapse is triggered by the event itself, not a population/stock floor.
    """
    underharvest_death = False
    for agent in state.alive_agents:
        if agent.payoff < 0:
            agent.alive = False
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


async def run_operationalization_discussion_phase(
    state: FisheryState,
    decisions: GovernanceDecisionSource,
    events: EventSink,
    se_agent: SEAgentClient,
    *,
    agent_id: str,
    community_proposal: str,
    rounds: int,
) -> str:
    """One proposing agent's private back-and-forth with the fishery SE
    agent about how to put `community_proposal` into practice -- one
    `opencode` session for the whole discussion (not a fresh session per
    turn), so the agent's own conversational memory carries across turns.
    It asks the opening question; each turn after that is just the villager's
    last reply relayed back into the same session, and the agent's
    (opencode-generated) response relayed back to the villager. The final
    turn's reply is finalized -- returned as the operationalization text --
    rather than relayed onward.
    """
    session_id = await se_agent.start_session(title=f"{state.config.fishery_id}-{agent_id}-r{state.round}")
    opening_message = f"""A villager just proposed this policy to their fishing community: "{community_proposal}"

Here's how the fishery currently works:
{active_norms_summary(state)}

Ask the villager how they think this policy should actually be operationalized -- put into practice."""
    councillor_message = await se_agent.ask(session_id, opening_message)

    transcript: list[tuple[str, str]] = []
    for turn in range(rounds):
        is_final_turn = turn == rounds - 1
        reply = await decisions.decide_councillor_reply(
            agent_id,
            state,
            community_proposal=community_proposal,
            transcript=transcript,
            councillor_message=councillor_message,
            is_final_turn=is_final_turn,
        )
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="operationalization_discussion",
                type=EventType.COUNCILLOR_QUESTION,
                actor_id=agent_id,
                payload={"message": councillor_message},
            )
        )
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="operationalization_discussion",
                type=EventType.COUNCILLOR_DISCUSSION_REPLY,
                actor_id=agent_id,
                payload={"message": reply},
            )
        )
        transcript.append(("councillor", councillor_message))
        transcript.append(("agent", reply))
        if is_final_turn:
            return reply
        councillor_message = await se_agent.ask(session_id, reply)

    return community_proposal  # unreachable: `rounds` is validated >= 1


async def run_propose_phase(
    state: FisheryState,
    decisions: GovernanceDecisionSource,
    events: EventSink,
    *,
    se_agent: SEAgentClient | None = None,
) -> dict[str, PolicyProposal]:
    """Each agent updates its own personal norm (private) and proposes a
    community policy (public candidate for this round's vote). When an SE
    agent is available, the proposing agent immediately
    discusses how to operationalize that policy
    (`run_operationalization_discussion_phase`); without one (e.g. existing
    tests, or a deployment that hasn't wired up opencode), operationalization
    is left blank rather than asked for in the same call it used to be.
    Returns agent_id -> proposed `PolicyProposal`.
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
        if se_agent is not None:
            operationalization = await run_operationalization_discussion_phase(
                state,
                decisions,
                events,
                se_agent,
                agent_id=agent.agent_id,
                community_proposal=decision.community_proposal,
                rounds=state.config.operationalization_discussion_rounds,
            )
        else:
            operationalization = ""
        proposal = PolicyProposal(
            community_proposal=decision.community_proposal, operationalization=operationalization
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


async def run_norm_adoption(
    state: FisheryState,
    decisions: GovernanceDecisionSource,
    events: EventSink,
    *,
    se_agent: SEAgentClient | None = None,
) -> bool:
    """Every proposal gets discussed and tuned (`run_propose_phase`), but only
    the single winner of this round's vote -- and only when it's genuinely
    new -- ever reaches an actual code change: `_implement_winning_norm` below
    is the one and only place `SEAgentClient.implement_norm` is called.
    Returns True exactly when that implementation step actually committed a
    code change this round -- the caller's cue that a process restart is
    needed to pick it up (see `run_round`/`FisheryRunner._loop`).
    """
    proposals = await run_propose_phase(state, decisions, events, se_agent=se_agent)
    winner = await run_vote_phase(state, decisions, events, proposals)
    if winner is None or winner.community_proposal == state.group_norm_text:
        return False

    state.group_norm_text = winner.community_proposal
    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="norm_adoption",
            type=EventType.NORM_ADOPTED,
            payload={
                "community_proposal": winner.community_proposal,
                "operationalization": winner.operationalization,
            },
        )
    )

    if se_agent is None:
        return False
    return await _implement_winning_norm(state, events, se_agent, winner)


async def _implement_winning_norm(
    state: FisheryState,
    events: EventSink,
    se_agent: SEAgentClient,
    winner: PolicyProposal,
) -> bool:
    """Tells the SE agent this specific proposal has won and to implement it
    as a real change under `backend/`, run the test suite, and commit --
    a fresh session, separate from any proposer's discussion session (a
    round can have several of those; this step belongs to none of them in
    particular). Success/failure is never taken on the agent's own say-so --
    see `SEAgentClient.implement_norm`'s docstring on why -- and a failure
    leaves `state.group_norm_text` as already set (the community's adopted
    policy text doesn't revert) but performs no restart and commits nothing,
    so the repo is never left partially edited. Returns whether it succeeded.
    """
    session_id = await se_agent.start_session(
        title=f"{state.config.fishery_id}-implement-r{state.round}"
    )
    instructions = f"""This proposal has now won the community's vote for fishery "{state.config.fishery_id}":

Policy: "{winner.community_proposal}"
Operationalization (from the community's own discussion of how to put it into practice): "{winner.operationalization}"

Implement this policy's mechanics as an actual code change to this simulation under backend/, run the backend test suite, and commit your changes with git once they pass."""
    result = await se_agent.implement_norm(session_id, instructions)
    if result.success:
        await events.record(
            Event.create(
                fishery_id=state.config.fishery_id,
                round=state.round,
                phase="norm_adoption",
                type=EventType.NORM_IMPLEMENTED,
                payload={"commit_sha": result.commit_sha, "reply": result.reply},
            )
        )
        return True

    logger.warning(
        "SE agent failed to implement the winning norm for fishery %s, round %s -- "
        "no code change, no restart. Reply: %s",
        state.config.fishery_id,
        state.round,
        result.reply,
    )
    await events.record(
        Event.create(
            fishery_id=state.config.fishery_id,
            round=state.round,
            phase="norm_adoption",
            type=EventType.NORM_IMPLEMENTATION_FAILED,
            payload={"reply": result.reply},
        )
    )
    return False


async def run_memory_write_phase(
    state: FisheryState,
    llm: LLMClient,
    memory_registry: MemoryBankRegistry,
    round_events: list[Event],
) -> None:
    """Turns this round's events into memory-stream writes, per agent, per
    build spec §3 ("every observation ... stored"). Visibility-filtered via
    `Event.is_visible_to` -- an agent only ever remembers what they were
    actually entitled to see, same rule prompt assembly uses.

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
    enable_governance: bool = False,
    llm: LLMClient | None = None,
    memory_registry: MemoryBankRegistry | None = None,
    se_agent: SEAgentClient | None = None,
) -> bool:
    """Returns whether this round's SE agent implementation step committed a
    code change -- the caller (`FisheryRunner._loop`) treats True as "stop
    this runner and trigger a process restart," per `sim/engine.py`'s module
    docstring on why restarts happen.
    """
    collector = RoundEventCollector(events)
    state.round += 1
    await run_strategy_phase(state, decisions, collector)
    await run_harvest_phase(state, collector)

    had_underharvest_death = await run_starvation_check(state, collector)

    restart_needed = False
    if enable_governance:
        restart_needed = await run_norm_adoption(state, decisions, collector, se_agent=se_agent)

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

    return restart_needed


async def run_simulation(
    state: FisheryState,
    decisions: DecisionSource,
    events: EventSink,
    *,
    enable_governance: bool = False,
    llm: LLMClient | None = None,
    memory_registry: MemoryBankRegistry | None = None,
    se_agent: SEAgentClient | None = None,
) -> int:
    """Run rounds until collapse or `max_rounds`; returns the survival time T_s.
    Ignores `run_round`'s restart signal -- this helper is for offline
    scripts/tests running the whole simulation in one process, not the live
    `FisheryRunner`, which is the one place that signal actually matters.
    """
    while not state.collapsed:
        if state.config.max_rounds is not None and state.round >= state.config.max_rounds:
            break
        await run_round(
            state,
            decisions,
            events,
            enable_governance=enable_governance,
            llm=llm,
            memory_registry=memory_registry,
            se_agent=se_agent,
        )
    return state.round
