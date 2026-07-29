"""Effort/governance decision sources.

`LLMDecisionSource` is the real production path (build spec §2: every agent
decision is a Claude structured tool call). `RuleBasedDecisionSource` is a
non-LLM stand-in used only to validate the engine mechanics themselves
(harvest/regrowth/starvation equations) against Gupta et al.'s own
framework-validation methodology (their §4.1 uses rule-based agents for
exactly this, before any LLM is involved) -- it is never used for a real run.

There is no `decide_punishment` anymore -- punishment (`PenalisePrimitive`)
is automatic, triggered directly by violations Harvest records, not a
per-agent "should I punish someone?" choice (project decision, see
`models/norms.py`'s `PenalisePrimitive` docstring).
"""

import random
from typing import Protocol

from genfishery.config.model_config import LLMCallType, ModelConfig
from genfishery.llm.client import LLMClient
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.sim.decisions import (
    CouncillorReplyDecision,
    EffortDecision,
    NominationDecision,
    PolicyProposal,
    ProposalDecision,
    build_councillor_reply_prompt,
    build_effort_prompt,
    build_election_vote_prompt,
    build_election_vote_response_model,
    build_nomination_prompt,
    build_proposal_prompt,
    build_vote_prompt,
    build_vote_response_model,
)
from genfishery.sim.state import NO_NORM_YET, FisheryState


class DecisionSource(Protocol):
    async def decide_effort(self, agent_id: str, state: FisheryState) -> float: ...


class GovernanceDecisionSource(Protocol):
    """Norm proposal/voting/election decisions (build spec §4-5) -- separate
    from `DecisionSource` since `RuleBasedDecisionSource` (validation-only, no
    natural-language reasoning) never implements these.
    """

    async def decide_proposal(self, agent_id: str, state: FisheryState) -> ProposalDecision: ...

    async def decide_vote(
        self, agent_id: str, state: FisheryState, *, candidates: list[PolicyProposal]
    ) -> PolicyProposal: ...

    async def decide_councillor_reply(
        self,
        agent_id: str,
        state: FisheryState,
        *,
        community_proposal: str,
        transcript: list[tuple[str, str]],
        councillor_message: str,
        is_final_turn: bool,
    ) -> str: ...

    async def decide_nomination(self, agent_id: str, state: FisheryState, *, role_name: str) -> bool: ...

    async def decide_election_vote(
        self, agent_id: str, state: FisheryState, *, role_name: str, candidates: list[str]
    ) -> str: ...


class LLMDecisionSource:
    """`memory_registry` is optional (default None, preserving the exact old
    behavior for callers/tests that don't care about memory) -- when
    supplied, every decision retrieves that agent's own top-k relevant
    memories (build spec §3's weighted recency+importance+relevance
    retrieval) against a query describing the decision at hand, and feeds
    them into the prompt. This is the retrieval half of the Smallville
    pattern; step 7 only wired the write half.
    """

    _MEMORY_RETRIEVAL_K = 5

    def __init__(
        self,
        llm_client: LLMClient,
        model_config: ModelConfig,
        memory_registry: MemoryBankRegistry | None = None,
    ) -> None:
        self._llm = llm_client
        self._model_config = model_config
        self._memory_registry = memory_registry

    async def _memory_block(self, agent_id: str, state: FisheryState, query: str) -> str:
        if self._memory_registry is None:
            return ""
        bank = self._memory_registry.get_or_create(agent_id)
        records = bank.retrieve(query, k=self._MEMORY_RETRIEVAL_K, current_round=state.round)
        if not records:
            return ""
        return "\n".join(f"- (round {r.round}) {r.content}" for r in records)

    async def decide_effort(self, agent_id: str, state: FisheryState) -> float:
        memories = await self._memory_block(
            agent_id, state, "How much fishing effort should I use this round, given my survival needs?"
        )
        system, prompt = build_effort_prompt(
            state=state,
            viewer_id=agent_id,
            agent_norm=state.agent_norms.get(agent_id, NO_NORM_YET),
            group_norm=state.group_norm_text,
            memories=memories,
        )
        decision = await self._llm.structured_call(
            call_type=LLMCallType.EFFORT_DECISION,
            system=system,
            prompt=prompt,
            response_model=EffortDecision,
        )
        return decision.effort

    async def decide_proposal(self, agent_id: str, state: FisheryState) -> ProposalDecision:
        memories = await self._memory_block(
            agent_id,
            state,
            "What should my personal strategy be, and what policy should I propose to the community?",
        )
        system, prompt = build_proposal_prompt(
            state=state,
            viewer_id=agent_id,
            agent_norm=state.agent_norms.get(agent_id, NO_NORM_YET),
            group_norm=state.group_norm_text,
            memories=memories,
        )
        return await self._llm.structured_call(
            call_type=LLMCallType.PROPOSAL,
            system=system,
            prompt=prompt,
            response_model=ProposalDecision,
        )

    async def decide_vote(
        self, agent_id: str, state: FisheryState, *, candidates: list[PolicyProposal]
    ) -> PolicyProposal:
        memories = await self._memory_block(
            agent_id, state, "Which proposed community policy should I vote for?"
        )
        system, prompt = build_vote_prompt(
            state=state,
            viewer_id=agent_id,
            candidates=candidates,
            agent_norm=state.agent_norms.get(agent_id, NO_NORM_YET),
            group_norm=state.group_norm_text,
            memories=memories,
        )
        vote_model = build_vote_response_model(candidates)
        decision = await self._llm.structured_call(
            call_type=LLMCallType.VOTE,
            system=system,
            prompt=prompt,
            response_model=vote_model,
        )
        return candidates[int(decision.chosen_id) - 1]

    async def decide_councillor_reply(
        self,
        agent_id: str,
        state: FisheryState,
        *,
        community_proposal: str,
        transcript: list[tuple[str, str]],
        councillor_message: str,
        is_final_turn: bool,
    ) -> str:
        memories = await self._memory_block(
            agent_id, state, "How should my proposed policy actually be put into practice?"
        )
        system, prompt = build_councillor_reply_prompt(
            state=state,
            viewer_id=agent_id,
            community_proposal=community_proposal,
            transcript=transcript,
            councillor_message=councillor_message,
            is_final_turn=is_final_turn,
            agent_norm=state.agent_norms.get(agent_id, NO_NORM_YET),
            group_norm=state.group_norm_text,
            memories=memories,
        )
        decision = await self._llm.structured_call(
            call_type=LLMCallType.COUNCILLOR_REPLY,
            system=system,
            prompt=prompt,
            response_model=CouncillorReplyDecision,
        )
        return decision.reply

    async def decide_nomination(self, agent_id: str, state: FisheryState, *, role_name: str) -> bool:
        memories = await self._memory_block(
            agent_id, state, f"Should I nominate myself for the role of {role_name}?"
        )
        system, prompt = build_nomination_prompt(
            state=state,
            viewer_id=agent_id,
            role_name=role_name,
            agent_norm=state.agent_norms.get(agent_id, NO_NORM_YET),
            group_norm=state.group_norm_text,
            memories=memories,
        )
        decision = await self._llm.structured_call(
            call_type=LLMCallType.ELECTION_DECISION,
            system=system,
            prompt=prompt,
            response_model=NominationDecision,
        )
        return decision.self_nominate

    async def decide_election_vote(
        self, agent_id: str, state: FisheryState, *, role_name: str, candidates: list[str]
    ) -> str:
        memories = await self._memory_block(agent_id, state, f"Who should I vote for as {role_name}?")
        system, prompt = build_election_vote_prompt(
            state=state,
            viewer_id=agent_id,
            role_name=role_name,
            candidates=candidates,
            agent_norm=state.agent_norms.get(agent_id, NO_NORM_YET),
            group_norm=state.group_norm_text,
            memories=memories,
        )
        vote_model = build_election_vote_response_model(candidates)
        decision = await self._llm.structured_call(
            call_type=LLMCallType.ELECTION_DECISION,
            system=system,
            prompt=prompt,
            response_model=vote_model,
        )
        return decision.chosen_agent_id


class RuleBasedDecisionSource:
    """Stochastic non-LLM agents matching Gupta et al. §4.1's validation setup.

    Each agent has a fixed personal harvest-threshold belief `g_i` (init
    Uniform(2, 8)). Effort is drawn once at init (Uniform(0, 1)) and then
    nudged each round toward the effort level that would realize `g_i` given
    the *current* stock, i.e. `g_i / (alpha * R(t))` -- a minimal reactive
    policy standing in for the paper's payoff-biased social learning (agents
    periodically adopting a higher-payoff peer's whole strategy tuple via a
    logit rule), which is NOT implemented here. Known consequence (see
    `scripts/validate_gupta_replication.py`): without that selection dynamic,
    this proxy does not reproduce the paper's punishment-sustains-cooperation
    pattern -- it exercises the engine's mechanics only, and is not itself a
    validated behavioral model.
    """

    def __init__(self, agent_ids: list[str], *, seed: int | None = None) -> None:
        rng = random.Random(seed)
        self._rng = rng
        self._belief_g: dict[str, float] = {aid: rng.uniform(2.0, 8.0) for aid in agent_ids}
        self._effort: dict[str, float] = {aid: rng.uniform(0.0, 1.0) for aid in agent_ids}

    def _ensure_initialized(self, agent_id: str) -> None:
        """Lazily draws parameters for an agent not seen at construction time
        -- e.g. one who migrated in from another fishery (build spec §7) and
        is now making decisions here for the first time.
        """
        if agent_id not in self._belief_g:
            self._belief_g[agent_id] = self._rng.uniform(2.0, 8.0)
            self._effort[agent_id] = self._rng.uniform(0.0, 1.0)

    async def decide_effort(self, agent_id: str, state: FisheryState) -> float:
        self._ensure_initialized(agent_id)
        alpha = state.config.alpha
        target = self._belief_g[agent_id] / max(alpha * state.stock, 1e-6)
        current = self._effort[agent_id]
        nudged = current + 0.3 * (target - current) + self._rng.uniform(-0.05, 0.05)
        nudged = min(1.0, max(0.0, nudged))
        self._effort[agent_id] = nudged
        return nudged
