"""Decision schemas and prompts (build spec §2, §4-5; paper Fig 8-11).

Prompt wording follows Gupta et al.'s Figures 8-11 as closely as possible;
placeholders are filled from live `FisheryState`. Responses are always
structured tool-call output, never free text -- per build spec §0/§2.

Every prompt states the viewer's own agent_id (so they know which row in the
observations block is "them" -- Smallville agents are always given an
explicit identity too) and, when memories are supplied, a retrieved-memories
block (build spec §3's retrieval score feeding back into the next decision,
Smallville-style -- see `policies.LLMDecisionSource` for the retrieval side).

The observations block only ever shows a villager their own effort/payoff --
there's no structured "transparency norm" mechanism anymore (see
`sim/engine.py`'s module docstring on why); an SE-agent-implemented norm that
wants to change what's visible would do so directly in code.

Each villager states two distinct things in the preamble: a fixed
"personality" (Gupta et al. Table 2 -- `state.persona_descriptions`, set once
at initialization and never overwritten) and a "personal strategy"
(`state.agent_norms` -- starts as `NO_NORM_YET` and evolves every round via
ProposeNormPhase). Conflating the two would mean a villager's underlying
disposition silently vanishes from the prompt the moment they revise their
own stated strategy.

The roster block lists every villager who has ever been in this fishery by
name, unconditionally: "active" if alive,
or "removed (underharvest)"/"removed (punished)" if they starved --
`state.starvation_reasons`, set once by `run_starvation_check` based on
whether their payoff was already negative right after harvest (underharvest)
or only went negative once a punishment penalty was applied (punished). This
means a villager's departure is always visible in every future prompt, not
just when a "starved" memory happens to be retrieved.

A migrated-in agent is tagged "arrived recently to the fishery" in the roster
for `NEWCOMER_ROUND_WINDOW` rounds (`state.migration_arrival_round`, set once
by `migrate_random_survivor`), then folds back into a plain "active" entry.
From the round after they arrive on, they're just another entry in
`state.alive_agents` and participate in every phase identically to everyone
else -- no special-casing anywhere in `engine.py`.

A `ProposalDecision` is norm-only (`personal_norm` + `community_proposal`) --
agents propose whatever policy they want with no operationalization question
attached. How to operationalize it is worked out afterward, one proposing
agent at a time, in a back-and-forth with the fishery councillor
(`sim.engine.run_operationalization_discussion_phase`): the councillor asks
how the norm should work in practice, the agent replies via
`decide_councillor_reply`/`build_councillor_reply_prompt` below, and on the
final configured round the agent is told to finalize a concrete,
operationalizable version -- that final reply becomes `.operationalization`.
Voting and compiling both carry the pair through together as one
`PolicyProposal` candidate (see `proposal_candidate_key` -- the display text
shown on the ballot; the vote itself picks a short 1-indexed ballot id, not
this text, since asking a model to reproduce a long string byte-for-byte
turned out to be fragile -- see `build_vote_response_model`), so voting for
a candidate commits to its enforcement detail too, not just its headline
text.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, create_model

from genfishery.sim.state import NO_NORM_YET, FisheryState

__all__ = [
    "NO_NORM_YET",
    "EffortDecision",
    "ProposalDecision",
    "PolicyProposal",
    "proposal_candidate_key",
    "NominationDecision",
    "CouncillorReplyDecision",
    "active_norms_summary",
    "build_effort_prompt",
    "build_proposal_prompt",
    "build_vote_prompt",
    "build_vote_response_model",
    "build_councillor_reply_prompt",
    "build_nomination_prompt",
    "build_election_vote_prompt",
    "build_election_vote_response_model",
]


class EffortDecision(BaseModel):
    effort: float = Field(ge=0.0, le=1.0)


class ProposalDecision(BaseModel):
    personal_norm: str = Field(description="Your updated personal strategy/belief.")
    community_proposal: str = Field(description="What you propose the whole community should do.")


class CouncillorReplyDecision(BaseModel):
    reply: str = Field(description="Your reply to the fishery councillor.")


@dataclass(frozen=True)
class PolicyProposal:
    """One agent's bundled (policy, how-to-enforce-it) proposal -- the unit
    that's voted on and compiled. Frozen/hashable so it can be deduplicated
    via `dict.fromkeys` and used directly as a vote-tally dict key, same as
    the plain proposal-text candidates were before this field existed.
    """

    community_proposal: str
    operationalization: str


def proposal_candidate_key(proposal: PolicyProposal) -> str:
    """The exact string a vote must reproduce to choose this candidate --
    both fields folded into one identity, so voting for a candidate commits
    to its operationalization detail too, not just its headline policy text.
    """
    return f'{proposal.community_proposal}\n(How to operationalize: {proposal.operationalization})'


class NominationDecision(BaseModel):
    self_nominate: bool = Field(description="Whether you nominate/volunteer yourself for the role.")


# How many rounds a migrated-in agent is tagged as a newcomer in the roster
# block before folding back into a plain "active" entry.
NEWCOMER_ROUND_WINDOW = 5


def _roster_block(state: FisheryState, viewer_id: str) -> str:
    lines = []
    for agent in state.agents.values():
        marker = " (you)" if agent.agent_id == viewer_id else ""
        if agent.alive:
            arrival_round = state.migration_arrival_round.get(agent.agent_id)
            if arrival_round is not None and state.round - arrival_round <= NEWCOMER_ROUND_WINDOW:
                lines.append(f"- {agent.agent_id}{marker}: arrived recently to the fishery")
            else:
                lines.append(f"- {agent.agent_id}{marker}: active")
        else:
            reason = state.starvation_reasons.get(agent.agent_id, "starved")
            lines.append(f"- {agent.agent_id}{marker}: removed ({reason})")
    return "\n".join(lines)


def _observations_block(state: FisheryState, viewer_id: str) -> str:
    agents = [state.agents[viewer_id]]
    lines = []
    for agent in agents:
        marker = " (you)" if agent.agent_id == viewer_id else ""
        if agent.last_effort is not None:
            lines.append(
                f"- {agent.agent_id}{marker}: effort={agent.last_effort:.2f}, payoff={agent.payoff:.2f}"
            )
        else:
            lines.append(f"- {agent.agent_id}{marker}: (no observations yet), payoff={agent.payoff:.2f}")
    return "\n".join(lines)


def active_norms_summary(state: FisheryState) -> str:
    """The "how this fishery currently works" grounding fed to the fishery
    SE agent -- just the community's currently-adopted norm text. There's no
    structured primitive breakdown anymore (see `sim/engine.py`'s module
    docstring): how a norm is actually enforced is whatever code the SE
    agent has itself written into the engine for it, which it already knows
    since it wrote it.
    """
    if state.group_norm_text == NO_NORM_YET:
        return "No formal rules are in place yet -- villagers fish under their own personal strategies only."
    return f'- The community\'s current policy is: "{state.group_norm_text}"'


def _preamble(
    state: FisheryState, viewer_id: str, agent_norm: str, group_norm: str, memories: str = ""
) -> str:
    memory_block = f"\n\nRelevant memories from your past experience:\n{memories}" if memories else ""
    observations_label = "Your own fishing effort and total payoff so far:"
    persona_description = state.persona_descriptions.get(viewer_id)
    personality_block = f'\n\nYour personality: "{persona_description}"' if persona_description else ""
    return f"""You are villager {viewer_id}, one of several villagers who fish from a shared lake
together. Each villager needs to consume {state.config.consumption:.2f} units of fish
daily to survive. When your payoff becomes negative, you die.{personality_block}

Villagers in this fishery:
{_roster_block(state, viewer_id)}

Each villager holds a personal strategy about what they should do,
and the community has also a shared policy.

Your personal strategy: "{agent_norm}"
Shared community policy: "{group_norm}"

{observations_label}
{_observations_block(state, viewer_id)}{memory_block}"""


def build_effort_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

Based on both your personal belief and the community policy, decide how much effort
you want to put into fishing today, as a number between 0.0 (no fishing at all)
and 1.0 (fishing as hard as you possibly can)."""
    return system, prompt


def build_proposal_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

Based on your observations:
1. Update your personal strategy about what you should do
2. Propose what the others should do in the community

Propose whatever policy you think is right -- you'll discuss how to actually
put it into practice with the fishery councillor afterward."""
    return system, prompt


def _transcript_block(transcript: list[tuple[str, str]]) -> str:
    if not transcript:
        return ""
    lines = [f'{"Councillor" if speaker == "councillor" else "You"}: "{text}"' for speaker, text in transcript]
    return "\n\nDiscussion so far:\n" + "\n".join(lines)


def build_councillor_reply_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    community_proposal: str,
    transcript: list[tuple[str, str]],
    councillor_message: str,
    is_final_turn: bool,
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    finalize_block = (
        "\n\nThis is the final round of this discussion. Finalize your norm now into one "
        "clear, concrete, operationalizable version -- this exact text becomes the fishery's "
        "official policy going to a vote."
        if is_final_turn
        else ""
    )
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

You proposed this policy to the community: "{community_proposal}"

You are now discussing with the fishery counsellor -- who knows how this fishery
currently works right now -- how your proposal should actually work in practice.{_transcript_block(transcript)}

The counsellor says: "{councillor_message}"{finalize_block}

Reply to the counsellor."""
    return system, prompt


def build_vote_response_model(candidates: list[PolicyProposal]) -> type[BaseModel]:
    """Builds a per-round response schema constraining the vote to a short
    1-indexed ballot id rather than the candidate's full text. An earlier
    version asked the model to reproduce a candidate's exact text
    (`proposal_candidate_key`) verbatim -- that broke two ways in practice:
    truncation on a tight token budget ("Unterminated string"), and, even
    with headroom, near-but-not-byte-identical reproduction (smart quotes,
    dashes, minor rewording) failing the `Literal` match. A short id has
    nothing to get subtly wrong.
    """
    ids = tuple(str(i) for i in range(1, len(candidates) + 1))
    return create_model(
        "VoteDecision",
        chosen_id=(
            Literal[ids],
            Field(description="The ballot number of the policy (and its operationalization) you vote for."),
        ),
    )


def build_vote_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    candidates: list[PolicyProposal],
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    candidate_block = "\n".join(
        f'{i}. "{proposal_candidate_key(c)}"' for i, c in enumerate(candidates, start=1)
    )
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

The following community policies (with their suggested operationalization) have
been proposed this round:
{candidate_block}

Based on your personal strategy and the current state of the lake, vote for which
proposed policy (and its operationalization) you think should become the new
shared policy, by its ballot number."""
    return system, prompt


def build_nomination_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    role_name: str,
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

The community has decided to establish the role of "{role_name}", chosen by election
from self-nominated candidates. Do you want to nominate yourself for this role?"""
    return system, prompt


def build_election_vote_response_model(candidates: list[str]) -> type[BaseModel]:
    """Same enum-constrained-tool-schema pattern as `build_vote_response_model`,
    over candidate agent IDs instead of proposal texts.
    """
    return create_model(
        "ElectionVoteDecision",
        chosen_agent_id=(Literal[tuple(candidates)], Field(description="Agent ID you vote for.")),
    )


def build_election_vote_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    role_name: str,
    candidates: list[str],
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    candidate_block = "\n".join(f"- {c}" for c in candidates)
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

The following villagers have nominated themselves for the role of "{role_name}":
{candidate_block}

Vote for who you think should hold this role."""
    return system, prompt


