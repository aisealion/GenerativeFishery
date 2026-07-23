"""Decision schemas and prompts (build spec §2, §4-5; paper Fig 8-11).

Prompt wording follows Gupta et al.'s Figures 8-11 as closely as possible;
placeholders are filled from live `FisheryState`. Responses are always
structured tool-call output, never free text -- per build spec §0/§2.

Every prompt states the viewer's own agent_id (so they know which row in the
observations block is "them" -- Smallville agents are always given an
explicit identity too) and, when memories are supplied, a retrieved-memories
block (build spec §3's retrieval score feeding back into the next decision,
Smallville-style -- see `policies.LLMDecisionSource` for the retrieval side).

The observations block only shows a villager their own effort/payoff by
default. It expands to show every villager's effort/payoff once a
`PeerObservability` primitive is active -- i.e. only once the community has
proposed and voted in a transparency norm that the NormCompiler compiled to
that primitive, same opt-in pattern every primitive uses (inert until
actually voted in).

Each villager states two distinct things in the preamble: a fixed
"personality" (Gupta et al. Table 2 -- `state.persona_descriptions`, set once
at initialization and never overwritten) and a "personal strategy"
(`state.agent_norms` -- starts as `NO_NORM_YET` and evolves every round via
ProposeNormPhase). Conflating the two would mean a villager's underlying
disposition silently vanishes from the prompt the moment they revise their
own stated strategy.

The roster block lists every villager who has ever been in this fishery by
name, unconditionally (independent of PeerObservability): "active" if alive,
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

Every proposal now bundles two things in one structured call
(`ProposalDecision.community_proposal` + `.operationalization`): what the
policy should be, and how it should actually be operationalized/enforced.
Voting and compiling both carry the pair through together as one
`PolicyProposal` candidate (see `proposal_candidate_key` -- the exact string
identity a vote must reproduce to choose a candidate, both fields folded
into one, so voting for a candidate commits to its enforcement detail too,
not just its headline text) -- not a separately-clustered/voted add-on.

`OperationalizationSuggestion`/`OperationalizationCluster` and the
`build_operationalization_*` prompt builders below are a superseded,
disabled-by-default alternate design (a separate per-aspect propose ->
classify -> vote pipeline that ran *after* the main vote) -- kept defined
and unit-tested, but `run_norm_adoption` no longer calls into them. The
bundled-in-one-proposal approach above replaced it.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, create_model

from genfishery.models.norms import PeerObservability
from genfishery.sim.state import NO_NORM_YET, FisheryState

__all__ = [
    "NO_NORM_YET",
    "EffortDecision",
    "ProposalDecision",
    "PolicyProposal",
    "proposal_candidate_key",
    "NominationDecision",
    "OperationalizationSuggestion",
    "OperationalizationCluster",
    "OperationalizationProposalDecision",
    "build_effort_prompt",
    "build_proposal_prompt",
    "build_vote_prompt",
    "build_vote_response_model",
    "build_operationalization_proposal_prompt",
    "build_operationalization_vote_prompt",
    "build_operationalization_vote_response_model",
    "build_nomination_prompt",
    "build_election_vote_prompt",
    "build_election_vote_response_model",
]


@dataclass
class OperationalizationSuggestion:
    """One agent's optional suggestion for how to put a just-adopted policy
    into practice. `suggestion_id` is the 1-indexed position ("1", "2", ...)
    in the round's suggestion list -- the same id the classifier, the
    per-cluster vote ballot, and the vote-result payload all reference, so
    there's exactly one identifier scheme end to end.
    """

    suggestion_id: str
    agent_id: str
    aspect_label: str
    suggestion_text: str


@dataclass
class OperationalizationCluster:
    """One aspect the classifier grouped suggestions under. `cluster_id` is a
    generated, guaranteed-valid Python identifier ("aspect_0", "aspect_1",
    ...) used as the per-cluster vote's dynamic response-model field name --
    `canonical_aspect` (the classifier's own free-text label) is what's
    actually shown to villagers.
    """

    cluster_id: str
    canonical_aspect: str
    suggestion_ids: list[str]


class OperationalizationProposalDecision(BaseModel):
    aspect: str | None = Field(
        default=None,
        description="Short label for the aspect of the policy you're addressing, or omit if you have nothing to propose.",
    )
    suggestion: str | None = Field(
        default=None,
        description="Your concrete proposal for how that aspect should work, or omit if you have nothing to propose.",
    )


class EffortDecision(BaseModel):
    effort: float = Field(ge=0.0, le=1.0)


class ProposalDecision(BaseModel):
    personal_norm: str = Field(description="Your updated personal strategy/belief.")
    community_proposal: str = Field(description="What you propose the whole community should do.")
    operationalization: str = Field(
        description="How this policy should actually be operationalized/enforced within the fishery."
    )


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


def _has_peer_observability(state: FisheryState) -> bool:
    return any(isinstance(p, PeerObservability) for p in state.active_norms)


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
    agents = state.alive_agents if _has_peer_observability(state) else [state.agents[viewer_id]]
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


def _preamble(
    state: FisheryState, viewer_id: str, agent_norm: str, group_norm: str, memories: str = ""
) -> str:
    memory_block = f"\n\nRelevant memories from your past experience:\n{memories}" if memories else ""
    observations_label = (
        "You observe each villager's fishing effort and total payoff:"
        if _has_peer_observability(state)
        else "Your own fishing effort and total payoff so far:"
    )
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
3. Suggest how that shared community policy should actually be operationalized or enforced
   within the fishery

When proposing how the policy should be operationalized, make your proposal
specific and actionable rather than abstract or vague. The community should
be able to put the proposed policy into practice using only the people and
resources already available within the fishery. Do not assume that external
authorities, outsiders, or additional people will come to help. Clearly
specify any concrete actions, responsibilities, limits, thresholds, or
consequences needed to make the proposed policy work in practice, including
who within the community is responsible for carrying them out."""
    return system, prompt


def build_vote_response_model(candidates: list[PolicyProposal]) -> type[BaseModel]:
    """Builds a per-round response schema constraining the vote to the exact
    candidate identity strings on the ballot -- "Respond with only the exact
    text of your chosen policy" (paper Fig 11), enforced by the tool schema
    itself rather than by post-hoc free-text matching. Each candidate's key
    (`proposal_candidate_key`) folds in both the policy text and its
    proposed operationalization, so choosing a candidate commits to both.
    """
    keys = tuple(proposal_candidate_key(c) for c in candidates)
    return create_model(
        "VoteDecision",
        chosen_text=(
            Literal[keys],
            Field(description="Exact text of the policy (and its operationalization) you vote for."),
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
    candidate_block = "\n".join(f'- "{proposal_candidate_key(c)}"' for c in candidates)
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

The following community policies (with their suggested operationalization) have
been proposed this round:
{candidate_block}

Based on your personal strategy and the current state of the lake, vote for which
proposed policy (and its operationalization) you think should become the new
shared policy."""
    return system, prompt


def build_operationalization_proposal_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    raw_text: str,
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

The community just voted, and this policy was chosen as the most-voted
option — it is now the fishery's shared policy, and you are helping decide
how to actually put it into practice: "{raw_text}"

Based on your personal strategy and the fishery's current situation, pick ONE
specific aspect of how this should actually work in practice, and describe
your suggestion for it. Some examples of aspects you might address (you don't
have to use these exact words — describe it your own way): timing (when does
this happen, and relative to what), a specific number or limit, what happens
to someone who doesn't follow it, who this applies to, how often it gets
revisited, or who is going to check that this is actually being followed.

Respond with:
- aspect: a short label, in your own words, for which part of the policy
  you're addressing
- suggestion: your concrete proposal for how that aspect should work

If you don't think this policy needs anything beyond what was already said,
you don't have to propose anything this round."""
    return system, prompt


def _operationalization_cluster_block(
    clusters: list[OperationalizationCluster], suggestions: list[OperationalizationSuggestion]
) -> str:
    suggestions_by_id = {s.suggestion_id: s for s in suggestions}
    lines = []
    for cluster in clusters:
        lines.append(f'Aspect: "{cluster.canonical_aspect}"')
        for suggestion_id in cluster.suggestion_ids:
            s = suggestions_by_id[suggestion_id]
            lines.append(f'  - [{suggestion_id}] {s.agent_id}: "{s.suggestion_text}"')
    return "\n".join(lines)


def build_operationalization_vote_response_model(clusters: list[OperationalizationCluster]) -> type[BaseModel]:
    """One field per cluster/aspect, each constrained to that cluster's own
    suggestion ids plus "abstain" -- same enum-constrained-tool-schema
    approach as `build_vote_response_model`, just one field per aspect
    instead of a single top-level choice.
    """
    fields = {
        cluster.cluster_id: (
            Literal[(*cluster.suggestion_ids, "abstain")],
            Field(description=f'Your choice for the aspect "{cluster.canonical_aspect}", or "abstain".'),
        )
        for cluster in clusters
    }
    return create_model("OperationalizationVoteDecision", **fields)


def build_operationalization_vote_prompt(
    *,
    state: FisheryState,
    viewer_id: str,
    raw_text: str,
    clusters: list[OperationalizationCluster],
    suggestions: list[OperationalizationSuggestion],
    agent_norm: str = NO_NORM_YET,
    group_norm: str = NO_NORM_YET,
    memories: str = "",
) -> tuple[str, str]:
    system = "You are a villager who fishes from a shared lake together with others in your community."
    aspect_choice_block = "\n".join(
        f'- "{cluster.canonical_aspect}": choose one of {[*cluster.suggestion_ids, "abstain"]}'
        for cluster in clusters
    )
    prompt = f"""{_preamble(state, viewer_id, agent_norm, group_norm, memories)}

The community voted, and this policy was chosen as the most-voted option;
you are now helping decide how to operationalize it: "{raw_text}"

Here's how villagers suggested putting it into practice, grouped by aspect:

{_operationalization_cluster_block(clusters, suggestions)}

Based on your personal strategy and the current state of the lake, for each
aspect below, either vote for the suggestion you think is best, or choose
"abstain" if you don't want to endorse any suggestion for that aspect —
abstaining is a valid choice, not a mistake to avoid.

{aspect_choice_block}"""
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


