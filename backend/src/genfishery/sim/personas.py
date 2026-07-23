"""Initial persona assignment (Gupta et al. Table 2).

Each agent is assigned one of these templates as a fixed personality
description (`FisheryState.persona_descriptions`), stated in every decision
prompt as the agent's underlying disposition -- distinct from and never
overwritten by `FisheryState.agent_norms` (the agent's own evolving personal
strategy, updated each round via ProposeNormPhase). `altruism_ratio` controls
the population composition: with N agents and ratio r, exactly round(N*r) are
altruistic (a controlled split, not a per-agent coin flip -- matching the
paper's framing of altruism ratio as an experimental variable), each then
assigned one random template from their type's list.
"""

import random

ALTRUISTIC_TEMPLATES = [
    "Fish only what you need to feed your family",
    "Preserve the lake for future generations",
    "Follow the community guidelines strictly",
    "Fish moderately to maintain the ecosystem",
    "Prioritize long-term sustainability over short-term gain",
    "Consider the needs of the entire community before fishing",
    "Leave enough fish for others and future generations",
    "Share the lake's resources fairly with all villagers",
]

SELFISH_TEMPLATES = [
    "Maximize your catch while the fish are abundant",
    "Take as much as you can before others do",
    "Your family's needs come first, fish as much as possible",
    "The lake is there to be used, take your maximum share",
    "Fish aggressively to ensure your own survival",
    "Get the most value from your fishing effort",
    "Compete to harvest more than others",
    "Focus on your immediate gains from the lake",
]


def assign_personas(
    agent_ids: list[str], altruism_ratio: float, *, rng: random.Random | None = None
) -> tuple[dict[str, str], dict[str, str]]:
    """Returns (persona_descriptions, persona_types): agent_id -> assigned
    template text, and agent_id -> "altruistic" | "selfish".
    """
    rng = rng or random.Random()
    n_altruistic = round(len(agent_ids) * altruism_ratio)
    shuffled = list(agent_ids)
    rng.shuffle(shuffled)
    altruistic_ids = set(shuffled[:n_altruistic])

    persona_descriptions: dict[str, str] = {}
    persona_types: dict[str, str] = {}
    for agent_id in agent_ids:
        if agent_id in altruistic_ids:
            persona_descriptions[agent_id] = rng.choice(ALTRUISTIC_TEMPLATES)
            persona_types[agent_id] = "altruistic"
        else:
            persona_descriptions[agent_id] = rng.choice(SELFISH_TEMPLATES)
            persona_types[agent_id] = "selfish"
    return persona_descriptions, persona_types
