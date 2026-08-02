"""Drives a live fishery forward in a background asyncio task, independent of
any other fishery's clock (build spec §1, step 9: two fisheries run as
independent async tasks, `asyncio.gather`-style -- explicitly sanctioned by
the spec as an alternative to two LangGraph graphs, and what this project
already had in place from step 8).

Migration (build spec §7) is wired in here: when a round leaves a fishery
collapsed, `migration_target` (its counterpart, set via `link_migration`) gets
a random survivor. If survivors remain in this fishery afterward, collapse
was just a trigger and the loop keeps going; if it emptied out, the task
returns and the fishery stays paused (no auto-restart).
"""

import asyncio
import logging
from collections.abc import Callable
from pathlib import Path

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import ModelConfig
from genfishery.llm.client import LLMClient
from genfishery.llm.logging_client import LoggingLLMClient
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.models.events import Event, EventType
from genfishery.se_agent.client import SEAgentClient
from genfishery.se_agent.logging_client import LoggingSEAgentClient
from genfishery.sim.engine import run_round
from genfishery.sim.events_sink import EventSink
from genfishery.sim.migration import handle_collapse_and_migrate
from genfishery.sim.policies import LLMDecisionSource
from genfishery.sim.state import FisheryState
from genfishery.sim.state_replay import reconstruct_state

logger = logging.getLogger(__name__)


class FisheryRunner:
    def __init__(
        self,
        config: FisheryConfig,
        llm_client: LLMClient,
        model_config: ModelConfig,
        events: EventSink,
        memory_registry: MemoryBankRegistry,
        *,
        round_interval_seconds: float = 4.0,
        log_dir: Path | None = None,
        se_agent_client: SEAgentClient | None = None,
        on_restart_needed: Callable[[], None] | None = None,
    ) -> None:
        self.state = FisheryState.initial(config)
        self.events = events
        # `log_dir` wraps every structured_call this fishery makes (effort,
        # propose/vote, councillor reply, memory importance-rating/reflection
        # -- all of it, since they all funnel through self.llm) into one
        # plain-text, in-order log file per fishery. None (the default --
        # what every existing test uses) preserves the old unwrapped
        # behavior exactly.
        self.llm = (
            llm_client
            if log_dir is None
            else LoggingLLMClient(
                llm_client,
                log_dir / f"{config.fishery_id}.log",
                fishery_id=config.fishery_id,
                get_round=lambda: self.state.round,
            )
        )
        # `se_agent_client` is optional -- absent it, `run_propose_phase`
        # skips the operationalization discussion entirely (see its own
        # docstring), same "None preserves old behavior" convention as `llm`.
        self.se_agent = (
            None
            if se_agent_client is None
            else (
                se_agent_client
                if log_dir is None
                else LoggingSEAgentClient(
                    se_agent_client,
                    log_dir / f"{config.fishery_id}_se_agent.log",
                    fishery_id=config.fishery_id,
                )
            )
        )
        self.memory_registry = memory_registry
        # Retrieval side of the memory loop (build spec §3): every decision
        # pulls this agent's own relevant memories back into its prompt.
        self.decisions = LLMDecisionSource(self.llm, model_config, memory_registry)
        self.round_interval_seconds = round_interval_seconds
        self.migration_target: FisheryRunner | None = None
        self._task: asyncio.Task | None = None
        self._bootstrapped = False
        # Called at most once, from inside `_loop`, the instant a round's SE
        # agent implementation step actually commits a code change -- the
        # process-level restart this triggers (see `api/app.py`) affects every
        # fishery this process runs, not just this one, so orchestrating the
        # actual restart is the caller's job, not this runner's.
        self._on_restart_needed = on_restart_needed

    async def _bootstrap(self) -> None:
        """Runs once, before this runner's first round: picks the simulation
        back up from its own event history if there is any (a restart, see
        `sim/engine.py`'s module docstring on why those happen), or -- for a
        genuinely fresh fishery -- records the one event a future restart's
        replay needs but can't otherwise recover (persona assignment is
        randomized at `FisheryState.initial`, so it has to be captured once,
        not re-rolled on every replay).
        """
        reconstructed = await reconstruct_state(
            self.state.config.fishery_id, self.events, self.state.config
        )
        if reconstructed is not None:
            self.state = reconstructed
            return
        await self.events.record(
            Event.create(
                fishery_id=self.state.config.fishery_id,
                round=0,
                phase="init",
                type=EventType.FISHERY_INITIALIZED,
                payload={
                    "persona_descriptions": self.state.persona_descriptions,
                    "persona_types": self.state.persona_types,
                },
            )
        )

    async def _loop(self) -> None:
        if not self._bootstrapped:
            await self._bootstrap()
            self._bootstrapped = True
        while True:
            if self.state.collapsed:
                if self.migration_target is not None:
                    await handle_collapse_and_migrate(
                        self.state,
                        self.migration_target.state,
                        self.memory_registry,
                        self.events,
                        self.migration_target.events,
                        source_llm=self.llm,
                        target_llm=self.migration_target.llm,
                    )
                if self.state.collapsed:
                    # No migration target, or the roster is now empty either
                    # way -- stays paused, per build spec §7 point 6.
                    return

            try:
                restart_needed = await run_round(
                    self.state,
                    self.decisions,
                    self.events,
                    enable_governance=True,
                    llm=self.llm,
                    memory_registry=self.memory_registry,
                    se_agent=self.se_agent,
                )
            except Exception:
                logger.exception(
                    "round %s failed for fishery %s -- stopping the runner",
                    self.state.round,
                    self.state.config.fishery_id,
                )
                return
            if restart_needed:
                logger.info(
                    "fishery %s implemented a winning norm at round %s -- "
                    "stopping this runner to let the process restart",
                    self.state.config.fishery_id,
                    self.state.round,
                )
                if self._on_restart_needed is not None:
                    self._on_restart_needed()
                return
            await asyncio.sleep(self.round_interval_seconds)

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None


def link_migration(runner_a: FisheryRunner, runner_b: FisheryRunner) -> None:
    """Makes two runners each other's migration target -- collapse in either
    sends a survivor to the other.
    """
    runner_a.migration_target = runner_b
    runner_b.migration_target = runner_a
