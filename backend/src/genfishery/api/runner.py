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
from pathlib import Path

from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import ModelConfig
from genfishery.councillor.client import CouncillorClient
from genfishery.councillor.logging_client import LoggingCouncillorClient
from genfishery.llm.client import LLMClient
from genfishery.llm.logging_client import LoggingLLMClient
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.sim.engine import run_round
from genfishery.sim.events_sink import EventSink
from genfishery.sim.migration import handle_collapse_and_migrate
from genfishery.sim.norm_compiler import NormCompiler
from genfishery.sim.policies import LLMDecisionSource
from genfishery.sim.state import FisheryState

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
        councillor_client: CouncillorClient | None = None,
    ) -> None:
        self.state = FisheryState.initial(config)
        self.events = events
        # `log_dir` wraps every structured_call this fishery makes (effort,
        # punishment, propose/vote, election, dispute, norm-compiler, memory
        # importance-rating/reflection -- all of it, since they all funnel
        # through self.llm) into one plain-text, in-order log file per
        # fishery. None (the default -- what every existing test uses)
        # preserves the old unwrapped behavior exactly.
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
        # `councillor_client` is optional -- absent it, `run_propose_phase`
        # skips the operationalization discussion entirely (see its own
        # docstring), same "None preserves old behavior" convention as `llm`.
        self.councillor = (
            None
            if councillor_client is None
            else (
                councillor_client
                if log_dir is None
                else LoggingCouncillorClient(
                    councillor_client,
                    log_dir / f"{config.fishery_id}_councillor.log",
                    fishery_id=config.fishery_id,
                )
            )
        )
        self.memory_registry = memory_registry
        # Retrieval side of the memory loop (build spec §3): every decision
        # pulls this agent's own relevant memories back into its prompt.
        self.decisions = LLMDecisionSource(self.llm, model_config, memory_registry)
        self.norm_compiler = NormCompiler(self.llm)
        self.round_interval_seconds = round_interval_seconds
        self.migration_target: FisheryRunner | None = None
        self._task: asyncio.Task | None = None

    async def _loop(self) -> None:
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
                await run_round(
                    self.state,
                    self.decisions,
                    self.events,
                    norm_compiler=self.norm_compiler,
                    llm=self.llm,
                    memory_registry=self.memory_registry,
                    councillor=self.councillor,
                )
            except Exception:
                logger.exception(
                    "round %s failed for fishery %s -- stopping the runner",
                    self.state.round,
                    self.state.config.fishery_id,
                )
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
