"""FastAPI app factory: GET/WS fishery endpoints, backed by one background
`FisheryRunner` per fishery and a shared Postgres LISTEN/NOTIFY bridge
(build spec step 8, extended in step 9 to run several fisheries concurrently
-- each an independent async task per build spec §1, `asyncio.gather`-style,
migrating survivors between each other on collapse per §7).

`create_app` takes `llm_client`/`embedder` overrides so tests can substitute a
`FakeLLMClient` and the lightweight test embedder without needing an
Anthropic API key or downloading the real sentence-transformers model, while
still exercising the real Postgres event log + LISTEN/NOTIFY path end to end.
"""

import logging
import os
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import yaml
from fastapi import FastAPI

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from genfishery.api.notify_bridge import NotifyBridge
from genfishery.api.routes import router
from genfishery.api.runner import FisheryRunner, link_migration
from genfishery.config.fishery_config import FisheryConfig
from genfishery.config.model_config import ModelConfig
from genfishery.councillor.client import CouncillorClient, HttpCouncillorClient
from genfishery.councillor.config import build_default_councillor_client
from genfishery.db.session import DEFAULT_DATABASE_URL
from genfishery.llm.client import LLMClient
from genfishery.llm.provider import build_default_llm_client
from genfishery.memory.embedder import get_embedder
from genfishery.memory.registry import MemoryBankRegistry
from genfishery.sim.events_sink import PostgresEventSink

CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
DEFAULT_DEMO_CONFIG_NAMES = ["live_demo_a.yaml", "live_demo_b.yaml"]

logger = logging.getLogger(__name__)


def _load_fishery_config(filename: str) -> FisheryConfig:
    raw = yaml.safe_load((CONFIGS_DIR / "fisheries" / filename).read_text())
    return FisheryConfig(**raw)


def _load_default_demo_configs() -> list[FisheryConfig]:
    """Which fishery YAML files to load, when `create_app` isn't given an
    explicit `fishery_configs` list. `FISHERY_CONFIGS` (comma-separated
    filenames under configs/fisheries/) overrides the two-fishery default --
    e.g. `FISHERY_CONFIGS=live_demo_a.yaml` to run just one fishery, with no
    migration counterpart (migration linking only applies at exactly 2).
    """
    names_env = os.environ.get("FISHERY_CONFIGS")
    names = [n.strip() for n in names_env.split(",")] if names_env else DEFAULT_DEMO_CONFIG_NAMES
    return [_load_fishery_config(name) for name in names]


def create_app(
    *,
    llm_client: LLMClient | None = None,
    model_config: ModelConfig | None = None,
    fishery_configs: list[FisheryConfig] | None = None,
    round_interval_seconds: float = 4.0,
    embedder: Callable[[str], np.ndarray] | None = None,
    log_dir: Path | None = Path("logs"),
    councillor_client: CouncillorClient | None = None,
) -> FastAPI:
    """`log_dir` (default "./logs", relative to wherever the process is run
    from): every prompt/response for a fishery is appended to
    `{log_dir}/{fishery_id}.log`, in call order -- see `LoggingLLMClient`.
    Pass None to disable (tests do this, to avoid writing real files). When a
    councillor is wired up (see `councillor_client` below), its discussion
    transcript is logged the same way, to `{log_dir}/{fishery_id}_councillor.log`.

    `councillor_client` is optional: pass one explicitly (tests, a shared
    instance across apps), or set `OPENCODE_SERVER_URL` to have one built
    automatically from `councillor.config.build_default_councillor_client`.
    Leave both unset to skip the operationalization discussion entirely --
    the default, and what every existing deployment/test does.
    """
    resolved_configs = fishery_configs or _load_default_demo_configs()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        database_url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
        if llm_client is not None:
            # Caller supplied their own client (tests, FakeLLMClient) --
            # model_config only matters for the model *string*, which a fake
            # client ignores, so an explicit override or a bare default is
            # both fine here.
            resolved_llm = llm_client
            resolved_model_config = model_config or ModelConfig.default()
        else:
            # LLM_PROVIDER-driven (default "litellm"): the client and its
            # model config are resolved together since a provider's model
            # strings are meaningless to any other provider's endpoint.
            resolved_llm, resolved_model_config = build_default_llm_client()
        resolved_embedder = embedder or get_embedder()
        # Unlike `llm_client`, this has no always-on default: opencode is an
        # optional add-on, and most deployments (every existing test
        # included) don't have a councillor server running. Only auto-build
        # one when the deployment has explicitly opted in via
        # OPENCODE_SERVER_URL -- otherwise stay None, which
        # `run_propose_phase` treats as "skip the discussion" (see its
        # docstring), same as before this feature existed.
        if councillor_client is not None:
            resolved_councillor = councillor_client
        elif os.environ.get("OPENCODE_SERVER_URL"):
            resolved_councillor = build_default_councillor_client(resolved_model_config)
        else:
            resolved_councillor = None

        # A dedicated engine per app instance (not `db.session`'s
        # process-wide `@lru_cache`'d one): a background FisheryRunner task
        # gets cancelled on shutdown, which can leave a connection
        # "idle in transaction" if cancelled mid-insert. Disposing this
        # engine at the end of the lifespan closes every connection in its
        # pool, so that can never leak into a *different* app instance's
        # pool -- which matters most for tests, which construct many apps in
        # one process.
        engine = create_async_engine(database_url, pool_pre_ping=True)
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

        # Shared across every fishery -- memory belongs to the agent, not the
        # fishery (build spec §3), so migration only works if the registry
        # (and, harmlessly, the event sink) are the same instances everywhere.
        events = PostgresEventSink(sessionmaker)
        memory_registry = MemoryBankRegistry(resolved_embedder)
        notify_bridge = NotifyBridge(database_url)
        await notify_bridge.start()

        runners = {
            config.fishery_id: FisheryRunner(
                config,
                resolved_llm,
                resolved_model_config,
                events,
                memory_registry,
                round_interval_seconds=round_interval_seconds,
                log_dir=log_dir,
                councillor_client=resolved_councillor,
            )
            for config in resolved_configs
        }
        # Build spec step 9 is specifically two fisheries migrating survivors
        # to each other; bidirectional linking only has one unambiguous
        # meaning at N=2. A topology for N>2 fisheries is a future decision,
        # not something to guess at here.
        runner_list = list(runners.values())
        if len(runner_list) == 2:
            link_migration(runner_list[0], runner_list[1])
        elif len(runner_list) > 2:
            logger.warning(
                "create_app got %d fishery configs; migration linking is only "
                "defined for exactly 2, so no fisheries were linked",
                len(runner_list),
            )

        for runner in runner_list:
            runner.start()

        app.state.runners = runners
        app.state.notify_bridge = notify_bridge
        try:
            yield
        finally:
            for runner in runner_list:
                await runner.stop()
            await notify_bridge.stop()
            await engine.dispose()
            # Only close it if this lifespan built it -- a caller-supplied
            # `councillor_client` override (tests, a shared instance across
            # apps) is theirs to close, not ours.
            if councillor_client is None and isinstance(resolved_councillor, HttpCouncillorClient):
                await resolved_councillor.aclose()

    app = FastAPI(title="GenFishery API", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
