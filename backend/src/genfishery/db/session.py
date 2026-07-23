import os
from functools import lru_cache

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

DEFAULT_DATABASE_URL = "postgresql+asyncpg://genfishery:genfishery@localhost:5432/genfishery"


@lru_cache
def get_engine() -> AsyncEngine:
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    return create_async_engine(url, pool_pre_ping=True)


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(get_engine(), expire_on_commit=False)
