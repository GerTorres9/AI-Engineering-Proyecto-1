import asyncio
import os
import pytest
import pytest_asyncio
from redis.asyncio import Redis
from app.config import Settings

os.environ["LANGSMITH_TRACING"] = "false"


@pytest.fixture
def settings():
    return Settings(_env_file=None,
        redis_url=os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15"),
        api_token="test-api", approval_token="test-human", openai_api_key="test-placeholder",
        langsmith_tracing=False, job_timeout_seconds=30)


@pytest_asyncio.fixture
async def redis(settings):
    client = Redis.from_url(settings.redis_url)
    # Prefijo tutor:; ejecuta las pruebas en DB dedicada, nunca producción.
    await client.ping()
    await client.flushdb()
    yield client
    await client.flushdb()
    await client.aclose()
