"""Persistencia asíncrona de conversaciones para LangGraph (Checkpointer).

Soporta dos backends de producción configurables vía variables de entorno:
1. `sqlite` (por defecto): `AsyncSqliteSaver` sobre `aiosqlite`, persistiendo en disco local
   para que las conversaciones sobrevivan reinicios del servidor sin depender de contenedores externos.
2. `redis`: `AsyncRedisCheckpointer` sobre `redis.asyncio`, compatible con Redis estándar.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from redis.asyncio import Redis

from intelligence_system.config import Settings, get_settings


class AsyncRedisCheckpointer(BaseCheckpointSaver[int]):
    """Checkpointer asíncrono compatible con Redis estándar (sin requerir RedisJSON)."""

    def __init__(self, redis_client: Redis) -> None:
        super().__init__()
        self.redis = redis_client

    def _prefix(self, config: dict[str, Any]) -> str:
        c = config["configurable"]
        thread = hashlib.sha256(str(c["thread_id"]).encode()).hexdigest()
        ns = hashlib.sha256(str(c.get("checkpoint_ns", "")).encode()).hexdigest()
        return f"intelligence:cp:{thread}:{ns}"

    def _dump(self, value: Any) -> str:
        kind, data = self.serde.dumps_typed(value)
        return json.dumps([kind, base64.b64encode(data).decode()])

    def _load(self, raw: str | bytes) -> Any:
        kind, data = json.loads(raw)
        return self.serde.loads_typed((kind, base64.b64decode(data)))

    async def aput(self, config, checkpoint, metadata, new_versions):
        prefix = self._prefix(config)
        cid = checkpoint["id"]
        document = {
            "checkpoint": self._dump(checkpoint),
            "metadata": self._dump(get_checkpoint_metadata(config, metadata)),
            "parent": get_checkpoint_id(config),
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_ns": config["configurable"].get("checkpoint_ns", ""),
        }
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(f"{prefix}:data:{cid}", json.dumps(document))
            pipe.zadd(prefix + ":index", {cid: 0})
            await pipe.execute()
        return {
            "configurable": {
                "thread_id": document["thread_id"],
                "checkpoint_ns": document["checkpoint_ns"],
                "checkpoint_id": cid,
            }
        }

    async def aput_writes(self, config, writes, task_id, task_path=""):
        key = f"{self._prefix(config)}:writes:{get_checkpoint_id(config)}"
        async with self.redis.pipeline(transaction=True) as pipe:
            for idx, (channel, value) in enumerate(writes):
                index = WRITES_IDX_MAP.get(channel, idx)
                field = json.dumps([task_id, index])
                payload = json.dumps([task_id, channel, self._dump(value), task_path])
                if index < 0:
                    pipe.hset(key, field, payload)
                else:
                    pipe.hsetnx(key, field, payload)
            await pipe.execute()

    async def aget_tuple(self, config) -> CheckpointTuple | None:
        prefix = self._prefix(config)
        cid = get_checkpoint_id(config)
        if not cid:
            ids = await self.redis.zrevrange(prefix + ":index", 0, 0)
            if not ids:
                return None
            cid = ids[0].decode() if isinstance(ids[0], bytes) else str(ids[0])
        raw = await self.redis.get(f"{prefix}:data:{cid}")
        if not raw:
            return None
        doc = json.loads(raw)
        c = {"thread_id": doc["thread_id"], "checkpoint_ns": doc["checkpoint_ns"]}
        writes = await self.redis.hgetall(f"{prefix}:writes:{cid}")
        pending = []
        for payload in writes.values():
            task_id, channel, value, _ = json.loads(payload)
            pending.append((task_id, channel, self._load(value)))
        return CheckpointTuple(
            config={"configurable": {**c, "checkpoint_id": cid}},
            checkpoint=self._load(doc["checkpoint"]),
            metadata=self._load(doc["metadata"]),
            pending_writes=pending,
            parent_config=(
                {"configurable": {**c, "checkpoint_id": doc["parent"]}}
                if doc["parent"]
                else None
            ),
        )

    async def alist(
        self, config, *, filter=None, before=None, limit=None
    ) -> AsyncIterator[CheckpointTuple]:
        if config:
            prefixes = [self._prefix(config)]
        else:
            prefixes = [
                (k.decode() if isinstance(k, bytes) else str(k)).removesuffix(":index")
                async for k in self.redis.scan_iter(match="intelligence:cp:*:index")
            ]
        candidates = []
        for prefix in prefixes:
            for raw_cid in await self.redis.zrevrange(prefix + ":index", 0, -1):
                cid = raw_cid.decode() if isinstance(raw_cid, bytes) else str(raw_cid)
                if before and cid >= (get_checkpoint_id(before) or ""):
                    continue
                if config and get_checkpoint_id(config) and cid != get_checkpoint_id(config):
                    continue
                raw = await self.redis.get(f"{prefix}:data:{cid}")
                if raw:
                    candidates.append((cid, json.loads(raw)))
        count = 0
        for cid, doc in sorted(candidates, key=lambda item: item[0], reverse=True):
            if limit is not None and count >= limit:
                break
            saved = await self.aget_tuple(
                {
                    "configurable": {
                        "thread_id": doc["thread_id"],
                        "checkpoint_ns": doc["checkpoint_ns"],
                        "checkpoint_id": cid,
                    }
                }
            )
            if saved is None:
                continue
            if filter and not all(saved.metadata.get(k) == v for k, v in filter.items()):
                continue
            yield saved
            count += 1


@asynccontextmanager
async def create_checkpointer_context(
    settings: Settings | None = None,
) -> AsyncIterator[BaseCheckpointSaver]:
    """Context manager asíncrono que inicializa y cierra limpiamente el Checkpointer."""
    cfg = settings or get_settings()
    if cfg.checkpointer_backend == "redis":
        redis_client = Redis.from_url(cfg.redis_url, decode_responses=False)
        await redis_client.ping()
        saver = AsyncRedisCheckpointer(redis_client)
        try:
            yield saver
        finally:
            await redis_client.aclose()
    else:
        db_path = Path(cfg.sqlite_checkpoint_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(str(db_path))
        saver = AsyncSqliteSaver(conn)
        await saver.setup()
        try:
            yield saver
        finally:
            await conn.close()
