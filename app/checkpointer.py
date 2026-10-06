"""Checkpointer async equivalente a RedisSaver, compatible con Redis estándar.

Guarda checkpoints completos, metadatos, padres y pending_writes de LangGraph.
No requiere RedisJSON/RediSearch; usa el contrato oficial BaseCheckpointSaver.
Uso exclusivamente async: aget_tuple/alist/aput/aput_writes/adelete_thread.
"""
import base64
import hashlib
import json
from collections.abc import AsyncIterator
from langgraph.checkpoint.base import (
    BaseCheckpointSaver, CheckpointTuple, WRITES_IDX_MAP,
    get_checkpoint_id, get_checkpoint_metadata,
)
from redis.asyncio import Redis


class AsyncRedisCheckpointer(BaseCheckpointSaver[int]):
    def __init__(self, redis: Redis):
        super().__init__()
        self.redis = redis

    def _prefix(self, config) -> str:
        c = config["configurable"]
        thread = hashlib.sha256(c["thread_id"].encode()).hexdigest()
        ns = hashlib.sha256(c.get("checkpoint_ns", "").encode()).hexdigest()
        return f"tutor:cp:{thread}:{ns}"

    def _dump(self, value) -> str:
        kind, data = self.serde.dumps_typed(value)
        return json.dumps([kind, base64.b64encode(data).decode()])

    def _load(self, raw):
        kind, data = json.loads(raw)
        return self.serde.loads_typed((kind, base64.b64decode(data)))

    async def aput(self, config, checkpoint, metadata, new_versions):
        prefix = self._prefix(config)
        cid = checkpoint["id"]
        document = {"checkpoint": self._dump(checkpoint),
                    "metadata": self._dump(get_checkpoint_metadata(config, metadata)),
                    "parent": get_checkpoint_id(config),
                    "thread_id": config["configurable"]["thread_id"],
                    "checkpoint_ns": config["configurable"].get("checkpoint_ns", "")}
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(f"{prefix}:data:{cid}", json.dumps(document))
            # Los UUID de checkpoints son ordenables. Score constante => orden lex.
            pipe.zadd(prefix + ":index", {cid: 0})
            await pipe.execute()
        return {"configurable": {"thread_id": document["thread_id"],
                                 "checkpoint_ns": document["checkpoint_ns"],
                                 "checkpoint_id": cid}}

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
            cid = ids[0].decode()
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
            metadata=self._load(doc["metadata"]), pending_writes=pending,
            parent_config={"configurable": {**c, "checkpoint_id": doc["parent"]}}
            if doc["parent"] else None,
        )

    async def alist(self, config, *, filter=None, before=None, limit=None) -> AsyncIterator:
        if config:
            prefixes = [self._prefix(config)]
        else:
            prefixes = [key.decode().removesuffix(":index")
                        async for key in self.redis.scan_iter(match="tutor:cp:*:index")]
        candidates = []
        for prefix in prefixes:
            for cid in await self.redis.zrevrange(prefix + ":index", 0, -1):
                cid = cid.decode()
                if before and cid >= get_checkpoint_id(before):
                    continue
                if config and get_checkpoint_id(config) and cid != get_checkpoint_id(config):
                    continue
                raw = await self.redis.get(f"{prefix}:data:{cid}")
                if raw:
                    doc = json.loads(raw)
                    candidates.append((cid, doc))
        count = 0
        for cid, doc in sorted(candidates, key=lambda item: item[0], reverse=True):
            if limit is not None and count >= limit:
                break
            saved = await self.aget_tuple({"configurable": {
                "thread_id": doc["thread_id"], "checkpoint_ns": doc["checkpoint_ns"],
                "checkpoint_id": cid}})
            if filter and not all(saved.metadata.get(k) == v for k, v in filter.items()):
                continue
            yield saved
            count += 1

    async def adelete_thread(self, thread_id: str):
        thread = hashlib.sha256(thread_id.encode()).hexdigest()
        async for key in self.redis.scan_iter(match=f"tutor:cp:{thread}:*"):
            await self.redis.unlink(key)
