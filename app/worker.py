"""Worker separado de HTTP: Redis Streams, concurrencia y recuperación de mensajes."""
import asyncio
import logging
import socket
from contextlib import suppress
from uuid import uuid4
from redis.asyncio import Redis
from redis.exceptions import RedisError, ResponseError
from langgraph.types import Command
from langsmith import traceable
from app.config import Settings
from app.models import Status
from app.store import JobStore, STREAM, GROUP
from app.checkpointer import AsyncRedisCheckpointer
from app.graph import build_graph
from app.observability import init_observability, create_llm, execution_context

logger = logging.getLogger(__name__)
TERMINAL = {Status.DONE, Status.FAILED, Status.REJECTED}


@traceable(name="task_execution", run_type="chain", process_inputs=lambda inputs: {
    "payload": inputs["payload"], "thread_id": inputs["config"]["configurable"]["thread_id"]})
async def invoke_graph(graph, payload, config):
    return await graph.ainvoke(payload, config=config)


class Worker:
    def __init__(self, settings, redis, graph):
        self.settings = settings
        self.redis = redis
        self.store = JobStore(redis)
        self.graph = graph
        self.identity = f"{socket.gethostname()}-{uuid4().hex}"
        # Se reclama solo después del timeout de ejecución + margen.
        self.claim_idle_ms = (settings.job_timeout_seconds + 60) * 1000

    async def ensure_group(self):
        try:
            await self.redis.xgroup_create(STREAM, GROUP, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def process(self, message_id, fields):
        job_id = fields[b"job_id"].decode()
        job = await self.store.get(job_id)
        if not job or job.status in TERMINAL or job.status == Status.WAITING_APPROVAL:
            await self.redis.xack(STREAM, GROUP, message_id)
            await self.redis.xdel(STREAM, message_id)
            return
        run_id = str(uuid4())
        job = await self.store.update(job, status=Status.RUNNING, error=None,
                                      run_ids=[*job.run_ids, run_id])
        config = {"configurable": {"thread_id": job.job_id},
                  "metadata": {"job_id": job.job_id, "thread_id": job.job_id,
                               "batch_id": job.task.batch_id},
                  "tags": ["preentrega-7", *([job.task.batch_id] if job.task.batch_id else [])]}
        try:
            snapshot = await self.graph.aget_state(config)
            paused = any(task.interrupts for task in snapshot.tasks)
            if paused:
                if job.decision is None:
                    raise RuntimeError("Falta una decisión para reanudar el interrupt")
                payload = Command(resume=job.decision)
            elif snapshot.values:
                payload = None  # recuperación tras caída: continúa último checkpoint
            else:
                payload = {"job_id": job.job_id, "task": job.task.model_dump()}
            with execution_context(self.settings, job.job_id, job.task.batch_id):
                async with asyncio.timeout(self.settings.job_timeout_seconds):
                    result = await invoke_graph(self.graph, payload, config,
                                               langsmith_extra={"run_id": run_id})
            interruptions = result.get("__interrupt__", [])
            if interruptions:
                job = await self.store.update(job, status=Status.WAITING_APPROVAL,
                                              interrupt=interruptions[0].value)
            else:
                final = result["result"]
                job = await self.store.update(job, result=final, interrupt=None,
                    status=Status.REJECTED if final.get("rejected") else Status.DONE)
        except RedisError:
            # Sin Redis no podemos marcar FAILED. No ACK: otro worker recuperará.
            raise
        except Exception as exc:
            logger.exception("Fallo del job %s", job.job_id)
            await self.store.update(job, status=Status.FAILED, interrupt=None,
                error={"type": type(exc).__name__,
                       "message": "La ejecución falló; revisa la traza y el log del worker."})
        await self.redis.xack(STREAM, GROUP, message_id)
        # La cola no crece indefinidamente: se elimina solo tras ACK.
        await self.redis.xdel(STREAM, message_id)

    async def consume(self, slot):
        consumer = f"{self.identity}-{slot}"
        cursor = "0-0"
        while True:
            try:
                claimed = await self.redis.xautoclaim(STREAM, GROUP, consumer,
                    min_idle_time=self.claim_idle_ms, start_id=cursor, count=1)
                cursor = claimed[0]
                messages = claimed[1]
                if not messages:
                    result = await self.redis.xreadgroup(GROUP, consumer, {STREAM: ">"},
                                                         count=1, block=1000)
                    messages = result[0][1] if result else []
                for message_id, fields in messages:
                    await self.process(message_id, fields)
            except asyncio.CancelledError:
                raise  # mensaje pendiente recuperable; no marcar DONE ni ACK
            except Exception:
                logger.exception("Error del consumidor; se reintentará")
                await asyncio.sleep(1)

    async def run(self):
        await self.ensure_group()
        async with asyncio.TaskGroup() as group:
            for slot in range(self.settings.worker_concurrency):
                group.create_task(self.consume(slot))


async def main():
    settings = Settings()
    init_observability(settings)
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    try:
        await redis.ping()
        graph = build_graph(settings, AsyncRedisCheckpointer(redis), redis, create_llm(settings))
        await Worker(settings, redis, graph).run()
    finally:
        await redis.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with suppress(KeyboardInterrupt):
        asyncio.run(main())
