import asyncio
import json
from contextlib import suppress
from uuid import uuid4
import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.runnables import RunnableLambda
from app.checkpointer import AsyncRedisCheckpointer
from app.graph import build_graph
from app.main import create_app
from app.models import TaskRequest, Status, ApprovalRequest
from app.store import JobStore, STREAM, GROUP
from app.worker import Worker


class FakeModel:
    """Solo en tests: respuestas deterministas, sin tokens ni costos ficticios."""
    def __init__(self, fail=False):
        self.fail = fail
        self.active = 0
        self.max_active = 0
        self.calls = 0
        self.budgets = []

    def bind(self, *, max_tokens):
        self.budgets.append(max_tokens)
        async def respond(prompt):
            self.calls += 1
            if self.fail:
                raise ValueError("fallo de prueba")
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            try:
                await asyncio.sleep(0.03)
                return "Una coroutine permite ceder el control mientras espera I/O."
            finally:
                self.active -= 1
        return RunnableLambda(respond)


async def message(redis, consumer="test"):
    entries = await redis.xreadgroup(GROUP, consumer, {STREAM: ">"}, count=1)
    return entries[0][1][0]


async def poll(client, job_id):
    for _ in range(100):
        job = (await client.get(f"/tasks/{job_id}")).json()
        if job["status"] in {"DONE", "FAILED", "REJECTED", "WAITING_APPROVAL"}:
            return job
        await asyncio.sleep(0.02)
    raise AssertionError("El trabajo no finalizó")


async def test_five_concurrent_requests(settings, redis):
    model = FakeModel()
    graph = build_graph(settings, AsyncRedisCheckpointer(redis), redis, model)
    worker = Worker(settings, redis, graph)
    running = asyncio.create_task(worker.run())
    app = create_app(settings)
    try:
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app), base_url="http://test",
                                    headers={"Authorization": "Bearer test-api"}) as client:
                responses = await asyncio.gather(*[
                    client.post("/tasks", json={"pregunta": "¿Qué es async?", "batch_id": "test-carga"})
                    for _ in range(5)])
                assert all(r.status_code == 202 for r in responses)
                ids = [r.json()["job_id"] for r in responses]
                assert len(set(ids)) == 5
                jobs = await asyncio.gather(*(poll(client, job_id) for job_id in ids))
                assert all(j["status"] == "DONE" for j in jobs)
                assert model.calls == 15
                assert model.max_active == 5
                assert set(model.budgets) == {400}
                assert all(len(j["run_ids"]) == 1 for j in jobs)
                assert (await client.get("/health")).status_code == 200
    finally:
        running.cancel()
        with suppress(asyncio.CancelledError):
            await running


async def test_hitl_survives_new_graph_and_single_approval(settings, redis):
    store = JobStore(redis)
    model = FakeModel()
    saver = AsyncRedisCheckpointer(redis)
    worker = Worker(settings, redis, build_graph(settings, saver, redis, model))
    await worker.ensure_group()
    job = await store.create(TaskRequest(pregunta="Explica async", accion="guardar_material"))
    await worker.process(*await message(redis))
    paused = await store.get(job.job_id)
    assert paused.status == Status.WAITING_APPROVAL
    assert model.calls == 0
    assert not await redis.exists("tutor:materials:" + job.job_id)
    assert (await saver.aget_tuple({"configurable": {"thread_id": job.job_id}})) is not None
    assert len([c async for c in saver.alist({"configurable": {"thread_id": job.job_id}})]) >= 2
    responses = await asyncio.gather(*[
        store.approve(job.job_id, ApprovalRequest(approved=True)) for _ in range(2)])
    assert sorted(responses) == [-1, 1]
    # Nueva instancia: no depende del grafo/estado en memoria del proceso anterior.
    worker2 = Worker(settings, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, model))
    await worker2.process(*await message(redis))
    done = await store.get(job.job_id)
    assert done.status == Status.DONE
    assert model.calls == 3
    assert await redis.exists(done.result["material_key"])
    assert len(done.run_ids) == 2
    await saver.adelete_thread(job.job_id)
    assert await saver.aget_tuple({"configurable": {"thread_id": job.job_id}}) is None


async def test_rejection_and_expensive_request_pause(settings, redis):
    model = FakeModel()
    worker = Worker(settings, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, model))
    await worker.ensure_group()
    job = await worker.store.create(TaskRequest(pregunta="Explica async", max_output_tokens=1200))
    await worker.process(*await message(redis))
    assert (await worker.store.get(job.job_id)).status == Status.WAITING_APPROVAL
    assert model.calls == 0
    await worker.store.approve(job.job_id, ApprovalRequest(approved=False, note="Costo alto"))
    await worker.process(*await message(redis))
    assert (await worker.store.get(job.job_id)).status == Status.REJECTED
    assert model.calls == 0


async def test_llm_error_becomes_failed_and_queue_acked(settings, redis):
    worker = Worker(settings, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, FakeModel(fail=True)))
    await worker.ensure_group()
    job = await worker.store.create(TaskRequest(pregunta="Explica async"))
    await worker.process(*await message(redis))
    failed = await worker.store.get(job.job_id)
    assert failed.status == Status.FAILED
    assert failed.error["type"] == "ValueError"
    assert (await redis.xpending(STREAM, GROUP))["pending"] == 0


async def test_api_auth_validation_and_approval_role(settings, redis):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test") as client:
            assert (await client.post("/tasks", json={"pregunta": "async"})).status_code in {401, 403}
            client.headers["Authorization"] = "Bearer test-api"
            assert (await client.post("/tasks", json={"pregunta": "x"})).status_code == 422
            assert (await client.get(f"/tasks/{uuid4()}")).status_code == 404
            assert (await client.get("/tasks/no-es-uuid")).status_code == 422
            response = await client.post("/tasks", json={"pregunta": "Explica async"})
            job_id = response.json()["job_id"]
            assert (await client.post(f"/tasks/{job_id}/approve", json={"approved": True})).status_code == 401
            client.headers["Authorization"] = "Bearer test-human"
            assert (await client.post(f"/tasks/{job_id}/approve", json={"approved": True})).status_code == 409
            assert (await client.post(f"/tasks/{uuid4()}/approve", json={"approved": True})).status_code == 404


async def test_abandoned_message_recovery(settings, redis):
    worker = Worker(settings, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, FakeModel()))
    await worker.ensure_group()
    job = await worker.store.create(TaskRequest(pregunta="Explica async"))
    msg_id, fields = await message(redis, consumer="worker-caido")
    await worker.store.update(job, status=Status.RUNNING)
    reclaimed = await redis.xautoclaim(STREAM, GROUP, "worker-nuevo", min_idle_time=0, start_id="0-0")
    assert len(reclaimed[1]) == 1
    await worker.process(*reclaimed[1][0])
    assert (await worker.store.get(job.job_id)).status == Status.DONE


async def test_endpoint_queues_without_worker(settings, redis):
    app = create_app(settings)
    async with app.router.lifespan_context(app):
        async with AsyncClient(transport=ASGITransport(app), base_url="http://test",
                headers={"Authorization": "Bearer test-api"}) as client:
            response = await client.post("/tasks", json={"pregunta": "Explica async"})
            assert response.status_code == 202
            job = (await client.get(response.json()["status_url"])).json()
            assert job["status"] == "PENDING"
            assert job["result"] is None
            assert await redis.xlen(STREAM) == 1


async def test_worker_timeout_marks_failed(settings, redis):
    short = settings.model_copy(update={"job_timeout_seconds": 0.005})
    worker = Worker(short, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, FakeModel()))
    await worker.ensure_group()
    job = await worker.store.create(TaskRequest(pregunta="Explica async"))
    await worker.process(*await message(redis))
    failed = await worker.store.get(job.job_id)
    assert failed.status == Status.FAILED
    assert failed.error["type"] == "TimeoutError"


async def test_cancelled_worker_continues_completed_nodes(settings, redis):
    model = FakeModel()
    worker = Worker(settings, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, model))
    await worker.ensure_group()
    job = await worker.store.create(TaskRequest(pregunta="Explica async"))
    msg = await message(redis, "worker-caido")
    active = asyncio.create_task(worker.process(*msg))
    # Cancela mientras el tutor trabaja: el planificador ya está checkpointed.
    for _ in range(200):
        if model.calls >= 2:
            break
        await asyncio.sleep(0.001)
    assert model.calls == 2
    active.cancel()
    with suppress(asyncio.CancelledError):
        await active
    assert (await worker.store.get(job.job_id)).status == Status.RUNNING
    recovered = await redis.xautoclaim(STREAM, GROUP, "nuevo", min_idle_time=0, start_id="0-0")
    newer = Worker(settings, redis, build_graph(settings, AsyncRedisCheckpointer(redis), redis, model))
    await newer.process(*recovered[1][0])
    assert (await newer.store.get(job.job_id)).status == Status.DONE
    assert model.calls == 4  # planificador 1, tutor cancelado 1, tutor + revisor 2
