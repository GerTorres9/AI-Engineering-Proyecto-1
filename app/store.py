"""Jobs + cola durable. Las mutaciones de estado y cola son atómicas."""
from datetime import datetime, timezone
from uuid import uuid4
from redis.asyncio import Redis
from app.models import Job, Status, TaskRequest, ApprovalRequest

STREAM = "tutor:queue"
GROUP = "tutor-workers"
JOBS = "tutor:jobs:"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self, redis: Redis):
        self.redis = redis

    async def create(self, task: TaskRequest) -> Job:
        job = Job(job_id=str(uuid4()), status=Status.PENDING, task=task,
                  created_at=now(), updated_at=now())
        async with self.redis.pipeline(transaction=True) as pipe:
            pipe.set(JOBS + job.job_id, job.model_dump_json())
            pipe.xadd(STREAM, {"job_id": job.job_id})
            await pipe.execute()
        return job

    async def get(self, job_id: str) -> Job | None:
        raw = await self.redis.get(JOBS + job_id)
        return Job.model_validate_json(raw) if raw else None

    async def update(self, job: Job, **fields) -> Job:
        updated = job.model_copy(update={**fields, "updated_at": now()})
        await self.redis.set(JOBS + job.job_id, updated.model_dump_json())
        return updated

    async def approve(self, job_id: str, decision: ApprovalRequest) -> int:
        # Un único evaluador puede cambiar WAITING_APPROVAL y publicar el mensaje.
        script = """
        local raw = redis.call('GET', KEYS[1])
        if not raw then return 0 end
        local job = cjson.decode(raw)
        if job.status ~= 'WAITING_APPROVAL' then return -1 end
        job.status = 'PENDING'
        job.decision = cjson.decode(ARGV[1])
        job.updated_at = ARGV[2]
        redis.call('SET', KEYS[1], cjson.encode(job))
        redis.call('XADD', KEYS[2], '*', 'job_id', ARGV[3])
        return 1
        """
        return await self.redis.eval(script, 2, JOBS + job_id, STREAM,
                                     decision.model_dump_json(), now(), job_id)
