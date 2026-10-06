from contextlib import asynccontextmanager
from secrets import compare_digest
from uuid import UUID
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from redis.asyncio import Redis
from redis.exceptions import RedisError
from fastapi.responses import JSONResponse
from app.config import Settings
from app.models import TaskRequest, ApprovalRequest, TaskAccepted, Job
from app.store import JobStore

bearer = HTTPBearer()


def create_app(settings: Settings | None = None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.settings = settings or Settings()
        app.state.redis = Redis.from_url(app.state.settings.redis_url, decode_responses=False)
        await app.state.redis.ping()
        app.state.store = JobStore(app.state.redis)
        try:
            yield
        finally:
            await app.state.redis.aclose()

    app = FastAPI(title="Tutor multi-agente — Pre-entrega 7", version="1.0.0", lifespan=lifespan)

    async def authorize(request: Request, auth: HTTPAuthorizationCredentials = Depends(bearer)):
        if not compare_digest(auth.credentials, request.app.state.settings.api_token.get_secret_value()):
            raise HTTPException(401, "Token de API inválido")

    async def authorize_approval(request: Request, auth: HTTPAuthorizationCredentials = Depends(bearer)):
        if not compare_digest(auth.credentials, request.app.state.settings.approval_token.get_secret_value()):
            raise HTTPException(401, "Se requiere el token del aprobador humano")

    @app.exception_handler(RedisError)
    async def redis_error(request, exc):
        return JSONResponse(status_code=503, content={"detail": "Redis no está disponible"})

    @app.get("/health")
    async def health(request: Request):
        await request.app.state.redis.ping()
        return {"status": "ok", "redis": "ok"}

    @app.post("/tasks", status_code=202, response_model=TaskAccepted, dependencies=[Depends(authorize)])
    async def submit(task: TaskRequest, request: Request):
        job = await request.app.state.store.create(task)
        return TaskAccepted(job_id=job.job_id, status_url=f"/tasks/{job.job_id}")

    @app.get("/tasks/{job_id}", response_model=Job, dependencies=[Depends(authorize)])
    async def get_job(job_id: UUID, request: Request):
        job = await request.app.state.store.get(str(job_id))
        if not job:
            raise HTTPException(404, "Job inexistente")
        return job

    @app.post("/tasks/{job_id}/approve", status_code=202, response_model=TaskAccepted,
              dependencies=[Depends(authorize_approval)])
    async def approve(job_id: UUID, decision: ApprovalRequest, request: Request):
        result = await request.app.state.store.approve(str(job_id), decision)
        if result == 0:
            raise HTTPException(404, "Job inexistente")
        if result == -1:
            raise HTTPException(409, "El job no está esperando aprobación")
        return TaskAccepted(job_id=str(job_id), status_url=f"/tasks/{job_id}")

    return app


app = create_app()
