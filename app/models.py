from enum import StrEnum
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class Status(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    FAILED = "FAILED"
    DONE = "DONE"
    REJECTED = "REJECTED"


class TaskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pregunta: str = Field(min_length=3, max_length=12000)
    accion: Literal["explicar", "guardar_material"] = "explicar"
    max_output_tokens: int = Field(default=400, ge=64, le=2000)
    solicitar_aprobacion: bool = False
    batch_id: str | None = Field(default=None, pattern=r"^[a-zA-Z0-9_-]{1,64}$")


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool
    note: str = Field(default="", max_length=1000)


class Job(BaseModel):
    job_id: str
    status: Status
    task: TaskRequest
    created_at: str
    updated_at: str
    result: dict | None = None
    error: dict | None = None
    interrupt: dict | None = None
    decision: dict | None = None
    run_ids: list[str] = Field(default_factory=list)


class TaskAccepted(BaseModel):
    job_id: str
    status: Status = Status.PENDING
    status_url: str
