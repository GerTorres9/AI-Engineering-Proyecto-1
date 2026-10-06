import os
from langsmith import traceable, tracing_context
from langchain_openai import ChatOpenAI
from app.config import Settings


def init_observability(settings: Settings) -> None:
    if settings.langsmith_tracing and not settings.langsmith_api_key:
        raise ValueError("LANGSMITH_API_KEY es obligatoria con LANGSMITH_TRACING=true")
    for name, value in {
        "LANGSMITH_TRACING": str(settings.langsmith_tracing).lower(),
        "LANGSMITH_PROJECT": settings.langsmith_project,
        "LANGSMITH_ENDPOINT": settings.langsmith_endpoint,
    }.items():
        os.environ[name] = value
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key.get_secret_value()
    if settings.langsmith_workspace_id:
        os.environ["LANGSMITH_WORKSPACE_ID"] = settings.langsmith_workspace_id


def create_llm(settings: Settings):
    # LangChain registra modelo/proveedor/usage_metadata para costo automático.
    return ChatOpenAI(model=settings.openai_model,
                      api_key=settings.openai_api_key.get_secret_value(),
                      temperature=0.3, timeout=30, max_retries=1)


def execution_context(settings: Settings, job_id: str, batch_id: str | None):
    return tracing_context(
        enabled=settings.langsmith_tracing, project_name=settings.langsmith_project,
        tags=["preentrega-7", *( [batch_id] if batch_id else [])],
        metadata={"job_id": job_id, "thread_id": job_id, "batch_id": batch_id},
    )
