"""Configuración centralizada basada en variables de entorno con validación Pydantic.

Evita cualquier valor hard-codeado de modelos, rutas de índices o endpoints.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Esquema validado de configuración de entorno para The Intelligence System."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    # --- LLM y Embeddings (sin hard-coding) ---
    openai_api_key: SecretStr = Field(default=SecretStr(""), alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small", alias="OPENAI_EMBEDDING_MODEL"
    )
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    llm_timeout_seconds: float = Field(
        default=30.0, ge=5.0, le=300.0, alias="LLM_TIMEOUT_SECONDS"
    )
    llm_max_retries: int = Field(default=2, ge=0, le=5, alias="LLM_MAX_RETRIES")

    # --- RAG y Base Vectorial (ChromaDB) ---
    chroma_persist_dir: Path = Field(
        default=PROJECT_ROOT / "storage" / "chroma", alias="CHROMA_PERSIST_DIR"
    )
    chroma_collection_name: str = Field(
        default="intelligence_system_kb", alias="CHROMA_COLLECTION_NAME"
    )
    knowledge_base_dir: Path = Field(
        default=PROJECT_ROOT / "data" / "knowledge_base", alias="KNOWLEDGE_BASE_DIR"
    )
    rag_top_k: int = Field(default=4, ge=1, le=20, alias="RAG_TOP_K")
    chunk_size: int = Field(default=500, ge=100, le=4000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, ge=0, le=1000, alias="CHUNK_OVERLAP")

    # --- Persistencia (LangGraph Checkpointer + DB Operativa) ---
    checkpointer_backend: Literal["sqlite", "redis"] = Field(
        default="sqlite", alias="CHECKPOINTER_BACKEND"
    )
    sqlite_checkpoint_path: Path = Field(
        default=PROJECT_ROOT / "storage" / "checkpoints.sqlite",
        alias="SQLITE_CHECKPOINT_PATH",
    )
    orders_db_path: Path = Field(
        default=PROJECT_ROOT / "storage" / "orders.sqlite",
        alias="ORDERS_DB_PATH",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0", alias="REDIS_URL"
    )

    # --- Orquestación Multi-Agente (Supervisor) ---
    max_validation_retries: int = Field(
        default=2, ge=0, le=5, alias="MAX_VALIDATION_RETRIES"
    )
    prompt_version: str = Field(
        default="intelligence-system-v1.0", alias="PROMPT_VERSION"
    )

    # --- API FastAPI ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, ge=1, le=65535, alias="API_PORT")
    api_bearer_token: str = Field(default="", alias="API_BEARER_TOKEN")

    # --- Observabilidad (Arize Phoenix / OpenTelemetry & LangSmith) ---
    phoenix_enabled: bool = Field(default=True, alias="PHOENIX_ENABLED")
    phoenix_collector_endpoint: str = Field(
        default="http://127.0.0.1:6006/v1/traces",
        alias="PHOENIX_COLLECTOR_ENDPOINT",
    )
    phoenix_project_name: str = Field(
        default="the-intelligence-system", alias="PHOENIX_PROJECT_NAME"
    )
    price_input_usd_per_million: float = Field(
        default=0.15, ge=0.0, alias="PRICE_INPUT_USD_PER_MILLION"
    )
    price_output_usd_per_million: float = Field(
        default=0.60, ge=0.0, alias="PRICE_OUTPUT_USD_PER_MILLION"
    )
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_api_key: SecretStr = Field(default=SecretStr(""), alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="the-intelligence-system", alias="LANGSMITH_PROJECT"
    )

    @property
    def has_valid_openai_key(self) -> bool:
        raw = self.openai_api_key.get_secret_value().strip()
        return bool(raw and raw != "pon_aqui_tu_api_key" and not raw.startswith("tu_"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Devuelve la instancia singleton de Settings."""
    return Settings()
