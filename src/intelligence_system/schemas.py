"""Contratos de datos Pydantic v2 para herramientas de agentes, estado y API FastAPI.

Garantiza validación estricta de todas las entradas y salidas del sistema.
"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


# ==============================================================================
# 1. Contratos de Herramientas y Agentes Especialistas
# ==============================================================================


class TriageInput(BaseModel):
    """Entrada validada para la herramienta de análisis de sentimiento y clasificación."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=2,
        max_length=8000,
        description="Mensaje o consulta del usuario tratado como dato no confiable.",
    )


class TriageAnalysis(BaseModel):
    """Salida estructurada del Agente Especialista de Sentimiento y Triage."""

    model_config = ConfigDict(extra="forbid")

    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Polaridad emocional detectada en el mensaje del cliente."
    )
    issue_type: Literal[
        "return_policy",
        "refund_status",
        "order_inquiry",
        "delivery_delay",
        "technical_issue",
        "security_privacy",
        "general_policy",
        "out_of_scope",
    ] = Field(description="Categoría principal del caso.")
    urgency: Literal["low", "medium", "high", "critical"] = Field(
        description="Nivel de urgencia operativa según SLA-003."
    )
    order_id: str | None = Field(
        default=None,
        description="Identificador del pedido detectado (ej. ORD-1001) o None.",
    )
    needs_order_lookup: bool = Field(
        default=False,
        description="True si el caso requiere consultar la base operativa de pedidos.",
    )
    needs_rag_lookup: bool = Field(
        default=True,
        description="True si requiere consultar políticas o procedimientos en ChromaDB.",
    )
    escalation_required: bool = Field(
        default=False,
        description="True si hay amenazas legales, fraude, riesgo crítico o solicitud fuera de límites.",
    )
    escalation_reason: str = Field(
        default="",
        description="Motivo detallado cuando escalation_required es True.",
    )
    confidence: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nivel de confianza del análisis entre 0.0 y 1.0.",
    )


class OrderLookupInput(BaseModel):
    """Entrada validada para la herramienta de consulta asíncrona de pedidos."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        min_length=3,
        max_length=32,
        pattern=r"^[A-Za-z0-9_-]+$",
        description="Identificador del pedido (ej. ORD-1001).",
    )
    customer_id: str | None = Field(
        default=None,
        max_length=64,
        description="Identificador opcional del cliente para verificación cruzada.",
    )

    @field_validator("order_id")
    @classmethod
    def normalize_order_id(cls, value: str) -> str:
        return value.strip().upper()


class OrderStatusResult(BaseModel):
    """Salida validada de la herramienta de consulta operativa de pedidos."""

    model_config = ConfigDict(extra="forbid")

    order_found: bool = Field(description="Indica si el pedido existe en la base autorizada.")
    order_id: str = Field(description="Identificador consultado.")
    customer_id: str | None = Field(default=None, description="ID del cliente asociado.")
    customer_email_masked: str | None = Field(
        default=None, description="Correo enmascarado por privacidad (SEC-004)."
    )
    product_name: str | None = Field(default=None, description="Producto asociado al pedido.")
    order_status: Literal[
        "preparing",
        "in_transit",
        "delayed",
        "delivered",
        "refund_pending",
        "refunded",
        "not_found",
    ] = Field(description="Estado oficial del pedido según SOP-002.")
    last_shipping_event: str = Field(
        default="", description="Último evento logístico registrado."
    )
    last_update: str = Field(
        default="", description="Marca de tiempo ISO-8601 de la última actualización."
    )
    refund_status: str = Field(
        default="none", description="Estado del reembolso si aplica."
    )
    source: str = Field(default="orders_sqlite_async", description="Fuente de datos verificada.")
    verified: bool = Field(default=False, description="True si los datos fueron confirmados.")


class RAGSearchInput(BaseModel):
    """Entrada validada para la herramienta de recuperación semántica RAG."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=4000,
        description="Consulta semántica para buscar en la base de conocimientos ChromaDB.",
    )
    category_filter: str | None = Field(
        default=None,
        description="Filtro opcional de metadatos por categoría (ej. devoluciones, reembolsos).",
    )
    top_k: int = Field(default=4, ge=1, le=10, description="Número de fragmentos a recuperar.")


class RetrievedDocument(BaseModel):
    """Fragmento documental recuperado del índice vectorial con metadatos."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(description="Identificador canónico del documento (ej. POL-001).")
    title: str = Field(description="Título oficial del documento.")
    category: str = Field(description="Categoría temática del documento.")
    page_content: str = Field(description="Contenido textual del fragmento recuperado.")
    chunk_index: int = Field(default=0, ge=0, description="Índice del fragmento.")


class RAGSearchOutput(BaseModel):
    """Salida validada de la herramienta de búsqueda vectorial RAG."""

    model_config = ConfigDict(extra="forbid")

    documents: list[RetrievedDocument] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    retrieval_ms: float = Field(default=0.0, ge=0.0)
    embedding_ms: float = Field(default=0.0, ge=0.0)
    vector_search_ms: float = Field(default=0.0, ge=0.0)


class CitationCheckResult(BaseModel):
    """Resultado de la validación determinística de citas bibliográficas."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    cited_ids: list[str] = Field(default_factory=list)
    invalid_ids: list[str] = Field(default_factory=list)
    missing_citation: bool = False
    is_abstention: bool = False
    issues: list[str] = Field(default_factory=list)


class GroundingReview(BaseModel):
    """Evaluación estructurada del Agente Crítico / Validador Semántico."""

    model_config = ConfigDict(extra="forbid")

    grounded: bool = Field(
        description="True si todas las afirmaciones factuales están respaldadas por el contexto o base de pedidos."
    )
    relevant: bool = Field(
        description="True si la respuesta atiende directamente la consulta del usuario."
    )
    policy_compliant: bool = Field(
        default=True,
        description="True si cumple SEC-004 (sin exponer PII) y SOP-002 (sin inventar estados).",
    )
    explanation: str = Field(
        description="Justificación concisa basada en las fuentes y evidencias."
    )
    unsupported_claims: list[str] = Field(
        default_factory=list,
        description="Lista de afirmaciones sin respaldo o citas inválidas a corregir.",
    )
    approved: bool = Field(
        default=False,
        description="Decisión final combinando revisión semántica y validación determinística.",
    )
    citation_check: CitationCheckResult | None = Field(default=None)


# ==============================================================================
# 2. Contratos de Entrada y Salida de la API Profesional (FastAPI)
# ==============================================================================


class ChatRequest(BaseModel):
    """Petición de conversación o resolución de caso hacia el sistema multi-agente."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        min_length=3,
        max_length=8000,
        description="Consulta o queja del cliente.",
        examples=[
            "Mi pedido ORD-1002 lleva días retrasado y nadie me responde. ¿Cuál es su estado y cuál es el SLA de atención? Cita las fuentes."
        ],
    )
    thread_id: str = Field(
        default="default-session",
        min_length=1,
        max_length=128,
        pattern=r"^[a-zA-Z0-9_.:-]+$",
        description="Identificador de sesión para persistir la conversación en el Checkpointer de LangGraph.",
    )
    customer_id: str | None = Field(
        default=None,
        max_length=64,
        description="Identificador opcional del cliente autenticado (ej. CUST-02).",
    )
    simulate_failure: bool = Field(
        default=False,
        description="Si es True, provoca un fallo controlado en recuperación para pruebas de observabilidad.",
    )


class ExecutionMetrics(BaseModel):
    """Métricas de latencia, uso de tokens y costo estimado por ejecución."""

    model_config = ConfigDict(extra="forbid")

    retrieval_ms: float = 0.0
    embedding_ms: float = 0.0
    vector_search_ms: float = 0.0
    order_lookup_ms: float = 0.0
    triage_ms: float = 0.0
    analyst_ms: float = 0.0
    reviewer_ms: float = 0.0
    total_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float | None = None


class ChatResponse(BaseModel):
    """Respuesta validada de la API tras ejecutar el grafo Supervisor Multi-Agente."""

    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(description="UUID único de la ejecución y traza OpenTelemetry.")
    thread_id: str = Field(description="Identificador del hilo persistido en el Checkpointer.")
    status: Literal["completed", "escalated_to_human", "error"] = Field(
        description="Estado final de la solicitud."
    )
    decision: Literal["publish", "escalate", "request_information"] = Field(
        description="Decisión tomada por el agente Supervisor."
    )
    answer: str = Field(description="Respuesta final validada o mensaje de derivación a soporte humano.")
    citations: list[str] = Field(
        default_factory=list,
        description="Lista de identificadores documentales citados y verificados.",
    )
    retrieved_sources: list[str] = Field(
        default_factory=list,
        description="IDs de documentos recuperados desde ChromaDB.",
    )
    triage: TriageAnalysis | None = Field(default=None)
    order_info: OrderStatusResult | None = Field(default=None)
    review: GroundingReview | None = Field(default=None)
    retry_count: int = Field(default=0, ge=0)
    supervisor_route_trace: list[str] = Field(
        default_factory=list,
        description="Secuencia de nodos especialistas coordinados por el Supervisor.",
    )
    metrics: ExecutionMetrics
    trace_project: str = Field(description="Nombre del proyecto en Arize Phoenix.")


class ThreadHistoryMessage(BaseModel):
    """Mensaje individual dentro del historial persistido en el Checkpointer."""

    model_config = ConfigDict(extra="forbid")

    role: str
    content: str


class ThreadStateResponse(BaseModel):
    """Estado recuperado desde el Checkpointer de LangGraph para un thread_id."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    exists: bool
    checkpoint_id: str | None = None
    messages_count: int = 0
    messages: list[ThreadHistoryMessage] = Field(default_factory=list)
    last_status: str | None = None
    last_decision: str | None = None
    last_answer: str | None = None
    is_escalated: bool = False
    retry_count: int = 0
    supervisor_route_trace: list[str] = Field(default_factory=list)


class HumanResolutionRequest(BaseModel):
    """Entrada validada para que un agente humano resuelva o apruebe un caso escalado."""

    model_config = ConfigDict(extra="forbid")

    approved: bool = Field(description="True para aprobar y publicar una respuesta supervisada.")
    supervisor_notes: str = Field(
        min_length=3,
        max_length=2000,
        description="Notas de auditoría o respuesta autorizada por el supervisor humano.",
    )


class HumanResolutionResponse(BaseModel):
    """Salida tras aplicar una resolución humana sobre un thread persistido."""

    model_config = ConfigDict(extra="forbid")

    thread_id: str
    status: Literal["resolved_by_human", "rejected_by_human"]
    decision: Literal["publish", "escalate"]
    final_answer: str
    supervisor_notes: str


class IngestRequest(BaseModel):
    """Petición para re-indexar documentos en la base vectorial ChromaDB."""

    model_config = ConfigDict(extra="forbid")

    force_rebuild: bool = Field(
        default=False,
        description="Si es True, recrea los fragmentos de la colección.",
    )


class IngestResponse(BaseModel):
    """Resultado de la ingesta asíncrona en ChromaDB."""

    model_config = ConfigDict(extra="forbid")

    collection_name: str
    documents_loaded: int
    chunks_indexed: int
    source_ids: list[str]
    duration_ms: float


class HealthResponse(BaseModel):
    """Respuesta del endpoint de salud del sistema."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    version: str
    checkpointer_backend: str
    vectorstore_chunks: int
    orders_count: int
    phoenix_enabled: bool
    phoenix_endpoint: str
    openai_configured: bool
