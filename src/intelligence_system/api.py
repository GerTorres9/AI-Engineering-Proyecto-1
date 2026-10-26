"""API Profesional FastAPI (100% Asíncrona) para The Intelligence System.

Expone endpoints documentados con OpenAPI/Swagger, validación estricta con Pydantic v2,
persistencia de conversaciones mediante Checkpointer de LangGraph, streaming SSE y
trazabilidad completa en Arize Phoenix (OpenTelemetry).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import json
from secrets import compare_digest
import time
import uuid
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.responses import StreamingResponse
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from opentelemetry.trace import Status, StatusCode

from intelligence_system import __version__
from intelligence_system.checkpointer import create_checkpointer_context
from intelligence_system.config import Settings, get_settings
from intelligence_system.database import get_order_by_id_async, init_orders_db
from intelligence_system.graph import build_intelligence_graph
from intelligence_system.observability import (
    configure_observability,
    estimate_cost_usd,
)
from intelligence_system.rag import AsyncRAGManager
from intelligence_system.schemas import (
    ChatRequest,
    ChatResponse,
    ExecutionMetrics,
    GroundingReview,
    HealthResponse,
    HumanResolutionRequest,
    HumanResolutionResponse,
    IngestRequest,
    IngestResponse,
    OrderLookupInput,
    OrderStatusResult,
    ThreadHistoryMessage,
    ThreadStateResponse,
    TriageAnalysis,
)

optional_bearer = HTTPBearer(auto_error=False)


async def execute_intelligence_request(
    graph_app: Any,
    tracer: Any,
    settings: Settings,
    payload: ChatRequest,
) -> ChatResponse:
    """Ejecuta una solicitud de forma asíncrona en el grafo Supervisor con trazabilidad OTel."""
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    config = {"configurable": {"thread_id": payload.thread_id}}

    initial_turn_state = {
        "messages": [HumanMessage(content=payload.query)],
        "request_id": request_id,
        "thread_id": payload.thread_id,
        "customer_id": payload.customer_id,
        "query": payload.query,
        "simulate_failure": payload.simulate_failure,
        "triage": None,
        "order_info": None,
        "rag_completed": False,
        "retrieved_docs": [],
        "answer": "",
        "review": None,
        "retry_count": 0,
        "validation_error": "",
        "is_escalated": False,
        "decision": "publish",
        "status": "completed",
        "next_agent": "triage_specialist",
        "supervisor_route_trace": [],
        "token_events": [],
        "triage_ms": 0.0,
        "order_lookup_ms": 0.0,
        "retrieval_ms": 0.0,
        "embedding_ms": 0.0,
        "vector_search_ms": 0.0,
        "analyst_ms": 0.0,
        "reviewer_ms": 0.0,
    }

    with tracer.start_as_current_span("multi_agent.request") as root_span:
        root_span.set_attribute("app.request_id", request_id)
        root_span.set_attribute("app.thread_id", payload.thread_id)
        root_span.set_attribute("app.prompt_version", settings.prompt_version)
        root_span.set_attribute("app.model", settings.openai_model)
        root_span.set_attribute("app.simulate_failure", payload.simulate_failure)
        root_span.set_attribute("input.query", payload.query)
        root_span.set_attribute("input.query_length", len(payload.query))

        try:
            final_state = await graph_app.ainvoke(initial_turn_state, config=config)
        except Exception as exc:
            total_ms = round((time.perf_counter() - started) * 1000, 2)
            root_span.record_exception(exc)
            root_span.set_status(Status(StatusCode.ERROR, str(exc)))
            root_span.set_attribute("app.latency_ms", total_ms)
            raise

        total_ms = round((time.perf_counter() - started) * 1000, 2)
        token_events = final_state.get("token_events") or []
        input_tokens = sum(int(e.get("input_tokens", 0)) for e in token_events)
        output_tokens = sum(int(e.get("output_tokens", 0)) for e in token_events)
        total_tokens = sum(int(e.get("total_tokens", 0)) for e in token_events)
        estimated_cost = estimate_cost_usd(token_events, settings=settings)

        review_dict = final_state.get("review")
        triage_dict = final_state.get("triage")
        order_dict = final_state.get("order_info")
        docs = final_state.get("retrieved_docs") or []

        review_model = GroundingReview.model_validate(review_dict) if review_dict else None
        triage_model = TriageAnalysis.model_validate(triage_dict) if triage_dict else None
        order_model = OrderStatusResult.model_validate(order_dict) if order_dict else None

        cited_ids = (
            review_model.citation_check.cited_ids
            if (review_model and review_model.citation_check)
            else []
        )
        retrieved_sources = list(dict.fromkeys(str(d["source_id"]) for d in docs))

        root_span.set_attribute(
            "eval.grounded", bool(review_model.grounded) if review_model else False
        )
        root_span.set_attribute(
            "eval.relevant", bool(review_model.relevant) if review_model else False
        )
        root_span.set_attribute(
            "eval.citation_valid",
            bool(review_model.citation_check.valid)
            if (review_model and review_model.citation_check)
            else False,
        )
        root_span.set_attribute("app.retry_count", int(final_state.get("retry_count", 0)))
        root_span.set_attribute("app.decision", str(final_state.get("decision", "publish")))
        root_span.set_attribute("llm.input_tokens", input_tokens)
        root_span.set_attribute("llm.output_tokens", output_tokens)
        root_span.set_attribute("llm.total_tokens", total_tokens)
        root_span.set_attribute("app.latency_ms", total_ms)

        metrics = ExecutionMetrics(
            retrieval_ms=round(float(final_state.get("retrieval_ms", 0.0)), 2),
            embedding_ms=round(float(final_state.get("embedding_ms", 0.0)), 2),
            vector_search_ms=round(float(final_state.get("vector_search_ms", 0.0)), 2),
            order_lookup_ms=round(float(final_state.get("order_lookup_ms", 0.0)), 2),
            triage_ms=round(float(final_state.get("triage_ms", 0.0)), 2),
            analyst_ms=round(float(final_state.get("analyst_ms", 0.0)), 2),
            reviewer_ms=round(float(final_state.get("reviewer_ms", 0.0)), 2),
            total_latency_ms=total_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        return ChatResponse(
            request_id=request_id,
            thread_id=payload.thread_id,
            status=final_state.get("status", "completed"),
            decision=final_state.get("decision", "publish"),
            answer=final_state.get("answer", ""),
            citations=cited_ids,
            retrieved_sources=retrieved_sources,
            triage=triage_model,
            order_info=order_model,
            review=review_model,
            retry_count=int(final_state.get("retry_count", 0)),
            supervisor_route_trace=list(final_state.get("supervisor_route_trace") or []),
            metrics=metrics,
            trace_project=settings.phoenix_project_name,
        )


def create_app(
    settings: Settings | None = None,
    *,
    llm: BaseChatModel | None = None,
    rag_manager: AsyncRAGManager | None = None,
) -> FastAPI:
    """Fábrica de la aplicación FastAPI con ciclo de vida asíncrono (lifespan)."""
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.settings = cfg
        app.state.tracer = configure_observability(cfg)
        app.state.orders_count = await init_orders_db(cfg)
        app.state.rag_manager = rag_manager or AsyncRAGManager(settings=cfg)
        await app.state.rag_manager.ingest_async(force_rebuild=False)

        async with create_checkpointer_context(cfg) as checkpointer:
            app.state.checkpointer = checkpointer
            app.state.graph = build_intelligence_graph(
                settings=cfg,
                rag_manager=app.state.rag_manager,
                checkpointer=checkpointer,
                llm=llm,
                tracer=app.state.tracer,
            )
            yield

    app = FastAPI(
        title="The Intelligence System — Multi-Agent RAG & Supervisor API",
        description=(
            "Sistema de IA de grado de producción que integra RAG Avanzado (ChromaDB), "
            "Arquitectura Multi-Agente con Supervisor en LangGraph, Persistencia de Conversaciones "
            "(AsyncSqliteSaver / AsyncRedisCheckpointer), Validación con Pydantic v2 y "
            "Observabilidad OpenTelemetry en Arize Phoenix."
        ),
        version=__version__,
        lifespan=lifespan,
    )

    async def verify_token(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(optional_bearer),
    ) -> None:
        expected = request.app.state.settings.api_bearer_token.strip()
        if not expected:
            return
        if credentials is None or not compare_digest(credentials.credentials, expected):
            raise HTTPException(
                status_code=401, detail="Token Bearer inválido o ausente."
            )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Infraestructura"],
        summary="Estado de salud del sistema, base vectorial, checkpointer y observabilidad",
    )
    async def health_check(request: Request) -> HealthResponse:
        s: Settings = request.app.state.settings
        chunks = await request.app.state.rag_manager.count_chunks_async()
        return HealthResponse(
            status="ok",
            version=__version__,
            checkpointer_backend=s.checkpointer_backend,
            vectorstore_chunks=chunks,
            orders_count=int(getattr(request.app.state, "orders_count", 0)),
            phoenix_enabled=s.phoenix_enabled,
            phoenix_endpoint=s.phoenix_collector_endpoint,
            openai_configured=s.has_valid_openai_key,
        )

    @app.post(
        "/api/v1/chat",
        response_model=ChatResponse,
        dependencies=[Depends(verify_token)],
        tags=["Agentes & RAG"],
        summary="Ejecuta el grafo multi-agente con Supervisor y persiste la conversación",
    )
    async def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
        try:
            return await execute_intelligence_request(
                graph_app=request.app.state.graph,
                tracer=request.app.state.tracer,
                settings=request.app.state.settings,
                payload=payload,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post(
        "/api/v1/chat/stream",
        dependencies=[Depends(verify_token)],
        tags=["Agentes & RAG"],
        summary="Streaming SSE en tiempo real de las decisiones del Supervisor y especialistas",
    )
    async def chat_stream_endpoint(
        payload: ChatRequest, request: Request
    ) -> StreamingResponse:
        graph_app = request.app.state.graph
        config = {"configurable": {"thread_id": payload.thread_id}}
        initial_state = {
            "messages": [HumanMessage(content=payload.query)],
            "request_id": str(uuid.uuid4()),
            "thread_id": payload.thread_id,
            "customer_id": payload.customer_id,
            "query": payload.query,
            "simulate_failure": payload.simulate_failure,
            "triage": None,
            "order_info": None,
            "rag_completed": False,
            "retrieved_docs": [],
            "answer": "",
            "review": None,
            "retry_count": 0,
            "validation_error": "",
            "is_escalated": False,
            "decision": "publish",
            "status": "completed",
            "next_agent": "triage_specialist",
            "supervisor_route_trace": [],
            "token_events": [],
        }

        async def event_generator() -> AsyncIterator[str]:
            async for event in graph_app.astream(
                initial_state, config=config, stream_mode="updates"
            ):
                for node_name, node_update in event.items():
                    safe_payload = {
                        "node": node_name,
                        "next_agent": node_update.get("next_agent"),
                        "status": node_update.get("status"),
                        "decision": node_update.get("decision"),
                        "answer_preview": (node_update.get("answer") or "")[:240],
                    }
                    yield f"data: {json.dumps(safe_payload, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(), media_type="text/event-stream"
        )

    @app.get(
        "/api/v1/threads/{thread_id}",
        response_model=ThreadStateResponse,
        dependencies=[Depends(verify_token)],
        tags=["Persistencia (Checkpointer)"],
        summary="Recupera el historial y estado persistido de una conversación en LangGraph",
    )
    async def get_thread_state(
        thread_id: str, request: Request
    ) -> ThreadStateResponse:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await request.app.state.graph.aget_state(config)
        values = snapshot.values if snapshot else {}
        if not values:
            return ThreadStateResponse(thread_id=thread_id, exists=False)

        raw_messages = values.get("messages") or []
        formatted_messages: list[ThreadHistoryMessage] = []
        for m in raw_messages:
            role = "user" if isinstance(m, HumanMessage) else "assistant"
            formatted_messages.append(
                ThreadHistoryMessage(role=role, content=str(m.content))
            )

        checkpoint_id = None
        if snapshot and snapshot.config:
            checkpoint_id = snapshot.config.get("configurable", {}).get("checkpoint_id")

        return ThreadStateResponse(
            thread_id=thread_id,
            exists=True,
            checkpoint_id=checkpoint_id,
            messages_count=len(formatted_messages),
            messages=formatted_messages,
            last_status=values.get("status"),
            last_decision=values.get("decision"),
            last_answer=values.get("answer"),
            is_escalated=bool(values.get("is_escalated", False)),
            retry_count=int(values.get("retry_count", 0)),
            supervisor_route_trace=list(values.get("supervisor_route_trace") or []),
        )

    @app.post(
        "/api/v1/threads/{thread_id}/resolve",
        response_model=HumanResolutionResponse,
        dependencies=[Depends(verify_token)],
        tags=["Persistencia (Checkpointer)"],
        summary="Permite a un supervisor humano resolver un caso escalado (HITL)",
    )
    async def resolve_escalated_thread(
        thread_id: str,
        payload: HumanResolutionRequest,
        request: Request,
    ) -> HumanResolutionResponse:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await request.app.state.graph.aget_state(config)
        if not snapshot or not snapshot.values:
            raise HTTPException(
                status_code=404,
                detail=f"El thread '{thread_id}' no existe en el Checkpointer.",
            )

        new_status = "resolved_by_human" if payload.approved else "rejected_by_human"
        new_decision = "publish" if payload.approved else "escalate"
        final_answer = (
            f"✅ [Resolución de Supervisor Humano]: {payload.supervisor_notes}"
            if payload.approved
            else f"🔒 [Caso retenido en cola especializada]: {payload.supervisor_notes}"
        )

        await request.app.state.graph.aupdate_state(
            config,
            {
                "answer": final_answer,
                "status": new_status,
                "decision": new_decision,
                "is_escalated": not payload.approved,
                "messages": [AIMessage(content=final_answer)],
            },
        )
        return HumanResolutionResponse(
            thread_id=thread_id,
            status=new_status,
            decision=new_decision,
            final_answer=final_answer,
            supervisor_notes=payload.supervisor_notes,
        )

    @app.post(
        "/api/v1/ingest",
        response_model=IngestResponse,
        dependencies=[Depends(verify_token)],
        tags=["RAG & Base de Conocimientos"],
        summary="Ejecuta la ingesta asíncrona de documentos Markdown hacia ChromaDB",
    )
    async def ingest_documents(
        payload: IngestRequest, request: Request
    ) -> IngestResponse:
        return await request.app.state.rag_manager.ingest_async(
            force_rebuild=payload.force_rebuild
        )

    @app.get(
        "/api/v1/orders/{order_id}",
        response_model=OrderStatusResult,
        dependencies=[Depends(verify_token)],
        tags=["Base Operativa de Pedidos"],
        summary="Consulta directa asíncrona del estado de un pedido con validación Pydantic",
    )
    async def get_order_endpoint(
        order_id: str, request: Request
    ) -> OrderStatusResult:
        validated = OrderLookupInput(order_id=order_id)
        return await get_order_by_id_async(
            validated, settings=request.app.state.settings
        )

    return app


app = create_app()
