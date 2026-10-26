"""Herramientas asíncronas de los agentes especialistas validadas con Pydantic."""

from __future__ import annotations

import re
import time
from opentelemetry import trace

from intelligence_system.config import Settings
from intelligence_system.database import get_order_by_id_async
from intelligence_system.rag import AsyncRAGManager
from intelligence_system.schemas import (
    OrderLookupInput,
    OrderStatusResult,
    RAGSearchInput,
    RAGSearchOutput,
)

_ORDER_ID_PATTERN = re.compile(r"\b(ORD-\d{3,6})\b", re.IGNORECASE)


def extract_order_id_from_text(text: str) -> str | None:
    """Extrae un identificador de pedido con formato `ORD-XXXX` si aparece en el mensaje."""
    match = _ORDER_ID_PATTERN.search(text or "")
    return match.group(1).upper() if match else None


async def lookup_order_tool_async(
    order_id: str,
    customer_id: str | None = None,
    *,
    settings: Settings | None = None,
    tracer: trace.Tracer | None = None,
) -> tuple[OrderStatusResult, float]:
    """Herramienta asíncrona del Agente Especialista de Pedidos con validación Pydantic y span OTel."""
    validated_input = OrderLookupInput(order_id=order_id, customer_id=customer_id)
    active_tracer = tracer or trace.get_tracer("intelligence_system.tracer")
    started = time.perf_counter()

    with active_tracer.start_as_current_span("tool.order_lookup") as span:
        span.set_attribute("tool.name", "lookup_order_sqlite_async")
        span.set_attribute("order.id", validated_input.order_id)
        result = await get_order_by_id_async(validated_input, settings=settings)
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        span.set_attribute("order.found", result.order_found)
        span.set_attribute("order.status", result.order_status)
        span.set_attribute("tool.duration_ms", duration_ms)

    return result, duration_ms


async def search_rag_tool_async(
    rag_manager: AsyncRAGManager,
    query: str,
    *,
    category_filter: str | None = None,
    top_k: int = 4,
    simulate_failure: bool = False,
    tracer: trace.Tracer | None = None,
) -> RAGSearchOutput:
    """Herramienta asíncrona del Agente Especialista RAG con validación Pydantic de entrada y salida."""
    validated_input = RAGSearchInput(
        query=query,
        category_filter=category_filter,
        top_k=top_k,
    )
    return await rag_manager.search_async(
        validated_input,
        tracer=tracer,
        simulate_failure=simulate_failure,
    )
