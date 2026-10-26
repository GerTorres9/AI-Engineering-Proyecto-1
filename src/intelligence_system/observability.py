"""Instrumentación de observabilidad con OpenTelemetry, OpenInference y Arize Phoenix.

Soporta trazas jerárquicas OTLP/HTTP hacia Arize Phoenix y trazado opcional en LangSmith,
además de extracción de métricas de tokens y cálculo de costos.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from openinference.instrumentation.langchain import LangChainInstrumentor

from intelligence_system.config import Settings, get_settings

logger = logging.getLogger(__name__)
_INSTRUMENTED = False


def configure_observability(settings: Settings | None = None) -> trace.Tracer:
    """Configura el TracerProvider de OpenTelemetry hacia Arize Phoenix y LangSmith opcional."""
    global _INSTRUMENTED
    cfg = settings or get_settings()

    if cfg.langsmith_tracing and cfg.langsmith_api_key.get_secret_value().strip():
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = cfg.langsmith_api_key.get_secret_value().strip()
        os.environ["LANGSMITH_PROJECT"] = cfg.langsmith_project

    os.environ["PHOENIX_PROJECT_NAME"] = cfg.phoenix_project_name

    current_provider = trace.get_tracer_provider()
    if cfg.phoenix_enabled and not isinstance(current_provider, TracerProvider):
        resource = Resource.create(
            {
                "service.name": cfg.phoenix_project_name,
                "openinference.project.name": cfg.phoenix_project_name,
                "service.version": cfg.prompt_version,
            }
        )
        provider = TracerProvider(resource=resource)
        try:
            exporter = OTLPSpanExporter(
                endpoint=cfg.phoenix_collector_endpoint,
                timeout=3,
            )
            provider.add_span_processor(SimpleSpanProcessor(exporter))
        except Exception as exc:
            logger.warning("No se pudo inicializar el exportador OTLP a Phoenix: %s", exc)

        trace.set_tracer_provider(provider)
        if not _INSTRUMENTED:
            try:
                LangChainInstrumentor().instrument(tracer_provider=provider)
                _INSTRUMENTED = True
            except Exception as exc:
                logger.warning("Instrumentación LangChain ya activa o no disponible: %s", exc)

    return trace.get_tracer("intelligence_system.tracer")


def extract_token_usage(message: Any) -> dict[str, int]:
    """Extrae contadores de tokens reales desde un AIMessage de LangChain."""
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    legacy = response_metadata.get("token_usage") or response_metadata.get("usage") or {}

    input_tokens = usage.get(
        "input_tokens",
        legacy.get("prompt_tokens", legacy.get("input_tokens", 0)),
    )
    output_tokens = usage.get(
        "output_tokens",
        legacy.get("completion_tokens", legacy.get("output_tokens", 0)),
    )
    total_tokens = usage.get(
        "total_tokens",
        legacy.get("total_tokens", (input_tokens or 0) + (output_tokens or 0)),
    )
    return {
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "total_tokens": int(total_tokens or 0),
    }


def estimate_cost_usd(
    token_events: list[dict[str, Any]], settings: Settings | None = None
) -> float | None:
    """Estima el costo en USD a partir de los eventos de tokens y tarifas configuradas."""
    cfg = settings or get_settings()
    if cfg.price_input_usd_per_million <= 0 and cfg.price_output_usd_per_million <= 0:
        return None
    inputs = sum(int(e.get("input_tokens", 0)) for e in token_events)
    outputs = sum(int(e.get("output_tokens", 0)) for e in token_events)
    cost = (
        inputs * cfg.price_input_usd_per_million
        + outputs * cfg.price_output_usd_per_million
    ) / 1_000_000
    return round(cost, 8)
