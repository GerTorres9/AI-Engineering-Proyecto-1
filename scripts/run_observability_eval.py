"""Script de evaluación y observabilidad: ejecuta 5+ pruebas integrales en The Intelligence System.

Registra trazas completas OpenTelemetry/OpenInference en Arize Phoenix (http://localhost:6006),
verifica la persistencia de conversaciones en el Checkpointer de LangGraph y exporta métricas
reales de latencia, tokens, grounding y costos en `results/summary.csv` y `results/runs.jsonl`.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from opentelemetry import trace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from intelligence_system.api import execute_intelligence_request  # noqa: E402
from intelligence_system.checkpointer import create_checkpointer_context  # noqa: E402
from intelligence_system.config import get_settings  # noqa: E402
from intelligence_system.database import init_orders_db  # noqa: E402
from intelligence_system.graph import build_intelligence_graph  # noqa: E402
from intelligence_system.observability import configure_observability  # noqa: E402
from intelligence_system.rag import AsyncRAGManager  # noqa: E402
from intelligence_system.schemas import ChatRequest  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FIVE_PRODUCTION_SCENARIOS: list[dict[str, object]] = [
    {
        "name": "Prueba 1 — Política de Devoluciones y Excepciones (RAG)",
        "thread_id": "eval-thread-01-devoluciones",
        "customer_id": "CUST-01",
        "query": (
            "Explica los requisitos para devolver un producto, el plazo para solicitarlo "
            "y las excepciones aplicables. Cita las fuentes."
        ),
    },
    {
        "name": "Prueba 2 — Pedido con Reembolso y Plazos Bancarios (RAG + Base de Pedidos ORD-1003)",
        "thread_id": "eval-thread-02-reembolso",
        "customer_id": "CUST-03",
        "query": (
            "Mi pedido ORD-1003 ya tiene devolución aprobada. ¿Cuál es el estado actual del pedido, "
            "cuánto tarda en procesarse el reembolso y por qué el banco podría reflejarlo después? "
            "Cita las fuentes."
        ),
    },
    {
        "name": "Prueba 3 — Pedido Retrasado y SLA de Atención (RAG + Base de Pedidos ORD-1002)",
        "thread_id": "eval-thread-03-multiturno",
        "customer_id": "CUST-02",
        "query": (
            "Mi pedido ORD-1002 aparece retrasado. ¿Cuál es su último evento de envío verificado, "
            "qué estados permite comunicar el sistema y cuál es el SLA de respuesta? Cita las fuentes."
        ),
    },
    {
        "name": "Prueba 4 — Continuación Multi-Turno con Checkpointer + Incidencia Técnica (RAG + Memoria)",
        "thread_id": "eval-thread-03-multiturno",
        "customer_id": "CUST-02",
        "query": (
            "Siguiendo con mi pedido anterior en esta conversación: si al recibir la laptop presenta una "
            "falla técnica, ¿qué datos debo reportar para el diagnóstico y cada cuánto se hacen los "
            "respaldos incrementales del sistema? Cita las fuentes."
        ),
    },
    {
        "name": "Prueba 5 — Caso Crítico de Doble Cobro y Escalamiento Humano (Supervisor -> SLA-003)",
        "thread_id": "eval-thread-05-escalamiento",
        "customer_id": "CUST-05",
        "query": (
            "Me aplicaron un doble cobro en mi pedido ORD-1005 y nadie me resuelve; presentaré una "
            "queja formal ante PROFECO hoy mismo por cargo duplicado. ¿Cuál es el estado del pedido "
            "y cómo procede el escalamiento? Cita las fuentes."
        ),
    },
]


def persist_eval_records(records: list[dict[str, object]]) -> None:
    """Guarda los resultados de las ejecuciones en JSONL y CSV."""
    jsonl_path = RESULTS_DIR / "runs.jsonl"
    csv_path = RESULTS_DIR / "summary.csv"

    with jsonl_path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if records:
        fields = list(records[0].keys())
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with csv_path.open("a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(records)


async def run_evaluation_suite(args: argparse.Namespace) -> int:
    settings = get_settings()
    tracer = configure_observability(settings)
    await init_orders_db(settings)
    rag_manager = AsyncRAGManager(settings=settings)
    ingest_stats = await rag_manager.ingest_async(force_rebuild=False)

    print("=" * 84)
    print("🚀 THE INTELLIGENCE SYSTEM — SUITE DE OBSERVABILIDAD Y TRAZABILIDAD (ARIZE PHOENIX)")
    print("=" * 84)
    print(f"• Modelo LLM configurado      : {settings.openai_model}")
    print(f"• OpenAI API Key activa       : {'SÍ (Modo LLM Real)' if settings.has_valid_openai_key else 'NO (Modo Determinístico Local)'}")
    print(f"• Checkpointer Backend        : {settings.checkpointer_backend} ({settings.sqlite_checkpoint_path})")
    print(f"• Fragmentos en ChromaDB      : {ingest_stats.chunks_indexed} ({', '.join(ingest_stats.source_ids)})")
    print(f"• Endpoint Arize Phoenix      : {settings.phoenix_collector_endpoint}")
    print(f"• Proyecto Phoenix            : {settings.phoenix_project_name}")
    print("=" * 84)

    if args.simulate_failure:
        scenarios = [
            {
                "name": "Prueba de Fallo Controlado en Vector Search",
                "thread_id": "eval-thread-fail",
                "customer_id": "CUST-01",
                "query": "[FAIL_RETRIEVAL] Simular caída controlada de recuperación vectorial.",
                "simulate_failure": True,
            }
        ]
    elif args.out_of_scope:
        scenarios = [
            {
                "name": "Prueba Fuera de Dominio (Abstención Segura)",
                "thread_id": "eval-thread-out-of-scope",
                "customer_id": "CUST-01",
                "query": "¿Cuál es el número de cuenta bancaria interna del comercio y su clave interbancaria?",
            }
        ]
    elif args.query:
        scenarios = [
            {
                "name": "Consulta Personalizada",
                "thread_id": args.thread_id,
                "customer_id": "CUST-01",
                "query": args.query,
            }
        ]
    else:
        scenarios = FIVE_PRODUCTION_SCENARIOS

    records: list[dict[str, object]] = []

    async with create_checkpointer_context(settings) as checkpointer:
        graph_app = build_intelligence_graph(
            settings=settings,
            rag_manager=rag_manager,
            checkpointer=checkpointer,
            tracer=tracer,
        )

        for idx, item in enumerate(scenarios, start=1):
            name = str(item["name"])
            thread_id = str(item["thread_id"])
            query = str(item["query"])
            simulate_failure = bool(item.get("simulate_failure", False))

            print(f"\n[{idx}/{len(scenarios)}] {name}")
            print(f"  Thread ID : {thread_id}")
            print(f"  Consulta  : {query}")

            req = ChatRequest(
                query=query,
                thread_id=thread_id,
                customer_id=str(item.get("customer_id") or "CUST-01"),
                simulate_failure=simulate_failure,
            )

            try:
                resp = await execute_intelligence_request(
                    graph_app=graph_app,
                    tracer=tracer,
                    settings=settings,
                    payload=req,
                )
                record = {
                    "request_id": resp.request_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "scenario": name,
                    "thread_id": resp.thread_id,
                    "query": query,
                    "model": settings.openai_model,
                    "prompt_version": settings.prompt_version,
                    "status": resp.status,
                    "decision": resp.decision,
                    "sentiment": resp.triage.sentiment if resp.triage else None,
                    "issue_type": resp.triage.issue_type if resp.triage else None,
                    "urgency": resp.triage.urgency if resp.triage else None,
                    "order_id": resp.order_info.order_id if resp.order_info else None,
                    "order_status": resp.order_info.order_status if resp.order_info else None,
                    "grounded": resp.review.grounded if resp.review else None,
                    "relevant": resp.review.relevant if resp.review else None,
                    "citations": ",".join(resp.citations),
                    "retrieved_sources": ",".join(resp.retrieved_sources),
                    "retry_count": resp.retry_count,
                    "supervisor_route": " -> ".join(resp.supervisor_route_trace),
                    "retrieval_ms": resp.metrics.retrieval_ms,
                    "order_lookup_ms": resp.metrics.order_lookup_ms,
                    "analyst_ms": resp.metrics.analyst_ms,
                    "reviewer_ms": resp.metrics.reviewer_ms,
                    "latency_ms": resp.metrics.total_latency_ms,
                    "input_tokens": resp.metrics.input_tokens,
                    "output_tokens": resp.metrics.output_tokens,
                    "total_tokens": resp.metrics.total_tokens,
                    "estimated_cost_usd": resp.metrics.estimated_cost_usd,
                    "answer": resp.answer,
                    "error": "",
                }
                records.append(record)
                print(f"  Ruta Sup. : {record['supervisor_route']}")
                print(
                    f"  Resultado : status={resp.status} | decision={resp.decision} | "
                    f"grounded={record['grounded']} | citations={resp.citations} | "
                    f"latency={resp.metrics.total_latency_ms}ms | tokens={resp.metrics.total_tokens}"
                )
                print(f"  Respuesta :\n    {resp.answer[:320]}...")
            except Exception as exc:
                record = {
                    "request_id": "error-span",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "scenario": name,
                    "thread_id": thread_id,
                    "query": query,
                    "model": settings.openai_model,
                    "prompt_version": settings.prompt_version,
                    "status": "error",
                    "decision": "escalate",
                    "sentiment": None,
                    "issue_type": None,
                    "urgency": None,
                    "order_id": None,
                    "order_status": None,
                    "grounded": False,
                    "relevant": False,
                    "citations": "",
                    "retrieved_sources": "",
                    "retry_count": 0,
                    "supervisor_route": "error",
                    "retrieval_ms": 0.0,
                    "order_lookup_ms": 0.0,
                    "analyst_ms": 0.0,
                    "reviewer_ms": 0.0,
                    "latency_ms": 0.0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                    "estimated_cost_usd": 0.0,
                    "answer": "",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                records.append(record)
                print(f"  [ERROR CONTROLADO REGISTRADO EN TRAZA]: {record['error']}")

        # Verificación explícita de persistencia en Checkpointer para el hilo multi-turno
        multi_turn_snapshot = await graph_app.aget_state(
            {"configurable": {"thread_id": "eval-thread-03-multiturno"}}
        )
        if multi_turn_snapshot and multi_turn_snapshot.values:
            persisted_msgs = multi_turn_snapshot.values.get("messages") or []
            print(
                f"\n✅ Persistencia verificada en Checkpointer ('eval-thread-03-multiturno'): "
                f"{len(persisted_msgs)} mensajes almacenados en disco."
            )

    provider = trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()

    persist_eval_records(records)
    print(f"\n📊 Resultados guardados en:\n  - {RESULTS_DIR / 'summary.csv'}\n  - {RESULTS_DIR / 'runs.jsonl'}")
    print(f"🔭 Abre Arize Phoenix en http://localhost:6006 y selecciona el proyecto '{settings.phoenix_project_name}'.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ejecuta las 5 pruebas de observabilidad de The Intelligence System"
    )
    parser.add_argument("--query", help="Ejecuta una consulta personalizada")
    parser.add_argument(
        "--thread-id",
        default="custom-thread-01",
        help="Identificador de hilo para persistencia en Checkpointer",
    )
    parser.add_argument(
        "--out-of-scope",
        action="store_true",
        help="Prueba una pregunta fuera del alcance del corpus para validar abstención",
    )
    parser.add_argument(
        "--simulate-failure",
        action="store_true",
        help="Provoca un fallo controlado en el span tool.vector_search",
    )
    args = parser.parse_args()
    return asyncio.run(run_evaluation_suite(args))


if __name__ == "__main__":
    raise SystemExit(main())
