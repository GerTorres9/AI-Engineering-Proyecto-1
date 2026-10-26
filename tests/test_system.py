"""Pruebas integrales asíncronas de RAG, Agentes con Supervisor, Checkpointer, Pydantic y FastAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError
import pytest

from intelligence_system.api import create_app
from intelligence_system.config import Settings
from intelligence_system.database import get_order_by_id_async, init_orders_db
from intelligence_system.rag import AsyncRAGManager, DeterministicKeywordEmbeddings
from intelligence_system.schemas import ChatRequest, OrderLookupInput
from intelligence_system.validation import validate_citations


class RepairCycleMockChatModel(BaseChatModel):
    """LLM asíncrono de prueba que simula un fallo de cita en el intento 0 y lo corrige en el reintento."""

    analyst_calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "repair-cycle-mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text = str(messages[0].content)
        if "Especialista de Sentimiento" in system_text:
            payload = (
                '{"sentiment": "neutral", "issue_type": "return_policy", '
                '"urgency": "medium", "order_id": null, "needs_order_lookup": false, '
                '"needs_rag_lookup": true, "escalation_required": false, '
                '"escalation_reason": "", "confidence": 0.95}'
            )
        elif "Crítico/Validador" in system_text:
            payload = (
                '{"grounded": true, "relevant": true, "policy_compliant": true, '
                '"explanation": "Evaluado por mock juez.", "unsupported_claims": []}'
            )
        else:
            self.analyst_calls += 1
            if self.analyst_calls == 1:
                # Primer borrador sin cita para disparar el ciclo de reparación del Supervisor -> Analista
                payload = "Tienes 30 días naturales para devolver el producto con su empaque original."
            else:
                # Segundo borrador corregido con cita válida
                payload = (
                    "Tienes 30 días naturales para solicitar una devolución con su empaque original "
                    "[source: POL-001]."
                )
        msg = AIMessage(
            content=payload,
            usage_metadata={"input_tokens": 120, "output_tokens": 45, "total_tokens": 165},
        )
        return ChatResult(generations=[ChatGeneration(message=msg)])


def _make_test_settings(tmp_path: Path) -> Settings:
    return Settings(
        OPENAI_API_KEY="",
        CHROMA_PERSIST_DIR=tmp_path / "chroma",
        SQLITE_CHECKPOINT_PATH=tmp_path / "checkpoints.sqlite",
        ORDERS_DB_PATH=tmp_path / "orders.sqlite",
        PHOENIX_ENABLED=False,
        MAX_VALIDATION_RETRIES=2,
    )


def test_citation_validation_rules() -> None:
    valid = validate_citations(
        "El reembolso tarda de 5 a 7 días hábiles [source: RET-005].",
        ["RET-005", "POL-001"],
    )
    assert valid.valid is True
    assert valid.cited_ids == ["RET-005"]

    invalid = validate_citations(
        "El reembolso es inmediato [source: FAKE-999].",
        ["RET-005"],
    )
    assert invalid.valid is False
    assert invalid.invalid_ids == ["FAKE-999"]

    missing = validate_citations("El reembolso tarda 5 días hábiles.", ["RET-005"])
    assert missing.valid is False
    assert missing.missing_citation is True

    abstention = validate_citations(
        "No encontré suficiente información en los documentos autorizados.", []
    )
    assert abstention.valid is True
    assert abstention.is_abstention is True


def test_pydantic_strict_validation() -> None:
    with pytest.raises(ValidationError):
        OrderLookupInput(order_id="ORD-1001; DROP TABLE orders;")

    with pytest.raises(ValidationError):
        ChatRequest(query="Hola", campo_no_permitido="hack")  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_async_orders_database_and_pii_masking(tmp_path: Path) -> None:
    settings = _make_test_settings(tmp_path)
    seeded = await init_orders_db(settings)
    assert seeded >= 5

    found = await get_order_by_id_async(
        OrderLookupInput(order_id="ord-1003"), settings=settings
    )
    assert found.order_found is True
    assert found.order_id == "ORD-1003"
    assert found.order_status == "refund_pending"
    assert found.customer_email_masked is not None
    assert "***@" in found.customer_email_masked

    missing = await get_order_by_id_async(
        OrderLookupInput(order_id="ORD-9999"), settings=settings
    )
    assert missing.order_found is False
    assert missing.order_status == "not_found"


@pytest.mark.asyncio
async def test_supervisor_repair_cycle_with_mock_llm(tmp_path: Path) -> None:
    """Verifica que el Supervisor reenvíe al Analista cuando el Crítico detecta falta de citas."""
    settings = _make_test_settings(tmp_path)
    mock_llm = RepairCycleMockChatModel()
    rag = AsyncRAGManager(settings=settings, embeddings=DeterministicKeywordEmbeddings())
    app = create_app(settings=settings, llm=mock_llm, rag_manager=rag)

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": "¿Cuál es el plazo y requisitos de la política de devoluciones?",
                    "thread_id": "thread-repair-test",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"
            assert data["decision"] == "publish"
            assert data["retry_count"] == 1
            assert "POL-001" in data["citations"]
            assert mock_llm.analyst_calls == 2


@pytest.mark.asyncio
async def test_checkpointer_persistence_across_server_restarts(tmp_path: Path) -> None:
    """Verifica que las conversaciones sobrevivan al apagar y volver a levantar el servidor FastAPI."""
    settings = _make_test_settings(tmp_path)
    thread_id = "persistent-customer-session-42"

    # --- Instancia 1 del servidor (antes del reinicio) ---
    app_instance_1 = create_app(settings=settings)
    async with app_instance_1.router.lifespan_context(app_instance_1):
        async with AsyncClient(
            transport=ASGITransport(app=app_instance_1), base_url="http://test"
        ) as client1:
            r1 = await client1.post(
                "/api/v1/chat",
                json={
                    "query": "Mi pedido ORD-1002 está retrasado. ¿Cuál es su estado y el SLA de atención?",
                    "thread_id": thread_id,
                    "customer_id": "CUST-02",
                },
            )
            assert r1.status_code == 200
            body1 = r1.json()
            assert body1["order_info"]["order_id"] == "ORD-1002"
            assert body1["order_info"]["order_status"] == "delayed"

    # --- Instancia 2 del servidor (simulando reinicio completo del proceso) ---
    app_instance_2 = create_app(settings=settings)
    async with app_instance_2.router.lifespan_context(app_instance_2):
        async with AsyncClient(
            transport=ASGITransport(app=app_instance_2), base_url="http://test"
        ) as client2:
            # 1. Consultar estado persistido tras reinicio
            state_resp = await client2.get(f"/api/v1/threads/{thread_id}")
            assert state_resp.status_code == 200
            state_data = state_resp.json()
            assert state_data["exists"] is True
            assert state_data["messages_count"] == 2
            assert state_data["checkpoint_id"] is not None

            # 2. Enviar segundo turno en la misma conversación
            r2 = await client2.post(
                "/api/v1/chat",
                json={
                    "query": "Gracias. Además, ¿cada cuánto hacen respaldos incrementales del sistema?",
                    "thread_id": thread_id,
                },
            )
            assert r2.status_code == 200

            # 3. Confirmar que el Checkpointer ahora acumula los 4 mensajes (2 turnos completos)
            state_resp_2 = await client2.get(f"/api/v1/threads/{thread_id}")
            assert state_resp_2.json()["messages_count"] == 4


@pytest.mark.asyncio
async def test_human_escalation_and_resolution_flow(tmp_path: Path) -> None:
    """Verifica que un caso crítico (PROFECO / doble cobro) se escale a soporte humano y pueda resolverse."""
    settings = _make_test_settings(tmp_path)
    app = create_app(settings=settings)
    thread_id = "escalation-thread-99"

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post(
                "/api/v1/chat",
                json={
                    "query": "Me hicieron doble cobro en ORD-1005 y pondré una demanda en PROFECO.",
                    "thread_id": thread_id,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "escalated_to_human"
            assert data["decision"] == "escalate"
            assert data["triage"]["escalation_required"] is True

            # Resolver desde el endpoint de supervisor humano
            resolve_resp = await client.post(
                f"/api/v1/threads/{thread_id}/resolve",
                json={
                    "approved": True,
                    "supervisor_notes": "Se verificó y liberó el cargo retenido de ORD-1005; reembolso confirmado.",
                },
            )
            assert resolve_resp.status_code == 200
            resolved_data = resolve_resp.json()
            assert resolved_data["status"] == "resolved_by_human"
            assert resolved_data["decision"] == "publish"
