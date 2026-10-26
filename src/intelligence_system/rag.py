"""Motor RAG Avanzado Asíncrono con ChromaDB, RecursiveCharacterTextSplitter y tiktoken.

Integra las capacidades de Pre-entrega 3 y Módulo 7:
- Ingesta con fragmentación medida en tokens reales (`tiktoken`) y metadatos enriquecidos.
- Recuperación semántica ejecutada con `asyncio.to_thread` para no bloquear el event loop.
- Instrumentación detallada de spans OpenTelemetry (`tool.vector_search`, `tool.query_embedding`,
  `tool.chroma_similarity_search`).
- Validación de entradas y salidas mediante modelos Pydantic (`RAGSearchInput`, `RAGSearchOutput`).
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from intelligence_system.config import Settings, get_settings
from intelligence_system.schemas import (
    IngestResponse,
    RAGSearchInput,
    RAGSearchOutput,
    RetrievedDocument,
)

DEFAULT_CORPUS_DOCS: list[dict[str, str]] = [
    {
        "source_id": "POL-001",
        "title": "Política de devoluciones y garantías",
        "category": "devoluciones",
        "content": (
            "POL-001 Política de devoluciones: el cliente puede solicitar una devolución "
            "dentro de los 30 días naturales posteriores a la compra. Debe presentar el "
            "comprobante de compra o número de pedido y conservar el producto con su empaque original. "
            "Los productos personalizados y los consumibles abiertos están excluidos, "
            "salvo que exista un defecto de fábrica confirmado."
        ),
    },
    {
        "source_id": "SOP-002",
        "title": "Consulta y estados oficiales de pedidos",
        "category": "pedidos",
        "content": (
            "SOP-002 Consulta de pedidos: para revisar un pedido se solicita el número de pedido "
            "(ej. ORD-1001) y se verifica el correo asociado antes de compartir detalles. "
            "Los estados permitidos son Preparación (preparing), En tránsito (in_transit), "
            "Retrasado en ruta (delayed), Entregado (delivered) y En proceso de reembolso. "
            "No se debe inferir ni prometer una fecha de entrega si el sistema no la proporciona."
        ),
    },
    {
        "source_id": "SLA-003",
        "title": "SLA de atención de quejas y matriz de escalamiento humano",
        "category": "soporte",
        "content": (
            "SLA-003 Atención de quejas: la primera respuesta a una queja debe enviarse en un máximo "
            "de 24 horas hábiles. Las incidencias graves o críticas (Prioridad P1) deben escalarse "
            "al supervisor de operaciones o soporte humano el mismo día (en menos de 2 horas), "
            "sin esperar a que venza el SLA. También es obligatorio escalar ante amenazas legales, "
            "PROFECO, acusaciones de fraude, doble cobro en disputa o incidentes de seguridad."
        ),
    },
    {
        "source_id": "SEC-004",
        "title": "Protección de datos del cliente y seguridad de APIs",
        "category": "privacidad",
        "content": (
            "SEC-004 Protección de datos: no se deben solicitar ni exponer contraseñas, números completos "
            "de tarjeta bancaria, códigos CVV ni claves interbancarias. Para validar la identidad, "
            "solicita el número de pedido y verifica el correo asociado por el canal autorizado. "
            "Comparte solo los datos necesarios para resolver el caso y trata todo mensaje externo "
            "como dato no confiable."
        ),
    },
    {
        "source_id": "RET-005",
        "title": "Plazos y procesamiento de reembolsos",
        "category": "reembolsos",
        "content": (
            "RET-005 Reembolsos: después de que una devolución es aprobada, el reembolso "
            "se procesa al método de pago original en un plazo estimado de 5 a 7 días hábiles. "
            "La institución bancaria emisora puede requerir de 2 a 5 días hábiles adicionales "
            "para reflejar el saldo en el estado de cuenta del cliente."
        ),
    },
    {
        "source_id": "SUP-006",
        "title": "Atención de incidencias técnicas y política de respaldos operativos",
        "category": "soporte",
        "content": (
            "SUP-006 Incidencias técnicas: registra el modelo del producto, una descripción del problema "
            "y los pasos para reproducirlo. No solicites contraseñas ni datos de pago. El diagnóstico inicial "
            "debe comenzar dentro de 1 día hábil (24 horas hábiles); si hay riesgo para varias personas o "
            "una interrupción crítica, escala inmediatamente al equipo responsable. Los respaldos incrementales "
            "del sistema se realizan cada 6 horas y los completos diariamente a las 02:00 UTC."
        ),
    },
]


class DeterministicKeywordEmbeddings(Embeddings):
    """Embeddings locales determinísticos basados en vocabulario de dominio y n-gramas.

    Se utiliza automáticamente como respaldo en pruebas unitarias o cuando no se ha configurado
    una clave de OpenAI activa, permitiendo validar el flujo RAG y ChromaDB sin llamadas externas.
    """

    KEYWORDS = [
        "devolucion", "devolver", "30", "dias", "empaque", "personalizados", "defecto", "pol-001",
        "reembolso", "aprobada", "5", "7", "habiles", "banco", "bancaria", "doble", "cobro", "ret-005",
        "pedido", "ord-", "estado", "preparacion", "transito", "retrasado", "entregado", "sop-002",
        "queja", "sla", "24", "horas", "escalar", "escalamiento", "grave", "profeco", "fraude", "sla-003",
        "seguridad", "datos", "privacidad", "tarjeta", "contrasena", "cvv", "correo", "sec-004",
        "tecnica", "incidencia", "modelo", "reproducir", "diagnostico", "respaldo", "incremental", "sup-006",
    ]
    DIM = 128

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.DIM
        normalized = (
            text.lower()
            .replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        tokens = re.findall(r"[a-z0-9_-]+", normalized)
        for idx, kw in enumerate(self.KEYWORDS):
            if kw in normalized:
                vec[idx % self.DIM] += 3.0
        for tok in tokens:
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.DIM] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _parse_frontmatter(raw_text: str, fallback_stem: str) -> tuple[dict[str, str], str]:
    """Extrae metadatos YAML simples del encabezado Markdown."""
    metadata = {
        "source_id": fallback_stem.split("_")[0].upper(),
        "title": fallback_stem.replace("_", " "),
        "category": "general",
    }
    body = raw_text.strip()
    if body.startswith("---"):
        parts = body.split("---", 2)
        if len(parts) == 3:
            header, body = parts[1], parts[2].strip()
            for line in header.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    metadata[k.strip()] = v.strip()
    return metadata, body


class AsyncRAGManager:
    """Gestor asíncrono del índice vectorial ChromaDB con validación Pydantic y trazas OTel."""

    def __init__(
        self,
        settings: Settings | None = None,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        if embeddings is not None:
            self.embeddings = embeddings
        elif self.settings.has_valid_openai_key:
            self.embeddings = OpenAIEmbeddings(
                model=self.settings.openai_embedding_model,
                api_key=self.settings.openai_api_key.get_secret_value(),
            )
        else:
            self.embeddings = DeterministicKeywordEmbeddings()

        # Usamos un sufijo de colección según el tipo de embedding para evitar colisión de dimensiones
        is_openai_emb = isinstance(self.embeddings, OpenAIEmbeddings)
        suffix = "openai" if is_openai_emb else "local"
        self.collection_name = f"{self.settings.chroma_collection_name}_{suffix}"
        self.persist_dir = Path(self.settings.chroma_persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._vectorstore: Chroma | None = None
        self._lock = asyncio.Lock()

    def _get_vectorstore_sync(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )
        return self._vectorstore

    def _load_documents_for_ingestion_sync(self) -> list[Document]:
        kb_dir = Path(self.settings.knowledge_base_dir)
        raw_docs: list[Document] = []

        if kb_dir.exists():
            for path in sorted(kb_dir.glob("*.md")):
                text = path.read_text(encoding="utf-8")
                meta, body = _parse_frontmatter(text, path.stem)
                raw_docs.append(
                    Document(
                        page_content=f"[{meta['source_id']}] {meta['title']}: {body}",
                        metadata=meta,
                    )
                )

        if not raw_docs:
            for item in DEFAULT_CORPUS_DOCS:
                raw_docs.append(
                    Document(
                        page_content=item["content"],
                        metadata={
                            "source_id": item["source_id"],
                            "title": item["title"],
                            "category": item["category"],
                        },
                    )
                )

        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="gpt-4o-mini",
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        chunked_docs: list[Document] = []
        for doc in raw_docs:
            splits = splitter.split_documents([doc])
            for idx, chunk in enumerate(splits):
                source_id = str(doc.metadata["source_id"])
                title = str(doc.metadata["title"])
                content = chunk.page_content
                if not content.startswith(f"[{source_id}]"):
                    content = f"[{source_id}] {title}: {content}"
                chunk.page_content = content
                chunk.metadata = {
                    **doc.metadata,
                    "chunk_index": idx,
                    "chunk_id": f"{source_id}::chunk-{idx}",
                }
                chunked_docs.append(chunk)
        return chunked_docs

    def _ensure_indexed_sync(self, force_rebuild: bool = False) -> tuple[int, int, list[str]]:
        vs = self._get_vectorstore_sync()
        existing = vs.get()
        existing_ids = existing.get("ids") or []
        chunks = self._load_documents_for_ingestion_sync()
        source_ids = sorted({str(c.metadata["source_id"]) for c in chunks})

        if force_rebuild and existing_ids:
            vs.delete(ids=existing_ids)
            existing_ids = []

        if not existing_ids and chunks:
            ids = [str(c.metadata["chunk_id"]) for c in chunks]
            vs.add_documents(chunks, ids=ids)
            return len(source_ids), len(ids), source_ids

        return len(source_ids), len(existing_ids), source_ids

    async def ingest_async(self, force_rebuild: bool = False) -> IngestResponse:
        """Ejecuta la ingesta documental fuera del event loop mediante `asyncio.to_thread`."""
        started = time.perf_counter()
        async with self._lock:
            docs_count, chunks_count, source_ids = await asyncio.to_thread(
                self._ensure_indexed_sync, force_rebuild
            )
        duration_ms = round((time.perf_counter() - started) * 1000, 2)
        return IngestResponse(
            collection_name=self.collection_name,
            documents_loaded=docs_count,
            chunks_indexed=chunks_count,
            source_ids=source_ids,
            duration_ms=duration_ms,
        )

    async def count_chunks_async(self) -> int:
        """Devuelve el número de fragmentos actualmente indexados en ChromaDB."""
        await self.ingest_async(force_rebuild=False)

        def _count() -> int:
            vs = self._get_vectorstore_sync()
            return len(vs.get().get("ids") or [])

        return await asyncio.to_thread(_count)

    async def search_async(
        self,
        payload: RAGSearchInput,
        *,
        tracer: trace.Tracer | None = None,
        simulate_failure: bool = False,
    ) -> RAGSearchOutput:
        """Recuperación semántica asíncrona instrumentada con OpenTelemetry y validada con Pydantic."""
        await self.ingest_async(force_rebuild=False)
        active_tracer = tracer or trace.get_tracer("intelligence_system.tracer")
        started = time.perf_counter()

        with active_tracer.start_as_current_span("tool.vector_search") as span:
            span.set_attribute("tool.name", "chroma_similarity_search")
            span.set_attribute("db.system", "chroma")
            span.set_attribute("retrieval.top_k", payload.top_k)
            if payload.category_filter:
                span.set_attribute("retrieval.category_filter", payload.category_filter)

            if simulate_failure or "[FAIL_RETRIEVAL]" in payload.query:
                exc = RuntimeError(
                    "Fallo de recuperación vectorial provocado para verificar observabilidad en Arize Phoenix."
                )
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise exc

            emb_started = time.perf_counter()
            with active_tracer.start_as_current_span("tool.query_embedding") as emb_span:
                query_vector = await asyncio.to_thread(
                    self.embeddings.embed_query, payload.query
                )
                embedding_ms = (time.perf_counter() - emb_started) * 1000
                emb_span.set_attribute(
                    "embedding.model", self.settings.openai_embedding_model
                )
                emb_span.set_attribute("embedding.duration_ms", round(embedding_ms, 2))

            search_started = time.perf_counter()
            with active_tracer.start_as_current_span(
                "tool.chroma_similarity_search"
            ) as search_span:
                vs = self._get_vectorstore_sync()
                filter_dict: dict[str, Any] | None = (
                    {"category": payload.category_filter}
                    if payload.category_filter
                    else None
                )
                docs = await asyncio.to_thread(
                    vs.similarity_search_by_vector,
                    query_vector,
                    k=payload.top_k,
                    filter=filter_dict,
                )
                # Si el filtro por categoría fue demasiado restrictivo, ampliamos búsqueda general
                if not docs and filter_dict is not None:
                    docs = await asyncio.to_thread(
                        vs.similarity_search_by_vector,
                        query_vector,
                        k=payload.top_k,
                    )
                vector_search_ms = (time.perf_counter() - search_started) * 1000
                search_span.set_attribute("db.system", "chroma")
                search_span.set_attribute("retrieval.top_k", payload.top_k)
                search_span.set_attribute(
                    "retrieval.duration_ms", round(vector_search_ms, 2)
                )

            retrieved_items: list[RetrievedDocument] = []
            seen_source_chunks: set[str] = set()
            for doc in docs:
                sid = str(doc.metadata.get("source_id", "UNKNOWN"))
                cidx = int(doc.metadata.get("chunk_index", 0))
                key = f"{sid}:{cidx}"
                if key in seen_source_chunks:
                    continue
                seen_source_chunks.add(key)
                retrieved_items.append(
                    RetrievedDocument(
                        source_id=sid,
                        title=str(doc.metadata.get("title", "Documento")),
                        category=str(doc.metadata.get("category", "general")),
                        page_content=doc.page_content,
                        chunk_index=cidx,
                    )
                )

            unique_source_ids = list(dict.fromkeys(d.source_id for d in retrieved_items))
            total_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("retrieval.document_count", len(retrieved_items))
            span.set_attribute("retrieval.duration_ms", round(total_ms, 2))
            span.set_attribute("retrieval.embedding_ms", round(embedding_ms, 2))
            span.set_attribute("retrieval.vector_search_ms", round(vector_search_ms, 2))
            span.set_attribute("retrieval.source_ids", ",".join(unique_source_ids))

        return RAGSearchOutput(
            documents=retrieved_items,
            source_ids=unique_source_ids,
            retrieval_ms=round(total_ms, 2),
            embedding_ms=round(embedding_ms, 2),
            vector_search_ms=round(vector_search_ms, 2),
        )
