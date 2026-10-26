"""Validación determinística de citas bibliográficas y políticas con Pydantic."""

from __future__ import annotations

import re
from typing import Iterable

from intelligence_system.schemas import CitationCheckResult

_CITATION_RE = re.compile(r"\[source:\s*([A-Za-z0-9_-]+)\s*\]", re.IGNORECASE)

_ABSTENTION_PHRASES = (
    "no encontré suficiente información",
    "no hay información suficiente",
    "no puedo confirmarlo con los documentos",
    "no puedo confirmar esa información",
    "no se encuentra en los documentos",
    "no cuento con evidencia suficiente",
    "no lo sé",
    "fuera del alcance",
)


def validate_citations(
    answer: str, valid_source_ids: Iterable[str]
) -> CitationCheckResult:
    """Valida de forma determinística que las citas en la respuesta existan en las fuentes recuperadas.

    Devuelve un objeto validado `CitationCheckResult` (Pydantic).
    """
    known_ids = {str(sid).strip().upper() for sid in valid_source_ids if sid}
    raw_cited = _CITATION_RE.findall(answer or "")
    cited_ids = sorted({cid.strip().upper() for cid in raw_cited if cid.strip()})
    invalid_ids = sorted(set(cited_ids) - known_ids)

    normalized_answer = (answer or "").casefold()
    is_abstention = any(phrase in normalized_answer for phrase in _ABSTENTION_PHRASES)
    missing_citation = (
        bool((answer or "").strip()) and not cited_ids and not is_abstention
    )

    issues: list[str] = []
    if invalid_ids:
        issues.append(
            "La respuesta cita identificadores que no están en los documentos recuperados: "
            + ", ".join(invalid_ids)
        )
    if missing_citation:
        issues.append(
            "La respuesta contiene afirmaciones factuales sin citas [source: ID] de las fuentes recuperadas."
        )

    return CitationCheckResult(
        valid=not issues,
        cited_ids=cited_ids,
        invalid_ids=invalid_ids,
        missing_citation=missing_citation,
        is_abstention=is_abstention,
        issues=issues,
    )
