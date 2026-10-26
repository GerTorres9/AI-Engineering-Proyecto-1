# Evidencias de Observabilidad — Arize Phoenix

Esta carpeta almacena las evidencias visuales y diagramas de trazas generados al ejecutar las **5 pruebas de observabilidad** del sistema (`./run.sh eval` o `PYTHONPATH=src python scripts/run_observability_eval.py`).

## Capturas recomendadas del Dashboard de Arize Phoenix (`http://localhost:6006`)

1. `arquitectura_trazas.png`: Diagrama de jerarquía de spans OpenTelemetry (`multi_agent.request` -> `agent.supervisor` -> especialistas y herramientas).
2. `01_dashboard-trazas.png`: Listado del proyecto `the-intelligence-system` con las 5+ ejecuciones registradas.
3. `02_traza-jerarquia.png`: Vista detallada de una traza mostrando `agent.supervisor`, `agent.triage_sentiment`, `tool.order_lookup`, `tool.vector_search`, `agent.analyst`, y `agent.semantic_review`.
4. `03_detalle-tokens.png`: Atributos `llm.input_tokens`, `llm.output_tokens`, `llm.total_tokens`, `eval.grounded`, `eval.relevant`, `eval.citation_valid` y `app.thread_id`.
5. `04_escalamiento-o-fallo.png`: Traza del Caso 5 (`agent.human_escalation` por `SLA-003`) o del fallo controlado (`--simulate-failure`).
