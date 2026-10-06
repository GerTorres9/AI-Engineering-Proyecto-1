Evidencia pendiente de corrida real
==================================

Esta carpeta debe contener capturas auténticas de LangSmith antes de entregar.
No se incluyen imágenes de ejemplo ni métricas simuladas.

Ejecuta `python -m scripts.load_test`, verifica cinco `DONE` y abre el dashboard
filtrado por el `batch_id` de esa corrida. Después ejecuta
`python -m scripts.monitoring` para contrastar costos, p95 y consumo por nodo.

Agrega estas capturas:

- `01-trazas-corrida.png`: lista de las cinco trazas raíz.
- `02-detalle-nodos.png`: árbol, tiempos y tokens de una ejecución.
- `03-costo-por-ejecucion.png`: costo automático de cada una de las cinco trazas.
- `04-latencia-p95.png`: p95 en segundos del mismo batch.
- `05-hitl.png`: pausa y reanudación de una tarea crítica aparte.

Si necesitas `--publish-p95`, el gráfico debe identificar la métrica como
“p95 derivado de cinco trazas”, según el README raíz. No sustituyas p95 por p99.
Las capturas son parte obligatoria de la pre-entrega: esta carpeta sin imágenes
todavía no acredita el criterio de observabilidad.
