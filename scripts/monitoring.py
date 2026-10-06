"""Lee costo y tiempos REALES de LangSmith. No usa precios ni tokens inventados.

Publica opcionalmente p95 como feedback para interfaces sin percentil 95 nativo.
"""
import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from uuid import UUID, uuid5, NAMESPACE_URL
from dotenv import load_dotenv
from langsmith import Client
from scripts.load_test import percentile95


def main(args):
    if not os.getenv("LANGSMITH_API_KEY"):
        raise SystemExit("Configura LANGSMITH_API_KEY en .env")
    batch = json.loads(Path(args.batch).read_text())
    if batch["count"] != 5 or any(j["status"] != "DONE" for j in batch["jobs"]):
        raise SystemExit("Se requieren cinco trabajos DONE de la misma corrida")
    if any(len(j["run_ids"]) != 1 for j in batch["jobs"]):
        raise SystemExit("Hubo reintentos/HITL; ejecuta otra corrida de cinco tareas normales")
    client = Client()
    rows = []
    nodes = defaultdict(lambda: {"tokens": 0, "llm_seconds": 0.0, "llm_calls": 0})
    for job in batch["jobs"]:
        run_id = job["run_ids"][0]
        deadline = time.monotonic() + args.wait
        while True:
            try:
                root = client.read_run(run_id)
                if root.end_time and root.total_cost is not None:
                    break
            except Exception:
                if time.monotonic() >= deadline:
                    raise
            if time.monotonic() >= deadline:
                raise SystemExit("LangSmith aún no registra costo/fin. Verifica trazas, "
                                 "modelo y precios en el dashboard y vuelve a ejecutar.")
            time.sleep(1)
        if root.error:
            raise SystemExit(f"La traza {run_id} reporta un error")
        rows.append({"job_id": job["job_id"], "run_id": run_id,
                     "cost_usd": float(root.total_cost),
                     "latency_seconds": (root.end_time - root.start_time).total_seconds(),
                     "input_tokens": root.prompt_tokens,
                     "output_tokens": root.completion_tokens})
        runs = list(client.list_runs(trace_id=UUID(run_id)))
        indexed = {str(r.id): r for r in runs}
        for r in runs:
            if r.run_type != "llm":
                continue
            parent = r
            name = "sin_clasificar"
            seen = set()
            while parent and str(parent.id) not in seen:
                seen.add(str(parent.id))
                if parent.name in {"agente_planificador", "agente_tutor", "agente_revisor"}:
                    name = parent.name
                    break
                parent = indexed.get(str(parent.parent_run_id))
            nodes[name]["tokens"] += (r.prompt_tokens or 0) + (r.completion_tokens or 0)
            nodes[name]["llm_calls"] += 1
            if r.end_time:
                nodes[name]["llm_seconds"] += (r.end_time - r.start_time).total_seconds()
    if sum(n["llm_calls"] for n in nodes.values()) != 15:
        raise SystemExit("Las 15 llamadas LLM todavía no están disponibles: reintenta tras sincronizar.")
    p95 = percentile95([r["latency_seconds"] for r in rows])
    output = {"batch_id": batch["batch_id"], "executions": rows,
              "trace_latency_p95_seconds": p95,
              "p95_method": "interpolación lineal sobre las cinco trazas raíz de LangSmith",
              "nodes": dict(nodes),
              "most_tokens_node": max(nodes, key=lambda n: nodes[n]["tokens"]),
              "most_llm_latency_node": max(nodes, key=lambda n: nodes[n]["llm_seconds"])}
    if args.publish_p95:
        for row in rows:
            client.create_feedback(row["run_id"], key="batch_latency_p95_seconds", score=p95,
                feedback_id=uuid5(NAMESPACE_URL, row["run_id"] + ":batch_latency_p95_seconds"),
                comment=f"Corrida {batch['batch_id']}; p95 lineal calculado desde 5 trazas raíz. "
                        "Promediar este feedback en el dashboard muestra el p95 de esta corrida.")
        output["dashboard_feedback_key"] = "batch_latency_p95_seconds"
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", default="artifacts/load-batch.json")
    parser.add_argument("--output", default="artifacts/langsmith-metrics.json")
    parser.add_argument("--wait", type=float, default=60)
    parser.add_argument("--publish-p95", action="store_true")
    main(parser.parse_args())
