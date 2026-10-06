"""Envía exactamente cinco POST concurrentes; hace polling hasta estado terminal."""
import argparse
import asyncio
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import httpx
from dotenv import load_dotenv

QUESTIONS = [
    "¿Qué es la programación asíncrona? Explica un ejemplo breve en Python.",
    "¿Qué diferencia hay entre una lista y una tupla en Python?",
    "¿Para qué sirve un endpoint REST y qué significa HTTP 202?",
    "¿Qué es Redis y cómo ayuda a guardar el estado de una tarea?",
    "¿Qué es una condición de carrera y cómo se evita?",
]


def percentile95(values):
    # Interpolación lineal: mismo método usado en scripts/monitoring.py.
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


async def run(args):
    token = os.getenv("API_TOKEN")
    if not token:
        raise SystemExit("Configura API_TOKEN en .env")
    batch_id = "carga-" + uuid4().hex[:12]
    started_at = datetime.now(timezone.utc).isoformat()
    gate = asyncio.Event()
    async with httpx.AsyncClient(base_url=args.url, timeout=15,
                                 headers={"Authorization": f"Bearer {token}"}) as client:
        async def one(question):
            await gate.wait()
            start = time.perf_counter()
            response = await client.post("/tasks", json={"pregunta": question,
                "batch_id": batch_id, "max_output_tokens": 400})
            response.raise_for_status()
            if response.status_code != 202:
                raise RuntimeError("POST /tasks no devolvió HTTP 202")
            job_id = response.json()["job_id"]
            admission_ms = (time.perf_counter() - start) * 1000
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                status = await client.get(f"/tasks/{job_id}")
                status.raise_for_status()
                job = status.json()
                if job["status"] in {"DONE", "FAILED", "REJECTED", "WAITING_APPROVAL"}:
                    return {"job_id": job_id, "status": job["status"],
                            "run_ids": job["run_ids"], "admission_ms": admission_ms,
                            "client_e2e_seconds": time.perf_counter() - start}
                await asyncio.sleep(0.2)
            raise TimeoutError(f"No finalizó {job_id} en {args.timeout} segundos")
        tasks = [asyncio.create_task(one(q)) for q in QUESTIONS]
        gate.set()
        results = await asyncio.gather(*tasks)
    output = {"batch_id": batch_id, "started_at": started_at,
              "finished_at": datetime.now(timezone.utc).isoformat(), "count": len(results),
              "jobs": results, "client_e2e_p95_seconds": percentile95([
                  job["client_e2e_seconds"] for job in results]),
              "note": "Latencia cliente incluye cola y polling; no sustituye p95 del dashboard."}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if any(job["status"] != "DONE" for job in results):
        raise SystemExit("La corrida no tiene cinco DONE; revisa el worker antes de capturar.")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", default="artifacts/load-batch.json")
    asyncio.run(run(parser.parse_args()))
