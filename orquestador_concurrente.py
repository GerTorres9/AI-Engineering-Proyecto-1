"""Cliente CLI compatible con el punto de entrada original del tutor."""
import asyncio
import os
import httpx
from dotenv import load_dotenv


async def main():
    load_dotenv()
    token = os.getenv("API_TOKEN")
    if not token:
        raise SystemExit("Configura API_TOKEN en .env y levanta la API y el worker.")
    async with httpx.AsyncClient(base_url=os.getenv("API_URL", "http://localhost:8000"),
            headers={"Authorization": f"Bearer {token}"}, timeout=15) as client:
        response = await client.post("/tasks", json={"pregunta":
            "programación asíncrona y cómo se diferencia de la programación síncrona"})
        response.raise_for_status()
        job_id = response.json()["job_id"]
        print(f"Consulta encolada: {job_id}")
        for _ in range(240):
            response = await client.get(f"/tasks/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["status"] == "DONE":
                print(job["result"]["answer"])
                return
            if job["status"] in {"FAILED", "REJECTED", "WAITING_APPROVAL"}:
                raise SystemExit(f"Estado: {job['status']}; consulta la API para los detalles.")
            await asyncio.sleep(1)
        raise SystemExit("Tiempo de espera agotado; el job conserva su estado en Redis.")


if __name__ == "__main__":
    asyncio.run(main())
