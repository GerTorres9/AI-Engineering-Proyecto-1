"""Prueba la integración LangChain/OpenAI con HTTP simulado, sin consumo pagado."""
import json
import httpx2
from langchain_openai import ChatOpenAI
from app.checkpointer import AsyncRedisCheckpointer
from app.graph import build_graph
from app.observability import create_llm
from app.models import TaskRequest


async def test_real_sdk_and_token_limit(settings, redis, monkeypatch):
    payloads = []
    def respond(request):
        payload = json.loads(request.content)
        payloads.append(payload)
        return httpx2.Response(200, json={
            "id": "chatcmpl-test", "object": "chat.completion", "created": 0,
            "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Explicación de prueba"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as async_client:
        with httpx2.Client(transport=httpx2.MockTransport(respond)) as sync_client:
            def factory(**kwargs):
                return ChatOpenAI(**kwargs, http_async_client=async_client, http_client=sync_client)
            monkeypatch.setattr("app.observability.ChatOpenAI", factory)
            model = create_llm(settings)
            graph = build_graph(settings, AsyncRedisCheckpointer(redis), redis, model)
            result = await graph.ainvoke({"job_id": "sdk-test", "task": TaskRequest(
                pregunta="Explica async", max_output_tokens=128).model_dump()},
                config={"configurable": {"thread_id": "sdk-test"}})
    assert result["result"]["answer"] == "Explicación de prueba"
    assert len(payloads) == 3
    assert all(p["model"] == "gpt-4o-mini" for p in payloads)
    assert all(p.get("max_completion_tokens", p.get("max_tokens")) == 128 for p in payloads)
