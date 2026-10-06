from langgraph.types import interrupt


async def approval_node(state):
    # No efectos secundarios antes del interrupt: el nodo se reejecuta al reanudar.
    if not state["critical_reasons"]:
        return {"approved": True}
    decision = interrupt({
        "question": "¿Autorizas esta ejecución crítica?",
        "reasons": state["critical_reasons"],
        "accion": state["task"]["accion"],
        "max_output_tokens_por_llamada": state["task"]["max_output_tokens"],
    })
    return {"approved": decision["approved"], "approval_note": decision.get("note", "")}
