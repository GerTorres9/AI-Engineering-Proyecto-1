#!/usr/bin/env bash
# ==============================================================================
# The Intelligence System — Script de Despliegue Local Unificado (run.sh)
# ==============================================================================
# Uso:
#   ./run.sh           -> Instala dependencias, levanta Arize Phoenix (6006),
#                         ejecuta las 5 pruebas de observabilidad y levanta FastAPI (8000).
#   ./run.sh eval      -> Levanta Phoenix (si no está activo) y corre las 5 pruebas.
#   ./run.sh test      -> Ejecuta la suite completa de pruebas unitarias e integración (pytest).
#   ./run.sh server    -> Levanta únicamente Arize Phoenix + Servidor FastAPI.
# ==============================================================================

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

MODE="${1:-all}"
export PYTHONPATH="$ROOT_DIR/src"

echo "=============================================================================="
echo "🧠 THE INTELLIGENCE SYSTEM — DESPLIEGUE LOCAL DE GRADO DE PRODUCCIÓN"
echo "=============================================================================="

# 1. Preparar archivo .env si no existe
if [[ ! -f "$ROOT_DIR/.env" ]]; then
    echo "📄 Creando archivo .env a partir de .env.example..."
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    echo "⚠️  Nota: Recuerda configurar tu OPENAI_API_KEY real en .env para usar gpt-4o-mini."
fi

# 2. Preparar entorno virtual Python 3.12+
if [[ ! -d "$ROOT_DIR/.venv" ]]; then
    echo "🐍 Creando entorno virtual Python 3.12+ en .venv..."
    if command -v uv >/dev/null 2>&1; then
        uv venv "$ROOT_DIR/.venv" --python 3.12
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        "$HOME/.local/bin/uv" venv "$ROOT_DIR/.venv" --python "$HOME/.local/bin/python3.12"
    else
        python3 -m venv "$ROOT_DIR/.venv"
    fi
fi

PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

# 3. Instalar dependencias si aún no están instaladas
if ! "$PYTHON_BIN" -c "import fastapi, langgraph, phoenix, chromadb, aiosqlite" >/dev/null 2>&1; then
    echo "📦 Instalando dependencias del sistema..."
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$PYTHON_BIN" -r "$ROOT_DIR/requirements.txt"
    elif [[ -x "$HOME/.local/bin/uv" ]]; then
        "$HOME/.local/bin/uv" pip install --python "$PYTHON_BIN" -r "$ROOT_DIR/requirements.txt"
    else
        "$PYTHON_BIN" -m pip install --upgrade pip
        "$PYTHON_BIN" -m pip install -r "$ROOT_DIR/requirements.txt"
    fi
fi

mkdir -p "$ROOT_DIR/storage" "$ROOT_DIR/results" "$ROOT_DIR/logs"

if [[ "$MODE" == "test" ]]; then
    echo "🧪 Ejecutando suite de pruebas automatizadas (pytest)..."
    exec "$ROOT_DIR/.venv/bin/pytest" -v
fi

# 4. Asegurar que Arize Phoenix esté corriendo en http://127.0.0.1:6006
start_phoenix_if_needed() {
    if curl -sSf http://127.0.0.1:6006 >/dev/null 2>&1; then
        echo "🔭 Arize Phoenix ya está activo en http://localhost:6006"
        return 0
    fi
    echo "🚀 Iniciando servidor de observabilidad Arize Phoenix en segundo plano (puerto 6006)..."
    nohup "$PYTHON_BIN" -m phoenix.server.main serve > "$ROOT_DIR/logs/phoenix.log" 2>&1 &
    PHOENIX_PID=$!
    echo "$PHOENIX_PID" > "$ROOT_DIR/logs/phoenix.pid"

    for _ in {1..30}; do
        if curl -sSf http://127.0.0.1:6006 >/dev/null 2>&1; then
            echo "✅ Arize Phoenix listo en http://localhost:6006 (PID: $PHOENIX_PID)"
            return 0
        fi
        sleep 1
    done
    echo "⚠️  Arize Phoenix tardó más de lo esperado en responder; revisa logs/phoenix.log"
}

start_phoenix_if_needed

if [[ "$MODE" == "eval" ]]; then
    echo "📊 Ejecutando las 5 pruebas de observabilidad y persistencia..."
    exec "$PYTHON_BIN" "$ROOT_DIR/scripts/run_observability_eval.py"
fi

if [[ "$MODE" == "all" ]]; then
    echo "📊 Ejecutando las 5 pruebas de observabilidad iniciales..."
    "$PYTHON_BIN" "$ROOT_DIR/scripts/run_observability_eval.py"
fi

echo ""
echo "🌐 Levantando servidor FastAPI en http://localhost:8000"
echo "   • Documentación interactiva (Swagger UI): http://localhost:8000/docs"
echo "   • Documentación ReDoc                   : http://localhost:8000/redoc"
echo "   • Healthcheck                           : http://localhost:8000/health"
echo "   • Dashboard Arize Phoenix               : http://localhost:6006"
echo "=============================================================================="

exec "$ROOT_DIR/.venv/bin/uvicorn" intelligence_system.api:app --host 0.0.0.0 --port 8000
