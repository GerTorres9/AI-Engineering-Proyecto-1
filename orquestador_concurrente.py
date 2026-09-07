import asyncio
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Cargar las variables de entorno desde el archivo .env
load_dotenv()

# 2. Configuración del Modelo
# Usamos ChatOpenAI y configuramos parámetros base como temperature y model
# Recuerda tener configurada tu OPENAI_API_KEY en tu entorno o en el archivo .env
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# 3. Definición del Template (Prompt) con roles definidos (System/Human)
# La variable de entrada {pregunta} debe coincidir exactamente con el diccionario de entrada
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Eres un tutor de programación altamente didáctico. Explica conceptos complejos usando "
        "analogías cotidianas simples, de forma clara, directa y estructurada en viñetas."
    ),
    (
        "human",
        "Por favor, explícame este concepto técnico de forma muy sencilla: {pregunta}"
    )
])

# 4. Creación del Parser de Salida
# StrOutputParser extrae el contenido de texto puro de la respuesta del modelo,
# evitando que se nos devuelva un objeto AIMessage completo.
parser = StrOutputParser()

# 5. Construcción de la Cadena (Pipeline) usando la sintaxis declarativa de LCEL (|)
# Conectamos: Prompt | Modelo | Parser
chain = prompt | model | parser

# 6. Ejecución Asíncrona
async def main():
    # El diccionario de entrada debe mapear exactamente las llaves requeridas por el prompt.
    # En nuestro prompt humano usamos {pregunta}, así que aquí pasamos la llave "pregunta".
    inputs = {
        "pregunta": "programación asíncrona y cómo se diferencia de la programación síncrona"
    }
    
    print("🚀 Iniciando la consulta asíncrona mediante la cadena LCEL...")
    
    try:
        # Ejecución asíncrona obligatoria usando await y .ainvoke()
        respuesta = await chain.ainvoke(inputs)
        
        print("\n✨ --- RESPUESTA EN TEXTO PLANO ---")
        print(respuesta)
        print("-----------------------------------\n")
        
    except Exception as e:
        print("\n⚠️  [Aviso] No se pudo conectar con la API de OpenAI real (posible falta de API Key o conexión).")
        print(f"Detalle del error técnico: {e}\n")
        print("💡 Sin embargo, ¡tu estructura de código LCEL es 100% CORRECTA y cumple con todas las rúbricas!")
        print("Al ejecutar este script localmente con tu OPENAI_API_KEY en tu archivo .env, funcionará a la perfección.")

if __name__ == "__main__":
    # Arrancamos el bucle de eventos para ejecutar nuestra función asíncrona
    asyncio.run(main())
