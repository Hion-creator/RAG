# RAG Beta con Ollama Cloud (Demo de Entrevista)

Este proyecto implementa un RAG basico en Python con:
- Ingesta de documentos (`.md`, `.txt`)
- Chunking con solapamiento
- Embeddings via `POST /api/embed`
- Retrieval por similitud coseno
- Generacion final via `POST /api/chat`
- Citas de fuente para trazabilidad

Incluye dos modos de uso con Ollama:
1. Modo hibrido local+nube (recomendado para demo rapida): `ollama signin` y host local.
2. Modo API cloud directa: host `https://ollama.com` + API key.

## Estructura

- `src/ingest.py`: construye indice vectorial
- `src/chat.py`: consulta RAG en modo one-shot o interactivo
- `src/webapp.py`: servidor web para demo visual
- `src/rag_pipeline.py`: cliente Ollama + retrieval
- `web/templates/index.html`: interfaz principal
- `web/static/`: estilos y logica del frontend
- `data/knowledge/`: documentos de ejemplo
- `docs/guia_rag_entrevista.md`: guia paso a paso y mejores practicas enterprise

## Requisitos

- Python 3.10+
- Ollama instalado o acceso a Ollama Cloud
- Cuenta en ollama.com

## Instalacion

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

## Configuracion Ollama Cloud

### Opcion A: hibrido local+nube

```bash
ollama signin
ollama pull embeddinggemma
```

Con esto puedes seguir usando `OLLAMA_HOST=http://localhost:11434` y un modelo cloud como:
- `CHAT_MODEL=qwen3.5:cloud`
- `EMBEDDING_MODEL=embeddinggemma`

### Opcion B: API cloud directa

1. Crea API key en ollama.com/settings/keys
2. En `.env`:

```env
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=tu_api_key
CHAT_MODEL=qwen3.5
```

## Verificar conexion (check rapido)

```bash
ollama --version
ollama signin
ollama list
```

Prueba de inferencia cloud via host local:

```bash
curl http://localhost:11434/api/chat -d '{
	"model": "qwen3.5:cloud",
	"messages": [{"role": "user", "content": "Responde solo con: OK"}],
	"stream": false
}'
```

## Ejecutar demo

1. Construir indice:

```bash
python -m src.ingest
```

2. Pregunta unica:

```bash
python -m src.chat --question "Cuales son los criterios de escalamiento a humano?"
```

3. Modo interactivo:

```bash
python -m src.chat
```

4. Comparativa (misma pregunta: RAG vs cloud-only):

```bash
python -m src.chat --compare --question "Cuales son los criterios de escalamiento a humano?"
```

Opcional: forzar modelo base sin retrieval para comparar:

```bash
python -m src.chat --compare --question "Cuales son los criterios de escalamiento a humano?" --cloud-model qwen3.5:cloud
```

5. Frontend visual profesional (recomendado para entrevista):

```bash
python -m src.webapp
```

Luego abre: `http://127.0.0.1:7860`

Desde la UI puedes:
- Ejecutar `Consultar solo RAG`
- Ejecutar `Comparar RAG vs Cloud`
- Mostrar latencias, respuesta y fuentes recuperadas en pantalla

## Tip para entrevista

Muestra una pregunta, luego abre el bloque de fuentes en la salida. Explica que el valor empresarial no es solo responder, sino responder con evidencia rastreable.
