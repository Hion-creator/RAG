# Guia Paso a Paso: Construir y Explicar un RAG Empresarial

## 1) Que es RAG en una frase
RAG (Retrieval-Augmented Generation) combina busqueda semantica sobre conocimiento de empresa con un LLM para responder con contexto real y verificable.

## 2) Arquitectura minima que debes saber explicar
1. Fuentes de conocimiento: politicas, manuales, SOPs, KB interna.
2. Ingesta: limpieza y normalizacion de texto.
3. Chunking: dividir documentos en partes con solapamiento.
4. Embeddings: convertir chunks en vectores numericos.
5. Indice vectorial: almacenar vectores + metadatos.
6. Retrieval: recuperar los chunks mas cercanos a la consulta.
7. Generacion: enviar pregunta + contexto recuperado al modelo.
8. Citas y observabilidad: mostrar fuentes, latencia y calidad.

## 3) Paso a paso tecnico (este repo)

### Paso 1. Configurar entorno
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Paso 2. Configurar Ollama Cloud
Tienes dos caminos oficiales:

1. Modo hibrido local+nube
```bash
ollama signin
ollama pull embeddinggemma
```
Mantienes `OLLAMA_HOST=http://localhost:11434` y usas modelos cloud (ejemplo verificado: `qwen3.5:cloud`).
Para embeddings en este modo: `EMBEDDING_MODEL=embeddinggemma`.

2. Modo API cloud directa
- Crear API key en `ollama.com/settings/keys`
- Configurar:
```env
OLLAMA_HOST=https://ollama.com
OLLAMA_API_KEY=tu_api_key
CHAT_MODEL=qwen3.5
```

### Paso 2.1 Verificar conexion antes de la demo
```bash
ollama --version
ollama signin
ollama list
```

Prueba de respuesta cloud (debe devolver "OK"):
```bash
curl http://localhost:11434/api/chat -d '{
	"model": "qwen3.5:cloud",
	"messages": [{"role": "user", "content": "Responde solo con: OK"}],
	"stream": false
}'
```

## Paso 3. Preparar datos
Coloca documentos en `data/knowledge/`.
Buenas practicas:
- Solo fuentes aprobadas.
- Versionado y fecha de vigencia.
- Anonimizar PII antes de indexar.

## Paso 4. Ingestar e indexar
```bash
python -m src.ingest
```
Esto realiza:
- Lectura de `.md` y `.txt`
- Chunking con overlap
- Llamadas a `POST /api/embed`
- Persistencia en `storage/index.json`

## Paso 5. Consultar RAG
```bash
python -m src.chat --question "Que KPIs recomienda el onboarding de IA?"
```
La app:
- Embebe la pregunta
- Recupera top-k chunks por similitud coseno
- Llama a `POST /api/chat`
- Devuelve respuesta + citas

### Paso 5.1 Comparar RAG vs cloud-only con la misma pregunta
```bash
python -m src.chat --compare --question "Cuales son los criterios de escalamiento a humano?"
```

Que muestra esta comparativa:
- Bloque 1: respuesta RAG con fuentes recuperadas.
- Bloque 2: respuesta cloud-only (sin retrieval).
- Tiempos de respuesta de ambos flujos.

Opcional:
```bash
python -m src.chat --compare --question "Cuales son los criterios de escalamiento a humano?" --cloud-model qwen3.5:cloud
```

### Paso 5.2 Demo visual con frontend web
```bash
python -m src.webapp
```

Abrir en navegador:
- `http://127.0.0.1:7860`

Flujo sugerido en vivo:
1. Ejecutar una pregunta en `Consultar solo RAG`.
2. Ejecutar la misma en `Comparar RAG vs Cloud`.
3. Resaltar que RAG entrega evidencia (fuentes) y cloud-only tiende a generalizar.

## Paso 6. Demo para entrevista (guion breve)
1. Explica problema: sin RAG, el modelo alucina o ignora contexto interno.
2. Ejecuta ingesta en vivo.
3. Haz 2 preguntas de negocio.
4. Ensena fuentes citadas.
5. Cierra con controles enterprise: seguridad, evaluacion y monitoreo.

## 4) Conceptos clave que SI o SI debes dominar

### Chunk size y overlap
- Chunk pequeno: mejor recall, peor contexto.
- Chunk grande: mas contexto, mayor ruido y costo.
- Punto de partida recomendado: 600-1,000 caracteres con overlap 10-20%.

### Top-k retrieval
- Top-k bajo: rapido pero puede perder evidencia.
- Top-k alto: mas cobertura pero mas ruido.
- Punto de partida: k=3 a k=6.

### Similarity threshold
Evita responder con chunks irrelevantes. Si la similitud no supera umbral, devolver "no hay contexto suficiente" y escalar.

### Grounding y citas
Toda respuesta debe mapearse a evidencia documental. Sin evidencia, no hay respuesta confiable para entorno empresarial.

## 5) Mejores practicas enterprise

1. Gobernanza de datos
- Catalogo de fuentes permitidas.
- Clasificacion de datos (publico/interno/confidencial/restringido).
- Politica de retencion y borrado.

2. Seguridad
- No indexar secretos ni credenciales.
- Cifrado en reposo y en transito.
- Control de acceso por roles (RBAC).
- Auditoria de consultas y respuestas.

3. Calidad y evaluacion
- Dataset dorado de preguntas/respuestas.
- Medir precision factual, cobertura de citas y alucinacion.
- Pruebas de regresion antes de cada release.

4. Observabilidad
- Metricas: latencia p50/p95, tasa de fallback, costo por consulta.
- Dashboards y alertas por drift o degradacion.

5. Resiliencia operativa
- Fallback a humano cuando baja confianza.
- Timeouts, retries y circuit breakers.
- Plan de incidentes y runbooks.

6. Cumplimiento
- Requisitos legales y regulatorios (PII, privacidad, auditoria).
- Evidencia trazable para revisiones internas y externas.

## 6) Errores comunes a evitar
- Indexar datos sin curacion ni permisos.
- No versionar embeddings y prompts.
- Mezclar documentos obsoletos con vigentes.
- Medir solo "que suene bien" en lugar de exactitud factual.

## 7) Como venderlo en entrevista
1. Enfatiza impacto de negocio: menos tiempo de busqueda, mejor consistencia.
2. Enfatiza control: evidencia citada y politicas de seguridad.
3. Enfatiza operacion: monitoreo, evaluacion continua y costos.

## 8) Checklist final pre-entrevista
- [ ] Ejecutar `python -m src.ingest` sin errores.
- [ ] Ejecutar `python -m src.chat --question ...` y validar citas.
- [ ] Tener preparado ejemplo de fallback cuando no hay contexto.
- [ ] Explicar diferencia entre "demo bonita" y "sistema enterprise confiable".
