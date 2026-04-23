from __future__ import annotations

import argparse
from pathlib import Path
import re
from time import perf_counter

from src.config import get_settings
from src.rag_pipeline import OllamaAPIError, OllamaClient, load_index, retrieve_chunks


SYSTEM_PROMPT = """Eres un asistente de RAG para entorno empresarial.
Responde solo con base en el contexto recuperado.
Si la respuesta no esta en el contexto, dilo de forma explicita.
Usa tono ejecutivo y sintetico.
No uses formato markdown ni cites texto literal largo del documento.
No menciones retrieval ni hagas alusion a documentos internos en la redaccion final.
Cuando menciones porcentajes usa siempre el simbolo % (ejemplo 85%), nunca escribas "por ciento".
Cuando haya varios puntos, usa saltos de linea y lista con prefijo '- '."""

CLOUD_ONLY_SYSTEM_PROMPT = """Eres un asistente empresarial.
Responde sin inventar informacion y se explicito cuando falte contexto.
No uses fuentes externas no verificables.
Usa tono ejecutivo y no uses markdown.
Cuando menciones porcentajes usa el simbolo % y no la frase "por ciento".
Cuando haya varios puntos, separalos con saltos de linea."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask questions over a RAG index")
    parser.add_argument("--index-path", default="storage/index.json", help="Index path")
    parser.add_argument("--question", default=None, help="Single question mode")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare same question in RAG vs cloud-only model",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Override retrieval top-k")
    parser.add_argument("--threshold", type=float, default=None, help="Override similarity threshold")
    parser.add_argument("--chat-model", default=None, help="Override chat model")
    parser.add_argument("--cloud-model", default=None, help="Override cloud-only baseline model")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model")
    parser.add_argument("--timeout", type=int, default=240, help="HTTP timeout in seconds")
    return parser.parse_args()


def _friendly_ollama_error(exc: OllamaAPIError) -> str:
    message = str(exc)
    hints: list[str] = []

    lowered = message.lower()
    if "model" in lowered and "not found" in lowered:
        hints.append("Modelo no disponible. Ajusta CHAT_MODEL/EMBEDDING_MODEL o descarga el modelo.")
    if "401" in lowered or "unauthorized" in lowered:
        hints.append("Autenticacion invalida. Ejecuta `ollama signin` o configura `OLLAMA_API_KEY`.")
    if "403" in lowered or "forbidden" in lowered:
        hints.append("Acceso denegado. Verifica permisos de cuenta y API key en ollama.com.")
    if "timed out" in lowered or "timeout" in lowered:
        hints.append("Timeout de red. Sube timeout o verifica conectividad.")
    if "no se pudo conectar" in lowered or "connection" in lowered:
        hints.append("No hay conexion con Ollama. Verifica host, daemon o internet.")

    if not hints:
        hints.append("Verifica OLLAMA_HOST, modelos configurados y conectividad de red.")

    return f"{message}\nSugerencias:\n- " + "\n- ".join(hints)


def _to_executive_plain_text(text: str) -> str:
    if not text:
        return ""

    cleaned = text.replace("\r\n", "\n")
    cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.replace("**", "")
    cleaned = cleaned.replace("__", "")
    cleaned = re.sub(r"^\s*[*+-]\s+", "- ", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"^\s*>\s?", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", cleaned)
    cleaned = re.sub(
        r"(?i)^\s*segun (el|la) (documento|documentacion|contexto recuperado)[^\n.]*[.:]\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)^\s*con base en (el|la) (documentacion|contexto)[^\n.]*[.:]\s*",
        "",
        cleaned,
    )
    cleaned = re.sub(r"(?i)\b(\d+(?:[.,]\d+)?)\s*por\s*ciento\b", r"\1%", cleaned)
    cleaned = re.sub(r"(?i)\b(\d+(?:[.,]\d+)?)\s+%", r"\1%", cleaned)

    if "\n" not in cleaned:
        sentences = [
            part.strip()
            for part in re.split(r"(?<=[.!?;])\s+", cleaned)
            if part.strip()
        ]
        if len(sentences) >= 2:
            has_percent = bool(re.search(r"\d+(?:[.,]\d+)?%", cleaned))
            if has_percent:
                cleaned = "\n".join(f"- {sentence}" for sentence in sentences)
            else:
                cleaned = "\n\n".join(sentences)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _build_context(retrieved: list[tuple]) -> tuple[str, list[str]]:
    blocks: list[str] = []
    citations: list[str] = []

    for chunk, score in retrieved:
        citation = f"{chunk.source}#{chunk.chunk_id}"
        citations.append(citation)
        blocks.append(
            "\n".join(
                [
                    f"Fuente: {chunk.source}",
                    f"Chunk: {chunk.chunk_id}",
                    f"Score: {score:.3f}",
                    f"Contenido: {chunk.text}",
                ]
            )
        )

    return "\n\n---\n\n".join(blocks), citations


def _build_client(timeout: int) -> OllamaClient:
    settings = get_settings()
    return OllamaClient(
        host=settings.ollama_host,
        api_key=settings.ollama_api_key,
        timeout_seconds=timeout,
    )


def answer_once(
    question: str,
    index_path: Path,
    top_k: int,
    threshold: float,
    chat_model: str,
    embedding_model: str,
    timeout: int,
) -> tuple[str, list[str]]:
    index = load_index(index_path)
    client = _build_client(timeout)

    try:
        query_embedding = client.embed(model=embedding_model, texts=[question])[0]
    except OllamaAPIError as exc:
        raise SystemExit(_friendly_ollama_error(exc)) from exc
    retrieved = retrieve_chunks(
        index=index,
        query_embedding=query_embedding,
        top_k=top_k,
        min_similarity=threshold,
    )

    if not retrieved:
        return (
            "No encontre contexto suficiente en la base documental para responder con confianza.",
            [],
        )

    context, citations = _build_context(retrieved)
    user_prompt = f"""Pregunta: {question}

Contexto recuperado:
{context}

Instrucciones:
1) Responde en espanol de forma clara y concreta.
2) No inventes datos fuera del contexto recuperado.
3) Usa texto estructurado con saltos de linea.
4) Si hay varios puntos, presentalos como lista con prefijo '- '.
5) Escribe porcentajes usando simbolo % (por ejemplo 85%), nunca "por ciento".
6) No uses markdown decorativo (sin **, #, tablas o bloques de codigo).
7) No copies texto literal largo del contexto; sintetiza en tono ejecutivo.
8) No incluyas citas tecnicas ni menciones retrieval o documentos internos."""

    try:
        answer = client.chat(
            model=chat_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )
    except OllamaAPIError as exc:
        raise SystemExit(_friendly_ollama_error(exc)) from exc
    return _to_executive_plain_text(answer), citations


def cloud_only_answer_once(
    question: str,
    cloud_model: str,
    timeout: int,
) -> str:
    client = _build_client(timeout)
    try:
        answer = client.chat(
            model=cloud_model,
            messages=[
                {"role": "system", "content": CLOUD_ONLY_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.1,
        )
        return _to_executive_plain_text(answer)
    except OllamaAPIError as exc:
        raise SystemExit(_friendly_ollama_error(exc)) from exc


def compare_once(
    question: str,
    index_path: Path,
    top_k: int,
    threshold: float,
    rag_model: str,
    cloud_model: str,
    embedding_model: str,
    timeout: int,
) -> dict[str, object]:
    rag_started_at = perf_counter()
    rag_answer, citations = answer_once(
        question=question,
        index_path=index_path,
        top_k=top_k,
        threshold=threshold,
        chat_model=rag_model,
        embedding_model=embedding_model,
        timeout=timeout,
    )
    rag_elapsed = perf_counter() - rag_started_at

    cloud_started_at = perf_counter()
    cloud_answer = cloud_only_answer_once(
        question=question,
        cloud_model=cloud_model,
        timeout=timeout,
    )
    cloud_elapsed = perf_counter() - cloud_started_at

    return {
        "rag_answer": rag_answer,
        "citations": citations,
        "rag_seconds": rag_elapsed,
        "cloud_answer": cloud_answer,
        "cloud_seconds": cloud_elapsed,
    }


def _print_comparison(
    question: str,
    result: dict[str, object],
    rag_model: str,
    cloud_model: str,
) -> None:
    rag_answer = str(result["rag_answer"])
    citations = list(result["citations"])
    rag_seconds = float(result["rag_seconds"])
    cloud_answer = str(result["cloud_answer"])
    cloud_seconds = float(result["cloud_seconds"])

    print("\n" + "=" * 72)
    print("COMPARATIVA MISMA PREGUNTA")
    print("=" * 72)
    print(f"Pregunta: {question}")

    print("\n[RAG con contexto recuperado]")
    print(f"Modelo: {rag_model}")
    print(f"Tiempo: {rag_seconds:.2f}s")
    print("Respuesta:")
    print(rag_answer)

    if citations:
        print("Fuentes recuperadas:")
        for citation in citations:
            print(f"- {citation}")

    print("\n[Cloud-only sin retrieval]")
    print(f"Modelo: {cloud_model}")
    print(f"Tiempo: {cloud_seconds:.2f}s")
    print("Respuesta:")
    print(cloud_answer)


def run_compare_interactive(
    index_path: Path,
    top_k: int,
    threshold: float,
    rag_model: str,
    cloud_model: str,
    embedding_model: str,
    timeout: int,
) -> None:
    print("Comparador interactivo RAG vs cloud-only. Escribe 'salir' para terminar.")

    while True:
        question = input("\nPregunta > ").strip()
        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            break

        result = compare_once(
            question=question,
            index_path=index_path,
            top_k=top_k,
            threshold=threshold,
            rag_model=rag_model,
            cloud_model=cloud_model,
            embedding_model=embedding_model,
            timeout=timeout,
        )
        _print_comparison(
            question=question,
            result=result,
            rag_model=rag_model,
            cloud_model=cloud_model,
        )


def run_interactive(
    index_path: Path,
    top_k: int,
    threshold: float,
    chat_model: str,
    embedding_model: str,
    timeout: int,
) -> None:
    print("RAG chat interactivo. Escribe 'salir' para terminar.")

    while True:
        question = input("\nPregunta > ").strip()
        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            break

        answer, citations = answer_once(
            question=question,
            index_path=index_path,
            top_k=top_k,
            threshold=threshold,
            chat_model=chat_model,
            embedding_model=embedding_model,
            timeout=timeout,
        )
        print("\nRespuesta:\n")
        print(answer)
        if citations:
            print("\nFuentes recuperadas:")
            for citation in citations:
                print(f"- {citation}")


def main() -> None:
    args = parse_args()
    settings = get_settings()

    index_path = Path(args.index_path)
    top_k = args.top_k or settings.top_k
    threshold = args.threshold or settings.similarity_threshold
    chat_model = args.chat_model or settings.chat_model
    cloud_model = args.cloud_model or chat_model
    embedding_model = args.embedding_model or settings.embedding_model

    if args.compare and args.question:
        result = compare_once(
            question=args.question,
            index_path=index_path,
            top_k=top_k,
            threshold=threshold,
            rag_model=chat_model,
            cloud_model=cloud_model,
            embedding_model=embedding_model,
            timeout=args.timeout,
        )
        _print_comparison(
            question=args.question,
            result=result,
            rag_model=chat_model,
            cloud_model=cloud_model,
        )
        return

    if args.compare:
        run_compare_interactive(
            index_path=index_path,
            top_k=top_k,
            threshold=threshold,
            rag_model=chat_model,
            cloud_model=cloud_model,
            embedding_model=embedding_model,
            timeout=args.timeout,
        )
        return

    if args.question:
        answer, citations = answer_once(
            question=args.question,
            index_path=index_path,
            top_k=top_k,
            threshold=threshold,
            chat_model=chat_model,
            embedding_model=embedding_model,
            timeout=args.timeout,
        )
        print(answer)
        if citations:
            print("\nFuentes:")
            for citation in citations:
                print(f"- {citation}")
        return

    run_interactive(
        index_path=index_path,
        top_k=top_k,
        threshold=threshold,
        chat_model=chat_model,
        embedding_model=embedding_model,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
