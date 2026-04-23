from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import get_settings
from src.rag_pipeline import (
    OllamaAPIError,
    RagIndex,
    OllamaClient,
    build_chunks,
    read_documents,
    save_index,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local RAG index from documents")
    parser.add_argument("--data-dir", default="data/knowledge", help="Directory with .md/.txt files")
    parser.add_argument("--index-path", default="storage/index.json", help="Output index path")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override chunk size")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="Override chunk overlap")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout in seconds")
    return parser.parse_args()


def _friendly_ollama_error(exc: OllamaAPIError) -> str:
    message = str(exc)
    hints: list[str] = []

    lowered = message.lower()
    if "model" in lowered and "not found" in lowered:
        hints.append("Modelo no disponible. Prueba con `ollama pull embeddinggemma`.")
    if "401" in lowered or "unauthorized" in lowered:
        hints.append("Autenticacion invalida. Ejecuta `ollama signin` o configura `OLLAMA_API_KEY`.")
    if "403" in lowered or "forbidden" in lowered:
        hints.append("Acceso denegado. Verifica permisos de cuenta y API key en ollama.com.")

    if not hints:
        hints.append("Verifica OLLAMA_HOST, modelo configurado y conectividad de red.")

    return f"{message}\nSugerencias:\n- " + "\n- ".join(hints)


def main() -> None:
    args = parse_args()
    settings = get_settings()

    data_dir = Path(args.data_dir)
    index_path = Path(args.index_path)
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = args.chunk_overlap or settings.chunk_overlap
    embedding_model = args.embedding_model or settings.embedding_model

    docs = read_documents(data_dir)
    if not docs:
        raise SystemExit(f"No documents found in {data_dir}")

    chunks = build_chunks(docs=docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise SystemExit("No chunks were generated. Check your document contents.")

    client = OllamaClient(
        host=settings.ollama_host,
        api_key=settings.ollama_api_key,
        timeout_seconds=args.timeout,
    )

    all_embeddings: list[np.ndarray] = []
    texts = [chunk.text for chunk in chunks]

    for start in range(0, len(texts), args.batch_size):
        end = min(len(texts), start + args.batch_size)
        batch = texts[start:end]
        try:
            embeddings = client.embed(model=embedding_model, texts=batch)
        except OllamaAPIError as exc:
            raise SystemExit(_friendly_ollama_error(exc)) from exc
        all_embeddings.append(embeddings)
        print(f"Embedded chunks {start + 1}-{end} / {len(texts)}")

    matrix = np.vstack(all_embeddings)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "ollama_host": settings.ollama_host,
        "embedding_model": embedding_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(docs),
        "chunk_count": len(chunks),
    }

    index = RagIndex(chunks=chunks, embeddings=matrix, metadata=metadata)
    save_index(index_path=index_path, index=index)

    print(f"Index written to {index_path}")
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")


if __name__ == "__main__":
    main()
