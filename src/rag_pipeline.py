from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Sequence

import numpy as np
import requests


SUPPORTED_EXTENSIONS = {".md", ".txt"}


@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    text: str


@dataclass
class RagIndex:
    chunks: list[ChunkRecord]
    embeddings: np.ndarray
    metadata: dict[str, Any]


class OllamaAPIError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str, api_key: str | None = None, timeout_seconds: int = 120) -> None:
        self.host = host.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.host}{endpoint}"

        # Cloud-backed models can be slow sporadically; retry once on read timeout.
        max_attempts = 2
        response: requests.Response | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout_seconds)
                break
            except requests.ReadTimeout as exc:
                if attempt >= max_attempts:
                    raise OllamaAPIError(
                        f"No se pudo conectar con Ollama en {url}: {exc}"
                    ) from exc
                time.sleep(0.8)
            except requests.RequestException as exc:
                raise OllamaAPIError(f"No se pudo conectar con Ollama en {url}: {exc}") from exc

        if response is None:
            raise OllamaAPIError(f"No se obtuvo respuesta de Ollama en {url}")

        if response.status_code >= 400:
            raise OllamaAPIError(f"Ollama API error {response.status_code}: {response.text}")
        try:
            return response.json()
        except ValueError as exc:
            raise OllamaAPIError(f"Invalid JSON response from Ollama endpoint {endpoint}") from exc

    def embed(self, model: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        payload = {
            "model": model,
            "input": list(texts),
        }
        data = self._post("/api/embed", payload)
        embeddings = data.get("embeddings")
        if not embeddings:
            raise OllamaAPIError("Embedding response does not include embeddings")
        return np.array(embeddings, dtype=np.float32)

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.1,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        data = self._post("/api/chat", payload)
        message = data.get("message", {})
        content = message.get("content", "")
        return content.strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def read_documents(data_dir: Path) -> list[tuple[str, str]]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    docs: list[tuple[str, str]] = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append((str(path), text))
    return docs


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = _normalize_whitespace(text)
    if not text:
        return []

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    chunks: list[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)

    while start < len(text):
        end = min(len(text), start + chunk_size)
        candidate = text[start:end]

        # Prefer sentence boundaries to avoid splitting in the middle of ideas.
        if end < len(text):
            boundary = candidate.rfind(". ")
            if boundary > int(chunk_size * 0.6):
                end = start + boundary + 1
                candidate = text[start:end]

        cleaned = candidate.strip()
        if cleaned:
            chunks.append(cleaned)

        if end >= len(text):
            break
        start += step

    return chunks


def build_chunks(
    docs: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    records: list[ChunkRecord] = []
    for source, text in docs:
        chunks = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, chunk in enumerate(chunks):
            records.append(
                ChunkRecord(
                    chunk_id=f"{Path(source).stem}-{idx}",
                    source=source,
                    text=chunk,
                )
            )
    return records


def save_index(index_path: Path, index: RagIndex) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": index.metadata,
        "chunks": [asdict(chunk) for chunk in index.chunks],
        "embeddings": index.embeddings.tolist(),
    }
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_path: Path) -> RagIndex:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    chunks = [ChunkRecord(**item) for item in payload["chunks"]]
    embeddings = np.array(payload["embeddings"], dtype=np.float32)
    metadata = payload.get("metadata", {})
    return RagIndex(chunks=chunks, embeddings=embeddings, metadata=metadata)


def cosine_similarity_scores(query_embedding: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if query_embedding.ndim != 1:
        query_embedding = query_embedding.reshape(-1)

    query_norm = np.linalg.norm(query_embedding) + 1e-12
    matrix_norms = np.linalg.norm(matrix, axis=1) + 1e-12
    scores = matrix @ query_embedding
    return scores / (matrix_norms * query_norm)


def retrieve_chunks(
    index: RagIndex,
    query_embedding: np.ndarray,
    top_k: int,
    min_similarity: float,
) -> list[tuple[ChunkRecord, float]]:
    if len(index.chunks) == 0:
        return []

    scores = cosine_similarity_scores(query_embedding, index.embeddings)
    ranked_indices = np.argsort(scores)[::-1]

    results: list[tuple[ChunkRecord, float]] = []
    for idx in ranked_indices:
        score = float(scores[idx])
        if score < min_similarity:
            continue
        results.append((index.chunks[int(idx)], score))
        if len(results) >= top_k:
            break

    return results
