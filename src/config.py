from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_host: str
    ollama_api_key: str | None
    chat_model: str
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    similarity_threshold: float


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    return int(value)


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    return float(value)


def get_settings() -> Settings:
    return Settings(
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/"),
        ollama_api_key=os.getenv("OLLAMA_API_KEY") or None,
        chat_model=os.getenv("CHAT_MODEL", "qwen3.5:cloud"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "embeddinggemma"),
        chunk_size=_get_int("CHUNK_SIZE", 900),
        chunk_overlap=_get_int("CHUNK_OVERLAP", 150),
        top_k=_get_int("TOP_K", 4),
        similarity_threshold=_get_float("SIMILARITY_THRESHOLD", 0.18),
    )
