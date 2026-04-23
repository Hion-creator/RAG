from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import Any

from flask import Flask, jsonify, render_template, request

from src.chat import answer_once, compare_once
from src.config import get_settings


BASE_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = BASE_DIR / "web" / "templates"
STATIC_DIR = BASE_DIR / "web" / "static"

app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)
app.config.setdefault("INDEX_PATH", "storage/index.json")


def _parse_int(value: Any, default: int, field_name: str, minimum: int = 1) -> int:
    if value in (None, ""):
        return default
    parsed = int(value)
    if parsed < minimum:
        raise ValueError(f"{field_name} debe ser >= {minimum}")
    return parsed


def _parse_float(
    value: Any,
    default: float,
    field_name: str,
    minimum: float = -1.0,
    maximum: float = 1.0,
) -> float:
    if value in (None, ""):
        return default
    parsed = float(value)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{field_name} debe estar entre {minimum} y {maximum}")
    return parsed


def _resolve_index_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = BASE_DIR / path
    return path


def _defaults_payload(index_path: str) -> dict[str, Any]:
    settings = get_settings()
    return {
        "index_path": index_path,
        "chat_model": settings.chat_model,
        "cloud_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "top_k": settings.top_k,
        "threshold": settings.similarity_threshold,
        "timeout": 240,
    }


def _extract_runtime_params(payload: dict[str, Any]) -> dict[str, Any]:
    settings = get_settings()

    index_path_raw = str(payload.get("index_path") or app.config["INDEX_PATH"])
    index_path = _resolve_index_path(index_path_raw)

    if not index_path.exists():
        raise FileNotFoundError(
            f"No se encontro el indice en {index_path}. Ejecuta primero: python -m src.ingest"
        )

    return {
        "index_path": index_path,
        "chat_model": str(payload.get("chat_model") or settings.chat_model),
        "cloud_model": str(payload.get("cloud_model") or settings.chat_model),
        "embedding_model": str(payload.get("embedding_model") or settings.embedding_model),
        "top_k": _parse_int(payload.get("top_k"), settings.top_k, "top_k", minimum=1),
        "threshold": _parse_float(
            payload.get("threshold"),
            settings.similarity_threshold,
            "threshold",
            minimum=-1.0,
            maximum=1.0,
        ),
        "timeout": _parse_int(payload.get("timeout"), 240, "timeout", minimum=1),
    }


def _error_response(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _unexpected_error(prefix: str, exc: Exception) -> str:
    details = str(exc).strip()
    if details:
        return f"{prefix}: {details}"
    return prefix


@app.get("/")
def index():
    defaults = _defaults_payload(app.config["INDEX_PATH"])
    return render_template("index.html", defaults=defaults)


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/ask")
def ask_rag():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    if not question:
        return _error_response("La pregunta no puede estar vacia")

    try:
        params = _extract_runtime_params(payload)

        started_at = perf_counter()
        answer, citations = answer_once(
            question=question,
            index_path=params["index_path"],
            top_k=params["top_k"],
            threshold=params["threshold"],
            chat_model=params["chat_model"],
            embedding_model=params["embedding_model"],
            timeout=params["timeout"],
        )
        elapsed = perf_counter() - started_at

        return jsonify(
            {
                "mode": "rag",
                "question": question,
                "rag": {
                    "model": params["chat_model"],
                    "answer": answer,
                    "citations": citations,
                    "seconds": elapsed,
                },
            }
        )
    except (ValueError, FileNotFoundError) as exc:
        return _error_response(str(exc), status_code=400)
    except SystemExit as exc:
        return _error_response(str(exc), status_code=400)
    except Exception as exc:
        app.logger.exception("Error inesperado en /api/ask")
        return _error_response(_unexpected_error("Error interno ejecutando RAG", exc), status_code=500)


@app.post("/api/compare")
def compare_rag_vs_cloud():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    if not question:
        return _error_response("La pregunta no puede estar vacia")

    try:
        params = _extract_runtime_params(payload)

        result = compare_once(
            question=question,
            index_path=params["index_path"],
            top_k=params["top_k"],
            threshold=params["threshold"],
            rag_model=params["chat_model"],
            cloud_model=params["cloud_model"],
            embedding_model=params["embedding_model"],
            timeout=params["timeout"],
        )

        return jsonify(
            {
                "mode": "compare",
                "question": question,
                "rag": {
                    "model": params["chat_model"],
                    "answer": str(result["rag_answer"]),
                    "citations": list(result["citations"]),
                    "seconds": float(result["rag_seconds"]),
                },
                "cloud": {
                    "model": params["cloud_model"],
                    "answer": str(result["cloud_answer"]),
                    "seconds": float(result["cloud_seconds"]),
                },
            }
        )
    except (ValueError, FileNotFoundError) as exc:
        return _error_response(str(exc), status_code=400)
    except SystemExit as exc:
        return _error_response(str(exc), status_code=400)
    except Exception as exc:
        app.logger.exception("Error inesperado en /api/compare")
        return _error_response(
            _unexpected_error("Error interno ejecutando comparativa", exc),
            status_code=500,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run visual frontend for RAG demo")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--index-path", default="storage/index.json", help="Path to RAG index")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app.config["INDEX_PATH"] = args.index_path
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
