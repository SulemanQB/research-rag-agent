from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.vector_store import LocalVectorStore


def load_questions(path: str | Path | None = None) -> List[Dict[str, Any]]:
    default_path = Path(__file__).resolve().parent / "questions.json"
    target = Path(path) if path else default_path
    if not target.exists():
        return []
    data = json.loads(target.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        if data.get("status") == "NOT_RUN":
            return []
        return data.get("questions", [])
    return data


def run_evaluation(path: str | Path | None = None, top_k: int = 3) -> Dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    questions = load_questions(path)
    if not questions:
        print("Evaluation status: NOT_RUN")
        print("The evaluation framework is present, but the actual corpus-backed question set is not ready to run.")
        return {
            "status": "NOT_RUN",
            "message": "The evaluation framework is present, but the actual corpus-backed question set is not ready to run.",
            "questions": 0,
            f"top_{top_k}_hits": 0,
            f"top_{top_k}_rate": 0.0,
        }

    corpus_dir = Path(__file__).resolve().parents[1] / "data" / "corpus"
    supported_files = [
        path for path in corpus_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt", ".md"}
        and path.name.lower() != "readme.md"
    ]
    if not supported_files:
        print("Evaluation status: NOT_RUN")
        print("No local corpus files were found under data/corpus, so evaluation could not run.")
        return {
            "status": "NOT_RUN",
            "message": "No local corpus files were found under data/corpus, so evaluation could not run.",
            "questions": len(questions),
            f"top_{top_k}_hits": 0,
            f"top_{top_k}_rate": 0.0,
        }

    real_questions = [item for item in questions if not str(item.get("expected_document", "")).startswith("replace_")]
    if not real_questions:
        print("Evaluation status: NOT_RUN")
        print("The evaluation file contains placeholder expected documents and is not yet aligned with a real corpus.")
        return {
            "status": "NOT_RUN",
            "message": "The evaluation file contains placeholder expected documents and is not yet aligned with a real corpus.",
            "questions": len(questions),
            f"top_{top_k}_hits": 0,
            f"top_{top_k}_rate": 0.0,
        }

    questions = real_questions

    vector_store = LocalVectorStore()
    hits = 0
    results: List[str] = []
    for item in questions:
        query = item["question"]
        expected_doc = item["expected_document"]
        response = vector_store.query(query, top_k=top_k)
        metas = response.get("metadatas", [[]])[0]
        top_sources = list(dict.fromkeys(meta.get("source") or meta.get("filename") or "unknown" for meta in metas))
        results.append(f"Expected document: {expected_doc}; Top-{top_k} documents: {top_sources}")
        expected_pages = set(item.get("expected_pages", []))
        page_match = not expected_pages or any(
            meta.get("page") in expected_pages
            for meta in metas
            if (meta.get("source") or meta.get("filename")) == expected_doc
        )
        if expected_doc in top_sources and page_match:
            hits += 1

    rate = (hits / len(questions)) * 100 if questions else 0.0
    print("Retrieval evaluation")
    print("--------------------")
    print(f"Questions: {len(questions)}")
    print(f"Top-{top_k} hits: {hits}/{len(questions)}")
    print(f"Top-{top_k} retrieval rate: {rate:.0f}%")
    for entry in results:
        print(entry)

    return {
        "status": "OK",
        "questions": len(questions),
        f"top_{top_k}_hits": hits,
        f"top_{top_k}_rate": rate,
    }
