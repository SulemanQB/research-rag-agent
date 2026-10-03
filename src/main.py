from __future__ import annotations

import argparse
from typing import Any

from src.pipeline import answer_question, run_ingestion


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _print_query_result(result: dict[str, Any]) -> None:
    print("\nROUTE")
    print("-----")
    print(f"Route: {result['route']}")
    print(f"Reason: {result['reason']}")
    if result.get("sources"):
        print("\nRetrieved sources:")
        for idx, source in enumerate(result["sources"], start=1):
            print(f"{idx}. {source}")
    print("\nAnswer:")
    print(result["answer"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Research RAG agent")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest the local corpus into Chroma")
    ingest_parser.add_argument("--reset", action="store_true", help="Rebuild the vector collection before indexing")

    query_parser = subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("question", nargs="+", help="Question to ask")
    query_parser.add_argument("--top-k", type=_positive_int, default=3, help="Number of relevant chunks to retrieve")

    chat_parser = subparsers.add_parser("chat", help="Interactive query loop")
    chat_parser.add_argument("--top-k", type=_positive_int, default=3, help="Number of relevant chunks to retrieve")

    evaluate_parser = subparsers.add_parser("evaluate", help="Run informal retrieval evaluation")

    args = parser.parse_args()

    if args.command == "ingest":
        run_ingestion(reset=args.reset)
        return

    if args.command == "query":
        question = " ".join(args.question)
        result = answer_question(question, top_k=args.top_k)
        _print_query_result(result)
        return

    if args.command == "chat":
        print("RAG Research Assistant")
        print("======================")
        while True:
            question = input("\nQuestion: ").strip()
            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break
            result = answer_question(question, top_k=args.top_k)
            _print_query_result(result)
        return

    if args.command == "evaluate":
        from evaluation.evaluate import run_evaluation

        result = run_evaluation()
        print(result)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
