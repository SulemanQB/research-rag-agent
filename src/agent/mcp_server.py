"""MCP server that exposes the document agent's tools over stdio.

Run it with ``python -m src.agent.mcp_server``. The retrieve tool uses the
local Chroma index unless a retriever is passed to ``build_server``.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.mcpserver import MCPServer

from src.agent.tools import (
    DocumentRetriever,
    VectorStoreRetriever,
    calculate,
    env_flag,
    lookup_term,
    search_arxiv,
)


def _dump(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)


def build_server(retriever: DocumentRetriever | None = None) -> MCPServer:
    """Create an MCP server with the agent's four tools registered."""
    corpus: DocumentRetriever = retriever or VectorStoreRetriever()
    server = MCPServer("research-rag-tools")

    @server.tool()
    def retrieve_corpus(query: str, top_k: int = 3) -> str:
        """Search the local research corpus and return matching passages with source metadata."""
        bounded = top_k if top_k > 0 else 3
        passages = corpus.retrieve(query, top_k=bounded)
        return _dump({"query": query, "passages": passages})

    @server.tool()
    def calculate_expression(expression: str) -> str:
        """Evaluate a safe arithmetic expression. Names, calls, and large exponents are rejected."""
        return _dump(calculate(expression))

    @server.tool()
    def lookup_glossary_term(term: str) -> str:
        """Look up a structured glossary entry for an AI risk-management term."""
        return _dump(lookup_term(term))

    @server.tool()
    def search_arxiv_papers(query: str, max_results: int = 3) -> str:
        """Search arXiv. Returns a disabled message unless AGENT_WEB_SEARCH=1."""
        return _dump(search_arxiv(query, enabled=env_flag("AGENT_WEB_SEARCH", default=False), max_results=max_results))

    return server


def main() -> None:
    """Serve the tools on stdio until the client disconnects."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
