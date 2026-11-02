"""An optional MCP server exposing the agent's tools (search_docs, read_document, calculate) to any MCP client.

Install the extra and point your MCP client (Claude Code, Claude Desktop, an IDE, ...) at it over stdio:

    pip install "agent-starter[mcp]"
    python -m agent_starter.mcp_server --docs ./docs-example

The tool names, descriptions and behavior are the same ones the built-in agent uses, so a document folder can be
searched by Claude through this starter's own loop or through any other MCP-capable host.
"""

# No `from __future__ import annotations` here: the MCP SDK reads the `Annotated[...]` parameter metadata of the
# tool functions below, which must be real objects, not strings.
import argparse
import sys
from pathlib import Path
from typing import Annotated

try:
    from mcp.server.mcpserver import MCPServer
    from mcp.types import ToolAnnotations
    from pydantic import Field
except ImportError as error:  # pragma: no cover - exercised only without the extra installed
    raise ImportError('The MCP server needs the optional extra: pip install "agent-starter[mcp]"') from error

from . import __version__
from .retrieval import RETRIEVERS, Retriever, build_retriever
from .tools import build_tools, calculate_impl, read_document_impl, search_docs_impl

SERVER_NAME = "agent-starter-docs"

_READ_ONLY = ToolAnnotations(readOnlyHint=True, destructiveHint=False, idempotentHint=True, openWorldHint=False)


def create_server(index: Retriever, name: str = SERVER_NAME) -> MCPServer:
    """An MCP server whose tools search `index`.

    Tool and parameter descriptions are copied from the agent's own `@beta_tool` definitions, so the two surfaces
    cannot drift apart.
    """
    specs = {tool.name: tool.to_dict() for tool in build_tools(index)}
    descriptions = {name: spec["description"] for name, spec in specs.items()}

    def arg(tool: str, param: str):
        return Field(description=specs[tool]["input_schema"]["properties"][param]["description"])

    server = MCPServer(
        name,
        version=__version__,
        instructions="Search the user's documents with search_docs and cite passages by their [source#n] tag.",
    )

    @server.tool(description=descriptions["search_docs"], annotations=_READ_ONLY)
    def search_docs(
        query: Annotated[str, arg("search_docs", "query")],
        top_k: Annotated[int, arg("search_docs", "top_k")] = 4,
    ) -> str:
        return search_docs_impl(index, query, top_k)

    @server.tool(description=descriptions["read_document"], annotations=_READ_ONLY)
    def read_document(source: Annotated[str, arg("read_document", "source")]) -> str:
        return read_document_impl(index, source)

    @server.tool(description=descriptions["calculate"], annotations=_READ_ONLY)
    def calculate(expression: Annotated[str, arg("calculate", "expression")]) -> str:
        return calculate_impl(expression)

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_starter.mcp_server", description="Serve document tools over MCP (stdio).")
    parser.add_argument("--docs", default="docs-example", help="Folder with .md/.txt/.rst files (default: docs-example).")
    parser.add_argument("--retriever", default="bm25", choices=RETRIEVERS, help="Search backend (default: bm25).")
    parser.add_argument("--max-chars", type=int, default=900, help="Soft chunk size (minimum 100). Evaluation citation tags assume the default 900.")
    args = parser.parse_args(argv)
    if args.max_chars < 100:
        parser.error("--max-chars must be at least 100")

    index = build_retriever(Path(args.docs), args.retriever, max_chars=args.max_chars)
    # stdout carries the MCP protocol over stdio, so diagnostics go to stderr.
    print(f"Serving {len(index.documents)} documents ({len(index.chunks)} chunks) over MCP stdio.", file=sys.stderr)
    create_server(index).run("stdio")
    return 0


if __name__ == "__main__":
    sys.exit(main())

