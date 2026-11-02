"""The agent's tools. Each one is a plain function (easy to unit-test) wrapped with the SDK's `@beta_tool`,
which builds the JSON schema from the signature and the docstring.

The agent's search tools return passages as `search_result` content blocks, so the API attaches native, checkable
citations to the answer (see `citations.py`). The `*_impl` text versions are kept for the MCP server, whose clients
expect plain text.
"""

from __future__ import annotations

import ast
import operator

from anthropic import beta_tool

from .citations import split_sentences
from .retrieval import Chunk, Retriever

MAX_DOCUMENT_CHARS = 20_000

NO_MATCH = "No passage in the documents matches that query. Try other words, or tell the user it is not covered."

_OPERATORS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def search_docs_impl(index: Retriever, query: str, top_k: int = 4) -> str:
    hits = index.search(query, k=max(1, min(top_k, 8)))
    if not hits:
        return NO_MATCH
    parts = []
    for chunk, score in hits:
        title = f" — {chunk.heading}" if chunk.heading else ""
        parts.append(f"[{chunk.ref}]{title} (score {score:.2f})\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def search_result_block(chunk: Chunk) -> dict:
    """One chunk as a `search_result` content block: one text block per sentence, citations enabled."""
    return {
        "type": "search_result",
        "source": chunk.ref,
        "title": chunk.heading or chunk.source,
        "content": [{"type": "text", "text": sentence} for sentence in split_sentences(chunk.text)],
        "citations": {"enabled": True},
    }


def search_results_impl(index: Retriever, query: str, top_k: int = 4) -> str | list[dict]:
    """The best passages for `query` as `search_result` blocks, best first; a plain message when nothing matches."""
    hits = index.search(query, k=max(1, min(top_k, 8)))
    return [search_result_block(chunk) for chunk, _ in hits] if hits else NO_MATCH


def document_results_impl(index: Retriever, source: str) -> str | list[dict]:
    """A whole document as `search_result` blocks, one per chunk, so it can be cited like a search result."""
    if source not in index.documents:
        available = ", ".join(sorted(index.documents)) or "(none)"
        return f"There is no document named {source!r}. Available: {available}"
    blocks, size = [], 0
    for chunk in (c for c in index.chunks if c.source == source):
        size += len(chunk.text)
        if size > MAX_DOCUMENT_CHARS:
            blocks.append({"type": "text", "text": f"[truncated: the document is longer than {MAX_DOCUMENT_CHARS} characters]"})
            break
        blocks.append(search_result_block(chunk))
    return blocks or f"The document {source!r} is empty."


def read_document_impl(index: Retriever, source: str) -> str:
    text = index.documents.get(source)
    if text is None:
        available = ", ".join(sorted(index.documents)) or "(none)"
        return f"There is no document named {source!r}. Available: {available}"
    if len(text) > MAX_DOCUMENT_CHARS:
        return text[:MAX_DOCUMENT_CHARS] + f"\n\n[truncated: showing {MAX_DOCUMENT_CHARS} of {len(text)} characters]"
    return text


def calculate_impl(expression: str) -> str:
    """Evaluates arithmetic safely: numbers and + - * / // % ** only. No names, calls or attributes."""
    def walk(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPERATORS:
            left, right = walk(node.left), walk(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100:
                raise ValueError("exponent too large")
            return _OPERATORS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPERATORS:
            return _OPERATORS[type(node.op)](walk(node.operand))
        raise ValueError("only numbers and + - * / // % ** are allowed")

    try:
        result = walk(ast.parse(expression, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (SyntaxError, ValueError) as error:
        return f"Error: {error}"
    return str(int(result)) if isinstance(result, float) and result.is_integer() else str(result)


def build_tools(index: Retriever) -> list:
    """The tool list handed to the tool runner, bound to one document index."""

    @beta_tool
    def search_docs(query: str, top_k: int = 4) -> str | list[dict]:
        """Search the user's documents and return the most relevant passages, each labelled with its source tag.

        Use this before answering any question about the documents. Search again with different words
        if the first results do not answer the question.

        Args:
            query: Keywords or a short question describing what to find.
            top_k: How many passages to return (1-8).
        """
        return search_results_impl(index, query, top_k)

    @beta_tool
    def read_document(source: str) -> str | list[dict]:
        """Read one whole document, when the passages from search_docs are not enough context.

        Args:
            source: The document name exactly as it appears in a citation tag, without the #number.
        """
        return document_results_impl(index, source)

    @beta_tool
    def calculate(expression: str) -> str:
        """Evaluate an arithmetic expression exactly. Use it for any calculation instead of doing math in your head.

        Args:
            expression: Arithmetic only, e.g. "1299 * 0.85" or "(30 - 7) / 2".
        """
        return calculate_impl(expression)

    return [search_docs, read_document, calculate]
