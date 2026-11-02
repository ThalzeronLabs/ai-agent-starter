"""Command line: python -m agent_starter --docs ./docs-example "How do refunds work?" """

from __future__ import annotations

import argparse
import sys

import anthropic

from .agent import DEFAULT_MODEL, ask
from .citations import Footnotes, render
from .retrieval import RETRIEVERS, build_retriever


def main(argv: list[str] | None = None, client: anthropic.Anthropic | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_starter", description="Ask questions about a folder of documents.")
    parser.add_argument("question", help="Your question, in any language.")
    parser.add_argument("--docs", default="docs-example", help="Folder with .md/.txt/.rst files (default: docs-example).")
    parser.add_argument("--retriever", default="bm25", choices=RETRIEVERS,
                        help="bm25 (default, no dependencies) or embeddings (local model, needs the [embeddings] extra).")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="medium", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=True,
                        help="Print the answer while it arrives (default) or all at once (--no-stream).")
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                        help="Use prompt caching (default) or not (--no-cache).")
    parser.add_argument("--show-tools", action="store_true", help="Print which tools the agent used.")
    parser.add_argument("--show-usage", action="store_true", help="Print token usage, including cache reads/writes.")
    parser.add_argument("--offline", action="store_true",
                        help="No API call: answer with the deterministic extractive stand-in used by the tests "
                             "(not a language model), to try the pipeline without credentials.")
    parser.add_argument("--max-chars", type=int, default=900, help="Soft chunk size (minimum 100). Evaluation citation tags assume the default 900.")
    args = parser.parse_args(argv)
    if args.max_chars < 100:
        parser.error("--max-chars must be at least 100")

    try:
        index = build_retriever(args.docs, args.retriever, max_chars=args.max_chars)
    except (FileNotFoundError, ImportError) as error:
        print(error, file=sys.stderr)
        return 2
    print(f"Indexed {len(index.documents)} documents into {len(index.chunks)} chunks.", file=sys.stderr)

    if args.offline and client is None:
        from .fakes import ExtractiveClient

        client = ExtractiveClient()
        print("Offline: answers come from the extractive stand-in, not from Claude.", file=sys.stderr)
    footnotes = Footnotes()

    def on_text(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    def on_citations(citations) -> None:
        on_text(footnotes.markers(citations))

    try:
        result = ask(args.question, index, client=client, model=args.model, effort=args.effort,
                     stream=args.stream, on_text=on_text, on_citations=on_citations, cache=args.cache)
    except anthropic.AuthenticationError:
        print("No valid Anthropic credentials. Set ANTHROPIC_API_KEY or run `ant auth login`.", file=sys.stderr)
        return 2
    except anthropic.APIConnectionError:
        print("\nCould not reach the Anthropic API. Check your connection.", file=sys.stderr)
        return 3

    if args.stream:
        print()  # end the streamed line
        if footnotes.sources():
            print(f"\nSources:\n{footnotes.sources()}")
        if result.refused:
            print(result.answer, file=sys.stderr)
    else:
        print(render(result.parts) if result.parts else result.answer)
    if args.show_tools:
        print(f"Tools used: {', '.join(result.tool_calls) or 'none'}", file=sys.stderr)
    if args.show_usage:
        print("Usage: " + ", ".join(f"{name}={value}" for name, value in result.usage.items()), file=sys.stderr)
    return 0 if not result.refused else 1


if __name__ == "__main__":
    sys.exit(main())
