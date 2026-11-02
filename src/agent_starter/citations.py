"""Native, verifiable citations.

The search tools return each passage as a `search_result` content block (source = the chunk tag, e.g.
`refunds.md#1`; one text block per sentence; citations enabled). The Claude API then attaches structured citations
to the text blocks of its answer: each one names the search result it comes from and carries `cited_text`, the exact
sentences it relies on (https://platform.claude.com/docs/en/build-with-claude/search-results).

So a citation is no longer a tag the model types and we parse back: it is data the API returns, and `cited_text`
can be checked against the document. `agent_starter.evals` does exactly that.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?=\s)")


@dataclass(frozen=True)
class Citation:
    ref: str
    """The cited chunk's tag, e.g. `refunds.md#1` (the `source` of its search result)."""

    title: str
    """The chunk's heading, or the document name when it has none."""

    cited_text: str
    """The exact text the answer relies on, as returned by the API."""


@dataclass(frozen=True)
class AnswerPart:
    """One text block of the answer and the citations the API attached to it."""

    text: str
    citations: tuple[Citation, ...] = ()


def split_sentences(text: str) -> list[str]:
    """Splits text into sentence-sized blocks that concatenate back to exactly `text`.

    Text blocks are the smallest unit the API can cite, so sentence-sized blocks make `cited_text` point at the
    sentence that supports a claim instead of the whole passage.
    """
    return [piece for piece in _SENTENCE_BOUNDARY.split(text) if piece]


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace, for comparing quoted text with the documents."""
    return " ".join(text.split()).lower()


def citations_of(block) -> tuple[Citation, ...]:
    """The search-result citations of one API text block (other citation kinds are ignored)."""
    found = []
    for citation in getattr(block, "citations", None) or ():
        if getattr(citation, "type", None) == "search_result_location":
            found.append(Citation(ref=citation.source, title=citation.title or "", cited_text=citation.cited_text))
    return tuple(found)


def parts_of(message) -> list[AnswerPart]:
    """The text blocks of an API message, with their citations."""
    return [AnswerPart(block.text, citations_of(block)) for block in message.content if block.type == "text"]


def unique_citations(parts: Iterable[AnswerPart]) -> list[Citation]:
    """Every citation in the answer, in order of first appearance, without duplicates."""
    return list(dict.fromkeys(citation for part in parts for citation in part.citations))


class Footnotes:
    """Numbers citations in order of first use, so an answer reads `... 14 days [1].` with sources listed below."""

    def __init__(self) -> None:
        self._numbers: dict[Citation, int] = {}

    def markers(self, citations: Iterable[Citation]) -> str:
        numbers = []
        for citation in citations:
            if citation not in self._numbers:
                self._numbers[citation] = len(self._numbers) + 1
            numbers.append(self._numbers[citation])
        return "".join(f"[{n}]" for n in dict.fromkeys(numbers))

    def sources(self) -> str:
        """The numbered source list, one line per citation: number, tag, heading and the quoted text."""
        lines = []
        for citation, number in self._numbers.items():
            quote = " ".join(citation.cited_text.split())
            lines.append(f'[{number}] {citation.ref} ({citation.title}): "{quote}"')
        return "\n".join(lines)


def render(parts: Iterable[AnswerPart]) -> str:
    """The answer with footnote markers after each cited block, followed by its sources."""
    footnotes = Footnotes()
    text = "".join(part.text + footnotes.markers(part.citations) for part in parts).strip()
    sources = footnotes.sources()
    return f"{text}\n\nSources:\n{sources}" if sources else text
