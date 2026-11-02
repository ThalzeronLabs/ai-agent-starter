"""Offline stand-ins for `anthropic.Anthropic`, used by the tests and by `python -m agent_starter.evals` (offline).

They implement only what `agent.ask` touches: `client.beta.messages.tool_runner(**kwargs)`, returning an iterable
of messages (or, with `stream=True`, of message streams that yield `text` and `content_block_stop` events and have
`get_final_message()`, like the SDK's `BetaMessageStream`).

- `ScriptedClient` replays messages you hand it. Use it to test the loop plumbing.
- `ExtractiveClient` behaves like a tiny, deterministic "model": it really calls the `search_docs` tool it was given,
  answers with the best-matching sentence of the first passage that covers the question, and attaches a native
  `search_result_location` citation in the API's format. When no passage covers the question it says so and cites
  nothing. It is not a language model, but it exercises retrieval, the tools and the citation pipeline end to end,
  which is what the offline evaluation measures in CI. No network, no API key.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from types import SimpleNamespace

from .retrieval import tokenize

NOT_COVERED = "The documents do not cover this question."


def text_block(text: str, citations: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text, citations=citations)


def search_result_citation(
    source: str, title: str, cited_text: str, search_result_index: int = 0, start_block_index: int = 0,
    end_block_index: int | None = None,
) -> SimpleNamespace:
    """A citation shaped like the API's `search_result_location`."""
    return SimpleNamespace(
        type="search_result_location", source=source, title=title, cited_text=cited_text,
        search_result_index=search_result_index, start_block_index=start_block_index,
        end_block_index=start_block_index + 1 if end_block_index is None else end_block_index,
    )


def tool_use_block(name: str, input: dict | None = None, id: str = "toolu_fake") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input or {}, id=id)


def message(*blocks, stop_reason: str = "end_turn", model: str = "claude-opus-5-5", usage=None) -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop_reason, model=model, usage=usage)


class FakeStream:
    """Mimics the per-turn `BetaMessageStream` the tool runner yields with `stream=True`."""

    def __init__(self, final_message, chunk_size: int = 8):
        self._final = final_message
        self._chunk_size = chunk_size

    def __iter__(self) -> Iterator[SimpleNamespace]:
        for block in self._final.content:
            if block.type == "text":
                for i in range(0, len(block.text), self._chunk_size):
                    yield SimpleNamespace(type="text", text=block.text[i : i + self._chunk_size])
            yield SimpleNamespace(type="content_block_stop", content_block=block)

    @property
    def text_stream(self) -> Iterator[str]:
        return (event.text for event in self if event.type == "text")

    def get_final_message(self):
        return self._final


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._tool_runner))

    def _tool_runner(self, **kwargs) -> Iterator:
        self.calls.append(kwargs)
        messages = self._run(kwargs)
        if kwargs.get("stream"):
            return (FakeStream(m) for m in messages)
        return iter(messages)

    def _run(self, kwargs: dict) -> Iterable:  # pragma: no cover - abstract
        raise NotImplementedError


class ScriptedClient(_FakeClient):
    """Replays `messages` in order, whatever the request was."""

    def __init__(self, messages: list) -> None:
        super().__init__()
        self._messages = messages

    def _run(self, kwargs: dict) -> Iterable:
        return list(self._messages)


_PASSAGE = re.compile(r"^\[(?P<ref>[^\]]+)\](?: — (?P<heading>.*?))? \(score (?P<score>-?[\d.]+)\)\n(?P<text>.*)\Z", re.S)


def parse_search_output(output: str) -> list[dict]:
    """Splits the text form of `search_docs` (as served over MCP) into `{"ref", "heading", "score", "text"}` dicts."""
    passages = []
    for part in output.split("\n\n---\n\n"):
        match = _PASSAGE.match(part.strip())
        if match:
            passages.append({**match.groupdict(), "score": float(match["score"])})
    return passages


def coverage(question: str, passage_text: str) -> float:
    """Share of the question's terms (after stopword removal) that appear in the passage."""
    terms = set(tokenize(question))
    return len(terms & set(tokenize(passage_text))) / len(terms) if terms else 0.0


class ExtractiveClient(_FakeClient):
    """Searches with the real tool, then answers extractively with a native citation.

    Args:
        min_coverage: A passage is used only if it contains at least this share of the question's terms; when no
            passage does, the answer is "not covered" with no citation. It stands in for the judgment a real model
            makes when it reads the search results.
    """

    def __init__(self, min_coverage: float = 0.5) -> None:
        super().__init__()
        self.min_coverage = min_coverage

    def _run(self, kwargs: dict) -> Iterable:
        tools = {tool.name: tool for tool in kwargs["tools"]}
        question = kwargs["messages"][-1]["content"]
        call = {"query": question, "top_k": 4}
        yield message(tool_use_block("search_docs", call), stop_reason="tool_use")
        results = tools["search_docs"].call(call)
        yield message(self.answer(question, results if isinstance(results, list) else []))

    def answer(self, question: str, results: list[dict]) -> SimpleNamespace:
        """The answer block for `question`, given the `search_result` blocks the tool returned (best first)."""
        terms = set(tokenize(question))
        for index, result in enumerate(results):
            if result.get("type") != "search_result":
                continue
            sentences = [block["text"] for block in result["content"]]
            if coverage(question, result["title"] + " " + " ".join(sentences)) < self.min_coverage:
                continue
            best = max(range(len(sentences)), key=lambda i: len(terms & set(tokenize(sentences[i]))))
            citation = search_result_citation(result["source"], result["title"], sentences[best], index, best)
            return text_block(" ".join(sentences[best].split()), [citation])
        return text_block(NOT_COVERED)
