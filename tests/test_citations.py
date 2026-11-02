"""Native citations: search_result blocks out, structured citations back, rendered as footnotes. All offline."""

from pathlib import Path
from types import SimpleNamespace

from agent_starter.agent import ask
from agent_starter.citations import (
    AnswerPart,
    Citation,
    Footnotes,
    citations_of,
    parts_of,
    render,
    split_sentences,
)
from agent_starter.fakes import ScriptedClient, message, search_result_citation, text_block, tool_use_block
from agent_starter.retrieval import BM25Index
from agent_starter.tools import NO_MATCH, build_tools

INDEX = BM25Index.from_folder(Path(__file__).resolve().parents[1] / "docs-example")
TOOLS = {tool.name: tool for tool in build_tools(INDEX)}
WINDOW = Citation("refunds.md#1", "Refund window", "Annual plans can be refunded in full within 14 days of purchase or renewal.")
EMAIL = Citation("refunds.md#2", "How to ask for a refund", "Write to billing@northwind-notes.example.")


# --- what the tools send -----------------------------------------------------------------------------------------


def test_sentences_concatenate_back_to_the_original_text():
    text = "Annual plans can be refunded. After 14 days, no.\n\nMonthly plans are not refunded!"
    sentences = split_sentences(text)
    assert len(sentences) == 3
    assert "".join(sentences) == text


def test_search_docs_returns_search_result_blocks_with_citations_enabled():
    results = TOOLS["search_docs"].call({"query": "refund annual plan days", "top_k": 2})
    first = results[0]
    assert first["type"] == "search_result"
    assert first["source"] == "refunds.md#1" and first["title"] == "Refund window"
    assert first["citations"] == {"enabled": True}
    chunk = next(c for c in INDEX.chunks if c.ref == "refunds.md#1")
    assert "".join(block["text"] for block in first["content"]) == chunk.text
    assert len(first["content"]) > 1  # sentence-sized blocks, so citations can point at one sentence
    assert all(r["citations"] == {"enabled": True} for r in results)  # the API needs the same setting on all


def test_search_docs_without_matches_says_so():
    assert TOOLS["search_docs"].call({"query": "submarine"}) == NO_MATCH


def test_read_document_returns_one_citable_block_per_chunk():
    blocks = TOOLS["read_document"].call({"source": "refunds.md"})
    assert [b["source"] for b in blocks] == ["refunds.md#0", "refunds.md#1", "refunds.md#2"]
    assert "There is no document" in TOOLS["read_document"].call({"source": "nope.md"})


# --- what the agent gets back ------------------------------------------------------------------------------------


def test_only_search_result_citations_are_read():
    other = SimpleNamespace(type="char_location", cited_text="x", document_title="d")
    block = text_block("14 days.", [search_result_citation(WINDOW.ref, WINDOW.title, WINDOW.cited_text), other])
    assert citations_of(block) == (WINDOW,)
    assert citations_of(text_block("no citations")) == ()
    assert parts_of(message(tool_use_block("search_docs"), block)) == [AnswerPart("14 days.", (WINDOW,))]


def test_ask_exposes_the_answer_parts_and_their_citations():
    final = message(
        text_block("Within 14 days", [search_result_citation(WINDOW.ref, WINDOW.title, WINDOW.cited_text)]),
        text_block(", by email", [search_result_citation(EMAIL.ref, EMAIL.title, EMAIL.cited_text)]),
        text_block(". Thanks."),
    )
    result = ask("Refund?", INDEX, client=ScriptedClient([final]))
    assert result.answer == "Within 14 days, by email. Thanks."
    assert result.citations == [WINDOW, EMAIL]
    assert [len(p.citations) for p in result.parts] == [1, 1, 0]


def test_streaming_reports_each_blocks_citations_when_the_block_ends():
    final = message(
        text_block("Within 14 days.", [search_result_citation(WINDOW.ref, WINDOW.title, WINDOW.cited_text)]),
        text_block(" Thanks."),
    )
    events = []
    result = ask("Refund?", INDEX, client=ScriptedClient([final]), stream=True,
                 on_text=lambda t: events.append(("text", t)), on_citations=lambda c: events.append(("cite", c)))
    cite_at = events.index(("cite", (WINDOW,)))
    assert "".join(t for kind, t in events[:cite_at] if kind == "text") == "Within 14 days."
    assert [kind for kind, _ in events].count("cite") == 1
    assert result.citations == [WINDOW]


def test_refusals_have_no_parts():
    result = ask("...", INDEX, client=ScriptedClient([message(text_block("x"), stop_reason="refusal")]))
    assert result.parts == [] and result.citations == []


# --- how they are shown ------------------------------------------------------------------------------------------


def test_footnotes_number_in_order_of_first_use_and_reuse_numbers():
    notes = Footnotes()
    assert notes.markers([WINDOW]) == "[1]"
    assert notes.markers([EMAIL, WINDOW]) == "[2][1]"
    assert notes.markers([WINDOW, WINDOW]) == "[1]"
    assert notes.sources().splitlines() == [
        '[1] refunds.md#1 (Refund window): "Annual plans can be refunded in full within 14 days of purchase or renewal."',
        '[2] refunds.md#2 (How to ask for a refund): "Write to billing@northwind-notes.example."',
    ]


def test_render_adds_markers_and_sources_only_when_there_are_citations():
    parts = [AnswerPart("Within 14 days", (WINDOW,)), AnswerPart(", by email", (EMAIL,)), AnswerPart(".")]
    assert render(parts).splitlines()[0] == "Within 14 days[1], by email[2]."
    assert "Sources:" in render(parts)
    assert render([AnswerPart("Not covered.")]) == "Not covered."
