from pathlib import Path

import pytest

from agent_starter.retrieval import BM25Index, chunk_document, load_folder, tokenize

EXAMPLE_DOCS = Path(__file__).resolve().parents[1] / "docs-example"


def test_tokenize_lowercases_folds_accents_and_drops_stopwords():
    assert tokenize("Como funciona a Cancelação do plano?") == ["funciona", "cancelacao", "plano"]
    assert tokenize("The REFUND window, in 14 days") == ["refund", "window", "14", "days"]


def test_chunking_follows_headings_and_keeps_them_as_context():
    text = "# Title\n\nIntro line.\n\n## Refunds\n\nWithin 14 days.\n\n## Security\n\nTLS 1.3."
    chunks = chunk_document("doc.md", text)
    assert [c.heading for c in chunks] == ["Title", "Refunds", "Security"]
    assert [c.ref for c in chunks] == ["doc.md#0", "doc.md#1", "doc.md#2"]
    assert chunks[1].text == "Within 14 days."


def test_long_sections_are_split_into_bounded_chunks():
    paragraphs = "\n\n".join(f"Paragraph {i} " + "word " * 40 for i in range(10))
    chunks = chunk_document("long.md", "# Big\n\n" + paragraphs, max_chars=500)
    assert len(chunks) > 1
    assert all(c.heading == "Big" for c in chunks)
    assert all(len(c.text) <= 500 + 260 for c in chunks)  # one paragraph may overflow the soft limit


def test_bm25_ranks_the_section_that_answers_the_question_first():
    index = BM25Index.from_folder(EXAMPLE_DOCS)
    top_chunk, _ = index.search("refund annual plan days", k=1)[0]
    assert top_chunk.source == "refunds.md"
    assert top_chunk.heading == "Refund window"


def test_bm25_is_accent_and_case_insensitive():
    index = BM25Index({"a.md": "# Cancelação\n\nComo cancelar o plano."})
    assert index.search("CANCELACAO")
    assert index.search("cancelação")


def test_queries_with_no_shared_terms_return_nothing():
    index = BM25Index.from_folder(EXAMPLE_DOCS)
    assert index.search("quantum chromodynamics") == []
    assert index.search("the of and") == []  # only stopwords


def test_bm25_min_score_drops_incidental_term_matches():
    """A lone shared word can score just above 0; min_score filters those weak hits (#2)."""
    docs = {
        "security.md": "# Authenticator app\n\nUse the authenticator app for 2FA.",
        "refunds.md": "# Refunds\n\nRefunds within 14 days of purchase.",
    }
    loose = BM25Index(docs)
    assert any(c.source == "security.md" for c, _ in loose.search("Is there a dark mode in the iOS app?"))

    # On this tiny corpus the incidental "app" hit is ~0.99; a real refund match is ~1.39.
    strict = BM25Index(docs, min_score=1.0)
    assert strict.search("Is there a dark mode in the iOS app?") == []
    assert strict.search("refund purchase days")


def test_bm25_min_score_fixes_example_docs_dark_mode_gap():
    """On docs-example, not-covered-dark-mode tops out ~2.26 while covered hits stay ≥ ~3.8."""
    loose = BM25Index.from_folder(EXAMPLE_DOCS)
    assert any(c.ref == "security.md#1" for c, _ in loose.search("Is there a dark mode in the iOS app?"))

    strict = BM25Index.from_folder(EXAMPLE_DOCS, min_score=3.0)
    assert strict.search("Is there a dark mode in the iOS app?") == []
    top_chunk, score = strict.search("refund annual plan days", k=1)[0]
    assert top_chunk.ref == "refunds.md#1" and score > 3.0


def test_bm25_default_min_score_keeps_previous_behaviour():
    index = BM25Index.from_folder(EXAMPLE_DOCS)
    assert index.min_score == 0.0
    assert index.search("refund annual plan days")


def test_k_limits_results_and_scores_are_sorted():
    index = BM25Index.from_folder(EXAMPLE_DOCS)
    hits = index.search("plan", k=3)
    assert len(hits) <= 3
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)


def test_empty_index_is_safe():
    assert BM25Index({}).search("anything") == []


def test_load_folder_reads_only_text_documents(tmp_path):
    (tmp_path / "a.md").write_text("# A", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("B", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    assert sorted(load_folder(tmp_path)) == ["a.md", "sub/b.txt"]


def test_missing_folder_raises():
    with pytest.raises(FileNotFoundError):
        load_folder("does-not-exist")
