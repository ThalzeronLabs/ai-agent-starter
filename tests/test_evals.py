from pathlib import Path

import pytest

from agent_starter.evals import (
    EvalCase,
    extract_citations,
    load_cases,
    main,
    run_eval,
    score_case,
)
from agent_starter.fakes import ExtractiveClient, parse_search_output
from agent_starter.retrieval import BM25Index
from agent_starter.tools import search_docs_impl

ROOT = Path(__file__).resolve().parents[1]
INDEX = BM25Index.from_folder(ROOT / "docs-example")
CASES = load_cases(ROOT / "evals" / "cases.jsonl")
CASE = EvalCase("t", "Where is data stored?", ["security.md#2"], ["Frankfurt"])


def test_selected_cases_and_markdown(capsys):
    import json

    args = ["--cases", str(ROOT / "evals/cases.jsonl"), "--docs", str(ROOT / "docs-example")]
    selected = CASES[0].id
    assert main([*args, "--only", selected, "--json", "--min-recall", "0", "--min-precision", "0"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [result["id"] for result in report["results"]] == [selected]
    assert report["summary"]["cases"] == 1
    main([*args, "--only", selected, "--markdown"])
    assert "| case | pass | P | R | cited |" in capsys.readouterr().out
    with pytest.raises(SystemExit) as error:
        main([*args, "--only", "unknown"])
    assert error.value.code == 2
    assert "Valid IDs:" in capsys.readouterr().err


# --- the case file itself --------------------------------------------------------------------------------------


def test_case_file_is_well_formed():
    assert len(CASES) >= 10
    assert len({c.id for c in CASES}) == len(CASES)
    assert any(not c.expected_citations for c in CASES), "keep at least one not-covered case"


def test_every_expected_citation_exists_and_contains_its_facts():
    chunks = {c.ref: c for c in INDEX.chunks}
    for case in CASES:
        for ref in case.expected_citations:
            assert ref in chunks, f"{case.id}: {ref} is not a chunk of docs-example"
        cited_text = " ".join(chunks[ref].text.lower() for ref in case.expected_citations)
        for fact in case.expected_facts:
            assert fact.lower() in cited_text, f"{case.id}: {fact!r} is not in {case.expected_citations}"


def test_invalid_case_lines_are_reported_with_their_position(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "a", "question": "q", "expected_citations": []}\n{"oops": 1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="cases.jsonl:2"):
        load_cases(path)


# --- scoring ---------------------------------------------------------------------------------------------------


def test_extract_citations_dedupes_and_keeps_order():
    answer = "A [b.md#2] and [a.md#0], again [b.md#2]. Not a tag: [note] or [x.md#y]."
    assert extract_citations(answer) == ["b.md#2", "a.md#0"]


def test_a_correct_cited_answer_passes():
    result = score_case(CASE, "Data lives in Frankfurt [security.md#2].", INDEX)
    assert (result.precision, result.recall, result.valid, result.supported, result.facts) == (1, 1, True, True, 1)
    assert result.passed


def test_citing_the_wrong_passage_is_not_supported():
    result = score_case(CASE, "Data lives in Frankfurt [security.md#0].", INDEX)
    assert result.recall == 0 and not result.supported and not result.passed


def test_invented_citations_are_invalid():
    result = score_case(CASE, "Frankfurt [security.md#2] [made-up.md#9].", INDEX)
    assert result.recall == 1 and result.precision == 0.5 and not result.valid and not result.passed


def test_missing_facts_fail_even_with_the_right_citation():
    result = score_case(CASE, "It is in Europe [security.md#2].", INDEX)
    assert result.facts == 0 and not result.passed


def test_not_covered_cases_must_cite_nothing():
    case = EvalCase("n", "Dark mode?", [], [])
    assert score_case(case, "The documents do not cover this.", INDEX).passed
    assert not score_case(case, "Yes [plans.md#0].", INDEX).passed


# --- the offline fake and the full run -------------------------------------------------------------------------


def test_search_output_round_trips_through_the_parser():
    passages = parse_search_output(search_docs_impl(INDEX, "students discount"))
    assert passages[0]["ref"] == "plans.md#4" and passages[0]["heading"] == "Discounts"
    assert passages[0]["score"] >= passages[-1]["score"]
    assert parse_search_output(search_docs_impl(INDEX, "submarine")) == []


def test_offline_eval_meets_the_ci_thresholds():
    summary = run_eval(CASES, INDEX, ExtractiveClient()).summary
    assert summary["valid_citations"] == 1.0
    assert summary["citation_recall"] >= 0.9
    assert summary["citation_precision"] >= 0.8
    assert summary["supported"] >= 0.9


def test_offline_eval_is_identical_with_streaming():
    plain = run_eval(CASES, INDEX, ExtractiveClient())
    streamed = run_eval(CASES, INDEX, ExtractiveClient(), stream=True)
    assert [r.answer for r in plain.results] == [r.answer for r in streamed.results]


def test_eval_cli_exit_codes(capsys):
    args = ["--cases", str(ROOT / "evals" / "cases.jsonl"), "--docs", str(ROOT / "docs-example")]
    assert main(args) == 0
    assert "citation_recall" in capsys.readouterr().out
    assert main([*args, "--min-recall", "1.01"]) == 1
    assert "Below threshold" in capsys.readouterr().err
    assert main([*args, "--json"]) == 0
    assert '"summary"' in capsys.readouterr().out
