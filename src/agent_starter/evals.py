"""Offline evaluation: do the answers cite the passages that actually contain the answer?

A case file is JSON Lines, one case per line:

    {"id": "refund-window", "question": "...", "expected_citations": ["refunds.md#1"], "expected_facts": ["14 days"]}

`expected_citations` lists the chunk tags a correct answer must cite (empty when the documents do not cover the
question, in which case the answer must cite nothing). `expected_facts` are short strings that a correct answer
contains (case-insensitive).

For every case the evaluator asks the agent and reads the answer's citations: the native ones the API attaches
(`search_result_location`, with the exact `cited_text`), or, when an answer has none, `[source#n]` tags typed in the
text. It scores:

- citation precision and recall against `expected_citations`;
- validity: every citation names a chunk that exists in the index, and every native `cited_text` is text that chunk
  really contains (no invented citations, no invented quotes);
- support: every expected fact appears in the answer *and* in the text of a cited chunk;
- grounding: every expected fact appears in the `cited_text` of a citation attached to the very sentence that states
  it. This is the check typed tags cannot give: a tag only says which chunk, so a claim that cites the right chunk
  but contradicts it still passes. With tags, grounding falls back to support;
- facts: the share of expected facts present in the answer.

A not-covered case (`expected_citations: []`) passes only if the answer cites nothing. Those cases are mandatory:
the run fails if any of them fails, whatever the thresholds.

Native citations are required too: the report shows each case's citation `mode` (native, tags or none), and the
run fails if a case the documents cover is answered without native citations, because the typed-tag fallback cannot
check claims and would hide that regression. `--allow-tags` lifts this for clients that cannot return them.

What the offline run measures: its answers come from a deterministic stand-in whose not-covered rule was tuned on
these cases, so its result is a baseline for the retrieval, tools and citation pipeline, not a score of the agent.
Only a `--live` run measures the agent.

Run it offline (the default; deterministic `ExtractiveClient`, no key, used in CI) or against the real API:

    python -m agent_starter.evals                    # offline, BM25, fails below the thresholds
    python -m agent_starter.evals --retriever embeddings
    python -m agent_starter.evals --live             # calls Claude; needs credentials and costs tokens
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .agent import DEFAULT_MODEL, ask
from .citations import AnswerPart, normalize, unique_citations
from .retrieval import RETRIEVERS, Retriever, build_retriever

CITATION = re.compile(r"\[([^\[\]\s]+#\d+)\]")

DEFAULT_CASES = "evals/cases.jsonl"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    expected_citations: list[str]
    expected_facts: list[str] = field(default_factory=list)


@dataclass
class CaseResult:
    id: str
    question: str
    answer: str
    cited: list[str]
    expected_citations: list[str]
    precision: float
    recall: float
    valid: bool
    supported: bool
    facts: float
    grounded: bool = True
    mode: str = "none"
    """How the answer cited: "native" (API citations), "tags" (typed `[source#n]`) or "none"."""

    @property
    def not_covered(self) -> bool:
        return not self.expected_citations

    @property
    def passed(self) -> bool:
        return self.recall == 1.0 and self.valid and self.supported and self.grounded and self.facts == 1.0


@dataclass
class EvalReport:
    results: list[CaseResult]

    def _mean(self, values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @property
    def summary(self) -> dict[str, float]:
        r = self.results
        return {
            "cases": len(r),
            "pass_rate": self._mean([float(x.passed) for x in r]),
            "citation_precision": self._mean([x.precision for x in r]),
            "citation_recall": self._mean([x.recall for x in r]),
            "valid_citations": self._mean([float(x.valid) for x in r]),
            "supported": self._mean([float(x.supported) for x in r]),
            "grounded": self._mean([float(x.grounded) for x in r]),
            "facts": self._mean([x.facts for x in r]),
        }

    @property
    def non_native(self) -> list[str]:
        """IDs of covered cases answered without native citations (typed tags or none at all)."""
        return [x.id for x in self.results if not x.not_covered and x.mode != "native"]

    @property
    def failed_not_covered(self) -> list[str]:
        """IDs of not-covered cases that cited something (these always fail the run)."""
        return [x.id for x in self.results if x.not_covered and not x.passed]


def load_cases(path: str | Path) -> list[EvalCase]:
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() and not line.lstrip().startswith("//"):
            try:
                cases.append(EvalCase(**json.loads(line)))
            except (json.JSONDecodeError, TypeError) as error:
                raise ValueError(f"{path}:{number}: invalid case: {error}") from error
    return cases


def extract_citations(answer: str) -> list[str]:
    """Citation tags in order of first appearance, without duplicates."""
    return list(dict.fromkeys(CITATION.findall(answer)))


def score_case(case: EvalCase, answer: str, index: Retriever, parts: list[AnswerPart] | None = None) -> CaseResult:
    """Scores one answer. Pass the agent's `parts` to use its native citations; without them, typed tags are read."""
    chunks = {chunk.ref: chunk for chunk in index.chunks}
    native = unique_citations(parts or [])
    if native:
        mode = "native"
        cited = list(dict.fromkeys(citation.ref for citation in native))
        valid = all(
            c.ref in chunks and normalize(c.cited_text) in normalize(chunks[c.ref].text) for c in native
        )
    else:
        cited = extract_citations(answer)
        mode = "tags" if cited else "none"
        valid = all(ref in chunks for ref in cited)
    expected = set(case.expected_citations)
    hits = expected & set(cited)

    if expected:
        precision = len(hits) / len(cited) if cited else 0.0
        recall = len(hits) / len(expected)
    else:  # not covered: the right answer cites nothing
        precision = recall = 1.0 if not cited else 0.0

    lowered = answer.lower()
    cited_text = " ".join(chunks[ref].text.lower() for ref in cited if ref in chunks)
    facts_found = [fact for fact in case.expected_facts if fact.lower() in lowered]
    supported = all(fact.lower() in cited_text for fact in facts_found) and len(facts_found) == len(case.expected_facts)
    if mode == "native":
        # Each fact must be quoted by a citation attached to the very text block that states it.
        grounded = supported and all(
            any(
                normalize(fact) in normalize(c.cited_text)
                for part in parts if normalize(fact) in normalize(part.text)
                for c in part.citations
            )
            for fact in facts_found
        )
    else:
        grounded = supported

    return CaseResult(
        id=case.id,
        question=case.question,
        answer=answer,
        cited=cited,
        expected_citations=case.expected_citations,
        precision=precision,
        recall=recall,
        valid=valid,
        supported=supported,
        facts=len(facts_found) / len(case.expected_facts) if case.expected_facts else 1.0,
        grounded=grounded,
        mode=mode,
    )


def run_eval(cases: list[EvalCase], index: Retriever, client, **ask_kwargs) -> EvalReport:
    """Asks every case through `agent.ask` with `client` and scores the answers."""
    results = []
    for case in cases:
        result = ask(case.question, index, client=client, **ask_kwargs)
        results.append(score_case(case, result.answer, index, result.parts))
    return EvalReport(results)


OFFLINE_TITLE = "Pipeline baseline (offline stand-in, not a score of the agent)"


def format_report(report: EvalReport, markdown: bool = False, title: str | None = None) -> str:
    if markdown:
        def cell(value: str) -> str:
            return value.replace("|", "\\|").replace("\n", " ").replace("\r", " ")

        lines = [f"**{cell(title)}**", ""] if title else []
        lines += ["| case | pass | mode | P | R | cited |", "| --- | --- | --- | --- | --- | --- |"]
        for r in report.results:
            lines.append(f"| {cell(r.id)} | {'ok' if r.passed else 'FAIL'} | {r.mode} | {r.precision:.2f} | {r.recall:.2f} | {cell(', '.join(r.cited) or '-')} |")
        lines += ["", "| metric | value |", "| --- | --- |"]
        lines += [f"| {name} | {value:.2f} |" for name, value in report.summary.items()]
        return "\n".join(lines)
    lines = [title, ""] if title else []
    lines.append(f"{'case':<28} {'pass':<5} {'mode':<7} {'P':>5} {'R':>5}  cited")
    for r in report.results:
        mark = "ok" if r.passed else "FAIL"
        lines.append(f"{r.id:<28} {mark:<5} {r.mode:<7} {r.precision:>5.2f} {r.recall:>5.2f}  {', '.join(r.cited) or '-'}")
    lines.append("")
    lines += [f"{name}: {value:.2f}" if isinstance(value, float) else f"{name}: {value}"
              for name, value in report.summary.items()]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent_starter.evals", description="Score answer citations on a case file.")
    parser.add_argument("--cases", default=DEFAULT_CASES, help=f"JSONL case file (default: {DEFAULT_CASES}).")
    parser.add_argument("--docs", default="docs-example", help="Document folder the cases are about.")
    parser.add_argument("--retriever", default="bm25", choices=RETRIEVERS)
    parser.add_argument("--live", action="store_true", help="Ask the real Claude API instead of the offline fake.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model for --live runs.")
    parser.add_argument("--min-recall", type=float, default=0.9, help="Fail below this mean citation recall.")
    parser.add_argument("--min-precision", type=float, default=0.8, help="Fail below this mean citation precision.")
    parser.add_argument("--min-supported", type=float, default=0.9, help="Fail below this share of supported answers.")
    parser.add_argument("--min-grounded", type=float, default=0.9,
                        help="Fail below this share of answers whose facts are quoted by their own citations.")
    parser.add_argument("--min-pass-rate", type=float, default=0.0,
                        help="Fail below this share of fully passing cases (CI uses 1.0).")
    parser.add_argument("--allow-tags", action="store_true",
                        help="Do not fail when covered cases are answered without native citations (typed tags).")
    parser.add_argument("--only", action="append", default=[], metavar="ID", help="Run only this case ID (repeatable).")
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    output.add_argument("--markdown", action="store_true", help="Print GitHub-flavored Markdown tables.")
    parser.add_argument("--max-chars", type=int, default=900, help="Soft chunk size (minimum 100). Evaluation citation tags assume the default 900.")
    args = parser.parse_args(argv)
    if args.max_chars < 100:
        parser.error("--max-chars must be at least 100")

    cases = load_cases(args.cases)
    valid_ids = {case.id for case in cases}
    unknown = set(args.only) - valid_ids
    if unknown:
        parser.error(f"Unknown case IDs: {', '.join(sorted(unknown))}. Valid IDs: {', '.join(sorted(valid_ids))}")
    if args.only:
        cases = [case for case in cases if case.id in args.only]
    index = build_retriever(args.docs, args.retriever, max_chars=args.max_chars)
    if args.live:
        import anthropic

        client, kwargs = anthropic.Anthropic(), {"model": args.model}
        title = f"Live run: {args.model}"
    else:
        from .fakes import ExtractiveClient

        client, kwargs = ExtractiveClient(), {}
        title = OFFLINE_TITLE
    report = run_eval(cases, index, client, **kwargs)

    if args.json:
        print(json.dumps({"kind": "live" if args.live else "pipeline-baseline", "title": title,
                          "summary": report.summary, "results": [asdict(r) for r in report.results]}, indent=2))
    else:
        print(format_report(report, markdown=args.markdown, title=title))

    summary = report.summary
    failures = [
        f"{name} {summary[name]:.2f} < {minimum:.2f}"
        for name, minimum in (
            ("citation_recall", args.min_recall),
            ("citation_precision", args.min_precision),
            ("supported", args.min_supported),
            ("grounded", args.min_grounded),
            ("pass_rate", args.min_pass_rate),
        )
        if summary[name] < minimum
    ]
    if report.failed_not_covered:
        failures.append("not-covered cases cited something: " + ", ".join(report.failed_not_covered))
    if report.non_native and not args.allow_tags:
        failures.append("answered without native citations: " + ", ".join(report.non_native))
    if failures:
        print("Below threshold: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
