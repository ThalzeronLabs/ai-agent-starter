"""Offline evaluation: do the answers cite the passages that actually contain the answer?

A case file is JSON Lines, one case per line:

    {"id": "refund-window", "question": "...", "expected_citations": ["refunds.md#1"], "expected_facts": ["14 days"]}

`expected_citations` lists the chunk tags a correct answer must cite (empty when the documents do not cover the
question, in which case the answer must cite nothing). `expected_facts` are short strings that a correct answer
contains (case-insensitive).

For every case the evaluator asks the agent, extracts the `[source#n]` tags from the answer, and scores:

- citation precision and recall against `expected_citations`;
- validity: every cited tag names a chunk that exists in the index (no invented citations);
- support: every expected fact appears in the answer *and* in the text of a cited chunk, i.e. the citation points
  at a passage that really says it;
- facts: the share of expected facts present in the answer.

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

    @property
    def passed(self) -> bool:
        return self.recall == 1.0 and self.valid and self.supported and self.facts == 1.0


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
            "facts": self._mean([x.facts for x in r]),
        }


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


def score_case(case: EvalCase, answer: str, index: Retriever) -> CaseResult:
    chunks = {chunk.ref: chunk for chunk in index.chunks}
    cited = extract_citations(answer)
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

    return CaseResult(
        id=case.id,
        question=case.question,
        answer=answer,
        cited=cited,
        expected_citations=case.expected_citations,
        precision=precision,
        recall=recall,
        valid=all(ref in chunks for ref in cited),
        supported=supported,
        facts=len(facts_found) / len(case.expected_facts) if case.expected_facts else 1.0,
    )


def run_eval(cases: list[EvalCase], index: Retriever, client, **ask_kwargs) -> EvalReport:
    """Asks every case through `agent.ask` with `client` and scores the answers."""
    results = []
    for case in cases:
        result = ask(case.question, index, client=client, **ask_kwargs)
        results.append(score_case(case, result.answer, index))
    return EvalReport(results)


def format_report(report: EvalReport) -> str:
    lines = [f"{'case':<28} {'pass':<5} {'P':>5} {'R':>5}  cited"]
    for r in report.results:
        mark = "ok" if r.passed else "FAIL"
        lines.append(f"{r.id:<28} {mark:<5} {r.precision:>5.2f} {r.recall:>5.2f}  {', '.join(r.cited) or '-'}")
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
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    parser.add_argument("--max-chars", type=int, default=900, help="Soft chunk size (minimum 100). Evaluation citation tags assume the default 900.")
    args = parser.parse_args(argv)
    if args.max_chars < 100:
        parser.error("--max-chars must be at least 100")

    index = build_retriever(args.docs, args.retriever, max_chars=args.max_chars)
    cases = load_cases(args.cases)
    if args.live:
        import anthropic

        client, kwargs = anthropic.Anthropic(), {"model": args.model}
    else:
        from .fakes import ExtractiveClient

        client, kwargs = ExtractiveClient(), {}
    report = run_eval(cases, index, client, **kwargs)

    if args.json:
        print(json.dumps({"summary": report.summary, "results": [asdict(r) for r in report.results]}, indent=2))
    else:
        print(format_report(report))

    summary = report.summary
    failures = [
        f"{name} {summary[name]:.2f} < {minimum:.2f}"
        for name, minimum in (
            ("citation_recall", args.min_recall),
            ("citation_precision", args.min_precision),
            ("supported", args.min_supported),
        )
        if summary[name] < minimum
    ]
    if failures:
        print("Below threshold: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

