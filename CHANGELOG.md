# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Optional `min_score` on `BM25Index` to filter weak retrieval matches. The default remains `0.0`.
- Native, verifiable citations. `search_docs` and `read_document` return `search_result` blocks (one text block per
  sentence, citations enabled), and the agent reads the API's `search_result_location` citations into
  `AgentResult.parts` / `AgentResult.citations` (`Citation(ref, title, cited_text)`, new module `citations.py`).
  The CLI prints footnote markers in place, also while streaming (new `on_citations` callback), and a source list.
- Evaluation: native `cited_text` must be text of the cited chunk (`valid`), and each expected fact must be quoted by
  the citation attached to the sentence that states it (`grounded`, new `--min-grounded`). New `--min-pass-rate`;
  not-covered cases are mandatory and fail the run if they cite anything. CI runs with `--min-pass-rate 1.0`.
- Evaluation: the report shows each case's citation mode (native, tags or none), and a covered case answered without
  native citations fails the run, for the offline stand-in and for real clients alike (`--allow-tags` to opt out).
  Offline reports are titled "Pipeline baseline (offline stand-in, not a score of the agent)"; JSON output has `kind`.
- `--offline` CLI flag: answers with the deterministic extractive stand-in, no API key needed.
- Tests: the real SDK against a local HTTP mock (request and streamed citations), and a check that the README's
  offline output block is the real output.
- Cache chunk embeddings on disk when a stable encoder identity is available.
- Expose the soft chunk size through all three command-line entry points.
- Add repeatable evaluation case selection and Markdown reports.
- Evaluation: six Portuguese cases in `evals/cases.pt.jsonl`, kept out of the CI gate; a test checks their citations and facts.

### Changed

- The offline stand-in (`fakes.ExtractiveClient`) answers only from a passage containing at least half of the
  question's terms and otherwise says "not covered" with no citation. The offline pipeline baseline now passes all 14
  cases with BM25 and with local embeddings (previously 13/14 each). The rule was tuned on these cases, so this is a
  baseline of the pipeline, not a score of the agent.
- The system prompt no longer asks for typed `[source#n]` tags; typed tags are still scored as a fallback.
- Rename the distribution to `ai-agent-starter` (the name `agent-starter` is taken on PyPI). The import name
  `agent_starter` and the `agent-starter`, `agent-starter-mcp` and `agent-starter-eval` commands are unchanged.

### Fixed

- Reject query/chunk embedding dimension mismatches instead of truncating similarity calculations.
- Return no passages for non-positive search limits.

### Known risk

- Native citations have not been run against the live Claude API from this repository (no credentials are used
  here). The request and response shapes follow the documented API and the SDK's types, and are tested through the
  real SDK against a local HTTP mock, but the model's behavior with them is not yet validated.

## [0.1.0] - 2026-11-02

First tagged release. Not published on PyPI: install from a clone or from the release assets.

### Added

- Document-grounded agent on the Anthropic SDK's tool runner (`client.beta.messages.tool_runner`) with three tools:
  `search_docs`, `read_document` and a safe AST `calculate`; inline `[source#chunk]` citations; refusal reporting
  with server-side fallbacks.
- `Retriever` protocol and `build_retriever(folder, kind)`; BM25 stays the dependency-free default.
- Optional local embedding retriever (`agent_starter.embeddings.EmbeddingIndex`, extra `[embeddings]`,
  sentence-transformers) selected with `--retriever embeddings`; any encoder function can be injected.
- Optional MCP server (`python -m agent_starter.mcp_server`, `agent-starter-mcp`, extra `[mcp]`) exposing the same
  tools, with descriptions copied from the agent's tool definitions and read-only annotations.
- Streaming: `ask(..., stream=True, on_text=...)` and streamed CLI output by default (`--no-stream` to turn off).
- Automatic prompt caching via top-level `cache_control` (on by default, `--no-cache` / `cache=False` to turn off),
  token usage summed in `AgentResult.usage`, and `--show-usage`.
- Offline evaluation (`python -m agent_starter.evals`, `agent-starter-eval`) with 14 cases in `evals/cases.jsonl`,
  scoring citation precision/recall, validity and support; runs in CI against a deterministic extractive fake;
  `--live` runs it against the API.
- `agent_starter.fakes`: offline client stand-ins (`ScriptedClient`, `ExtractiveClient`, `FakeStream`).
- Packaging: extras `mcp`, `embeddings`, `dev`; console scripts; SPDX license metadata; sdist includes the example
  docs and eval cases. CI builds the sdist and wheel and smoke-tests the wheel in a clean environment.
- Manual release workflow (`.github/workflows/release.yml`): checks, builds, tags and creates the GitHub release
  with the built files attached.
- CHANGELOG and CONTRIBUTING.

[Unreleased]: https://github.com/ThalzeronLabs/ai-agent-starter/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ThalzeronLabs/ai-agent-starter/releases/tag/v0.1.0
