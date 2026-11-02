# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

- Expose the soft chunk size through all three command-line entry points.

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

