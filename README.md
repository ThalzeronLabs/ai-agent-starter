# ai-agent-starter

[![CI](https://github.com/ThalzeronLabs/ai-agent-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/ThalzeronLabs/ai-agent-starter/actions/workflows/ci.yml)

**A small, tested starter for document-grounded Claude agents: retrieval + tool use + citations, with an offline
evaluation, an optional MCP server and optional local embeddings. Clone it, point it at your docs, change what you
need.**

Point it at a folder of Markdown/text files and ask questions. Claude searches the documents with tools, reads more
when it needs to, does arithmetic with a safe calculator, and answers with **native, verifiable citations**: every
cited sentence of the answer comes back from the API with the exact text of the document it relies on. Answers are
streamed to your terminal as they are written.

```bash
git clone https://github.com/ThalzeronLabs/ai-agent-starter && cd ai-agent-starter
pip install -e ".[dev]"
python -m agent_starter --offline "Where is my data stored by default?"     # no API key needed
python -m agent_starter --show-tools "Can I get my money back on a yearly Pro plan after 10 days?"
```

## Verifiable citations

The search tools hand each passage to Claude as a `search_result` block, one text block per sentence. The API then
attaches citations to the answer as data (`search_result_location`, with the exact `cited_text`), not as tags the
model types, so code can check that each claim is quoted by its own citation and that the quote is in the document.

Real output of the offline mode. `--offline` answers with the deterministic extractive stand-in that the tests and CI
use, not with Claude; Claude words its answers differently but returns citations in this same format:

<!-- offline-output:start -->
```
$ python -m agent_starter --offline "Where is my data stored by default?"
Data is stored in the European Union (Frankfurt) by default.[1]

Sources:
[1] security.md#2 (Data location): "Data is stored in the European Union (Frankfurt) by default."

$ python -m agent_starter --offline "Is there a dark mode in the iOS app?"
The documents do not cover this question.
```
<!-- offline-output:end -->

`tests/test_readme.py` re-runs these commands and fails if this block differs from the real output.

What this buys, as a test (`tests/test_evals.py::test_a_claim_that_contradicts_its_own_citation_is_caught_only_by_native_citations`):
the answer "Monthly plans are refunded within 14 days" cites `refunds.md#1`. That chunk does contain "14 days" (for
annual plans), so a typed tag `[refunds.md#1]` passes every check. The native citation shows that the claim rests on
"Monthly plans are not refunded…", and the evaluator rejects it.

Everything in this repository is tested **offline**: no API key, no network, no model download. Only real CLI runs
and `--live` evaluations call the Claude API. Native citations follow the documented API and are tested through the
real SDK against a local mock of the HTTP API (`tests/test_sdk_wire.py`), but have not yet been run against the live
API from this repository.

## How it works

```
question ──▶ Claude (claude-opus-5-5, tool runner, streaming) ──▶ search_docs ──▶ Retriever: BM25 (default)
                       │   ▲                                      read_document      or local embeddings
                       │   └──────────── tool results ◀────────── calculate ──▶ safe AST arithmetic
                       ▼
      answer + native citations (cited_text)        the same three tools ──▶ MCP server (optional)
```

| Piece | File | Notes |
|---|---|---|
| Agent loop | `agent.py` | The official SDK's **tool runner** (`client.beta.messages.tool_runner`), streaming, prompt caching, refusal fallbacks. |
| Tools | `tools.py` | Plain functions wrapped with `@beta_tool`, which builds the JSON schemas from type hints and docstrings. The search tools return `search_result` blocks. |
| Citations | `citations.py` | Reads the API's `search_result_location` citations into `Citation(ref, title, cited_text)` and renders footnotes. |
| Retrieval | `retrieval.py`, `embeddings.py` | A three-member `Retriever` protocol; BM25 (zero dependencies) and an optional local embedding adapter. |
| MCP server | `mcp_server.py` | The same tools for any MCP client (Claude Code, Claude Desktop, IDEs). |
| Evaluation | `evals.py`, `evals/cases.jsonl` | Checks that each claim is quoted by its own citation and that unanswerable questions cite nothing. Gates CI. |
| Offline fakes | `fakes.py` | Stand-ins for the Anthropic client used by tests and offline evals. |
| CLI | `__main__.py` | `python -m agent_starter` / `agent-starter`. |

- **Grounding:** answers cite search results natively, and the system prompt asks for an explicit "not covered",
  with no citation, when no search result answers the question.
- **Safety:** the calculator walks the Python AST and only allows numbers and arithmetic operators; no `eval`.
- **Robustness:** refusals (`stop_reason == "refusal"`) are reported, with server-side fallbacks enabled
  (`fallbacks="default"`); auth and connection errors get clear CLI messages.

## Install

```bash
pip install -e .                    # core: the agent, BM25, the CLI
pip install -e ".[mcp]"             # + the MCP server
pip install -e ".[embeddings]"      # + local embeddings (sentence-transformers; pulls in PyTorch)
pip install -e ".[dev]"             # + pytest, build and mcp, to run the test suite
```

The distribution is named `ai-agent-starter`; the Python package you import is `agent_starter`.
Python 3.10+. Credentials come from `ANTHROPIC_API_KEY` or an `ant auth login` profile.

## CLI

```bash
python -m agent_starter --docs path/to/your/handbook "What is the vacation policy?"
```

| Flag | Default | What it does |
|---|---|---|
| `--docs DIR` | `docs-example` | Folder of `.md` / `.txt` / `.rst` files (recursive). |
| `--retriever bm25\|embeddings` | `bm25` | Search backend (see [Retrieval](#retrieval)). |
| `--stream / --no-stream` | stream | Print the answer while it arrives, or all at once. |
| `--cache / --no-cache` | cache | Prompt caching (see [Prompt caching](#prompt-caching)). |
| `--effort low…max` | `medium` | Trades depth for cost. |
| `--show-tools`, `--show-usage` | off | Print the tools used, and token usage including cache reads/writes (to stderr). |
| `--offline` | off | No API call: answer with the deterministic extractive stand-in (not a model), to try the pipeline without a key. |
| `--max-chars N` | `900` | Soft chunk size (minimum 100). Also accepted by `agent_starter.evals` and `agent_starter.mcp_server`. Changing it changes the citation tags; the bundled evaluation cases assume 900. |

### Streaming

With streaming on, the tool runner is called with `stream=True`; each turn of the loop is then a message stream.
Its `text` events are printed as they arrive; when a text block ends (`content_block_stop`), its citations are passed
on, so the CLI prints the footnote markers in place; `get_final_message()` gives the complete message for the loop
(see the SDK's [tool runner docs](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-runner)).
From Python:

```python
from agent_starter import ask, build_retriever

index = build_retriever("docs-example")
result = ask("How do refunds work?", index, stream=True, on_text=lambda t: print(t, end="", flush=True))
for citation in result.citations:
    print(f"\n{citation.ref}: {citation.cited_text}")
```

`ask()` returns the same `AgentResult` with or without streaming: `answer`, `parts` (each text block with its
citations), `citations`, `tool_calls` and `usage`.

### Prompt caching

`ask()` sends a top-level `cache_control={"type": "ephemeral"}` (turn it off with `cache=False` / `--no-cache`).
This is the API's [automatic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching):
the breakpoint goes on the last cacheable block of each request and moves forward as the tool loop grows, so every
model call after the first can read the tools, the system prompt and the earlier turns from cache.

What keeps it working:

- The cache is a **prefix match** in the order `tools` → `system` → `messages`. The tool list and `SYSTEM_PROMPT` are
  constant; keep them that way (no timestamps, user names or per-request data in them; put those in the question).
- Prefixes below the model's minimum (**512 tokens** for `claude-opus-5-5`) are not cached, silently. A first call
  with only the tools and system prompt may fall under it; the turns with search results usually don't.
- The default entry lives 5 minutes; each hit refreshes it.
- Check it with `--show-usage`: `cache_creation_input_tokens` is what was written, `cache_read_input_tokens` what was
  read back. If reads stay at 0 across turns, something in the prefix is changing.

## Retrieval

Tools only need the `Retriever` protocol: `documents` (name → text), `chunks`, and `search(query, k)`. Both backends
chunk the same way, so citation tags such as `refunds.md#1` are identical whichever you pick.

- **`bm25`** (default): Okapi BM25 over heading-aware chunks, accent- and case-insensitive, English + Portuguese
  stopwords. No dependencies, deterministic, strong on exact terms (prices, product names, error codes).
- **`embeddings`**: cosine similarity over vectors from a **local** sentence-transformers model
  (`sentence-transformers/all-MiniLM-L6-v2` by default; downloaded once from Hugging Face, then cached; no paid API).
  Better on paraphrases ("money back" → "refund"). For non-English questions pass a multilingual model:

  ```python
  from agent_starter.embeddings import EmbeddingIndex
  index = EmbeddingIndex.from_folder("docs", model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
  ```

  Any encoder works: `EmbeddingIndex(documents, encoder=my_fn)` where `my_fn(list[str]) -> list[list[float]]`.
  The tests use a hashing encoder so they never download a model.

To plug in something else (a vector database, hybrid search, a reranker), implement the three members and pass the
object to `ask()` or `create_server()`.

## MCP server

The same `search_docs`, `read_document` and `calculate` tools, with the same descriptions, served over stdio by the
official [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) (`mcp` 2.x):

```bash
pip install -e ".[mcp]"
python -m agent_starter.mcp_server --docs /absolute/path/to/docs     # or: agent-starter-mcp --docs ...
```

Register it with Claude Code:

```bash
claude mcp add docs -- python -m agent_starter.mcp_server --docs /absolute/path/to/docs
```

or in any client that takes a JSON config (for example Claude Desktop's `claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "docs": {
      "command": "python",
      "args": ["-m", "agent_starter.mcp_server", "--docs", "/absolute/path/to/docs"]
    }
  }
}
```

All three tools are marked read-only. `--retriever embeddings` works here too.

## Evaluation

`evals/cases.jsonl` holds questions about `docs-example` with the chunks a correct answer must cite and short facts
it must contain; two cases are not covered by the docs and must cite nothing.

```json
{"id": "data-location", "question": "Where is my data stored by default?", "expected_citations": ["security.md#2"], "expected_facts": ["Frankfurt"]}
```

```bash
python -m agent_starter.evals                          # offline (CI), BM25
python -m agent_starter.evals --retriever embeddings   # offline, local embeddings
python -m agent_starter.evals --live                   # real Claude API: needs credentials, costs tokens
```

For each case it scores:

- citation **precision** and **recall** against the expected chunks;
- **validity**: every citation names a real chunk, and every native `cited_text` is text that chunk contains;
- **support**: each expected fact is in a chunk the answer cites;
- **grounding**: each expected fact is in the `cited_text` of the citation attached to the sentence that states it
  (with typed tags, which cannot say this, grounding falls back to support);
- **facts**: the share of expected facts in the answer.

A case passes when all of these hold. Not-covered cases are mandatory: if any of them cites something, the run fails
whatever the thresholds. Otherwise it exits non-zero below `--min-pass-rate` (default 0), `--min-recall` (0.9),
`--min-precision` (0.8), `--min-supported` (0.9) or `--min-grounded` (0.9). **CI runs it with `--min-pass-rate 1.0`:
every case must pass.**

Offline, the answers come from `fakes.ExtractiveClient`, a deterministic stand-in that really calls `search_docs`,
answers with the best sentence of the first passage that contains at least half of the question's terms, and
attaches a native citation in the API's format. When no passage does, it answers "not covered" and cites nothing;
that rule stands in for the judgment a model makes when it reads the results. It is not a language model: it
measures the retrieval, the tools and the citation pipeline, so a regression there fails CI.

| Retriever | Pass | Precision | Recall | Supported | Grounded |
|---|---|---|---|---|---|
| BM25 (CI) | 14/14 | 1.00 | 1.00 | 1.00 | 1.00 |
| Local embeddings (MiniLM, run locally) | 14/14 | 1.00 | 1.00 | 1.00 | 1.00 |

Before native citations, this table read 13/14 for both: the BM25 run cited `security.md#1` for the dark-mode
question because "app" matches "authenticator app". The coverage rule fixes that without changing retrieval. It was
chosen on these 14 cases, so treat it as a baseline, not a benchmark: a `--live` run measures what Claude does with
the same results. Add your own cases next to your own documents.

## Tests

```bash
pytest
```

94 tests run offline: tokenization, chunking, BM25, the embedding adapter (with a hashing encoder), every tool and
the calculator's sandbox, the generated tool schemas, the `search_result` blocks and native citations, the agent loop
with and without streaming, prompt-caching and usage plumbing, the CLI, the README's offline output, the real SDK
against a local HTTP mock, the MCP server (in-process and over stdio, through a real MCP client), and the evaluator
and its case file. CI runs them on Python 3.10, 3.12 and 3.13, runs the offline evaluation with every case required
to pass, and builds and smoke-tests the wheel.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) and the issues labelled
[good first issue](https://github.com/ThalzeronLabs/ai-agent-starter/labels/good%20first%20issue). Changes are listed
in [CHANGELOG.md](CHANGELOG.md).

## License

Apache-2.0 © Esdras Souza (Thalzeron Labs). The example documents describe a fictional company.
