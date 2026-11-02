"""The real Anthropic SDK end to end, with the network replaced by a local mock transport.

No request leaves the machine and no credential is used (the api_key below is a placeholder string the mock never
checks). What this proves: the SDK's tool runner sends our `search_result` blocks inside the `tool_result`, and the
agent reads native citations from responses shaped like the documented API ones, with and without streaming.
"""

import json
from pathlib import Path

import anthropic
import httpx2
from anthropic import DefaultHttpxClient

from agent_starter.agent import ask
from agent_starter.citations import Citation
from agent_starter.retrieval import BM25Index

INDEX = BM25Index.from_folder(Path(__file__).resolve().parents[1] / "docs-example")
QUOTE = "Data is stored in the European Union (Frankfurt) by default."
CITATION = {
    "type": "search_result_location", "source": "security.md#2", "title": "Data location", "cited_text": QUOTE,
    "search_result_index": 0, "start_block_index": 0, "end_block_index": 1,
}
USAGE = {"input_tokens": 10, "output_tokens": 5, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0}
TOOL_USE = {"type": "tool_use", "id": "toolu_1", "name": "search_docs", "input": {"query": "data stored"}}


def _message(number, content, stop_reason):
    return {"id": f"msg_{number}", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
            "content": content, "stop_reason": stop_reason, "stop_sequence": None, "usage": USAGE}


def _sse(number, content, stop_reason):
    """The same message as a server-sent event stream."""
    events = [("message_start", {"type": "message_start", "message": {**_message(number, [], None)}})]
    for i, block in enumerate(content):
        if block["type"] == "text":
            events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {"type": "text", "text": "", "citations": None}}))
            for citation in block.get("citations", []):
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i,
                                                       "delta": {"type": "citations_delta", "citation": citation}}))
            for start in range(0, len(block["text"]), 6):
                events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                    "type": "text_delta", "text": block["text"][start:start + 6]}}))
        else:
            events.append(("content_block_start", {"type": "content_block_start", "index": i,
                                                   "content_block": {**block, "input": {}}}))
            events.append(("content_block_delta", {"type": "content_block_delta", "index": i, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(block["input"])}}))
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason,
                                                                        "stop_sequence": None}, "usage": USAGE}))
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def _client(bodies):
    turns = [([TOOL_USE], "tool_use"), ([{"type": "text", "text": "In Frankfurt by default.", "citations": [CITATION]}],
                                        "end_turn")]

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        content, stop_reason = turns[len(bodies) - 1]
        if body.get("stream"):
            return httpx2.Response(200, text=_sse(len(bodies), content, stop_reason),
                                   headers={"content-type": "text/event-stream"})
        return httpx2.Response(200, json=_message(len(bodies), content, stop_reason))

    transport = httpx2.MockTransport(handler)
    return anthropic.Anthropic(api_key="offline-placeholder", http_client=DefaultHttpxClient(transport=transport),
                               max_retries=0)


def test_the_sdk_sends_search_result_blocks_and_reads_native_citations():
    bodies = []
    result = ask("Where is my data stored?", INDEX, client=_client(bodies))

    first, second = bodies
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["model"] == "claude-opus-5-5"
    tool_result = next(b for b in second["messages"][-1]["content"] if b["type"] == "tool_result")
    assert {block["type"] for block in tool_result["content"]} == {"search_result"}
    assert tool_result["content"][0]["source"] == "security.md#2"
    assert tool_result["content"][0]["citations"] == {"enabled": True}
    assert result.citations == [Citation("security.md#2", "Data location", QUOTE)]


def test_streaming_through_the_sdk_reports_citations_when_the_block_ends():
    bodies, text, cited = [], [], []
    result = ask("Where is my data stored?", INDEX, client=_client(bodies), stream=True,
                 on_text=text.append, on_citations=cited.append)

    assert all(body["stream"] is True for body in bodies)
    assert "".join(text) == "In Frankfurt by default."
    assert cited == [(Citation("security.md#2", "Data location", QUOTE),)]
    assert result.citations == [Citation("security.md#2", "Data location", QUOTE)]
