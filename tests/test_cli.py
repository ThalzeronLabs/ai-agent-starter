from pathlib import Path

from agent_starter.__main__ import main
from agent_starter.fakes import ExtractiveClient, ScriptedClient, message, text_block, tool_use_block

DOCS = str(Path(__file__).resolve().parents[1] / "docs-example")


def _client():
    return ScriptedClient([
        message(tool_use_block("search_docs"), stop_reason="tool_use"),
        message(text_block("Within 14 days [refunds.md#1].")),
    ])


def test_cli_streams_the_answer_by_default(capsys):
    client = _client()
    assert main(["--docs", DOCS, "Refund?"], client=client) == 0
    out, err = capsys.readouterr()
    assert out == "Within 14 days [refunds.md#1].\n"
    assert "Indexed 3 documents" in err
    assert client.calls[0]["stream"] is True
    assert client.calls[0]["cache_control"] == {"type": "ephemeral"}


def test_cli_no_stream_and_no_cache(capsys):
    client = _client()
    assert main(["--docs", DOCS, "--no-stream", "--no-cache", "--show-tools", "Refund?"], client=client) == 0
    out, err = capsys.readouterr()
    assert out == "Within 14 days [refunds.md#1].\n"
    assert "Tools used: search_docs" in err
    assert "stream" not in client.calls[0] and "cache_control" not in client.calls[0]


def test_cli_show_usage(capsys):
    assert main(["--docs", DOCS, "--show-usage", "Refund?"], client=_client()) == 0
    assert "Usage: input_tokens=0" in capsys.readouterr().err


def test_cli_refusal_exits_1(capsys):
    client = ScriptedClient([message(text_block("..."), stop_reason="refusal")])
    assert main(["--docs", DOCS, "Hi"], client=client) == 1
    assert "declined" in capsys.readouterr().err


def test_cli_missing_folder_exits_2(capsys):
    assert main(["--docs", "does-not-exist", "Hi"], client=_client()) == 2
    assert "No such folder" in capsys.readouterr().err


def test_cli_end_to_end_with_the_extractive_fake(capsys):
    assert main(["--docs", DOCS, "Where is my data stored by default?"], client=ExtractiveClient()) == 0
    assert "Frankfurt" in capsys.readouterr().out


def test_cli_prints_footnotes_and_sources_the_same_way_with_and_without_streaming(capsys):
    question = "Where is my data stored by default?"
    assert main(["--docs", DOCS, question], client=ExtractiveClient()) == 0
    streamed = capsys.readouterr().out
    assert main(["--docs", DOCS, "--no-stream", question], client=ExtractiveClient()) == 0
    assert capsys.readouterr().out == streamed
    assert streamed.splitlines()[0].endswith("by default.[1]")
    assert '[1] security.md#2 (Data location): "Data is stored in the European Union (Frankfurt) by default."' in streamed
