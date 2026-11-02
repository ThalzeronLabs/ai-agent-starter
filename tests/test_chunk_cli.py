import pytest

from agent_starter import __main__, evals, mcp_server
from agent_starter.fakes import ExtractiveClient
from agent_starter.retrieval import BM25Index


@pytest.mark.parametrize("module", [__main__, evals, mcp_server])
def test_small_chunk_size_is_rejected(module):
    args = ["question"] if module is __main__ else []
    with pytest.raises(SystemExit) as error:
        module.main([*args, "--max-chars", "99"])
    assert error.value.code == 2


@pytest.mark.parametrize("module", [__main__, evals, mcp_server])
def test_chunk_size_reaches_real_index(module, monkeypatch, tmp_path):
    (tmp_path / "a.md").write_text("# Topic\n\n" + "alpha " * 20 + "\n\n" + "beta " * 20)
    counts = []

    def build(folder, kind, **kwargs):
        index = BM25Index.from_folder(folder, **kwargs)
        counts.append(len(index.chunks))
        return index

    monkeypatch.setattr(module, "build_retriever", build)
    if module is mcp_server:
        class Server:
            def run(self, transport):
                assert transport == "stdio"
        monkeypatch.setattr(module, "create_server", lambda index: Server())
    if module is evals:
        cases = tmp_path / "cases.jsonl"
        cases.write_text('{"id":"empty","question":"zzz","expected_citations":[]}')
    for size in (900, 100):
        args = ["--docs", str(tmp_path), "--max-chars", str(size)]
        if module is __main__:
            module.main([*args, "--no-stream", "alpha"], client=ExtractiveClient())
        elif module is evals:
            module.main([*args, "--cases", str(cases)])
        else:
            module.main(args)
    assert counts == [1, 2]
