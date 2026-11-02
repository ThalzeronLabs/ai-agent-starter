"""The README's offline output block must be exactly what the commands print: nothing in it is written by hand."""

import re
import shlex
from pathlib import Path

from agent_starter.__main__ import main

ROOT = Path(__file__).resolve().parents[1]


def _blocks():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = re.search(r"<!-- offline-output:start -->\n```\n(.*?)```\n<!-- offline-output:end -->", readme, re.S)
    assert block, "README.md lost its offline-output block"
    for chunk in re.split(r"^\$ ", block.group(1), flags=re.M)[1:]:
        command, _, output = chunk.partition("\n")
        yield shlex.split(command), output.rstrip("\n")


def test_readme_offline_output_is_the_real_output(capsys, monkeypatch):
    monkeypatch.chdir(ROOT)  # the commands use the default --docs docs-example
    blocks = list(_blocks())
    assert len(blocks) >= 2
    for argv, expected in blocks:
        assert argv[:3] == ["python", "-m", "agent_starter"] and "--offline" in argv
        assert main(argv[3:]) == 0
        assert capsys.readouterr().out.rstrip("\n") == expected, " ".join(argv)
