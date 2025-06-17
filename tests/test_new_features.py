import os
import json
from typer.testing import CliRunner
from crmd import app, _index_file, _path

runner = CliRunner()


def test_web(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    called = {}
    monkeypatch.setattr("webbrowser.open", lambda url: called.setdefault("url", url))
    res = runner.invoke(app, ["web"])
    assert res.exit_code == 0
    assert "index.html" in called["url"]


def test_mobile_shell(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    seq = iter(["exit"])
    monkeypatch.setattr("crmd.Prompt.ask", lambda *_: next(seq))
    res = runner.invoke(app, ["mobile-shell"])
    assert res.exit_code == 0


def test_record_voice(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com"])
    called = {}
    def fake_run(cmd, check=False):
        called['cmd'] = cmd
    monkeypatch.setattr("subprocess.run", fake_run)
    res = runner.invoke(app, ["record-voice", "A", "--seconds", "1"])
    assert res.exit_code == 0
    assert "sox" in called['cmd'][0]


def test_build_index_and_fast_search(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "Alice", "--email", "a@b.com"])
    runner.invoke(app, ["add", "Bob", "--email", "b@b.com"])
    runner.invoke(app, ["log", "Bob", "--summary", "hi"])
    runner.invoke(app, ["build-index"])
    assert _index_file().exists()
    res = runner.invoke(app, ["fast-search", "bob"])
    assert "Bob" in res.stdout


def test_dedupe(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "Ann", "--email", "x@y.com"])
    runner.invoke(app, ["add", "Anne", "--email", "x@y.com"])
    res = runner.invoke(app, ["dedupe"])
    assert "same email" in res.stdout
