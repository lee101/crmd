import os
import yaml
from typer.testing import CliRunner
from crmd import app, data_dir, _path

runner = CliRunner()


def test_add_and_list(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    result = runner.invoke(app, ["add", "Test User", "--email", "test@example.com"])
    assert result.exit_code == 0
    assert _path("Test User").exists()

    result = runner.invoke(app, ["list"])
    assert "Test User" in result.stdout


def test_log(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "U", "--email", "u@e.com"])
    result = runner.invoke(app, ["log", "U", "--summary", "hi"])
    assert result.exit_code == 0
    data = yaml.safe_load(_path("U").read_text())
    assert len(data["interactions"]) == 1


def test_search(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "Alice", "--email", "a@b.com"])
    res = runner.invoke(app, ["search", "Alice"])
    assert "Alice" in res.stdout


def test_remind_due(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["remind", "Alice", "hi", "--in", "0"])
    res = runner.invoke(app, ["due"])
    assert "Alice" in res.stdout


def test_schedule_and_worker(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    monkeypatch.setattr(
        "crmd._ai",
        lambda s, u: {
            "run_at": "2020-01-01 00:00",
            "to": "U",
            "subject": "hi",
            "body": "body",
        },
    )
    runner.invoke(app, ["add", "U", "--email", "u@e.com"])
    called = {}

    def fake_open(url):
        called["url"] = url

    monkeypatch.setattr("webbrowser.open", fake_open)
    runner.invoke(app, ["schedule", "email U"])
    runner.invoke(app, ["worker", "--once"])
    assert "mail.google.com" in called["url"]


def test_schedule_post(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    called = {}

    def fake_open(url):
        called["url"] = url

    monkeypatch.setattr("webbrowser.open", fake_open)
    runner.invoke(
        app,
        ["schedule-post", "hello world", "--platform", "x", "--at", "2020-01-01 00:00"],
    )
    runner.invoke(app, ["worker", "--once"])
    assert "twitter.com" in called["url"]


def test_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "Bob", "--email", "b@b.com"])
    monkeypatch.setattr("crmd._ai", lambda s, u: {"summary": "hi"})
    res = runner.invoke(app, ["summary", "Bob"])
    assert "hi" in res.stdout


def test_import_csv(tmp_path, monkeypatch):
    import csv, tempfile

    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    csv_path = tmp_path / "c.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "email", "tags"])
        w.writeheader()
        w.writerow({"name": "Carla", "email": "c@c.com", "tags": "vip"})
    runner.invoke(app, ["import-csv", str(csv_path)])
    assert _path("Carla").exists()


def test_notify_due(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["remind", "Alice", "hi", "--in", "0"])
    called = {}

    def fake_run(cmd, check=False):
        called["cmd"] = cmd

    monkeypatch.setattr("subprocess.run", fake_run)
    runner.invoke(app, ["notify-due"])
    assert called["cmd"][0] == "notify-send"


def test_graphs(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "Alice", "--email", "a@b.com"])
    runner.invoke(app, ["log", "Alice", "--summary", "a"])
    monkeypatch.setattr("textual.app.App.run", lambda self: None)
    res = runner.invoke(app, ["graphs", "Alice"])
    assert res.exit_code == 0


def test_undo_and_stats(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "U", "--email", "u@e.com"])
    subprocess.run(["git", "-C", str(tmp_path), "init"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], check=True)
    runner.invoke(app, ["log", "U", "--summary", "hello"])
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "log"], check=True)
    runner.invoke(app, ["undo"])
    data = yaml.safe_load(_path("U").read_text())
    assert len(data["interactions"]) == 0
    res = runner.invoke(app, ["stats"])
    assert "contacts" in res.stdout


def test_tag_all(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com"])
    runner.invoke(app, ["add", "B", "--email", "b@b.com"])
    runner.invoke(app, ["tag-all", "vip", "A", "B"])
    a = yaml.safe_load(_path("A").read_text())
    b = yaml.safe_load(_path("B").read_text())
    assert "vip" in a["tags"] and "vip" in b["tags"]


def test_export_import_delete_backup(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "X", "--email", "x@e.com"])
    export_path = tmp_path / "x.json"
    runner.invoke(app, ["export-json", "X", str(export_path)])
    assert export_path.exists()

    runner.invoke(app, ["delete", "X"])
    assert not _path("X").exists()

    runner.invoke(app, ["import-json", str(export_path)])
    assert _path("X").exists()

    backup_path = tmp_path / "b.tar.gz"
    runner.invoke(app, ["backup", str(backup_path)])
    assert backup_path.exists()


def test_merge_export_csv_edit_list_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com", "--tags", "vip"])
    runner.invoke(app, ["add", "B", "--email", "b@b.com", "--tags", "new"])
    runner.invoke(app, ["merge", "A", "B"])
    assert not _path("A").exists()
    data = yaml.safe_load(_path("B").read_text())
    assert "vip" in data["tags"]
    csv_path = tmp_path / "all.csv"
    runner.invoke(app, ["export-csv", str(csv_path)])
    assert csv_path.exists()
    called = {}
    monkeypatch.setenv("EDITOR", "vim")
    monkeypatch.setattr("subprocess.run", lambda cmd: called.setdefault("cmd", cmd))
    runner.invoke(app, ["edit", "B"])
    assert called["cmd"][0] == "vim"
    res = runner.invoke(app, ["list-tags"])
    assert "vip" in res.stdout and "new" in res.stdout


def test_config_and_ai(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = {}

    class FakeOpenAI:
        class chat:
            class completions:
                @staticmethod
                def create(**_):
                    calls["used"] = True

                    class C:
                        pass

                    c = C()
                    c.message = C()
                    c.message.content = "{}"
                    result = C()
                    result.choices = [c]
                    return result

    monkeypatch.setattr("crmd.openai", FakeOpenAI)
    runner.invoke(app, ["add", "B", "--email", "b@b.com"])
    runner.invoke(app, ["config", "set-api-key", "k"])
    res = runner.invoke(
        app, ["plan", "B"], env={"OPENAI_API_KEY": "", "CRMD_HOME": str(tmp_path)}
    )
    assert res.exit_code != 1


def test_list_filter(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com", "--tags", "x"])
    runner.invoke(app, ["add", "B", "--email", "b@b.com", "--tags", "y"])
    res = runner.invoke(app, ["list", "--tag", "x"])
    assert "A" in res.stdout and "B" not in res.stdout


def test_reminders_and_tasks(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["remind", "A", "hi", "--in", "0"])
    res = runner.invoke(app, ["reminders"])
    assert "A" in res.stdout
    runner.invoke(app, ["reminders", "--clear", "0"])
    res = runner.invoke(app, ["reminders"])
    assert "A" not in res.stdout

    runner.invoke(app, ["schedule-browser", "http://e.com", "--at", "2020-01-01 00:00"])
    res = runner.invoke(app, ["tasks"])
    assert "http://e.com" in res.stdout
    runner.invoke(app, ["tasks", "--clear", "0"])
    res = runner.invoke(app, ["tasks"])
    assert "http://e.com" not in res.stdout


def test_send_email(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com"])

    sent = {}

    class FakeSMTP:
        def __init__(self, host, port):
            sent["server"] = (host, port)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def starttls(self):
            sent["tls"] = True

        def login(self, user, pw):
            sent["login"] = (user, pw)

        def send_message(self, msg):
            sent["to"] = msg["To"]

    monkeypatch.setattr("smtplib.SMTP", FakeSMTP)
    monkeypatch.setenv("SMTP_SERVER", "host")
    monkeypatch.setenv("SMTP_USER", "u")
    monkeypatch.setenv("SMTP_PASS", "p")
    res = runner.invoke(app, ["send-email", "A", "--subject", "s", "--body", "b"])
    assert res.exit_code == 0
    assert sent["to"] == "a@b.com"


def test_create_event(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com"])
    res = runner.invoke(
        app,
        ["create-event", "A", "--summary", "Meet", "--when", "2020-01-01 00:00"],
    )
    assert res.exit_code == 0
    assert list(tmp_path.glob("*.ics"))


def test_fullscreen(tmp_path, monkeypatch):
    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com"])
    monkeypatch.setattr("textual.app.App.run", lambda self: None)
    res = runner.invoke(app, ["fullscreen"])
    assert res.exit_code == 0


def test_sync_cloud(tmp_path, monkeypatch):
    import subprocess

    monkeypatch.setenv("CRMD_HOME", str(tmp_path))
    runner.invoke(app, ["add", "A", "--email", "a@b.com"])
    subprocess.run(["git", "-C", str(tmp_path), "init"], check=True)
    called = []
    monkeypatch.setattr("subprocess.run", lambda cmd, check=False: called.append(cmd))
    runner.invoke(app, ["sync-cloud"])
    assert any("push" in c for c in called[2])
