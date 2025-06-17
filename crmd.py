#!/usr/bin/env python3
"""CRMD - terminal CRM with local YAML storage and optional OpenAI integration."""

from __future__ import annotations
import os
import sys
import json
import datetime
from pathlib import Path
from typing import List

import yaml
import typer
from rich import print
from rich.table import Table
from rich.prompt import Prompt
import webbrowser
import time
import re
import subprocess
from collections import defaultdict
from difflib import SequenceMatcher

try:
    import openai
except ImportError:  # allow running without the dependency in tests
    openai = None


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------
def data_dir() -> Path:
    d = Path(os.environ.get("CRMD_HOME", Path.home() / ".crmd"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def reminders_file() -> Path:
    return data_dir() / "reminders.txt"


def config_path() -> Path:
    return data_dir() / "config.yaml"


def load_config() -> dict:
    p = config_path()
    if p.exists():
        return yaml.safe_load(p.read_text()) or {}
    return {}


def save_config(cfg: dict) -> None:
    config_path().write_text(yaml.safe_dump(cfg, sort_keys=False))


def _path(name: str) -> Path:
    safe = name.lower().replace(" ", "_")
    return data_dir() / f"{safe}.yaml"


def _load(name: str) -> dict:
    p = _path(name)
    if not p.exists():
        typer.echo(f"[red]No contact named '{name}'")
        raise typer.Exit(1)
    return yaml.safe_load(p.read_text())


def _save(record: dict) -> None:
    _path(record["name"]).write_text(yaml.safe_dump(record, sort_keys=False))


def _merge(a: dict, b: dict) -> None:
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v


# ---------------------------------------------------------------------------
# OpenAI helper
# ---------------------------------------------------------------------------


def _ai(system: str, user: str) -> dict:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        cfg = load_config()
        api_key = cfg.get("openai_api_key")
    if not api_key:
        print("[red]OpenAI key not configured")
        raise typer.Exit(1)
    if openai is None:
        print("[red]openai package missing")
        raise typer.Exit(1)
    openai.api_key = api_key
    res = openai.chat.completions.create(
        model="o3",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
        temperature=0.3,
    )
    return json.loads(res.choices[0].message.content)


# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------
app = typer.Typer(add_help_option=True, rich_markup_mode="rich")
config_app = typer.Typer()
app.add_typer(config_app, name="config")


@config_app.command("set-api-key")
def config_set_api_key(key: str):
    """Store OPENAI_API_KEY in the config file."""
    cfg = load_config()
    cfg["openai_api_key"] = key
    save_config(cfg)
    print("[green]✓ key saved")


@config_app.command("show")
def config_show():
    """Print current configuration."""
    cfg = load_config()
    print(yaml.safe_dump(cfg, sort_keys=False))


@app.command()
def add(
    name: str,
    email: str = typer.Option(..., "--email", "-e"),
    tags: str = typer.Option("", "--tags", "-t", help="comma separated"),
):
    """Add a new contact"""
    if _path(name).exists():
        print(f"[red]{name} already exists")
        raise typer.Exit(1)
    rec = {
        "name": name,
        "email": email,
        "tags": [t.strip() for t in tags.split(",") if t.strip()],
        "created": datetime.date.today().isoformat(),
        "interactions": [],
        "notes": [],
    }
    _save(rec)
    print(f"[green]✓ added {name}")


@app.command(name="list")
def list_contacts(tag: str = typer.Option(None, "--tag")):
    """List contacts, optionally filtering by tag."""
    tab = Table(title="Contacts")
    tab.add_column("Name")
    tab.add_column("Email")
    tab.add_column("Last")
    tab.add_column("#Int")
    for f in sorted(data_dir().glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        if tag and tag not in c.get("tags", []):
            continue
        last = c["interactions"][-1]["date"] if c["interactions"] else "-"
        tab.add_row(c["name"], c.get("email", ""), last, str(len(c["interactions"])))
    print(tab)


@app.command()
def log(
    name: str,
    summary: str = typer.Option(..., "--summary", "-s"),
    channel: str = typer.Option("email", "--channel", "-c"),
):
    """Log an interaction"""
    rec = _load(name)
    rec["interactions"].append(
        {
            "date": datetime.date.today().isoformat(),
            "channel": channel,
            "summary": summary,
        }
    )
    _save(rec)
    print("[green]✓ logged")


@app.command()
def plan(name: str):
    """AI next actions"""
    rec = _load(name)
    system = (
        "You are an assistant suggesting concise next actions for a contact in a CRM. "
        'Return JSON like {"actions":[{"date":"YYYY-MM-DD","action":"text"}]}'
    )
    result = _ai(system, json.dumps(rec))
    print(json.dumps(result, indent=2))


@app.command()
def chat(name: str):
    """Interactive AI chat that can update the record"""
    rec = _load(name)
    system = (
        "You manage a YAML CRM record. Always reply JSON. "
        "To update the record, include a 'patch' key with a JSON merge-patch."
    )
    while True:
        msg = Prompt.ask("[bold cyan]You[/]")
        if msg.lower() in {"/exit", "/quit", "/q"}:
            break
        resp = _ai(system + "\nCurrent record:\n" + json.dumps(rec), msg)
        patch = resp.get("patch")
        if patch:
            _merge(rec, patch)
            _save(rec)
            print("[green]record updated")
        if "assistant_reply" in resp:
            print(resp["assistant_reply"])


@app.command()
def dashboard():
    """Read-only Textual table"""
    try:
        from textual.app import App as TApp
        from textual.widgets import DataTable
    except Exception:
        print("[red]Install textual to use dashboard")
        raise typer.Exit(1)

    class Dash(TApp):
        BINDINGS = [("q", "quit", "Quit")]

        def compose(self):
            tbl = DataTable(zebra_stripes=True)
            tbl.add_columns("Name", "Email", "Last", "#Int")
            for f in sorted(data_dir().glob("*.yaml")):
                c = yaml.safe_load(f.read_text())
                last = c["interactions"][-1]["date"] if c["interactions"] else "-"
                tbl.add_row(
                    c["name"], c.get("email", ""), last, str(len(c["interactions"]))
                )
            yield tbl

    Dash().run()


@app.command()
def search(keyword: str):
    """Search contacts and interactions for a keyword."""
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    for f in sorted(data_dir().glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        haystack = json.dumps(c)
        if pattern.search(haystack):
            print(c["name"])


@app.command()
def remind(name: str, message: str, in_days: int = typer.Option(0, "--in")):
    """Add a reminder entry"""
    date = (datetime.date.today() + datetime.timedelta(days=in_days)).isoformat()
    rf = reminders_file()
    with rf.open("a") as fh:
        fh.write(f"{date} {name}: {message}\n")
    print("[green]✓ reminder added")


@app.command()
def due():
    """Show due reminders"""
    rf = reminders_file()
    if not rf.exists():
        return
    today = datetime.date.today().isoformat()
    for line in rf.read_text().splitlines():
        date, rest = line.split(" ", 1)
        if date <= today:
            print(line)


@app.command("reminders")
def reminders(clear: int = typer.Option(None, "--clear", help="index to remove")):
    """List all reminders or remove one by index."""
    rf = reminders_file()
    if not rf.exists():
        return
    lines = rf.read_text().splitlines()
    if clear is not None:
        if 0 <= clear < len(lines):
            del lines[clear]
            rf.write_text("\n".join(lines) + ("\n" if lines else ""))
            print("[green]✓ removed")
        return
    for i, line in enumerate(lines):
        print(f"{i}: {line}")


@app.command()
def sync():
    """Run git add/commit/push in the data directory if it's a repo."""
    if not (data_dir() / ".git").exists():
        print("[red]No git repository in data dir")
        raise typer.Exit(1)
    subprocess.run(["git", "-C", str(data_dir()), "add", "."])
    subprocess.run(["git", "-C", str(data_dir()), "commit", "-m", "sync"], check=False)
    subprocess.run(["git", "-C", str(data_dir()), "push"], check=False)


@app.command()
def draft_email(name: str, subject: str = typer.Option(..., "--subject")):
    """Use OpenAI to draft an email"""
    rec = _load(name)
    system = (
        "You are an assistant drafting a short email. " 'Return JSON {"email":"text"}'
    )
    prompt = f"Compose an email to {rec['name']} about: {subject}."
    result = _ai(system, prompt)
    print(result.get("email", ""))


@app.command()
def schedule(instruction: str):
    """Parse INSTRUCTION with OpenAI to create a timed Gmail draft task."""
    sys_msg = (
        "You are a parser that extracts scheduling info from the user's sentence.\n"
        "Return JSON with keys run_at (YYYY-MM-DD HH:MM 24h), to (name), subject, body."
    )
    parsed = _ai(sys_msg, instruction)
    run_at = parsed["run_at"]
    contact = _load(parsed["to"])
    tasks = _read_tasks()
    tasks.append(
        {
            "time": run_at,
            "url": _gmail_url(contact.get("email"), parsed["subject"], parsed["body"]),
            "done": False,
        }
    )
    _save_tasks(tasks)
    print(f"[green]✓ task scheduled for {run_at}")


def tasks_file() -> Path:
    return data_dir() / "tasks.json"


def _read_tasks() -> list:
    f = tasks_file()
    if f.exists():
        return json.loads(f.read_text())
    return []


def _save_tasks(tasks: list) -> None:
    tasks_file().write_text(json.dumps(tasks, indent=2))


def _gmail_url(email: str, subject: str, body: str) -> str:
    from urllib.parse import quote_plus

    return (
        "https://mail.google.com/mail/?view=cm&fs=1&tf=1"
        f"&to={quote_plus(email)}&su={quote_plus(subject)}&body={quote_plus(body)}"
    )


@app.command("schedule-browser")
def schedule_browser(
    url: str, at: str = typer.Option(..., "--at", help="YYYY-MM-DD HH:MM")
):
    """Store a browser-opening task to run later."""
    tasks = _read_tasks()
    tasks.append({"time": at, "url": url, "done": False})
    _save_tasks(tasks)
    print("[green]✓ task scheduled")


@app.command("schedule-post")
def schedule_post(
    text: str,
    at: str = typer.Option(..., "--at", help="YYYY-MM-DD HH:MM"),
    platform: str = typer.Option("x", "--platform", "-p"),
):
    """Schedule a social media post on X or Facebook."""
    from urllib.parse import quote_plus

    plat = platform.lower()
    if plat in {"x", "twitter"}:
        url = f"https://twitter.com/intent/tweet?text={quote_plus(text)}"
    elif plat in {"facebook", "fb"}:
        url = f"https://www.facebook.com/sharer/sharer.php?quote={quote_plus(text)}"
    else:
        print("[red]Unsupported platform")
        raise typer.Exit(1)
    tasks = _read_tasks()
    tasks.append({"time": at, "url": url, "done": False})
    _save_tasks(tasks)
    print("[green]✓ post scheduled")


@app.command()
def worker(once: bool = False, interval: int = 60):
    """Execute due browser tasks. Use --once for cron."""

    def process_once():
        tasks = _read_tasks()
        now = datetime.datetime.now()
        changed = False
        for t in tasks:
            if not t.get("done") and datetime.datetime.fromisoformat(t["time"]) <= now:
                webbrowser.open(t["url"])
                t["done"] = True
                changed = True
        if changed:
            _save_tasks(tasks)

    if once:
        process_once()
    else:
        while True:
            process_once()
            time.sleep(interval)


@app.command("tasks")
def tasks_cmd(clear: int = typer.Option(None, "--clear", help="index to remove")):
    """List scheduled tasks or remove one by index."""
    tasks = _read_tasks()
    if clear is not None:
        if 0 <= clear < len(tasks):
            del tasks[clear]
            _save_tasks(tasks)
            print("[green]✓ removed")
        return
    for i, t in enumerate(tasks):
        status = "done" if t.get("done") else "pending"
        print(f"{i}: {t['time']} {status} {t['url']}")


@app.command()
def graphs(name: str):
    """Show monthly interaction counts using a Textual sparkline."""
    try:
        from textual.app import App as TApp
        from textual.widgets import Sparkline
    except Exception:
        print("[red]Install textual to use graphs")
        raise typer.Exit(1)
    rec = _load(name)
    counts = defaultdict(int)
    for i in rec.get("interactions", []):
        dt = datetime.date.fromisoformat(i["date"])
        key = dt.strftime("%Y-%m")
        counts[key] += 1
    months = sorted(counts)
    values = [counts[m] for m in months]

    class Graph(TApp):
        BINDINGS = [("q", "quit", "Quit")]

        def compose(self):
            yield Sparkline(values=values, width=len(values) * 4, title="Interactions")

    Graph().run()


@app.command()
def undo():
    """Revert the last commit in the data directory."""
    if not (data_dir() / ".git").exists():
        print("[red]No git repository in data dir")
        raise typer.Exit(1)
    subprocess.run(
        ["git", "-C", str(data_dir()), "revert", "--no-edit", "HEAD"], check=False
    )


@app.command()
def stats():
    """Print total contacts, interactions and due reminders."""
    contacts = list(data_dir().glob("*.yaml"))
    num_contacts = len(contacts)
    interactions = sum(
        len(yaml.safe_load(f.read_text()).get("interactions", [])) for f in contacts
    )
    due = 0
    rf = reminders_file()
    if rf.exists():
        today = datetime.date.today().isoformat()
        due = sum(
            1 for line in rf.read_text().splitlines() if line.split(" ", 1)[0] <= today
        )
    print(
        json.dumps(
            {
                "contacts": num_contacts,
                "interactions": interactions,
                "due_reminders": due,
            },
            indent=2,
        )
    )


@app.command("tag-all")
def tag_all(tag: str, names: List[str]):
    """Add TAG to multiple contacts."""
    for name in names:
        rec = _load(name)
        if tag not in rec.get("tags", []):
            rec.setdefault("tags", []).append(tag)
            _save(rec)
    print("[green]✓ tags updated")


@app.command()
def summary(name: str):
    """Summarize a contact with AI."""
    rec = _load(name)
    resp = _ai(
        'Summarize this CRM record in <=100 words. Return JSON {"summary":"text"}',
        json.dumps(rec),
    )
    print(resp.get("summary", ""))


@app.command("import-csv")
def import_csv(path: str):
    """Import contacts from a CSV with columns name,email,tags."""
    import csv

    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("name")
            email = row.get("email", "")
            tags = row.get("tags", "")
            if name:
                if not _path(name).exists():
                    rec = {
                        "name": name,
                        "email": email,
                        "tags": [t.strip() for t in tags.split(",") if t.strip()],
                        "created": datetime.date.today().isoformat(),
                        "interactions": [],
                        "notes": [],
                    }
                    _save(rec)


@app.command("notify-due")
def notify_due():
    """Send desktop notifications for due reminders."""
    rf = reminders_file()
    if not rf.exists():
        return
    today = datetime.date.today().isoformat()
    for line in rf.read_text().splitlines():
        date, rest = line.split(" ", 1)
        if date <= today:
            subprocess.run(["notify-send", "CRMD", rest], check=False)


@app.command()
def encrypt():
    """Archive and encrypt the data directory with gpg."""
    tar = data_dir().with_suffix(".tar.gz")
    subprocess.run(["tar", "czf", str(tar), "-C", str(data_dir()), "."], check=True)
    subprocess.run(["gpg", "-c", str(tar)], check=False)


@app.command("export-json")
def export_json(name: str, path: str):
    """Export a contact to PATH as JSON."""
    rec = _load(name)
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=2)
    print("[green]✓ exported")


@app.command("import-json")
def import_json(path: str):
    """Import a contact from a JSON file."""
    with open(path) as fh:
        rec = json.load(fh)
    name = rec.get("name")
    if not name:
        print("[red]Missing name field")
        raise typer.Exit(1)
    if _path(name).exists():
        print("[red]Contact exists")
        raise typer.Exit(1)
    _save(rec)
    print("[green]✓ imported")


@app.command()
def delete(name: str):
    """Delete a contact."""
    p = _path(name)
    if p.exists():
        p.unlink()
        print("[green]✓ deleted")
    else:
        print("[red]Contact not found")


@app.command()
def backup(dest: str = typer.Argument("crmd_backup.tar.gz")):
    """Create a gzipped backup of the data directory."""
    subprocess.run(["tar", "czf", dest, "-C", str(data_dir()), "."], check=True)
    print(f"[green]✓ backup written to {dest}")


@app.command()
def merge(source: str, target: str):
    """Merge SOURCE contact into TARGET and delete SOURCE."""
    src = _load(source)
    tgt = _load(target)
    tgt.setdefault("interactions", []).extend(src.get("interactions", []))
    tgt.setdefault("notes", []).extend(src.get("notes", []))
    tags = set(tgt.get("tags", [])) | set(src.get("tags", []))
    tgt["tags"] = sorted(tags)
    _save(tgt)
    _path(source).unlink(missing_ok=True)
    print("[green]✓ merged")


@app.command("export-csv")
def export_csv(path: str):
    """Export all contacts to a CSV file."""
    import csv

    contacts = []
    for f in sorted(data_dir().glob("*.yaml")):
        c = yaml.safe_load(f.read_text())
        contacts.append(
            {
                "name": c.get("name"),
                "email": c.get("email", ""),
                "tags": ",".join(c.get("tags", [])),
            }
        )
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["name", "email", "tags"])
        w.writeheader()
        w.writerows(contacts)
    print("[green]✓ exported")


@app.command()
def edit(name: str):
    """Open a contact in $EDITOR."""
    editor = os.environ.get("EDITOR", "nano")
    subprocess.run([editor, str(_path(name))])


@app.command("send-email")
def send_email(
    name: str,
    subject: str = typer.Option(..., "--subject"),
    body: str = typer.Option(..., "--body"),
):
    """Send an email to NAME using environment SMTP settings."""
    rec = _load(name)
    to_addr = rec.get("email")
    if not to_addr:
        print("[red]No email for contact")
        raise typer.Exit(1)
    smtp_server = os.environ.get("SMTP_SERVER")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    if not smtp_server:
        print("[red]Set SMTP_SERVER")
        raise typer.Exit(1)
    from email.message import EmailMessage
    import smtplib

    msg = EmailMessage()
    msg["From"] = smtp_user or to_addr
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_server, smtp_port) as s:
        if smtp_user:
            s.starttls()
            s.login(smtp_user, smtp_pass or "")
        s.send_message(msg)
    print("[green]✓ email sent")


@app.command("create-event")
def create_event(
    name: str,
    summary: str = typer.Option(..., "--summary"),
    when: str = typer.Option(..., "--when", help="YYYY-MM-DD HH:MM"),
):
    """Write an ICS calendar event for NAME."""
    rec = _load(name)
    dt = datetime.datetime.fromisoformat(when)
    path = data_dir() / f"{dt.date()}_{rec['name'].replace(' ', '_')}.ics"
    ics = (
        "BEGIN:VCALENDAR\n"
        "VERSION:2.0\n"
        "BEGIN:VEVENT\n"
        f"DTSTAMP:{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}\n"
        f"DTSTART:{dt.strftime('%Y%m%dT%H%M%S')}\n"
        f"SUMMARY:{summary}\n"
        f"ATTENDEE:{rec.get('email','')}\n"
        "END:VEVENT\n"
        "END:VCALENDAR\n"
    )
    path.write_text(ics)
    print(f"[green]✓ event saved to {path}")


@app.command()
def fullscreen():
    """Full-screen Textual UI."""
    try:
        from textual.app import App, ComposeResult
        from textual.widgets import DataTable, Header, Footer
    except Exception:
        print("[red]Install textual first: pip install textual")
        raise typer.Exit(1)

    class Full(App):
        BINDINGS = [("q", "quit", "Quit")]

        def compose(self) -> ComposeResult:
            yield Header()
            tbl = DataTable(zebra_stripes=True)
            tbl.add_columns("Name", "Email", "Tags")
            for f in sorted(data_dir().glob("*.yaml")):
                c = yaml.safe_load(f.read_text())
                tbl.add_row(c["name"], c.get("email", ""), ",".join(c.get("tags", [])))
            yield tbl
            yield Footer()

    Full().run()


@app.command("sync-cloud")
def sync_cloud(remote: str = typer.Option("origin", "--remote")):
    """Push the data directory to a git remote."""
    if not (data_dir() / ".git").exists():
        print("[red]No git repository in data dir")
        raise typer.Exit(1)
    subprocess.run(["git", "-C", str(data_dir()), "add", "."], check=False)
    subprocess.run(["git", "-C", str(data_dir()), "commit", "-m", "sync"], check=False)
    subprocess.run(["git", "-C", str(data_dir()), "push", remote], check=False)
    print("[green]✓ pushed")


@app.command("list-tags")
def list_tags():
    """List all unique tags."""
    tags = set()
    for f in data_dir().glob("*.yaml"):
        c = yaml.safe_load(f.read_text())
        tags.update(c.get("tags", []))
    for t in sorted(tags):
        print(t)


@app.command("web")
def open_web():
    """Open the bundled index.html in a browser."""
    html = Path(__file__).with_name("index.html").resolve()
    webbrowser.open(str(html))


@app.command("mobile-shell")
def mobile_shell():
    """Minimal interactive shell for mobile devices."""
    while True:
        cmd = Prompt.ask("crmd>")
        if not cmd or cmd.lower() in {"exit", "quit", "q"}:
            break
        try:
            app(cmd.split())
        except SystemExit:
            continue


@app.command("record-voice")
def record_voice(name: str, seconds: int = typer.Option(5, "--seconds", "-s")):
    """Record a short voice note and attach to NAME."""
    rec = _load(name)
    path = data_dir() / f"{name.replace(' ', '_')}_{int(time.time())}.wav"
    subprocess.run(["sox", "-d", str(path), "trim", "0", str(seconds)], check=False)
    rec.setdefault("voice_notes", []).append(str(path))
    _save(rec)
    print(f"[green]✓ voice note saved to {path}")


def _index_file() -> Path:
    return data_dir() / "search_index.json"


@app.command("build-index")
def build_index():
    """Create a simple word->contact search index."""
    idx = defaultdict(set)
    for f in data_dir().glob("*.yaml"):
        c = yaml.safe_load(f.read_text())
        tokens = [c["name"]]
        tokens.extend(c.get("tags", []))
        tokens.append(c.get("email", ""))
        for i in c.get("interactions", []):
            tokens.extend(i.get("summary", "").split())
        for t in tokens:
            t = t.lower()
            if t:
                idx[t].add(c["name"])
    _index_file().write_text(json.dumps({k: sorted(v) for k, v in idx.items()}, indent=2))
    print("[green]✓ index built")


@app.command("fast-search")
def fast_search(word: str):
    """Search contacts using the local index."""
    if not _index_file().exists():
        print("[red]Index not built")
        raise typer.Exit(1)
    idx = json.loads(_index_file().read_text())
    for name in idx.get(word.lower(), []):
        print(name)


@app.command()
def dedupe(threshold: float = typer.Option(0.8, "--threshold")):
    """Suggest duplicate contacts."""
    contacts = []
    for f in data_dir().glob("*.yaml"):
        c = yaml.safe_load(f.read_text())
        contacts.append((c["name"], c.get("email", "")))
    for i, (n1, e1) in enumerate(contacts):
        for n2, e2 in contacts[i + 1 :]:
            if e1 and e1 == e2:
                print(f"{n1} <-> {n2} (same email)")
            else:
                if (
                    SequenceMatcher(None, n1.lower(), n2.lower()).ratio()
                    >= threshold
                ):
                    print(f"{n1} <-> {n2} (name match)")


def worker_cmd():
    """Entry point for systemd timers."""
    worker()


if __name__ == "__main__":
    app()
