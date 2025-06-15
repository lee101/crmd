# crmd

**crmd** is a terminal‑native CRM that keeps your customer data in flat YAML files and
uses OpenAI on demand for planning and summaries.

Features at a glance:

- Fast commands: `add`, `list`, `log`, `plan`, `chat`, and a Textual dashboard.
- Reminders, git‑powered undo, and simple stats.
- Natural‑language scheduling with `schedule` and a persistent background `worker`.
- Browser automation for Gmail drafts and one‑click social posts.
- Built-in email sending and calendar event export.
- Full-screen Textual UI and optional cloud sync.
- Import/export, CSV import, and encrypted backups.
- Everything lives in one file (`crmd.py`) so packaging stays lightweight.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.in  # dependencies

# Or install system-wide once published:
pip install crmd  # works on Linux, macOS and Windows
```

To install globally via pip once packaged, run `pip install crmd`.
You can also build from source using `python -m build` which relies on the
provided `pyproject.toml` and `setup.py`.

To create a `.deb` package install `dpkg-deb` and run `scripts/build_deb.sh`.

## Usage

Export your OpenAI key once:

```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively store it once with:

```bash
crmd config set-api-key sk-...
```

Example workflow:

```bash
crmd add "Jordan Lee" --email jordan@example.com --tags vip
crmd log "Jordan Lee" --summary "Intro email" --channel email
crmd list
crmd plan "Jordan Lee"
crmd chat "Jordan Lee"    # interactive, `/exit` to quit
crmd remind "Jordan Lee" "Check in" --in 30
crmd due
crmd search "Jordan"
crmd draft-email "Jordan Lee" --subject "Hello"
```

### Additional commands

- `summary NAME` – use OpenAI to generate a short summary of a contact.
- `import-csv FILE` – bulk import contacts from a CSV file.
- `notify-due` – send desktop notifications for due reminders.
- `encrypt` – archive and encrypt your data directory with GPG.
- `export-json NAME PATH` – save a contact as JSON.
- `import-json PATH` – load a contact from JSON.
- `delete NAME` – remove a contact.
- `backup [PATH]` – tarball all data for backup.
- `schedule-browser URL --at TIME` – open a URL later.
- `schedule-post TEXT --at TIME [--platform x|facebook]` – post on social media later.
- `worker` – background loop that executes scheduled tasks.
- `graphs NAME` – show interaction counts with Textual.
- `undo` – revert the last change to a contact using git.
- `stats` – JSON totals for contacts, interactions and reminders.
- `tag-all TAG NAMES...` – add a tag to multiple contacts.
- `merge SRC DEST` – combine two contacts into one.
- `export-csv PATH` – save all contacts to a CSV file.
- `edit NAME` – open a contact YAML in `$EDITOR`.
- `send-email NAME --subject S --body B` – send an email via SMTP.
- `create-event NAME --when TIME --summary TEXT` – export an ICS file.
- `fullscreen` – full-screen Textual UI.
- `sync-cloud` – git push to a remote.
- `list-tags` – list all unique tags.
- `config set-api-key KEY` – store your OpenAI key in a config file.
- `list --tag TAG` – filter contacts by tag.
- `reminders [--clear N]` – list or delete reminders.
- `tasks [--clear N]` – list or delete scheduled tasks.

## Daemon / Service

To run `crmd` periodically (for example to send reminders) you can use a
systemd service. Create `~/.config/systemd/user/crmd.service`:

```ini
[Unit]
Description=CRMD background service

[Service]
Type=simple
ExecStart=/usr/bin/crmd dashboard
Restart=on-failure

[Install]
WantedBy=default.target
```

Then enable with:

```bash
systemctl --user enable crmd.service
systemctl --user start crmd.service
```

This example runs the dashboard at login; adjust `ExecStart` for other tasks.

## Scheduling Browser Actions

`crmd` can understand natural language and turn it into a timed Gmail draft. Use
`schedule` to create a task and `worker` to execute it in the background:

```bash
crmd schedule "draft follow up email to Aravind about the browser launch tomorrow at 08:30"
crmd worker
```

This will open a Gmail draft at the specified time (if your system is running).

You can also schedule social media posts:

```bash
crmd schedule-post "Check out our new release" --platform x --at "2025-06-21 10:00"
```

`worker` will then launch the relevant share URL when the time is reached.

## Git Sync

If your `CRMD_HOME` directory is a git repository, run:

```bash
crmd sync
```

This will add, commit, and push all changes so you can access them across devices.

## Development

Run tests with:

```bash
source .venv/bin/activate
pytest
```

CI is configured via GitHub Actions in `.github/workflows/ci.yml`. Tagged
releases will automatically publish to PyPI if the `PYPI_API_TOKEN` secret is
set.

For a quick visual overview, open `index.html` in your browser to see a minimal demo site styled like a terminal.

## Next nice-to-haves

- Mobile-friendly shell wrapper
- Voice note recording
- Local search index for speed
- Contact deduplication suggestions
