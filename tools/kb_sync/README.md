# Knowledge Base Sync Tool

CLI tool for managing files in an Open WebUI knowledge base via the REST API.

## Setup

```bash
cd tools/kb_sync
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API key and KB ID
```

## Usage

```bash
# List files currently in the knowledge base
python sync_kb.py list

# Add a new file
python sync_kb.py add ./path/to/document.pdf

# Replace an existing file (matches by filename)
python sync_kb.py replace ./path/to/updated_document.pdf

# Remove a file (archived to .trash/ before deletion)
python sync_kb.py remove old_document.pdf

# Sync a directory with the KB (preview first with --dry-run)
python sync_kb.py sync ./context_files/ --dry-run
python sync_kb.py sync ./context_files/
```

## Safety

All destructive operations (replace, remove, sync with removals) archive the
file locally before deleting it from Open WebUI. Archived files are saved to
`.trash/{kb_id}/{timestamp}_{filename}` with a manifest at
`.trash/trash_manifest.json`.

The `sync` command requires explicit confirmation before removing files.

Use `--dry-run` on any command to preview changes without making them.

## Configuration

Set via `.env` file or command-line flags:

| Variable | Flag | Description |
|---|---|---|
| `OPENWEBUI_BASE_URL` | `--base-url` | Open WebUI URL (default: https://case.fedenia.us) |
| `OPENWEBUI_API_KEY` | `--api-key` | API key for authentication |
| `OPENWEBUI_KB_ID` | `--kb-id` | Knowledge base ID to manage |
