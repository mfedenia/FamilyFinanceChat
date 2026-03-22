"""
Knowledge Base Sync Tool for Open WebUI

Manages files in an Open WebUI knowledge base via the REST API.
Supports listing, adding, replacing, and removing files with a
local trash/archive for safe recovery of removed files.

Usage:
    python sync_kb.py list
    python sync_kb.py add <file_path>
    python sync_kb.py replace <file_path>
    python sync_kb.py remove <filename>
    python sync_kb.py sync <directory>

All destructive commands support --dry-run.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("kb_sync")

TRASH_DIR = Path(__file__).parent / ".trash"
MANIFEST_PATH = TRASH_DIR / "trash_manifest.json"


class KBSyncClient:
    """Client for managing Open WebUI knowledge base files."""

    def __init__(self, base_url: str, api_key: str, kb_id: str):
        self.base_url = base_url.rstrip("/")
        self.kb_id = kb_id
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        self.session.timeout = 120

    # -- API helpers --

    def get_knowledge_base(self) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/api/v1/knowledge/{self.kb_id}")
        resp.raise_for_status()
        return resp.json()

    def get_file_info(self, file_id: str) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/api/v1/files/{file_id}")
        resp.raise_for_status()
        return resp.json()

    def download_file_content(self, file_id: str) -> Optional[bytes]:
        resp = self.session.get(
            f"{self.base_url}/api/v1/files/{file_id}/content",
            headers={**self.session.headers, "Accept": "application/octet-stream"},
        )
        if resp.ok:
            return resp.content
        log.warning(f"Could not download file {file_id}: {resp.status_code}")
        return None

    def upload_file(self, file_path: Path) -> str:
        data = {
            "metadata": json.dumps({"source": "kb_sync"}),
            "process": "true",
            "process_in_background": "false",
        }
        with file_path.open("rb") as f:
            resp = self.session.post(
                f"{self.base_url}/api/v1/files/",
                data=data,
                files={"file": (file_path.name, f)},
            )
        resp.raise_for_status()
        file_id = resp.json()["id"]
        log.info(f"Uploaded {file_path.name} -> file_id: {file_id}")
        return file_id

    def add_file_to_kb(self, file_id: str) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/api/v1/knowledge/{self.kb_id}/file/add",
            json={"file_id": file_id},
        )
        resp.raise_for_status()
        log.info(f"Added file_id {file_id} to KB {self.kb_id}")
        return resp.json()

    def remove_file_from_kb(self, file_id: str) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/api/v1/knowledge/{self.kb_id}/file/remove",
            json={"file_id": file_id},
        )
        resp.raise_for_status()
        log.info(f"Removed file_id {file_id} from KB {self.kb_id}")
        return resp.json()

    def delete_file(self, file_id: str) -> bool:
        resp = self.session.delete(f"{self.base_url}/api/v1/files/{file_id}")
        if resp.ok:
            log.info(f"Deleted file_id {file_id} from Open WebUI")
            return True
        log.warning(f"Failed to delete file_id {file_id}: {resp.status_code}")
        return False

    # -- KB file listing --

    def list_kb_files(self) -> List[Dict[str, Any]]:
        kb = self.get_knowledge_base()
        kb_files = kb.get("files") or []
        if not kb_files:
            kb_files = [{"id": fid} for fid in (kb.get("data") or {}).get("file_ids", [])]

        files = []
        for f in kb_files:
            fid = f.get("id") or f
            fname = (
                f.get("filename")
                or (f.get("meta") or {}).get("name")
            )
            fsize = (f.get("meta") or {}).get("size")

            if not fname:
                try:
                    info = self.get_file_info(fid)
                    fname = info.get("filename", info.get("meta", {}).get("name", "unknown"))
                    fsize = info.get("meta", {}).get("size")
                except requests.HTTPError:
                    fname = "<unavailable>"

            files.append({
                "id": fid,
                "filename": fname or "unknown",
                "size": fsize,
                "created_at": f.get("created_at"),
            })
        return files

    # -- Trash / archive --

    def archive_file(self, file_id: str, filename: str) -> Optional[Path]:
        kb_trash = TRASH_DIR / self.kb_id
        kb_trash.mkdir(parents=True, exist_ok=True)

        content = self.download_file_content(file_id)
        if content is None:
            log.warning(f"Could not archive {filename} (download failed), proceeding anyway")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"{timestamp}_{filename}"
        archive_path = kb_trash / archive_name
        archive_path.write_bytes(content)
        log.info(f"Archived {filename} -> {archive_path}")

        self._update_manifest(file_id, filename, str(archive_path))
        return archive_path

    def _update_manifest(self, file_id: str, filename: str, archive_path: str):
        TRASH_DIR.mkdir(parents=True, exist_ok=True)
        manifest = []
        if MANIFEST_PATH.exists():
            manifest = json.loads(MANIFEST_PATH.read_text())

        manifest.append({
            "file_id": file_id,
            "filename": filename,
            "kb_id": self.kb_id,
            "archive_path": archive_path,
            "removed_at": datetime.now().isoformat(),
        })
        MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


# -- Commands --

def cmd_list(client: KBSyncClient, args):
    files = client.list_kb_files()
    if not files:
        print("Knowledge base is empty.")
        return

    print(f"\nFiles in KB {client.kb_id}:\n")
    print(f"  {'Filename':<40} {'File ID':<40} {'Size':>10}")
    print(f"  {'-'*40} {'-'*40} {'-':->10}")
    for f in files:
        size = f.get("size")
        size_str = f"{size:>10,}" if size else "     n/a"
        print(f"  {f['filename']:<40} {f['id']:<40} {size_str}")
    print(f"\n  Total: {len(files)} file(s)")


def cmd_add(client: KBSyncClient, args):
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        log.error(f"File not found: {file_path}")
        sys.exit(1)

    if args.dry_run:
        print(f"[DRY RUN] Would upload {file_path.name} and add to KB {client.kb_id}")
        return

    file_id = client.upload_file(file_path)
    client.add_file_to_kb(file_id)
    print(f"Added {file_path.name} to KB (file_id: {file_id})")


def cmd_replace(client: KBSyncClient, args):
    file_path = Path(args.file).resolve()
    if not file_path.exists():
        log.error(f"File not found: {file_path}")
        sys.exit(1)

    target_name = file_path.name
    existing = client.list_kb_files()
    matches = [f for f in existing if f["filename"] == target_name]

    if not matches:
        log.warning(f"No existing file named '{target_name}' in KB. Use 'add' instead.")
        sys.exit(1)

    if args.dry_run:
        for m in matches:
            print(f"[DRY RUN] Would archive and remove {m['filename']} (id: {m['id']})")
        print(f"[DRY RUN] Would upload new version of {target_name}")
        return

    for old_file in matches:
        client.archive_file(old_file["id"], old_file["filename"])
        client.remove_file_from_kb(old_file["id"])
        client.delete_file(old_file["id"])

    file_id = client.upload_file(file_path)
    client.add_file_to_kb(file_id)
    print(f"Replaced {target_name} in KB (new file_id: {file_id})")


def cmd_remove(client: KBSyncClient, args):
    target_name = args.filename
    existing = client.list_kb_files()
    matches = [f for f in existing if f["filename"] == target_name]

    if not matches:
        log.error(f"No file named '{target_name}' found in KB")
        sys.exit(1)

    if args.dry_run:
        for m in matches:
            print(f"[DRY RUN] Would archive and remove {m['filename']} (id: {m['id']})")
        return

    for old_file in matches:
        client.archive_file(old_file["id"], old_file["filename"])
        client.remove_file_from_kb(old_file["id"])
        client.delete_file(old_file["id"])
        print(f"Removed {old_file['filename']} from KB (archived to .trash/)")


def cmd_sync(client: KBSyncClient, args):
    dir_path = Path(args.directory).resolve()
    if not dir_path.is_dir():
        log.error(f"Not a directory: {dir_path}")
        sys.exit(1)

    local_files = {f.name: f for f in dir_path.iterdir() if f.is_file()}
    remote_files = client.list_kb_files()
    remote_by_name = {f["filename"]: f for f in remote_files}

    to_add = [name for name in local_files if name not in remote_by_name]
    to_remove = [name for name in remote_by_name if name not in local_files]
    to_replace = [name for name in local_files if name in remote_by_name]

    print(f"\nSync plan for KB {client.kb_id}:")
    print(f"  Directory: {dir_path}")
    print(f"  Add:     {len(to_add)} file(s)")
    for name in to_add:
        print(f"           + {name}")
    print(f"  Remove:  {len(to_remove)} file(s)")
    for name in to_remove:
        print(f"           - {name}")
    print(f"  Exists:  {len(to_replace)} file(s) (no action, use 'replace' to update)")
    print()

    if args.dry_run:
        print("[DRY RUN] No changes made.")
        return

    if to_remove:
        confirm = input(f"Remove {len(to_remove)} file(s) from KB? (yes/no): ")
        if confirm.strip().lower() != "yes":
            print("Aborted removal. Adding new files only.")
            to_remove = []

    for name in to_remove:
        rf = remote_by_name[name]
        client.archive_file(rf["id"], rf["filename"])
        client.remove_file_from_kb(rf["id"])
        client.delete_file(rf["id"])
        print(f"  Removed {name}")

    for name in to_add:
        file_id = client.upload_file(local_files[name])
        client.add_file_to_kb(file_id)
        print(f"  Added {name}")

    print("\nSync complete.")


def main():
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "--base-url",
        default=os.getenv("OPENWEBUI_BASE_URL", "https://case.fedenia.us"),
    )
    shared.add_argument(
        "--api-key",
        default=os.getenv("OPENWEBUI_API_KEY", ""),
    )
    shared.add_argument(
        "--kb-id",
        default=os.getenv("OPENWEBUI_KB_ID", ""),
    )
    shared.add_argument("--dry-run", action="store_true", help="Show what would happen without making changes")
    shared.add_argument("-v", "--verbose", action="store_true")

    parser = argparse.ArgumentParser(
        description="Manage Open WebUI knowledge base files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[shared],
        epilog="""
Examples:
  python sync_kb.py list
  python sync_kb.py add ./new_document.pdf
  python sync_kb.py replace ./updated_document.pdf
  python sync_kb.py remove old_document.pdf
  python sync_kb.py sync ./context_files/
  python sync_kb.py sync ./context_files/ --dry-run
        """,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List files in the knowledge base")

    add_parser = subparsers.add_parser("add", help="Upload and add a file to the KB")
    add_parser.add_argument("file", help="Path to the file to add")

    replace_parser = subparsers.add_parser("replace", help="Replace an existing file (by filename match)")
    replace_parser.add_argument("file", help="Path to the new version of the file")

    remove_parser = subparsers.add_parser("remove", help="Remove a file from the KB (archived to .trash/)")
    remove_parser.add_argument("filename", help="Name of the file to remove")

    sync_parser = subparsers.add_parser("sync", help="Sync a local directory with the KB")
    sync_parser.add_argument("directory", help="Path to the directory to sync")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.api_key:
        log.error("API key required. Set OPENWEBUI_API_KEY or use --api-key")
        sys.exit(1)
    if not args.kb_id:
        log.error("Knowledge base ID required. Set OPENWEBUI_KB_ID or use --kb-id")
        sys.exit(1)

    client = KBSyncClient(args.base_url, args.api_key, args.kb_id)

    commands = {
        "list": cmd_list,
        "add": cmd_add,
        "replace": cmd_replace,
        "remove": cmd_remove,
        "sync": cmd_sync,
    }
    try:
        commands[args.command](client, args)
    except requests.ConnectionError as e:
        log.error(f"Cannot connect to {args.base_url}: {e}")
        sys.exit(1)
    except requests.HTTPError as e:
        log.error(f"API error: {e.response.status_code} {e.response.text[:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
