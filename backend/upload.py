# test_upload.py — upload a file, then attach it to a Knowledge Base (KB)
import json
import pathlib
import requests

BASE_URL = "http://127.0.0.1:3000"  # talk to OpenWebUI on the VM
API_KEY  = ""
KB_ID    = "d2d01280-b703-4c94-8d53-058e9b3ff3b1"  # your KB id

FILE_PATH = pathlib.Path("/opt/openwebui-src/sample.txt")

def main():
    if not FILE_PATH.exists():
        raise SystemExit(f"File not found: {FILE_PATH}")

    headers = {"Authorization": f"Bearer {API_KEY}"}

    # 1) Upload file (multipart/form-data); let backend process it
    url_upload = f"{BASE_URL}/api/v1/files/"
    data = {
        "metadata": json.dumps({"source": "test_upload.py"}),  # no collection_name here
        "process": "true",
        "process_in_background": "false",
    }
    with FILE_PATH.open("rb") as f:
        files = {"file": (FILE_PATH.name, f, "text/plain")}
        up = requests.post(url_upload, headers=headers, data=data, files=files, timeout=120)
    up.raise_for_status()
    uploaded = up.json()
    file_id = uploaded["id"]
    print("Uploaded file_id:", file_id)

    # (optional) check processing status
    info = requests.get(f"{BASE_URL}/api/v1/files/{file_id}", headers=headers, timeout=60).json()
    print("File status:", info.get("data", {}).get("status"), "| path:", info.get("path"))

    # 2) Attach that file to your KB (adds to vector collection AND to KB file_ids)
    url_add = f"{BASE_URL}/api/v1/knowledge/{KB_ID}/file/add"
    add = requests.post(url_add, headers={**headers, "Content-Type": "application/json"},
                        data=json.dumps({"file_id": file_id}), timeout=120)

    if add.ok:
        print("KB add OK")
        print(add.json())
        return

    # 3) Fallback if add complains about duplicates: merge file_id into KB list
    if add.status_code == 400:
        print("KB add reported 400; falling back to KB update merge…")

        kb = requests.get(f"{BASE_URL}/api/v1/knowledge/{KB_ID}", headers=headers, timeout=60).json()
        current_ids = (kb.get("data") or {}).get("file_ids", [])
        if file_id not in current_ids:
            current_ids.append(file_id)

        body = {
            "name": kb["name"],
            "description": kb.get("description", ""),
            "data": {"file_ids": current_ids},
            "access_control": kb.get("access_control"),
        }
        upd = requests.post(f"{BASE_URL}/api/v1/knowledge/{KB_ID}/update",
                            headers={**headers, "Content-Type": "application/json"},
                            data=json.dumps(body), timeout=120)
        upd.raise_for_status()
        print("KB update OK")
        print(upd.json())
    else:
        # propagate unexpected errors
        add.raise_for_status()

if __name__ == "__main__":
    main()
