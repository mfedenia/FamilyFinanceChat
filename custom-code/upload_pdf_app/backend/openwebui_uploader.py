import json
import pathlib
import requests
import logging
from typing import Optional, Dict, Any

class OpenWebUIUploader:
    def __init__(self, base_url: str = "http://127.0.0.1:3000", api_key: str = "", kb_id: str = ""):
        self.base_url = base_url
        self.api_key = api_key
        self.kb_id = kb_id
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def upload_file(self, file_path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Upload a file to OpenWebUI and return its file_id"""
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        url_upload = f"{self.base_url}/api/v1/files/"
        data = {
            "metadata": json.dumps(metadata or {"source": "db_upload_menu"}),
            "process": "true",
            "process_in_background": "false",
        }
        
        with file_path.open("rb") as f:
            files = {"file": (file_path.name, f, "application/pdf")}
            response = requests.post(
                url_upload, 
                headers=self.headers, 
                data=data, 
                files=files, 
                timeout=120
            )
        
        response.raise_for_status()
        uploaded = response.json()
        file_id = uploaded["id"]
        logging.info(f"Uploaded {file_path.name}, file_id: {file_id}")
        return file_id
    
    def add_to_knowledge_base(self, file_id: str) -> Dict[str, Any]:
        """Add an uploaded file to the knowledge base"""
        url_add = f"{self.base_url}/api/v1/knowledge/{self.kb_id}/file/add"
        
        add_response = requests.post(
            url_add,
            headers={**self.headers, "Content-Type": "application/json"},
            data=json.dumps({"file_id": file_id}),
            timeout=120
        )
        
        if add_response.ok:
            logging.info(f"Successfully added file_id {file_id} to KB")
            return add_response.json()
        
        # Handle duplicate files
        if add_response.status_code == 400:
            logging.warning(f"File might be duplicate, attempting merge for file_id {file_id}")
            
            # Get current KB state
            kb = requests.get(
                f"{self.base_url}/api/v1/knowledge/{self.kb_id}",
                headers=self.headers,
                timeout=60
            ).json()
            
            current_ids = (kb.get("data") or {}).get("file_ids", [])
            
            if file_id not in current_ids:
                current_ids.append(file_id)
                
                body = {
                    "name": kb["name"],
                    "description": kb.get("description", ""),
                    "data": {"file_ids": current_ids},
                    "access_control": kb.get("access_control"),
                }
                
                update_response = requests.post(
                    f"{self.base_url}/api/v1/knowledge/{self.kb_id}/update",
                    headers={**self.headers, "Content-Type": "application/json"},
                    data=json.dumps(body),
                    timeout=120
                )
                
                update_response.raise_for_status()
                logging.info(f"Successfully merged file_id {file_id} into KB")
                return update_response.json()
            else:
                logging.info(f"File_id {file_id} already in KB")
                return {"status": "already_exists", "file_id": file_id}
        else:
            add_response.raise_for_status()
            return add_response.json()
    
    def upload_and_add_to_kb(self, file_path: pathlib.Path) -> Dict[str, Any]:
        """Upload a file and add it to the knowledge base in one go"""
        file_id = self.upload_file(file_path)
        result = self.add_to_knowledge_base(file_id)
        return {
            "file_id": file_id,
            "filename": file_path.name,
            "kb_result": result
        }