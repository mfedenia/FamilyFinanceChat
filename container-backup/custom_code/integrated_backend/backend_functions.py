#!/usr/bin/env python3
"""
Backend functions for PDF processing - container version
No FastAPI server, just functions to be called directly
"""
import json
import shutil
import subprocess
import sys
import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Container paths - use /app/backend/data for OpenWebUI's data
DATA_DIR = Path("/app/backend/data/pdf_processing")
SCRAPED = DATA_DIR / "webscraped"
KB = DATA_DIR / "knowledge_base"
THUMBNAILS = DATA_DIR / "thumbnails"
STATE_FILE = DATA_DIR / "pdf_state.json"
INPUT_DIR = DATA_DIR / "input_files"
WEBSCRAPING_DIR = Path("/app/custom_code/integrated_backend/Webscraping")

# Create directories
for dir_path in [DATA_DIR, SCRAPED, KB, THUMBNAILS, INPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def cleanup_on_startup():
    """Clean up old temporary files on startup"""
    logging.info("=== Cleaning up on startup ===")
    
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        logging.info("Cleared old state file")
    
    for pdf in SCRAPED.glob("*.pdf"):
        pdf.unlink()
        logging.info(f"Cleared old PDF: {pdf.name}")
    
    for thumb in THUMBNAILS.glob("*.png"):
        thumb.unlink()
        logging.info(f"Cleared old thumbnail: {thumb.name}")
    
    for pdf in INPUT_DIR.glob("*.pdf"):
        pdf.unlink()
        logging.info(f"Cleared old input: {pdf.name}")
    
    logging.info("Startup cleanup complete")

def generate_thumbnail(pdf_path: Path, thumbnail_dir: Path) -> Optional[Path]:
    """Generate a thumbnail for a PDF file"""
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumbnail_dir / f"{pdf_path.stem}.png"
        
        if output_path.exists():
            logging.info(f"Thumbnail already exists: {output_path}")
            return output_path
        
        images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=72)
        if images:
            img = images[0]
            img.thumbnail((200, 260), Image.Resampling.LANCZOS)
            img.save(output_path, "PNG")
            logging.info(f"Generated thumbnail: {output_path}")
            return output_path
        
        logging.warning(f"No pages found in PDF: {pdf_path}")
        return None
        
    except Exception as e:
        logging.error(f"Error generating thumbnail for {pdf_path}: {e}")
        return None

def load_state():
    """Load state from file"""
    if STATE_FILE.exists():
        try:
            saved_state = json.loads(STATE_FILE.read_text())
            return saved_state
        except Exception as e:
            logging.error(f"Error loading state: {e}")
            return []
    return []

def save_state(data):
    """Save state to file"""
    STATE_FILE.write_text(json.dumps(data, indent=2))

def list_pdfs() -> List[Dict[str, Any]]:
    """List all PDFs in the webscraped directory"""
    pdf_files = list(SCRAPED.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDFs in {SCRAPED}")
    
    if not pdf_files:
        if STATE_FILE.exists():
            STATE_FILE.unlink()
        return []
    
    saved_state = load_state()
    old_exclusions = {item["name"]: item.get("excluded", False) for item in saved_state}
    
    files = []
    for pdf in pdf_files:
        logging.info(f"Processing PDF: {pdf.name}")
        
        thumb_path = THUMBNAILS / f"{pdf.stem}.png"
        if not thumb_path.exists():
            generated_thumb = generate_thumbnail(pdf, THUMBNAILS)
            if generated_thumb:
                thumb_path = generated_thumb
            else:
                thumb_path = None
        
        file_info = {
            "name": pdf.name,
            "size_kb": round(pdf.stat().st_size / 1024, 1),
            "preview_url": f"/api/v1/custom/thumbnail/{thumb_path.name}" if thumb_path and thumb_path.exists() else None,
            "excluded": old_exclusions.get(pdf.name, False)
        }
        
        files.append(file_info)
    
    new_state = [{"name": f["name"], "excluded": f["excluded"]} for f in files]
    save_state(new_state)
    
    return files

def toggle_pdf_exclusion(pdf_name: str, excluded: bool) -> Dict[str, Any]:
    """Toggle PDF exclusion status"""
    state = load_state()
    
    for item in state:
        if item["name"] == pdf_name:
            item["excluded"] = excluded
            save_state(state)
            logging.info(f"Updated {pdf_name}: excluded={excluded}")
            return {"status": "success", "name": pdf_name, "excluded": excluded}
    
    return {"status": "error", "message": f"PDF {pdf_name} not found"}

def upload_and_crawl(files: List[Any]) -> Dict[str, Any]:
    """Process uploaded PDFs and trigger web crawling"""
    
    # Clear all old data
    logging.info("=== Clearing old data from previous runs ===")
    
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    
    for old_pdf in SCRAPED.glob("*.pdf"):
        old_pdf.unlink()
    
    for old_thumb in THUMBNAILS.glob("*.png"):
        old_thumb.unlink()
    
    for old_kb_pdf in KB.glob("*.pdf"):
        old_kb_pdf.unlink()
    
    for old_file in INPUT_DIR.glob("*.pdf"):
        old_file.unlink()
    
    # Save uploaded files
    saved_files = []
    for file_data in files:
        if file_data['filename'].endswith('.pdf'):
            file_path = INPUT_DIR / file_data['filename']
            # Write the file content
            with open(file_path, "wb") as f:
                f.write(file_data['content'])
            saved_files.append(str(file_path))
            logging.info(f"Saved uploaded file: {file_path}")
    
    if not saved_files:
        return {"status": "error", "message": "No PDF files uploaded"}
    
    # Run the link_downloader script
    script_path = WEBSCRAPING_DIR / "link_downloader.py"
  
    if not script_path.exists():
        logging.error(f"Script not found: {script_path}")
        return {
            "status": "error",
            "message": "Web scraping script not found in container"
        }
    
    SCRAPED.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(script_path),
        str(INPUT_DIR),
        "--out", str(SCRAPED),
        "--depth", "0",
        "--render-pages",
        "--skip-existing",
        "--max-from-page", "3",
        "-v"
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(WEBSCRAPING_DIR),
            timeout=300
        )
        
        logging.info(f"Crawler stdout: {result.stdout}")
        if result.stderr:
            logging.error(f"Crawler stderr: {result.stderr}")
        
        pdfs_found = len(list(SCRAPED.glob("*.pdf")))
        logging.info(f"PDFs found after crawling: {pdfs_found}")
        
        return {
            "status": "success",
            "message": f"Uploaded {len(saved_files)} files, found {pdfs_found} PDFs",
            "uploaded": len(saved_files),
            "pdfs_found": pdfs_found
        }
        
    except subprocess.TimeoutExpired:
        logging.error("Crawler timed out after 5 minutes")
        return {"status": "error", "message": "Crawling timed out"}
    except Exception as e:
        logging.error(f"Error running crawler: {e}")
        return {"status": "error", "message": str(e)}

def finalize_upload_to_openwebui() -> Dict[str, Any]:
    """Move non-excluded PDFs to OpenWebUI knowledge base"""
    import requests
    
    state = load_state()
    
    # Get environment variables
    OPENWEBUI_API_KEY = os.getenv("OPENWEBUI_API_KEY", "sk-c1811e512d9d4bf0b324b0482ccb18a9")
    OPENWEBUI_KB_ID = os.getenv("OPENWEBUI_KB_ID", "caf7b373-60ad-4238-9a06-8511cd6ce05a")
    
    # Only include PDFs that are not excluded
    include = [pdf for pdf in state if not pdf["excluded"]]
    moved = []
    uploaded_to_openwebui = []
    upload_errors = []
    
    for pdf_data in include:
        source = SCRAPED / pdf_data["name"]
        if source.exists():
            try:
                # Upload to OpenWebUI via internal API
                with open(source, "rb") as f:
                    files = {"file": (pdf_data["name"], f, "application/pdf")}
                    headers = {"Authorization": f"Bearer {OPENWEBUI_API_KEY}"}
                    
                    # Upload file
                    response = requests.post(
                        "http://localhost:8080/api/v1/files/",
                        headers=headers,
                        files=files,
                        data={
                            "metadata": json.dumps({"source": "pdf_crawler"}),
                            "process": "true"
                        }
                    )
                    
                    if response.ok:
                        result = response.json()
                        file_id = result["id"]
                        
                        # Add to knowledge base
                        kb_response = requests.post(
                            f"http://localhost:8080/api/v1/knowledge/{OPENWEBUI_KB_ID}/files",
                            headers=headers,
                            json={"file_ids": [file_id]}
                        )
                        
                        uploaded_to_openwebui.append({
                            "filename": pdf_data["name"],
                            "file_id": file_id
                        })
                        moved.append(pdf_data["name"])
                        logging.info(f"Uploaded to OpenWebUI: {pdf_data['name']}")
                
            except Exception as e:
                logging.error(f"Error uploading {pdf_data['name']}: {e}")
                upload_errors.append({
                    "filename": pdf_data["name"],
                    "error": str(e)
                })
    
    # Clear the scraped folder after successful upload
    for pdf in SCRAPED.glob("*.pdf"):
        pdf.unlink()
    
    STATE_FILE.unlink() if STATE_FILE.exists() else None
    
    return {
        "status": "success",
        "message": f"Uploaded {len(uploaded_to_openwebui)} PDFs to OpenWebUI",
        "moved": moved,
        "uploaded_to_openwebui": uploaded_to_openwebui,
        "upload_errors": upload_errors
    }

# Initialize on import
cleanup_on_startup()