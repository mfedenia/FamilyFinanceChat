"""
Custom PDF Router - Integrated into Open WebUI
Handles PDF upload, web crawling, and knowledge base integration
"""

import logging
import json
import shutil
import subprocess
import sys
import os
import io
import uuid
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, BackgroundTasks
from fastapi.responses import FileResponse, Response, PlainTextResponse

from pydantic import BaseModel

from open_webui.utils.auth import get_verified_user
from open_webui.models.knowledge import Knowledges
from open_webui.models.files import Files, FileForm
from open_webui.storage.provider import Storage
from open_webui.routers.retrieval import ProcessFileForm, process_file

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

if not log.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

router = APIRouter()

executor = ThreadPoolExecutor(max_workers=2)

# ============================================================================
# Configuration
# ============================================================================

def get_data_dir() -> Path:
    """Get the data directory"""
    if os.path.exists("/app/backend/data"):
        base = Path("/app/backend/data")
    else:
        base = Path(__file__).parent.parent.parent / "data"
    
    custom_dir = base / "custom_pdf_crawler"
    custom_dir.mkdir(parents=True, exist_ok=True)
    return custom_dir

def get_paths():
    """Get all required paths"""
    data_dir = get_data_dir()
    return {
        "data_dir": data_dir,
        "scraped": data_dir / "webscraped",
        "thumbnails": data_dir / "thumbnails", 
        "input_dir": data_dir / "input_files",
        "state_file": data_dir / "pdf_state.json",
        "job_file": data_dir / "job_status.json"
    }

paths = get_paths()
for key in ["scraped", "thumbnails", "input_dir"]:
    paths[key].mkdir(parents=True, exist_ok=True)

# ============================================================================
# Pydantic Models
# ============================================================================

class PDFItem(BaseModel):
    name: str
    excluded: bool = False

class PDFListItem(BaseModel):
    name: str
    size_kb: float
    excluded: bool = False
    preview_url: Optional[str] = None

class UploadResponse(BaseModel):
    message: str
    job_id: str
    status: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    message: str
    pdfs_found: int = 0
    progress: int = 0

class FinalizeRequest(BaseModel):
    knowledge_id: Optional[str] = None

class FinalizeResponse(BaseModel):
    message: str
    moved: List[str]
    uploaded_to_openwebui: List[str] = []
    added_to_knowledge: List[str] = []
    upload_errors: List[dict] = []

# ============================================================================
# Job Management
# ============================================================================

def load_job_status() -> dict:
    paths = get_paths()
    if paths["job_file"].exists():
        try:
            with open(paths["job_file"], "r") as f:
                return json.load(f)
        except:
            pass
    return {"job_id": None, "status": "idle", "message": "", "pdfs_found": 0, "progress": 0}

def save_job_status(job_id: str, status: str, message: str, pdfs_found: int = 0, progress: int = 0):
    paths = get_paths()
    data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "pdfs_found": pdfs_found,
        "progress": progress
    }
    with open(paths["job_file"], "w") as f:
        json.dump(data, f)

# ============================================================================
# State Management
# ============================================================================

def load_state() -> List[dict]:
    paths = get_paths()
    if paths["state_file"].exists():
        try:
            with open(paths["state_file"], "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_state(data: List[dict]):
    paths = get_paths()
    with open(paths["state_file"], "w") as f:
        json.dump(data, f)

# ============================================================================
# Thumbnail Generation
# ============================================================================

def generate_thumbnail(pdf_path: Path, thumbnail_dir: Path) -> Optional[Path]:
    """Generate a thumbnail for a PDF file using PyMuPDF (fitz)"""
    try:
        import fitz  # PyMuPDF
        
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumbnail_dir / f"{pdf_path.stem}.png"
        
        # Check if thumbnail already exists
        if output_path.exists():
            log.debug(f"Thumbnail already exists: {output_path}")
            return output_path
        
        log.info(f"Generating thumbnail for: {pdf_path}")
        
        # Open the PDF
        doc = fitz.open(pdf_path)
        
        if len(doc) == 0:
            log.warning(f"PDF has no pages: {pdf_path}")
            doc.close()
            return None
        
        # Get the first page
        page = doc[0]
        
        # Render page to image (pixmap)
        # Use a matrix to scale - 0.5 gives a reasonable thumbnail size
        mat = fitz.Matrix(0.5, 0.5)
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        pix.save(str(output_path))
        
        doc.close()
        
        log.info(f"Generated thumbnail: {output_path}")
        return output_path
        
    except ImportError:
        log.error("PyMuPDF (fitz) not installed, cannot generate thumbnails")
        return None
    except Exception as e:
        log.error(f"Error generating thumbnail for {pdf_path}: {e}")
        return None

# ============================================================================
# Web Scraping
# ============================================================================

def find_link_downloader() -> Optional[Path]:
    """Find link_downloader.py in various possible locations"""
    script_locations = [
        # Check relative to this router file
        Path(__file__).parent / "Webscraping" / "link_downloader.py",
        # Check in the routers directory (if copied there)
        Path("/app/backend/open_webui/routers/Webscraping/link_downloader.py"),
        # Check in custom_code integrated_backend
        Path("/app/custom_code/integrated_backend/Webscraping/link_downloader.py"),
        # Check in custom_code root
        Path("/app/custom_code/Webscraping/link_downloader.py"),
        # Check in custom-code (with hyphen) upload_pdf_app backend
        Path("/app/custom-code/upload_pdf_app/backend/Webscraping/link_downloader.py"),
        # Check in custom-code integrated_backend
        Path("/app/custom-code/integrated_backend/Webscraping/link_downloader.py"),
    ]
    
    for loc in script_locations:
        log.debug(f"Checking for link_downloader.py at: {loc}")
        if loc.exists():
            log.info(f"Found link_downloader.py at: {loc}")
            return loc
    
    log.warning("link_downloader.py not found in any known location")
    return None

def run_crawl_job(job_id: str, input_dir: Path, output_dir: Path):
    """Run the link_downloader.py script to crawl PDFs"""
    log.info(f"[Job {job_id}] Starting crawl job...")
    save_job_status(job_id, "running", "Starting web crawler...", 0, 10)
    
    script_path = find_link_downloader()
    
    if not script_path:
        log.error(f"[Job {job_id}] link_downloader.py not found")
        # Copy input files to output as fallback
        for pdf in input_dir.glob("*.pdf"):
            shutil.copy2(pdf, output_dir / pdf.name)
        pdf_count = len(list(output_dir.glob("*.pdf")))
        save_job_status(job_id, "completed", f"Crawler not found. Using {pdf_count} uploaded files.", pdf_count, 100)
        return
    
    save_job_status(job_id, "running", "Extracting links from PDFs...", 0, 20)
    
    input_files = list(input_dir.glob("*.pdf"))
    if not input_files:
        save_job_status(job_id, "failed", "No PDF files found", 0, 100)
        return
    
    log.info(f"[Job {job_id}] Found {len(input_files)} input PDFs: {[f.name for f in input_files]}")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # IMPORTANT: Copy input PDFs to output directory FIRST
    # This ensures the original uploaded files are always included
    for pdf in input_files:
        dest = output_dir / pdf.name
        if not dest.exists():
            shutil.copy2(pdf, dest)
            log.info(f"[Job {job_id}] Copied input PDF to output: {pdf.name}")
    
    cmd = [
        sys.executable,
        str(script_path),
        str(input_dir),
        "--out", str(output_dir),
        "--depth", "1",
        "--skip-existing",
        "--max-from-page", "10",
        "-v"
    ]
    
    log.info(f"[Job {job_id}] Running: {' '.join(cmd)}")
    log.info(f"[Job {job_id}] Working directory: {script_path.parent}")
    save_job_status(job_id, "running", "Downloading PDFs from links...", 0, 30)
    
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(script_path.parent),
            env=env
        )
        
        progress = 30
        stdout_lines = []
        
        # Read stdout
        for line in process.stdout:
            line = line.strip()
            if line:
                log.info(f"[Job {job_id}] STDOUT: {line}")
                stdout_lines.append(line)
                
                if "Downloading" in line or "Downloaded" in line or "Found" in line:
                    progress = min(progress + 5, 90)
                    pdf_count = len(list(output_dir.glob("*.pdf")))
                    save_job_status(job_id, "running", line[:100], pdf_count, progress)
        
        # Wait for process to complete
        process.wait()
        
        # Read any stderr
        stderr = process.stderr.read()
        if stderr:
            log.error(f"[Job {job_id}] STDERR: {stderr}")
        
        log.info(f"[Job {job_id}] Process exited with code: {process.returncode}")
        
        pdf_count = len(list(output_dir.glob("*.pdf")))
        log.info(f"[Job {job_id}] PDFs in output directory: {pdf_count}")
        log.info(f"[Job {job_id}] Output files: {[f.name for f in output_dir.glob('*.pdf')]}")
        
        # If somehow no PDFs in output (shouldn't happen since we copied input first)
        if pdf_count == 0:
            log.info(f"[Job {job_id}] No PDFs in output, copying uploaded files")
            for pdf in input_dir.glob("*.pdf"):
                dest = output_dir / pdf.name
                shutil.copy2(pdf, dest)
                log.info(f"[Job {job_id}] Copied: {pdf.name} -> {dest}")
            pdf_count = len(list(output_dir.glob("*.pdf")))
        
        log.info(f"[Job {job_id}] Completed with {pdf_count} PDFs")
        save_job_status(job_id, "completed", f"Found {pdf_count} PDFs", pdf_count, 100)
        
    except Exception as e:
        log.error(f"[Job {job_id}] Error: {e}")
        import traceback
        log.error(f"[Job {job_id}] Traceback: {traceback.format_exc()}")
        
        # Make sure input files are in output even on error
        for pdf in input_dir.glob("*.pdf"):
            dest = output_dir / pdf.name
            if not dest.exists():
                shutil.copy2(pdf, dest)
        pdf_count = len(list(output_dir.glob("*.pdf")))
        
        save_job_status(job_id, "completed", f"Crawler error. Using {pdf_count} uploaded files.", pdf_count, 100)

# ============================================================================
# Knowledge Base Integration
# ============================================================================

def add_file_to_knowledge_base(file_id: str, knowledge_id: str, user_id: str) -> dict:
    """Add a file to a knowledge base, similar to openwebui_uploader.py approach"""
    try:
        knowledge = Knowledges.get_knowledge_by_id(knowledge_id)
        if not knowledge:
            return {"success": False, "error": "Knowledge base not found"}
        
        # Get current file IDs
        current_data = knowledge.data or {}
        current_file_ids = current_data.get("file_ids", [])
        
        if file_id in current_file_ids:
            log.info(f"File {file_id} already in knowledge base {knowledge_id}")
            return {"success": True, "status": "already_exists"}
        
        # Add the new file ID
        current_file_ids.append(file_id)
        
        # Update the knowledge base
        updated_data = {**current_data, "file_ids": current_file_ids}
        result = Knowledges.update_knowledge_data_by_id(knowledge_id, updated_data)
        
        if result:
            log.info(f"Successfully added file {file_id} to knowledge base {knowledge_id}")
            return {"success": True, "status": "added"}
        else:
            return {"success": False, "error": "Failed to update knowledge base"}
            
    except Exception as e:
        log.error(f"Error adding file to knowledge base: {e}")
        return {"success": False, "error": str(e)}


# ============================================================================
# API Endpoints
# ============================================================================

@router.post("/pdf-upload", response_model=UploadResponse)
async def upload_and_crawl(
    request: Request,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    user=Depends(get_verified_user)
):
    """Upload PDFs and start background crawling job"""
    paths = get_paths()
    
    log.info(f"=== PDF Upload started by user {user.id} ===")
    
    job_id = str(uuid.uuid4())[:8]
    
    for old_file in [paths["state_file"], paths["job_file"]]:
        if old_file.exists():
            old_file.unlink()
    
    for folder in [paths["scraped"], paths["thumbnails"], paths["input_dir"]]:
        for f in folder.glob("*"):
            if f.is_file():
                f.unlink()
    
    saved_count = 0
    for file in files:
        if file.filename and file.filename.lower().endswith('.pdf'):
            file_path = paths["input_dir"] / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_count += 1
            log.info(f"Saved: {file.filename}")
    
    if saved_count == 0:
        raise HTTPException(status_code=400, detail="No PDF files uploaded")
    
    save_job_status(job_id, "pending", f"Uploaded {saved_count} files, starting crawler...", 0, 5)
    
    background_tasks.add_task(
        run_crawl_job,
        job_id,
        paths["input_dir"],
        paths["scraped"]
    )
    
    return UploadResponse(
        message=f"Uploaded {saved_count} files, crawling started",
        job_id=job_id,
        status="pending"
    )


@router.get("/pdf-job-status", response_model=JobStatusResponse)
async def get_job_status(
    request: Request,
    user=Depends(get_verified_user)
):
    """Get the current crawling job status"""
    status = load_job_status()
    return JobStatusResponse(**status)


@router.get("/pdf-list", response_model=List[PDFListItem])
async def list_pdfs(
    request: Request,
    user=Depends(get_verified_user)
):
    """List all crawled PDFs"""
    paths = get_paths()
    
    pdf_files = list(paths["scraped"].glob("*.pdf"))
    log.info(f"Listing {len(pdf_files)} PDFs from {paths['scraped']}")
    
    if not pdf_files:
        return []
    
    saved_state = load_state()
    exclusions = {item["name"]: item.get("excluded", False) for item in saved_state}
    
    result = []
    new_state = []
    
    for pdf in pdf_files:
        thumb_path = paths["thumbnails"] / f"{pdf.stem}.png"
        thumbnail_generated = False
        
        if not thumb_path.exists():
            generated = generate_thumbnail(pdf, paths["thumbnails"])
            thumbnail_generated = generated is not None
        else:
            thumbnail_generated = True
        
        is_excluded = exclusions.get(pdf.name, False)
        
        new_state.append({"name": pdf.name, "excluded": is_excluded})
        
        preview_url = None
        if thumbnail_generated and thumb_path.exists():
            preview_url = f"/api/v1/custom/pdf-thumbnail/{pdf.stem}.png"
        
        result.append(PDFListItem(
            name=pdf.name,
            size_kb=round(pdf.stat().st_size / 1024, 1),
            excluded=is_excluded,
            preview_url=preview_url
        ))
    
    save_state(new_state)
    return result


@router.get("/pdf-thumbnail/{filename}")
async def get_thumbnail(
    filename: str,
    request: Request,
    user=Depends(get_verified_user)
):
    """Serve thumbnail images"""
    paths = get_paths()
    
    if not filename.endswith('.png'):
        filename = f"{filename}.png"
    
    thumb_path = paths["thumbnails"] / filename
    
    if not thumb_path.exists():
        raise HTTPException(status_code=404, detail=f"Thumbnail not found: {filename}")
    
    return FileResponse(
        thumb_path, 
        media_type="image/png",
        headers={"Cache-Control": "max-age=3600"}
    )


@router.patch("/pdf-toggle/{name}")
async def toggle_exclusion(
    name: str,
    item: PDFItem,
    request: Request,
    user=Depends(get_verified_user)
):
    """Toggle PDF exclusion status"""
    state = load_state()
    
    found = False
    for pdf in state:
        if pdf["name"] == name:
            pdf["excluded"] = item.excluded
            found = True
            break
    
    if not found:
        state.append({"name": name, "excluded": item.excluded})
    
    save_state(state)
    return {"name": name, "excluded": item.excluded}

@router.post("/pdf-finalize", response_model=FinalizeResponse)
async def finalize_upload(
    request: Request,
    form_data: Optional[FinalizeRequest] = None,
    user=Depends(get_verified_user)
):
    """
    Upload selected PDFs to Open WebUI and optionally add to a knowledge base.
    Similar to openwebui_uploader.py but integrated directly.
    """
    paths = get_paths()
    state = load_state()
    
    log.info(f"=== Finalize Upload ===")
    log.info(f"Loaded state: {state}")
    log.info(f"Scraped dir: {paths['scraped']}")
    log.info(f"Files in scraped: {[f.name for f in paths['scraped'].glob('*.pdf')]}")
    
    # Get knowledge_id from request body if provided
    knowledge_id = None
    if form_data and form_data.knowledge_id:
        knowledge_id = form_data.knowledge_id
    
    # If state is empty, build it from files in scraped directory
    if not state:
        log.warning("State is empty, building from scraped directory...")
        for pdf in paths["scraped"].glob("*.pdf"):
            state.append({"name": pdf.name, "excluded": False})
        log.info(f"Built state from files: {state}")
    
    included = [pdf for pdf in state if not pdf.get("excluded", False)]
    log.info(f"Included PDFs: {included}")
    
    if not included:
        raise HTTPException(status_code=400, detail="No PDFs selected")
    
    moved = []
    uploaded = []
    added_to_kb = []
    errors = []
    
    for pdf_data in included:
        source = paths["scraped"] / pdf_data["name"]
        log.info(f"Processing: {pdf_data['name']}, source exists: {source.exists()}")
        
        if not source.exists():
            log.error(f"Source file not found: {source}")
            errors.append({"filename": pdf_data["name"], "error": "File not found"})
            continue
        
        try:
            # Read file content
            with open(source, "rb") as f:
                content = f.read()
            
            log.info(f"Read {len(content)} bytes from {source}")
            
            # Generate unique ID (like files.py does)
            file_id = str(uuid.uuid4())
            original_name = pdf_data["name"]
            
            # Create storage filename with UUID prefix (like files.py line 202)
            storage_filename = f"{file_id}_{original_name}"
            
            # Upload to storage
            log.info(f"Uploading to storage: {storage_filename}")
            
            file_obj = io.BytesIO(content)
            tags = {
                "OpenWebUI-User-Email": user.email,
                "OpenWebUI-User-Id": user.id,
                "OpenWebUI-User-Name": user.name,
                "OpenWebUI-File-Id": file_id,
            }
            
            _, file_path = Storage.upload_file(file_obj, storage_filename, tags)
            log.info(f"Storage path: {file_path}")
            
            # Create file record with proper metadata structure (matching files.py lines 214-229)
            file_record = Files.insert_new_file(
                user.id,
                FileForm(
                    id=file_id,
                    filename=original_name,  # Original filename for display
                    path=file_path,          # Storage path with UUID prefix
                    data={
                        "status": "pending"  # Will be updated after processing
                    },
                    meta={
                        "name": original_name,           # CRITICAL: This is what displays in UI
                        "content_type": "application/pdf",
                        "size": len(content),
                        "source": "pdf_crawler",
                    }
                )
            )
            
            if file_record:
                moved.append(pdf_data["name"])
                uploaded.append(file_id)
                log.info(f"SUCCESS: Created file record for {original_name} with ID {file_id}")
                
                # CRITICAL: Process the file to extract content for RAG
                # This is what openwebui_uploader.py does via the API with process=true
                try:
                    log.info(f"Processing file for content extraction: {file_id}")
                    process_file(
                        request,
                        ProcessFileForm(file_id=file_id),
                        user=user
                    )
                    log.info(f"File processed successfully: {file_id}")
                except Exception as proc_error:
                    log.error(f"Error processing file {file_id}: {proc_error}")
                    # Update file status to failed
                    Files.update_file_data_by_id(
                        file_id,
                        {
                            "status": "failed",
                            "error": str(proc_error)
                        }
                    )
                
                # If knowledge_id provided, add to knowledge base
                if knowledge_id:
                    try:
                        # Use the knowledge router's add file endpoint logic
                        from open_webui.routers.knowledge import add_file_to_knowledge_by_id, KnowledgeFileIdForm
                        
                        # Create a mock form_data for the knowledge endpoint
                        kb_form = KnowledgeFileIdForm(file_id=file_id)
                        
                        # Call the knowledge base add function
                        kb_result = add_file_to_knowledge_by_id(
                            request=request,
                            id=knowledge_id,
                            form_data=kb_form,
                            user=user
                        )
                        added_to_kb.append(file_id)
                        log.info(f"Added {original_name} to knowledge base {knowledge_id}")
                    except Exception as kb_error:
                        log.error(f"Error adding to KB: {kb_error}")
                        errors.append({
                            "filename": pdf_data["name"], 
                            "error": f"KB add failed: {str(kb_error)}"
                        })
            else:
                log.error(f"Failed to create file record for {original_name}")
                errors.append({"filename": pdf_data["name"], "error": "Failed to create file record"})
                
        except Exception as e:
            log.error(f"Error processing {pdf_data['name']}: {e}")
            import traceback
            log.error(traceback.format_exc())
            errors.append({"filename": pdf_data["name"], "error": str(e)})
    
    # Cleanup
    log.info("Cleaning up...")
    for pdf in paths["scraped"].glob("*.pdf"):
        pdf.unlink()
        log.info(f"Deleted: {pdf}")
    
    for f in [paths["state_file"], paths["job_file"]]:
        if f.exists():
            f.unlink()
    
    message = f"Uploaded {len(moved)} PDFs"
    if knowledge_id and added_to_kb:
        message += f", added {len(added_to_kb)} to knowledge base"
    
    log.info(f"=== Finalize Complete: {message} ===")
    log.info(f"Moved: {moved}, Errors: {errors}")
    
    return FinalizeResponse(
        message=message,
        moved=moved,
        uploaded_to_openwebui=uploaded,
        added_to_knowledge=added_to_kb,
        upload_errors=errors
    )

@router.delete("/pdf-reset")
async def reset_state(
    request: Request,
    user=Depends(get_verified_user)
):
    """Reset the crawler state"""
    paths = get_paths()
    
    for f in [paths["state_file"], paths["job_file"]]:
        if f.exists():
            f.unlink()
    
    for folder in [paths["thumbnails"], paths["scraped"], paths["input_dir"]]:
        for file in folder.glob("*"):
            if file.is_file():
                file.unlink()
    
    return {"message": "State reset"}


@router.get("/debug")
async def debug_info(
    request: Request,
    user=Depends(get_verified_user)
):
    """Debug endpoint"""
    paths = get_paths()
    script_path = find_link_downloader()
    job_status = load_job_status()
    
    return {
        "paths": {k: str(v) for k, v in paths.items()},
        "link_downloader_found": script_path is not None,
        "link_downloader_path": str(script_path) if script_path else None,
        "job_status": job_status,
        "scraped_pdfs": [f.name for f in paths["scraped"].glob("*.pdf")],
        "input_pdfs": [f.name for f in paths["input_dir"].glob("*.pdf")],
    }

# ============================================================================
# Script Injection Endpoint
# ============================================================================

@router.get("/inject-script")
async def get_injection_script():
    """Returns JavaScript that loads the PDF crawler UI with knowledge base integration"""
    
    script = r'''
(function() {
    if (window.__pdfCrawlerLoaded) return;
    window.__pdfCrawlerLoaded = true;
    
    console.log('[PDF Crawler] Loading...');
    
    const API_PREFIX = '/api/v1/custom';
    
    let floatingButton = null;
    let uploadModal = null;
    let crawledPDFs = [];
    let excludedPDFs = new Set();
    let pollInterval = null;
    let currentKnowledgeId = null;
    
    function detectKnowledgeId() {
        const match = window.location.pathname.match(/\/knowledge\/([a-f0-9-]+)/);
        return match ? match[1] : null;
    }
    
    function getAuthHeaders() {
        const token = localStorage.getItem('token');
        return { 'Authorization': 'Bearer ' + token };
    }
    
    async function fetchWithAuth(url, options = {}) {
        options.headers = { ...options.headers, ...getAuthHeaders() };
        return fetch(url, options);
    }
    
    function createStyles() {
        if (document.getElementById('pdf-crawler-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'pdf-crawler-styles';
        style.textContent = `
            #pdf-crawler-btn {
                position: fixed;
                bottom: 24px;
                right: 24px;
                z-index: 9999;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 24px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
                transition: all 0.3s ease;
            }
            #pdf-crawler-btn:hover {
                transform: scale(1.1);
                box-shadow: 0 6px 20px rgba(0,0,0,0.4);
            }
            #pdf-crawler-btn.on-kb-page {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            .pdf-modal-overlay {
                position: fixed;
                top: 0; left: 0; right: 0; bottom: 0;
                background: rgba(0,0,0,0.8);
                z-index: 10000;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .pdf-modal {
                background: #1e1e1e;
                border-radius: 16px;
                width: 90%;
                max-width: 700px;
                max-height: 85vh;
                overflow: hidden;
                display: flex;
                flex-direction: column;
            }
            .pdf-modal-header {
                padding: 20px 24px;
                border-bottom: 1px solid #333;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .pdf-modal-header h2 { margin: 0; color: #fff; font-size: 1.25rem; }
            .pdf-modal-close {
                background: none; border: none; color: #888;
                font-size: 24px; cursor: pointer; padding: 0; line-height: 1;
            }
            .pdf-modal-close:hover { color: #fff; }
            .pdf-modal-body { padding: 24px; overflow-y: auto; flex: 1; }
            .pdf-modal-footer {
                padding: 16px 24px;
                border-top: 1px solid #333;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .pdf-kb-notice {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid #10b981;
                border-radius: 8px;
                padding: 12px 16px;
                margin-bottom: 16px;
                color: #10b981;
                font-size: 0.9rem;
            }
            .pdf-kb-notice.warning {
                background: rgba(245, 158, 11, 0.1);
                border-color: #f59e0b;
                color: #f59e0b;
            }
            .pdf-upload-zone {
                border: 2px dashed #444;
                border-radius: 12px;
                padding: 48px 24px;
                text-align: center;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .pdf-upload-zone:hover {
                border-color: #667eea;
                background: rgba(102, 126, 234, 0.05);
            }
            .pdf-upload-zone h3 { color: #fff; margin: 0 0 8px 0; }
            .pdf-upload-zone p { color: #888; margin: 0 0 24px 0; }
            .pdf-upload-btn {
                display: inline-block;
                padding: 12px 32px;
                background: #667eea;
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: 500;
                cursor: pointer;
            }
            .pdf-upload-btn:hover { background: #5a6fd6; }
            .pdf-spinner {
                width: 48px; height: 48px;
                border: 4px solid #333;
                border-top-color: #667eea;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                margin: 0 auto 24px;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            .pdf-progress-bar {
                height: 8px; background: #333;
                border-radius: 4px; overflow: hidden; margin: 16px 0;
            }
            .pdf-progress-fill {
                height: 100%; background: #667eea;
                border-radius: 4px; transition: width 0.3s ease;
            }
            .pdf-status-msg {
                color: #888;
                font-size: 0.9rem;
                margin-top: 8px;
                max-height: 60px;
                overflow: hidden;
                text-overflow: ellipsis;
            }
            .pdf-list { max-height: 400px; overflow-y: auto; }
            .pdf-item {
                display: flex; align-items: center;
                padding: 12px; background: #2a2a2a;
                border-radius: 8px; margin-bottom: 8px;
            }
            .pdf-item.excluded { opacity: 0.5; }
            .pdf-item.excluded .pdf-name { text-decoration: line-through; }
            .pdf-thumb {
                width: 48px; height: 64px; background: #333;
                border-radius: 4px; margin-right: 12px; 
                object-fit: cover;
                display: flex; align-items: center; justify-content: center;
                color: #666; font-size: 24px;
                flex-shrink: 0;
            }
            .pdf-thumb img {
                width: 100%; height: 100%;
                object-fit: cover;
                border-radius: 4px;
            }
            .pdf-info { flex: 1; min-width: 0; }
            .pdf-name { 
                color: #fff; font-weight: 500; display: block;
                white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
            }
            .pdf-size { color: #888; font-size: 0.85rem; }
            .pdf-toggle {
                background: none; border: none;
                font-size: 20px; cursor: pointer; padding: 8px;
                flex-shrink: 0;
            }
            .pdf-submit-btn {
                padding: 12px 24px; background: #667eea;
                color: white; border: none; border-radius: 8px;
                font-weight: 500; cursor: pointer;
            }
            .pdf-submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
            .pdf-submit-btn.kb-mode {
                background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            }
            .pdf-count { color: #888; }
            .pdf-notification {
                position: fixed; bottom: 100px; right: 24px;
                padding: 16px 24px; border-radius: 8px;
                color: white; z-index: 10001;
                animation: slideIn 0.3s ease;
            }
            .pdf-notification.success { background: #10b981; }
            .pdf-notification.error { background: #ef4444; }
            @keyframes slideIn {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
        `;
        document.head.appendChild(style);
    }
    
    function showNotification(message, type) {
        const notif = document.createElement('div');
        notif.className = 'pdf-notification ' + (type || 'success');
        notif.textContent = message;
        document.body.appendChild(notif);
        setTimeout(function() { notif.remove(); }, 3000);
    }
    
    function createButton() {
        if (floatingButton) return;
        floatingButton = document.createElement('button');
        floatingButton.id = 'pdf-crawler-btn';
        floatingButton.innerHTML = '🕸️';
        floatingButton.title = 'PDF Web Crawler';
        floatingButton.onclick = openModal;
        document.body.appendChild(floatingButton);
        updateButtonStyle();
        console.log('[PDF Crawler] Button created');
    }
    
    function updateButtonStyle() {
        currentKnowledgeId = detectKnowledgeId();
        if (floatingButton) {
            if (currentKnowledgeId) {
                floatingButton.classList.add('on-kb-page');
                floatingButton.title = 'PDF Web Crawler (will add to this Knowledge Base)';
            } else {
                floatingButton.classList.remove('on-kb-page');
                floatingButton.title = 'PDF Web Crawler';
            }
        }
    }
    
    let lastPath = window.location.pathname;
    setInterval(function() {
        if (window.location.pathname !== lastPath) {
            lastPath = window.location.pathname;
            updateButtonStyle();
        }
    }, 500);
    
    function openModal() {
        if (uploadModal) return;
        currentKnowledgeId = detectKnowledgeId();
        
        uploadModal = document.createElement('div');
        uploadModal.className = 'pdf-modal-overlay';
        uploadModal.innerHTML = '<div class="pdf-modal">' +
            '<div class="pdf-modal-header">' +
            '<h2>📄 PDF Web Crawler</h2>' +
            '<button class="pdf-modal-close" id="pdf-close-btn">×</button>' +
            '</div>' +
            '<div class="pdf-modal-body" id="pdf-modal-body"></div>' +
            '<div class="pdf-modal-footer" id="pdf-modal-footer" style="display:none"></div>' +
            '</div>';
        document.body.appendChild(uploadModal);
        document.getElementById('pdf-close-btn').onclick = closeModal;
        uploadModal.onclick = function(e) { if (e.target === uploadModal) closeModal(); };
        showUploadStep();
    }
    
    function closeModal() {
        if (pollInterval) { clearInterval(pollInterval); pollInterval = null; }
        if (uploadModal) { uploadModal.remove(); uploadModal = null; }
        crawledPDFs = [];
        excludedPDFs = new Set();
    }
    
    function getKbNoticeHtml() {
        if (currentKnowledgeId) {
            return '<div class="pdf-kb-notice">✓ Files will be added to the current Knowledge Base</div>';
        } else {
            return '<div class="pdf-kb-notice warning">⚠ Navigate to a Knowledge Base page to auto-add files, or files will only be uploaded to your workspace.</div>';
        }
    }
    
    function showUploadStep() {
        var body = document.getElementById('pdf-modal-body');
        body.innerHTML = getKbNoticeHtml() +
            '<div class="pdf-upload-zone" id="pdf-drop-zone">' +
            '<h3>Upload PDF Files</h3>' +
            '<p>Select PDFs to extract and crawl linked documents</p>' +
            '<button class="pdf-upload-btn" id="pdf-choose-btn">Choose Files</button>' +
            '<input type="file" id="pdf-file-input" multiple accept=".pdf" style="display:none">' +
            '</div>';
        document.getElementById('pdf-choose-btn').onclick = function(e) {
            e.stopPropagation();
            document.getElementById('pdf-file-input').click();
        };
        document.getElementById('pdf-drop-zone').onclick = function(e) {
            if (e.target.id !== 'pdf-choose-btn') {
                document.getElementById('pdf-file-input').click();
            }
        };
        document.getElementById('pdf-file-input').onchange = function() {
            handleFiles(this.files);
        };
        document.getElementById('pdf-modal-footer').style.display = 'none';
    }
    
    function showProgress(message, progress, statusMsg) {
        var body = document.getElementById('pdf-modal-body');
        body.innerHTML = '<div style="text-align:center">' +
            '<div class="pdf-spinner"></div>' +
            '<h3 style="color:#fff;margin:0 0 8px">' + message + '</h3>' +
            '<div class="pdf-progress-bar"><div class="pdf-progress-fill" style="width:' + progress + '%"></div></div>' +
            '<p style="color:#888">' + progress + '%</p>' +
            (statusMsg ? '<p class="pdf-status-msg">' + statusMsg + '</p>' : '') +
            '</div>';
    }
    
    function showReviewStep() {
        var body = document.getElementById('pdf-modal-body');
        var footer = document.getElementById('pdf-modal-footer');
        
        if (crawledPDFs.length === 0) {
            body.innerHTML = '<div style="text-align:center;padding:48px">' +
                '<p style="color:#888;font-size:1.1rem">No PDFs were found from crawling.</p>' +
                '<button class="pdf-upload-btn" id="pdf-retry-btn" style="margin-top:24px">Try Again</button></div>';
            document.getElementById('pdf-retry-btn').onclick = showUploadStep;
            footer.style.display = 'none';
            return;
        }
        
        var html = getKbNoticeHtml() +
            '<div><p style="color:#888;margin:0 0 16px">Found ' + crawledPDFs.length + ' PDFs. Click ❌ to exclude:</p><div class="pdf-list">';
        crawledPDFs.forEach(function(pdf) {
            var isExcluded = excludedPDFs.has(pdf.name);
            var thumbContent = pdf.preview_url 
                ? '<img src="' + pdf.preview_url + '" alt="" onerror="this.parentElement.innerHTML=\'📄\'">'
                : '📄';
            
            html += '<div class="pdf-item' + (isExcluded ? ' excluded' : '') + '" data-name="' + escapeHtml(pdf.name) + '">' +
                '<div class="pdf-thumb">' + thumbContent + '</div>' +
                '<div class="pdf-info"><span class="pdf-name" title="' + escapeHtml(pdf.name) + '">' + escapeHtml(pdf.name) + '</span>' +
                '<span class="pdf-size">' + (pdf.size_kb || '?') + ' KB</span></div>' +
                '<button class="pdf-toggle" data-name="' + escapeHtml(pdf.name) + '">' + (isExcluded ? '✓' : '❌') + '</button></div>';
        });
        html += '</div></div>';
        body.innerHTML = html;
        
        document.querySelectorAll('.pdf-toggle').forEach(function(btn) {
            btn.onclick = function() { togglePDF(this.getAttribute('data-name')); };
        });
        
        updateFooter();
        footer.style.display = 'flex';
    }
    
    function escapeHtml(text) {
        var div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    function updateFooter() {
        var footer = document.getElementById('pdf-modal-footer');
        var selected = crawledPDFs.filter(function(p) { return !excludedPDFs.has(p.name); }).length;
        var btnClass = currentKnowledgeId ? 'pdf-submit-btn kb-mode' : 'pdf-submit-btn';
        var btnText = currentKnowledgeId ? 'Add to Knowledge Base' : 'Upload to Open WebUI';
        
        footer.innerHTML = '<span class="pdf-count">' + selected + ' of ' + crawledPDFs.length + ' selected</span>' +
            '<button class="' + btnClass + '" id="pdf-finalize-btn"' + (selected === 0 ? ' disabled' : '') + '>' + btnText + '</button>';
        document.getElementById('pdf-finalize-btn').onclick = finalize;
    }
    
    async function handleFiles(files) {
        if (!files || files.length === 0) return;
        showProgress('Uploading files...', 5, '');
        
        var formData = new FormData();
        for (var i = 0; i < files.length; i++) {
            formData.append('files', files[i]);
        }
        
        try {
            var response = await fetchWithAuth(API_PREFIX + '/pdf-upload', {
                method: 'POST',
                body: formData
            });
            
            var data = await response.json();
            console.log('[PDF Crawler] Upload response:', data);
            
            if (!response.ok) throw new Error(data.detail || 'Upload failed');
            startPolling();
        } catch (error) {
            console.error('[PDF Crawler] Upload error:', error);
            showNotification('Upload failed: ' + error.message, 'error');
            showUploadStep();
        }
    }
    
    function startPolling() {
        showProgress('Starting crawler...', 10, 'Please wait, this may take a few minutes...');
        
        pollInterval = setInterval(async function() {
            try {
                var response = await fetchWithAuth(API_PREFIX + '/pdf-job-status');
                var status = await response.json();
                console.log('[PDF Crawler] Job status:', status);
                
                if (status.status === 'completed') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    showProgress('Loading results...', 95, 'Generating thumbnails...');
                    await loadPDFs();
                    showReviewStep();
                } else if (status.status === 'failed') {
                    clearInterval(pollInterval);
                    pollInterval = null;
                    showNotification('Crawling failed: ' + status.message, 'error');
                    showUploadStep();
                } else {
                    showProgress('Crawling PDFs...', status.progress || 30, status.message || 'Downloading linked PDFs...');
                }
            } catch (error) {
                console.error('[PDF Crawler] Poll error:', error);
            }
        }, 2000);
    }
    
    async function loadPDFs() {
        var response = await fetchWithAuth(API_PREFIX + '/pdf-list');
        if (!response.ok) throw new Error('Failed to load PDFs');
        crawledPDFs = await response.json();
        console.log('[PDF Crawler] Loaded PDFs:', crawledPDFs);
        excludedPDFs = new Set(crawledPDFs.filter(function(p) { return p.excluded; }).map(function(p) { return p.name; }));
    }
    
    async function togglePDF(name) {
        var isExcluded = !excludedPDFs.has(name);
        if (isExcluded) excludedPDFs.add(name);
        else excludedPDFs.delete(name);
        
        try {
            await fetchWithAuth(API_PREFIX + '/pdf-toggle/' + encodeURIComponent(name), {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: name, excluded: isExcluded })
            });
            var item = document.querySelector('.pdf-item[data-name="' + name + '"]');
            if (item) {
                item.classList.toggle('excluded', isExcluded);
                item.querySelector('.pdf-toggle').textContent = isExcluded ? '✓' : '❌';
            }
            var pdf = crawledPDFs.find(function(p) { return p.name === name; });
            if (pdf) pdf.excluded = isExcluded;
            updateFooter();
        } catch (error) {
            console.error('[PDF Crawler] Toggle error:', error);
            if (isExcluded) excludedPDFs.delete(name);
            else excludedPDFs.add(name);
        }
    }
    
    async function finalize() {
        var actionText = currentKnowledgeId ? 'Adding to Knowledge Base...' : 'Uploading to Open WebUI...';
        showProgress(actionText, 50, '');
        
        try {
            var body = {};
            if (currentKnowledgeId) body.knowledge_id = currentKnowledgeId;
            
            var response = await fetchWithAuth(API_PREFIX + '/pdf-finalize', { 
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            
            if (!response.ok) throw new Error('Finalize failed');
            var result = await response.json();
            
            var successMsg = 'Successfully uploaded ' + result.moved.length + ' PDFs!';
            if (result.added_to_knowledge && result.added_to_knowledge.length > 0) {
                successMsg = 'Added ' + result.added_to_knowledge.length + ' PDFs to Knowledge Base!';
            }
            
            showNotification(successMsg, 'success');
            
            if (currentKnowledgeId) {
                setTimeout(function() { closeModal(); window.location.reload(); }, 1500);
            } else {
                setTimeout(closeModal, 1500);
            }
        } catch (error) {
            console.error('[PDF Crawler] Finalize error:', error);
            showNotification('Failed to upload: ' + error.message, 'error');
            showReviewStep();
        }
    }
    
    createStyles();
    createButton();
    console.log('[PDF Crawler] Ready!');
})();
'''
    
    from starlette.responses import Response as StarletteResponse
    return StarletteResponse(
        content=script,
        media_type="application/javascript",
        headers={
            "Content-Type": "application/javascript; charset=utf-8",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )