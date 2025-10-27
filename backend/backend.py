from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import json, shutil
import subprocess
import sys
import os
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_thumbnail(pdf_path: Path, thumbnail_dir: Path) -> Optional[Path]:
    """Generate a thumbnail for a PDF file"""
    try:
        from pdf2image import convert_from_path
        
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumbnail_dir / f"{pdf_path.stem}.png"
        
        # Check if thumbnail already exists
        if output_path.exists():
            logging.info(f"Thumbnail already exists: {output_path}")
            return output_path
        
        logging.info(f"Generating thumbnail for: {pdf_path}")
        
        # Convert first page to image with specific DPI for better quality
        pages = convert_from_path(
            str(pdf_path), 
            first_page=1, 
            last_page=1, 
            dpi=100,  # Higher DPI for better quality
            fmt='png'
        )
        
        if pages and len(pages) > 0:
            # Resize to thumbnail size
            from PIL import Image
            img = pages[0]
            # Calculate size maintaining aspect ratio
            max_width, max_height = 200, 250
            img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            img.save(output_path, "PNG", optimize=True)
            
            logging.info(f"Successfully created thumbnail: {output_path}")
            return output_path
        else:
            logging.error(f"No pages found in PDF: {pdf_path}")
            return None
            
    except ImportError as e:
        logging.error(f"pdf2image not installed: {e}")
        logging.error("Install with: pip install pdf2image pillow")
        logging.error("On Mac, also run: brew install poppler")
        return None
    except Exception as e:
        logging.error(f"Failed to create thumbnail for {pdf_path.name}: {e}")
        return None

DATA_DIR = Path("Webscraping/openwebui/data")
SCRAPED = DATA_DIR / "webscraped"
KB = DATA_DIR / "knowledge_base"
THUMBNAILS = DATA_DIR / "thumbnails"
STATE_FILE = DATA_DIR / "pdf_state.json"
INPUT_DIR = Path("/Users/mishiev/FamilyFinanceChat/backend/Webscraping/input_files")

# Create directories
for dir_path in [DATA_DIR, SCRAPED, KB, THUMBNAILS, INPUT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PDF Review Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount thumbnails directory as static files
app.mount("/thumbnails", StaticFiles(directory=str(THUMBNAILS)), name="thumbnails")

class PDFItem(BaseModel):
    name: str
    excluded: bool = False

class CrawlRequest(BaseModel):
    pdf_paths: List[str]
    depth: int = 3

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
    STATE_FILE.write_text(json.dumps(data, indent=2))

@app.get("/api/pdfs")
def list_pdfs():
    """List all PDFs in the webscraped directory"""
    # Get all PDFs currently in the scraped folder
    pdf_files = list(SCRAPED.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDFs in {SCRAPED}")
    
    if not pdf_files:
        logging.warning(f"No PDFs found in {SCRAPED}")
        # Clear the state file if no PDFs exist
        if STATE_FILE.exists():
            STATE_FILE.unlink()
            logging.info("Cleared state file as no PDFs exist")
        return []
    
    # Load existing state (for exclusion status only)
    saved_state = load_state()
    old_exclusions = {item["name"]: item.get("excluded", False) for item in saved_state}
    
    # Build fresh list based on what's actually in the directory
    files = []
    existing_pdf_names = set()
    
    for pdf in pdf_files:
        existing_pdf_names.add(pdf.name)
        logging.info(f"Processing PDF: {pdf.name}")
        
        # Generate or check for thumbnail
        thumb_path = THUMBNAILS / f"{pdf.stem}.png"
        
        # Try to generate thumbnail if it doesn't exist
        if not thumb_path.exists():
            logging.info(f"Thumbnail missing, generating for: {pdf.name}")
            generated_thumb = generate_thumbnail(pdf, THUMBNAILS)
            if generated_thumb and generated_thumb.exists():
                thumb_path = generated_thumb
            else:
                logging.warning(f"Failed to generate thumbnail for {pdf.name}")
                thumb_path = None
        else:
            logging.info(f"Thumbnail exists: {thumb_path}")
        
        # Build the file info
        file_info = {
            "name": pdf.name,
            "size_kb": round(pdf.stat().st_size / 1024, 1),
            "preview_url": f"/thumbnails/{thumb_path.name}" if thumb_path and thumb_path.exists() else None,
            "excluded": old_exclusions.get(pdf.name, False)  # Preserve exclusion status if it existed
        }
        
        files.append(file_info)
        logging.info(f"Added file: {file_info['name']} (excluded: {file_info['excluded']})")
    
    # Create new state with only PDFs that actually exist
    new_state = [{"name": f["name"], "excluded": f["excluded"]} for f in files]
    
    # Save the cleaned state
    save_state(new_state)
    logging.info(f"Saved state for {len(new_state)} PDFs")
    
    # Log what we're returning
    logging.info(f"Returning {len(files)} PDFs to frontend")
    for f in files:
        logging.info(f"  - {f['name']} (excluded: {f['excluded']})")
    
    return files

@app.get("/thumbnails/{filename}")
async def get_thumbnail(filename: str):
    """Serve thumbnail images"""
    file_path = THUMBNAILS / filename
    logging.info(f"Thumbnail requested: {filename}")
    
    if file_path.exists() and file_path.is_file():
        logging.info(f"Serving thumbnail: {file_path}")
        return FileResponse(
            file_path, 
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    
    logging.warning(f"Thumbnail not found: {file_path}")
    raise HTTPException(status_code=404, detail="Thumbnail not found")

@app.patch("/api/pdfs/{name}")
def toggle_exclusion(name: str, item: PDFItem):
    """Toggle PDF exclusion status"""
    state = load_state()
    found = False
    
    for pdf in state:
        if pdf["name"] == name:
            pdf["excluded"] = item.excluded
            found = True
            break
    
    # If not found in state, add it
    if not found:
        state.append({"name": name, "excluded": item.excluded})
    
    save_state(state)
    logging.info(f"{'Excluded' if item.excluded else 'Included'}: {name}")
    return {"message": f"{'Excluded' if item.excluded else 'Included'} {name}"}

@app.post("/api/upload")
async def upload_and_crawl(files: List[UploadFile] = File(...)):
    """Upload PDFs and trigger web crawling"""
    
    # Clear the state file for fresh start
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        logging.info("Cleared state file for new upload")
    
    # Clear input directory first
    for old_file in INPUT_DIR.glob("*.pdf"):
        old_file.unlink()
    
    # Save uploaded files
    saved_files = []
    for file in files:
        if file.filename.endswith('.pdf'):
            file_path = INPUT_DIR / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            saved_files.append(str(file_path))
            logging.info(f"Saved uploaded file: {file_path}")
    
    if not saved_files:
        raise HTTPException(400, "No PDF files uploaded")
    
    # Run the link_downloader script
    script_path = Path("/Users/mishiev/FamilyFinanceChat/backend/Webscraping/link_downloader.py")
    
    # Make sure output directory exists
    SCRAPED.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        str(script_path),
        str(INPUT_DIR),
        "--out", str(SCRAPED),
        "--depth", "0",  # Low depth for testing
        "--render-pages",
        "--skip-existing",
        "--max-from-page", "3",  # Limit for testing
        "-v"
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the crawler
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            cwd=str(Path("/Users/mishiev/FamilyFinanceChat/backend/Webscraping"))
        )
        
        logging.info(f"Crawler stdout: {result.stdout}")
        if result.stderr:
            logging.warning(f"Crawler stderr: {result.stderr}")
        logging.info(f"Return code: {result.returncode}")
        
        # Check if any PDFs were created
        pdf_count = len(list(SCRAPED.glob("*.pdf")))
        logging.info(f"PDFs in output directory: {pdf_count}")
        
        return {
            "message": f"Crawling completed. Found {pdf_count} PDFs",
            "files_uploaded": len(saved_files),
            "pdfs_found": pdf_count,
            "output": result.stdout[:500] if result.stdout else "No output"
        }
        
    except Exception as e:
        logging.error(f"Error running crawler: {str(e)}")
        return {
            "message": f"Crawling attempted but had errors: {str(e)}",
            "files_uploaded": len(saved_files),
            "pdfs_found": 0
        }

@app.post("/api/finalize")
def finalize_upload():
    """Move non-excluded PDFs to knowledge base"""
    state = load_state()
    
    # Only include PDFs that are not excluded
    include = [pdf for pdf in state if not pdf["excluded"]]
    KB.mkdir(parents=True, exist_ok=True)
    moved = []
    
    for pdf_data in include:
        source = SCRAPED / pdf_data["name"]
        if source.exists():
            dest = KB / pdf_data["name"]
            shutil.move(str(source), str(dest))
            moved.append(pdf_data["name"])
            logging.info(f"Moved to KB: {pdf_data['name']}")
    
    # Clean up: remove excluded PDFs from scraped folder
    for pdf_data in state:
        if pdf_data["excluded"]:
            source = SCRAPED / pdf_data["name"]
            if source.exists():
                source.unlink()
                logging.info(f"Deleted excluded PDF: {pdf_data['name']}")
    
    # Clear the state after finalizing
    if STATE_FILE.exists():
        STATE_FILE.unlink()
    
    return {"message": f"Moved {len(moved)} PDFs to Knowledge Base", "moved": moved}

@app.delete("/api/reset")
def reset_state():
    """Reset the application state completely"""
    # Clear state file
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        logging.info("Deleted state file")
    
    # Clear thumbnails
    for thumb in THUMBNAILS.glob("*.png"):
        thumb.unlink()
    logging.info(f"Cleared thumbnails")
    
    # Clear webscraped folder
    for pdf in SCRAPED.glob("*.pdf"):
        pdf.unlink()
    logging.info(f"Cleared webscraped folder")
    
    # Clear input files
    for pdf in INPUT_DIR.glob("*.pdf"):
        pdf.unlink()
    logging.info(f"Cleared input files")
    
    logging.info("Reset application state completely")
    return {"message": "State reset successfully"}

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server...")
    print(f"Input directory: {INPUT_DIR}")
    print(f"Output directory: {SCRAPED}")
    print(f"Thumbnails directory: {THUMBNAILS}")
    print(f"Knowledge Base directory: {KB}")
    print(f"State file: {STATE_FILE}")
    
    # Check current state
    if STATE_FILE.exists():
        state = load_state()
        print(f"Current state has {len(state)} entries")
    
    # Check what's actually in directories
    pdfs_in_scraped = list(SCRAPED.glob("*.pdf"))
    print(f"PDFs in webscraped: {len(pdfs_in_scraped)}")
    if pdfs_in_scraped:
        for pdf in pdfs_in_scraped[:5]:  # Show first 5
            print(f"  - {pdf.name}")
    
    # Check if required libraries are installed
    try:
        import pdf2image
        print("✓ pdf2image installed")
    except ImportError:
        print("✗ pdf2image NOT installed - run: pip install pdf2image")
    
    try:
        from PIL import Image
        print("✓ PIL/Pillow installed")
    except ImportError:
        print("✗ PIL/Pillow NOT installed - run: pip install Pillow")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)