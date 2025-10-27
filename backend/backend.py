from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import json, shutil
from pdf2image import convert_from_path
from backend.utils.generate_thumbnail import generate_thumbnail
 
DATA_DIR = Path("openwebui/data")
SCRAPED = DATA_DIR / "webscraped"
KB = DATA_DIR / "knowledge_base"
THUMBNAILS = DATA_DIR / "thumbnails" # folder that has the images for each pdf
STATE_FILE = DATA_DIR / "pdf_state.json" # Mini db
 
app = FastAPI(title="PDF Review Backend")
 
app.mount("./thumbnails", StaticFiles(directory=THUMBNAILS), name="thumbnails")
 
 
# Interface of JSON struct
class PDFItem(BaseModel):
    name: str
    excluded: bool = False
 
def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text)
    return []
 
def save_state(data):
    STATE_FILE.write_text(json.dumps(data, indent=2))
 
@app.get("/api/pdfs")
def list_pdfs():
    state = {pdf["name"]: pdf["excluded"] for pdf in load_state()}
    files = []
 
    # Find all pdfs in the scraped folder
    for pdf in SCRAPED.glob("*.pdf"):
        # Create thumbnail
        thumb = generate_thumbnail(pdf, THUMBNAILS)
       
        # add into the mini db
        files.append({
            "name": pdf.name,
            "size_kb": round(pdf.stat().st_size / 1024, 1),
            "preview_url": f"/thumbnails{pdf.stem}.png" if thumb else None,  
            "excluded": state.get(pdf.name, False)
        })
 
        save_state([{"name": f["name"], "excluded" : f["excluded"]} for f in files])
        return files
 
@app.patch("/api/pdfs/{name}")
def toggle_exclusion(name: str, excluded: bool):
    state = load_state()
    for pdf in state:
        if pdf["name"] == name:
            pdf["excluded"] = excluded
            save_state(state)
            return {"message": f"{'Excluded' if excluded else 'Included'} {name}"}
 
    raise HTTPException(404, f'PDF {name} not found')
 
 
@app.post("/api/finalize")
def finalize_upload():
    state = load_state()
 
    include = [pdf for pdf in state if not pdf["excluded"]]
    KB.mkdir(exist_ok=True)
    moved = []
 
    for pdf in include:
        source = SCRAPED / pdf
        if source.exists():
            # savely copy it into KB from source
            shutil.move(src=str(source), dst=KB / pdf)
            moved.append(pdf)
       
    return {"message" : f"Moved {len(moved)} PDFs to Knowledge Base"}
 
