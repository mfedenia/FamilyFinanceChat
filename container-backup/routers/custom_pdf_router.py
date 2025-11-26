#!/usr/bin/env python3
"""
Custom router for PDF processing in OpenWebUI
"""
from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import FileResponse
from open_webui.utils.auth import get_verified_user
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import sys
from pathlib import Path

# Add the backend functions
sys.path.insert(0, "/app/custom_code/integrated_backend")
import backend_functions as bf

log = logging.getLogger(__name__)
router = APIRouter()

class PDFToggleRequest(BaseModel):
    name: str
    excluded: bool

@router.get("/pdf-list")
async def get_pdf_list():## user=Depends(get_verified_user)
    """Get list of crawled PDFs"""
    try:
        pdfs = bf.list_pdfs()
        return pdfs
    except Exception as e:
        log.error(f"Error listing PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf-upload")
async def upload_pdfs(
    request: Request,
    files: List[UploadFile] = File(...),
    ## user=Depends(get_verified_user)
):
    """Upload PDFs and trigger crawling"""
    try:
        # Convert UploadFile objects to dict format expected by backend
        file_data = []
        for file in files:
            content = await file.read()
            file_data.append({
                "filename": file.filename,
                "content": content
            })
        
        result = bf.upload_and_crawl(file_data)
        
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
    except Exception as e:
        log.error(f"Error uploading PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/pdf-toggle/{pdf_name}")
async def toggle_pdf(
    pdf_name: str,
    toggle_request: PDFToggleRequest,
    ## user=Depends(get_verified_user)
):
    """Toggle PDF exclusion status"""
    try:
        result = bf.toggle_pdf_exclusion(pdf_name, toggle_request.excluded)
        if result["status"] == "error":
            raise HTTPException(status_code=404, detail=result["message"])
        return result
    except Exception as e:
        log.error(f"Error toggling PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf-finalize")
async def finalize_pdfs():## user=Depends(get_verified_user)
    """Finalize and upload PDFs to OpenWebUI"""
    try:
        result = bf.finalize_upload_to_openwebui()
        if result["status"] == "error":
            raise HTTPException(status_code=400, detail=result["message"])
        return result
    except Exception as e:
        log.error(f"Error finalizing PDFs: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/thumbnail/{filename}")
async def get_thumbnail(filename: str):
    """Serve thumbnail images"""
    try:
        thumb_path = bf.THUMBNAILS / filename
        if not thumb_path.exists():
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        return FileResponse(str(thumb_path))
    except Exception as e:
        log.error(f"Error serving thumbnail: {e}")
        raise HTTPException(status_code=500, detail=str(e))