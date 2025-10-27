from pathlib import Path
import logging

def generate_thumbnail(pdf_path: Path, thumbnail_dir: Path, width: int = 200, height: int = 250) -> Path | None:
    # Generates a PNG Thumbnail for the first page of a PDF File
    try:
        from pdf2image import convert_from_path
        
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumbnail_dir / f"{pdf_path.stem}.png"
 
        # only generate if it doesn't already exist
        if output_path.exists():
            logging.info(f"Thumbnail already exists: {output_path}")
            return output_path
 
        # Convert first page to image
        pages = convert_from_path(str(pdf_path), first_page=1, last_page=1, dpi=72, size=(width, height))
        
        if pages:
            pages[0].save(output_path, "PNG")
            logging.info(f"Created thumbnail: {output_path}")
            return output_path
        else:
            logging.warning(f"No pages found in PDF: {pdf_path}")
            return None
 
    except ImportError as e:
        logging.error(f"pdf2image not installed: {e}")
        logging.error("Install with: pip install pdf2image")
        logging.error("On Mac, also run: brew install poppler")
        return None
    except Exception as e:
        logging.error(f"Failed to create thumbnail for {pdf_path.name}: {e}")
        return None