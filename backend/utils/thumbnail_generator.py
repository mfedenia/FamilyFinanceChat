from pathlib import Path
from pdf2image import convert_from_path
 
def generate_thumbnail(pdf_path: Path, thumbnail_dir: Path, width: int = 200, height: int = 250) -> Path | None:
    '''
    Generates a PNG Thumbnail for the first page of a PDF File
    '''
    try:
        thumbnail_dir.mkdir(parents=True, exist_ok=True)
        output_path = thumbnail_dir / f"{pdf_path.stem}.png"
 
        # only generate if it doesn't already exist
        if output_path.exists():
            return output_path
 
        pages = convert_from_path(str(pdf_path), first_page = 1, last_page = 1, size=(width, height))
        pages[0].save(output_path, "PNG")
        return output_path
 
    except Exception as e:
        print(f"[Thumbnail Generator] Failed to create thumbnail for {pdf_path.name}: {e}")
        return None
 