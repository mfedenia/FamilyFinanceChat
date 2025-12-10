# Custom PDF Crawler & Upload Menu

A custom module integrated directly into Open WebUI that provides a floating PDF crawler button for uploading, crawling, and managing PDF documents within Knowledge Bases.

## Overview

This feature adds a convenient floating button to the Open WebUI interface that allows users to:

- **Upload PDFs** directly from the browser
- **Automatically crawl** linked PDFs from uploaded documents
- **Preview PDFs** with auto-generated thumbnails
- **Select/exclude** files before finalizing
- **Add to Knowledge Bases** automatically when on a Knowledge Base page

---

## Setup Instructions

### Step 1: Docker Compose Configuration

The setup is handled entirely through Docker volume mounts. The `docker-compose.yml` file mounts the custom code into the Open WebUI container:

```yaml
open-webui:
  image: ghcr.io/open-webui/open-webui:main
  volumes:
    # Replace main.py with our modified version
    - /home/mishiev_wisc_edu/FamilyFinanceChat/custom-code/main.py:/app/backend/open_webui/main.py:ro
    # Mount the entire custom-code folder
    - /home/mishiev_wisc_edu/FamilyFinanceChat/custom-code:/app/custom-code:ro
    # Mount custom_pdf_router.py into routers folder
    - /home/mishiev_wisc_edu/FamilyFinanceChat/custom-code/integrated_backend/custom_pdf_router.py:/app/backend/open_webui/routers/custom_pdf_router.py:ro
```

Simply run:
```bash
docker-compose up -d
```

### Step 2: Browser Bookmark Setup

To access the PDF crawler UI, create a browser bookmark:

1. **Open your browser's Bookmark menu** and add a new bookmark
2. **Name it** whatever you like (e.g., "PDF Crawler")
3. **Set the URL** to the following JavaScript command:

```javascript
javascript:fetch('/api/v1/custom/inject-script?_='+Date.now()).then(r=>r.text()).then(eval)
```

4. **Click the bookmark** while on the Open WebUI page to load the floating button!

> **Tip**: The `Date.now()` parameter prevents caching, ensuring you always get the latest version of the script.

---

## Future Version Compatibility

Even if the layout of Open WebUI's `main.py` changes in future versions, you only need to ensure:

1. **Import the router** where other routers are imported:
   ```python
   from open_webui.routers import custom_pdf_router
   ```

2. **Include the router** in the same pattern as other routers:
   ```python
   app.include_router(custom_pdf_router.router, prefix="/api/v1/custom", tags=["custom_pdf"])
   ```

---

## Architecture

```
custom-code/
├── main.py                         # Modified Open WebUI main.py with router integration
├── integrated_backend/
│   ├── custom_pdf_router.py        # Core API router (all endpoints + injected JS UI)
│   └── backend_functions.py        # Placeholder trigger file
└── upload_pdf_app/                 # Standalone backup app (may need fixes)
    └── backend/
        └── Webscraping/
            └── link_downloader.py  # PDF link extraction and download engine
```

---

## How It Works

### Workflow

1. **Upload**: User clicks the floating button and selects PDF files
2. **Crawl**: Backend extracts URLs from PDFs and downloads linked documents
3. **Review**: User sees all found PDFs with thumbnails, can exclude unwanted files
4. **Finalize**: Selected PDFs are uploaded to Open WebUI and optionally added to a Knowledge Base

### Knowledge Base Integration

When the floating button is clicked while on a Knowledge Base page (`/knowledge/{id}`):
- The button turns **green** to indicate KB mode
- Finalized PDFs are automatically added to that Knowledge Base
- Files are processed for RAG (content extraction + embeddings)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/custom/pdf-upload` | POST | Upload PDFs and start background crawling |
| `/api/v1/custom/pdf-job-status` | GET | Check crawling job progress |
| `/api/v1/custom/pdf-list` | GET | List all crawled PDFs with thumbnails |
| `/api/v1/custom/pdf-thumbnail/{filename}` | GET | Serve thumbnail images |
| `/api/v1/custom/pdf-toggle/{name}` | PATCH | Toggle PDF inclusion/exclusion |
| `/api/v1/custom/pdf-finalize` | POST | Upload selected PDFs to Open WebUI |
| `/api/v1/custom/pdf-reset` | DELETE | Reset crawler state |
| `/api/v1/custom/debug` | GET | Debug information |
| `/api/v1/custom/inject-script` | GET | Returns the JS UI code |

---

## Code Documentation

### `custom_pdf_router.py` - In-Depth Explanation

This is the main FastAPI router that handles all PDF crawler functionality.

#### Configuration & Path Management

```python
def get_data_dir() -> Path:
    """Get the data directory - checks for Docker path first, falls back to local"""
    
def get_paths() -> dict:
    """Returns all required paths: data_dir, scraped, thumbnails, input_dir, state_file, job_file"""
```

#### Job Management Functions

| Function | Description |
|----------|-------------|
| `load_job_status()` | Reads the current crawl job status from `job_status.json`. Returns job_id, status, message, pdfs_found, and progress. |
| `save_job_status(job_id, status, message, pdfs_found, progress)` | Persists job status to JSON file for tracking background crawl operations. |

#### State Management Functions

| Function | Description |
|----------|-------------|
| `load_state()` | Loads the list of PDFs with their inclusion/exclusion status from `pdf_state.json`. |
| `save_state(data)` | Saves the PDF list state, tracking which files the user wants to include or exclude. |

#### Thumbnail Generation

```python
def generate_thumbnail(pdf_path: Path, thumbnail_dir: Path) -> Optional[Path]:
    """
    Generate a PNG thumbnail of the first page of a PDF.
    
    Uses PyMuPDF (fitz) to:
    1. Open the PDF document
    2. Get the first page
    3. Render it to a pixmap at 0.5x scale
    4. Save as PNG
    
    Returns the path to the thumbnail or None if generation fails.
    """
```

#### Web Scraping Integration

```python
def find_link_downloader() -> Optional[Path]:
    """
    Searches for link_downloader.py in multiple possible locations:
    - Relative to the router file
    - In /app/backend/open_webui/routers/Webscraping/
    - In /app/custom_code/ directories
    - In /app/custom-code/ directories
    
    Returns the path if found, None otherwise.
    """

def run_crawl_job(job_id: str, input_dir: Path, output_dir: Path):
    """
    Background task that runs the web crawler.
    
    1. Copies uploaded PDFs to output directory (ensures originals are preserved)
    2. Launches link_downloader.py as a subprocess
    3. Monitors stdout for progress updates
    4. Updates job status throughout the process
    5. Handles errors gracefully, falling back to uploaded files only
    """
```

#### Knowledge Base Integration

```python
async def add_file_to_knowledge_base_async(file_id: str, knowledge_id: str, user) -> dict:
    """
    Add a file to a Knowledge Base using Open WebUI's KnowledgeFile junction table.
    
    1. Retrieves the knowledge base by ID
    2. Checks user permissions (must be owner or admin)
    3. Creates a KnowledgeFile record linking the file to the KB
    
    Returns: {"success": True} or {"success": False, "error": "..."}
    """
```

#### Main API Endpoint Functions

| Endpoint Function | Description |
|-------------------|-------------|
| `upload_and_crawl()` | Handles PDF file uploads. Clears previous state, saves uploaded files, and starts background crawl job. |
| `get_job_status()` | Returns current crawl job progress (status, message, pdfs_found, progress percentage). |
| `list_pdfs()` | Lists all crawled PDFs with file sizes, exclusion status, and thumbnail URLs. Generates thumbnails on-demand. |
| `get_thumbnail()` | Serves PNG thumbnail images for PDF previews. |
| `toggle_exclusion()` | Toggles whether a PDF is included/excluded from the final upload. |
| `finalize_upload()` | Uploads selected PDFs to Open WebUI storage, processes them for RAG, and optionally adds to Knowledge Base. |
| `reset_state()` | Clears all crawler state files and directories for a fresh start. |
| `debug_info()` | Returns diagnostic information including paths, job status, and file lists. |
| `get_injection_script()` | Returns the complete JavaScript UI code that creates the floating button and modal interface. |

#### Finalize Upload - Detailed Flow

```python
async def finalize_upload(...):
    """
    1. Load state to get list of included PDFs
    2. For each included PDF:
       a. Read file content
       b. Upload to Open WebUI storage (Storage.upload_file)
       c. Create file record in database (Files.insert_new_file)
       d. Process file for RAG indexing (process_file) - extracts text & creates embeddings
       e. If knowledge_id provided, add to Knowledge Base
    3. Cleanup: delete temporary files
    4. Return summary of uploaded/added/errored files
    """
```

---

### `link_downloader.py` - In-Depth Explanation

This is the web scraping engine that extracts links from PDFs and downloads linked documents.

#### Constants & Regex Patterns

```python
PDF_REGEX = re.compile(r"\.pdf(?:[?#].*)?$", re.IGNORECASE)  # Matches .pdf URLs
URL_REGEX = re.compile(r"""https?://[^\s<>()'"]+""", re.IGNORECASE)  # Extracts URLs from text
GOOGLE_DRIVE_REGEX = re.compile(r"^/file/d/([^/]+)/view", re.IGNORECASE)  # Google Drive file IDs
DOC_ID_REGEX = re.compile(r"^/(?:document|spreadsheets|presentation)/d/([^/]+)", re.IGNORECASE)  # Google Docs IDs
```

#### Helper Functions

| Function | Description |
|----------|-------------|
| `setup_logging(verbosity)` | Configures logging level (0=warnings, 1=info, 2+=debug) |
| `sanitize_filename(name)` | Cleans filenames by removing invalid characters, limiting to 200 chars |
| `filename_from_cd(content_disposition)` | Extracts filename from Content-Disposition HTTP header |
| `derive_filename_from_url(url)` | Creates a filename from a URL path when no header is available |

#### Link Extraction from PDFs

```python
def extract_annotation_links(doc: fitz.Document) -> Set[str]:
    """
    Extract clickable hyperlinks from PDF annotations.
    
    Iterates through all pages and extracts URIs from link annotations.
    Only includes http:// and https:// links.
    """

def extract_text_links(doc: fitz.Document) -> Set[str]:
    """
    Extract URLs from the raw text content of a PDF.
    
    Uses regex to find URL patterns in the extracted text.
    Handles URLs that may be broken across lines or have trailing punctuation.
    """
```

#### HTTP Download Functions

```python
def is_pdf_response(resp: requests.Response) -> bool:
    """Check if a response contains a PDF by Content-Type header or URL extension."""

def stream_download_pdf(session, url, out_dir, skip_existing, timeout=30) -> Optional[Path]:
    """
    Download a PDF file with streaming (memory-efficient).
    
    1. Make GET request with streaming enabled
    2. Check if response is actually a PDF
    3. Determine filename from Content-Disposition or URL
    4. Skip if file exists and skip_existing=True
    5. Write to disk in 64KB chunks
    
    Returns: Path to saved file or None if failed
    """
```

#### Google Drive/Docs Handling

```python
def google_direct_download_url(url: str) -> Optional[str]:
    """
    Convert Google Drive/Docs viewer URLs to direct download URLs.
    
    Handles:
    - drive.google.com/file/d/{id}/view → /uc?export=download&id={id}
    - drive.google.com/open?id={id} → /uc?export=download&id={id}
    - docs.google.com/document/d/{id}/... → /export?format=pdf
    - docs.google.com/spreadsheets/d/{id}/... → /export?format=pdf
    - docs.google.com/presentation/d/{id}/... → /export/pdf
    - Published Google Sheets (pubhtml) → pub?output=pdf
    """

def google_drive_fetch_with_confirm(session, uc_url, out_dir, skip_existing) -> Optional[Path]:
    """
    Handle Google Drive virus scan confirmation page.
    
    Large files trigger a "virus scan" warning page. This function:
    1. Makes initial request
    2. If blocked, parses HTML for confirmation link
    3. Follows confirmation URL to download actual file
    """

def playwright_download_from_drive(url, out_dir, user_agent, skip_existing) -> Optional[Path]:
    """
    Last resort: use headless browser to click Download button.
    
    Opens Google Drive viewer in Playwright/Chromium and clicks the
    download button directly. Useful when other methods fail.
    """
```

#### HTML Page Scraping

```python
def collect_pdf_links_from_page(session, page_url, timeout=30) -> List[str]:
    """
    Scrape a webpage for PDF links.
    
    1. Download the page HTML
    2. Parse with BeautifulSoup
    3. Find all <a href="..."> links ending in .pdf
    4. Find PDFs in <embed> and <iframe> tags
    5. Convert relative URLs to absolute
    6. Return deduplicated list
    """

def render_html_to_pdf_playwright(url, out_dir, ...) -> Optional[Path]:
    """
    Convert a webpage to PDF using headless Chrome.
    
    Features:
    - Auto-scrolling to trigger lazy loading
    - Waits for fonts, network idle, custom selectors
    - Handles loading indicators
    - Screenshot fallback if PDF generation fails
    - Configurable paper size (Letter/A4)
    """
```

#### Main Processing Functions

```python
def process_link(session, url, out_dir, ..., current_depth=0, max_depth=1, visited_urls=None) -> Tuple[int, int, int]:
    """
    Process a single URL with recursive crawling.
    
    Logic flow:
    1. Skip if already visited or max depth exceeded
    2. Try direct PDF download
    3. If PDF downloaded and not at max depth, extract links and recurse
    4. If Google Drive/Docs URL, use special handlers
    5. If regular webpage, scrape for PDF links and download them
    6. Optionally render the webpage itself to PDF
    7. If not at max depth, extract links and recurse
    
    Returns: (attempted_downloads, successful_downloads, rendered_pages)
    """

def process_input_pdf(pdf_path, out_dir, session, args) -> Tuple[int, int, int, int]:
    """
    Process a single input PDF file.
    
    1. Open PDF with PyMuPDF
    2. Extract annotation links and text links
    3. Deduplicate and sort links
    4. Process each link with process_link()
    
    Returns: (total_links, attempted, succeeded, rendered)
    """

def gather_pdfs(input_path, recursive) -> List[Path]:
    """Collect PDF files from a file or directory (optionally recursive)."""

def main():
    """
    CLI entry point with argument parsing.
    
    Arguments:
    - input: PDF file or directory
    - --out: Output directory
    - --depth: Max crawl depth (default: 3)
    - --delay: Seconds between downloads
    - --max-from-page: Limit PDFs per webpage
    - --render-pages: Convert webpages to PDF
    - --skip-existing: Don't re-download existing files
    - --user-agent: Browser user agent string
    - And many more for fine-tuning render behavior
    """
```

---

## File Storage

The crawler uses the following directories (created automatically):

```
/app/backend/data/custom_pdf_crawler/
├── input_files/     # Uploaded PDFs before crawling
├── webscraped/      # Crawled/downloaded PDFs
├── thumbnails/      # Generated PNG thumbnails
├── pdf_state.json   # Tracks included/excluded files
└── job_status.json  # Current crawl job status
```

---

## Troubleshooting

### Button not appearing
- Click the bookmark while on the Open WebUI page
- Check browser console for `[PDF Crawler] Ready!`
- Verify the API is accessible: `GET /api/v1/custom/debug`

### Crawling not working
- Check if `link_downloader.py` exists in expected locations
- View container logs for `[Job {id}]` messages
- Use `/api/v1/custom/debug` to see crawler status

### Thumbnails not generating
- Ensure `PyMuPDF` is installed in the container
- Check file permissions in the data directory

### Files not added to Knowledge Base
- Verify you're on a Knowledge Base page (URL contains `/knowledge/`)
- Check the button color (should be green on KB pages)

---

## Standalone App (Backup Option)

The `upload_pdf_app/` directory contains a standalone version with:
- Separate FastAPI backend
- React frontend
- Docker support

> ⚠️ **Note**: The standalone app may require some fixes to work properly. It's provided as a backup option if the integrated approach doesn't suit your deployment.

To try the standalone app:
```bash
cd custom-code/upload_pdf_app
chmod +x run_local.sh
./run_local.sh
```

---

## License

Part of the FamilyFinanceChat project.
