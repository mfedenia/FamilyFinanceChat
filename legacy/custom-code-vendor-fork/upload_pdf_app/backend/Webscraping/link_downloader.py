from __future__ import annotations
import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse, unquote
import fitz
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from playwright.sync_api import sync_playwright
from PIL import Image

# ============================== constants/regex ===============================

PDF_REGEX = re.compile(r"\.pdf(?:[?#].*)?$", re.IGNORECASE) # gets the strings ending with .pdf
URL_REGEX = re.compile(r"""https?://[^\s<>()'"]+""", re.IGNORECASE) # gets url from text
GOOGLE_DRIVE_REGEX = re.compile(r"^/file/d/([^/]+)/view", re.IGNORECASE) # gets the file id from google drive url
DOC_ID_REGEX = re.compile(r"^/(?:document|spreadsheets|presentation)/d/([^/]+)", re.IGNORECASE) # gets doc id from google docs/sheets/slides url

# ============================== Logging / helpers ==============================

def setup_logging(verbosity: int) -> None:
    # which logging level
    if verbosity == 0:
        # only show warnings and errors
        log_level = logging.WARNING
    elif verbosity == 1:
        # show info messages too
        log_level = logging.INFO
    else:
        # show all debug messages
        log_level = logging.DEBUG
    
    # Configure the logging system with our chosen settings
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",  # "12:34:56 | INFO | message"
        datefmt="%H:%M:%S"  # Hour:Minute:Second
    )

def sanitize_filename(name: str) -> str:
    name = unquote(name).strip().replace("\n", " ").replace("\r", " ")
    name = re.sub(r"[<>:\"/\\|?*\x00-\x1F]", "_", name)
    return name[:200] or "file.pdf"

def filename_from_cd(content_disposition: Optional[str]) -> Optional[str]:
    if not content_disposition:
        return None
    m = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^\";]+)"?', content_disposition, flags=re.IGNORECASE)
    return sanitize_filename(m.group(1)) if m else None

def derive_filename_from_url(url: str, default_ext: str = ".pdf") -> str:
    tail = Path(urlparse(url).path).name or "download"
    if default_ext and not tail.lower().endswith(default_ext):
        tail += default_ext
    return sanitize_filename(tail)

# ======================= Extract links from the source PDF =====================

def extract_annotation_links(doc: fitz.Document) -> Set[str]:
    found_links = set()
    total_pages = len(doc) # # of pages
    # Go thru each page in the PDF
    for page_number in range(total_pages):
        current_page = doc[page_number]
        # get clickable links on this page
        links_on_page = current_page.get_links()
        # Check each link on the page
        for link in links_on_page:
            # get the URL from the link
            url = link.get("uri")
            # Check if we actually got a URL
            if url:
                # Convert to lowercase for checking
                url_lowercase = url.lower()
                # Check if it's a web link (starts with http:// or https://)
                if url_lowercase.startswith("http://") or url_lowercase.startswith("https://"):
                    # remove extra spaces
                    clean_url = url.strip()
                    # add
                    found_links.add(clean_url)
    return found_links

def extract_text_links(doc: fitz.Document) -> Set[str]:
    found_links = set()
    total_pages = len(doc) # # of pages
    # Go thru each page in the PDF
    for page_number in range(total_pages):
        current_page = doc[page_number]
        # get all the text
        page_text = current_page.get_text("text")
        if page_text is None:
            page_text = "" # empty

        url_matches = URL_REGEX.finditer(page_text) # find url's in the text
        # Check each link on the page
        for match in url_matches:
            # get the URL from the link
            url = match.group(0)
            url = url.strip() # clean & standardize
            url = url.rstrip(").,]")
            url_lowercase = url.lower()
            # Check if we actually got a URL
            if url_lowercase.startswith("http://") or url_lowercase.startswith("https://"):
                    found_links.add(url)
    return found_links

# ================================ HTTP helpers ================================

def is_pdf_response(resp: requests.Response) -> bool:
    content_type = resp.headers.get("Content-Type", "")  # get header, or do empty string
    content_type = content_type.lower() # make lowercase to make everything standardized
    has_pdf_content_type = "application/pdf" in content_type # check

    url = (resp.url or "") # check if there's a .pdf at end of url
    url_ends_with_pdf = PDF_REGEX.search(url) is not None # check

    return  has_pdf_content_type or url_ends_with_pdf #return true if either is true

# Download url as a PDF & returns new path or None
def stream_download_pdf(session: requests.Session, url: str, out_dir: Path, skip_existing: bool, timeout: int = 30) -> Optional[Path]:
    try:
        # request to get file
        response = session.get(url, stream=True, timeout=timeout, allow_redirects=True)
        response.raise_for_status() # Check if the request went okay
        
        # check if pdf
        is_actually_pdf = is_pdf_response(response)
        if not is_actually_pdf:
            logging.debug("This URL doesn't contain a PDF file: %s", url)
            return None
        
        # file name
        suggested_filename = filename_from_cd(response.headers.get("Content-Disposition"))
        # make name from the URL
        if suggested_filename is None:
            suggested_filename = derive_filename_from_url(response.url)
        # full save path
        full_file_path = out_dir / suggested_filename
        # check if file already exists
        file_already_exists = full_file_path.exists()
        should_skip_download = skip_existing and file_already_exists
        
        if should_skip_download:
            logging.info("File already exists, skipping download: %s", full_file_path)
            return full_file_path
        
        output_file = open(full_file_path, "wb")
        try: # Download file chunk by chunk so we don't use too much memory
            chunk_size = 1024 * 64
            for data_chunk in response.iter_content(chunk_size):
                if data_chunk:
                    output_file.write(data_chunk)
        
        finally:
            output_file.close()
        logging.info("Successfully downloaded and saved: %s", full_file_path)
        return full_file_path

    except requests.RequestException as network_error:
        # Something went wrong with the internet connection/server
        logging.warning("Network error downloading %s: %s", url, network_error)
        return None
    except OSError as file_error:
        # Something went wrong with saving file
        logging.warning("File system error downloading %s: %s", url, file_error)
        return None
    except Exception as unexpected_error:
        # Something else happened
        logging.warning("Unexpected error downloading %s: %s", url, unexpected_error)
        return None
    
# =========================== Google Drive / Docs fix ==========================

# try and get the download/export endpoints for the viewer & editor urls
def google_direct_download_url(url: str) -> Optional[str]:
    # break URL into parts
    parsed_url = urlparse(url)
    website_name = parsed_url.netloc  # ex: "drive.google.com"
    path_part = parsed_url.path        # ex: "/file/d/test/view"
    query_part = parsed_url.query      # ex: "id=test&export=download"
    # if Google Drive URL
    is_google_drive = website_name.endswith("drive.google.com")
    
    if is_google_drive:
        # get file ID in path like "/file/d/FILE_ID/view"
        drive_file_match = GOOGLE_DRIVE_REGEX.match(path_part)
        if drive_file_match:
            file_id = drive_file_match.group(1)
            # make the download URL
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            return download_url
        # look for file ID in query like "/open?id=FILE_ID"
        is_open_link = path_part.endswith("/open")
        if is_open_link:
            query_params = parse_qs(query_part)
            # Check if there's an 'id' parameter
            if "id" in query_params:
                file_id = query_params["id"][0]  # first value
                # make download URL
                download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                return download_url
    
    # check if this is a Google Docs/Sheets/Slides URL
    is_google_docs = website_name.endswith("docs.google.com")
    
    if is_google_docs:
        # for published Google Sheets (since those act diff)
        # ex: https://docs.google.com/spreadsheets/d/e/test/pubhtml
        is_published_sheet = ("/spreadsheets/" in path_part and 
                             "/d/e/" in path_part and 
                             "pubhtml" in path_part)
        
        if is_published_sheet:
            # Convert pubhtml to pub and add PDF output
            new_path = path_part.replace("/pubhtml", "/pub")
            # Parse existing query params
            query_params = parse_qs(query_part)
            # PDF output
            query_params["output"] = ["pdf"]
            # query string
            query_dict = {}
            for key, value_list in query_params.items():
                query_dict[key] = value_list[0]
            new_query = urlencode(query_dict)
            # make new URL
            download_url = urlunparse((
                parsed_url.scheme,  # "https"
                parsed_url.netloc,  # "docs.google.com"
                new_path,          # changed path
                "",                # blank
                new_query,         # Query with output=pdf
                ""                 # blank
            ))
            return download_url
        
        # Regular Google Docs/Sheets/Slides
        # Look for document ID in path like "/document/d/DOC_ID/..."
        doc_id_match = DOC_ID_REGEX.match(path_part)
        if doc_id_match:
            # get doc ID
            document_id = doc_id_match.group(1)
            
            # Check what type of document it is & make export URL
            if "/document/" in path_part:
                # Google Doc
                download_url = f"https://docs.google.com/document/d/{document_id}/export?format=pdf"
                return download_url
            
            elif "/spreadsheets/" in path_part:
                # Google Sheet
                download_url = f"https://docs.google.com/spreadsheets/d/{document_id}/export?format=pdf"
                return download_url
            
            elif "/presentation/" in path_part:
                # Google Slides presentation
                download_url = f"https://docs.google.com/presentation/d/{document_id}/export/pdf"
                return download_url
    
    # couldn't convert the URL
    return None

# deals with the google drive confirm page for the virus scan message, returns saved files or None
def google_drive_fetch_with_confirm(session: requests.Session, uc_url: str, out_dir: Path, skip_existing: bool) -> Optional[Path]:

    try:
        # try to download file directly
        first_response = session.get(uc_url, allow_redirects=True)
        first_response.raise_for_status()
        
        # if we got a PDF directly, then we can just save
        is_direct_pdf = is_pdf_response(first_response)
        if is_direct_pdf:
            # name
            filename = filename_from_cd(first_response.headers.get("Content-Disposition"))
            if filename is None:
                filename = derive_filename_from_url(first_response.url)
            
            # Create save path
            output_file_path = out_dir / filename
            
            # skip existing files
            if skip_existing and output_file_path.exists():
                return output_file_path
            
            # Save
            output_file = open(output_file_path, "wb")
            try:
                # Download in chunks
                chunk_size = 1024 * 64  # 64KB
                for data_chunk in first_response.iter_content(chunk_size):
                    if data_chunk:
                        output_file.write(data_chunk)
            finally:
                output_file.close()
            return output_file_path

        # if stuck on virus scan page, find confirmation link in the html
        html_content = first_response.text
        # ex: href="/uc?export=download&confirm=XXXX&id=YYYY"
        confirm_link_pattern = r'href="(/uc\?export=download[^"]*?confirm=[^"&]+[^"]*?)"'
        match = re.search(confirm_link_pattern, html_content)
        if not match:
            return None
        
        # make confirmation URL
        confirm_path = match.group(1)
        confirm_path = confirm_path.replace("&amp;", "&") # replace &amp with &
        confirm_url = f"https://drive.google.com{confirm_path}" # full url

        # try downloading with new confirmation URL
        second_response = session.get(confirm_url, stream=True, allow_redirects=True)
        second_response.raise_for_status()
        is_confirmed_pdf = is_pdf_response(second_response)
        if not is_confirmed_pdf:
            # still nom pdf; something went wrong
            return None
        
        # file name & path
        filename = filename_from_cd(second_response.headers.get("Content-Disposition"))
        if filename is None:
            filename = derive_filename_from_url(second_response.url)
        output_file_path = out_dir / filename
        
        # skip existing files
        if skip_existing and output_file_path.exists():
            return output_file_path
        
        # Save
        output_file = open(output_file_path, "wb")
        try:
            chunk_size = 1024 * 64  # 64KB chunks
            for data_chunk in second_response.iter_content(chunk_size):
                if data_chunk:
                    output_file.write(data_chunk)
        finally:
            output_file.close()
        return output_file_path
    
    except requests.RequestException as network_error:
        # error with network
        return None

# if normal options dont work, just open Drive viewer and click 'Download' with Playwright to get the file
def playwright_download_from_drive(url: str, out_dir: Path, user_agent: str, skip_existing: bool) -> Optional[Path]:
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            ctx = browser.new_context(user_agent=user_agent, accept_downloads=True, ignore_https_errors=True)
            page = ctx.new_page()
            page.goto(url, wait_until="load", timeout=60000)
            selectors = [
                "a[aria-label='Download']",
                "div[aria-label='Download']",
                "div[guidedhelpid='download']",
                "div[aria-label*='Download']",
                "button[aria-label='Download']",
            ]
            for sel in selectors:
                try:
                    page.wait_for_selector(sel, timeout=5000)
                    with page.expect_download(timeout=30000) as dl_info:
                        page.click(sel)
                    download = dl_info.value
                    suggested = sanitize_filename(download.suggested_filename or "download.pdf")
                    outfile = out_dir / (suggested if suggested.lower().endswith(".pdf") else (suggested + ".pdf"))
                    if skip_existing and outfile.exists():
                        logging.info("Exists, skipping: %s", outfile)
                    else:
                        download.save_as(str(outfile))
                        logging.info("Downloaded from Drive via Playwright: %s", outfile)
                    ctx.close(); browser.close()
                    return outfile
                except Exception:
                    continue
            logging.warning("Could not find Drive 'Download' button.")
            ctx.close(); browser.close()
            return None
    except Exception as e:
        logging.warning("Playwright Drive download failed: %s", e)
        return None

# ============================= HTML scrape & render ============================

# get all the all the pdf links on a website
def collect_pdf_links_from_page(session: requests.Session, page_url: str, timeout: int = 30) -> List[str]:
    try:
        response = session.get(page_url, timeout=timeout, allow_redirects=True)
        response.raise_for_status()
        
        # check if URL already for a PDF
        is_direct_pdf = is_pdf_response(response)
        if is_direct_pdf:
            return [response.url]
        
        # parse the HTML for PDF links
        soup = BeautifulSoup(response.text, "html.parser")
        pdf_links_found = []
        # ex: <a href="...">
        all_link_tags = soup.find_all("a", href=True)
        for link_tag in all_link_tags:
            link_url = link_tag["href"].strip()
            # Convert relative URLs to absolute URLs
            absolute_url = urljoin(response.url, link_url)
            # Check if it is for a PDF
            if PDF_REGEX.search(absolute_url):
                pdf_links_found.append(absolute_url)
        
        # PDFs in embedded content & iframes
        embed_and_iframe_tags = soup.find_all(["embed", "iframe"])
        for tag in embed_and_iframe_tags:
            source_url = tag.get("src")
            if source_url:
                source_url = source_url.strip()
                absolute_url = urljoin(response.url, source_url)
                if PDF_REGEX.search(absolute_url):
                    pdf_links_found.append(absolute_url)
        
        # remove duplicates
        unique_pdf_links = []
        urls_already_seen = set()
        for pdf_url in pdf_links_found:
            if pdf_url not in urls_already_seen:
                # Mark as seen & add
                urls_already_seen.add(pdf_url)
                unique_pdf_links.append(pdf_url)
        
        return unique_pdf_links
    
    except requests.RequestException as network_error:
        return []

def _auto_scroll(page, max_steps: int = 10, step_px: int = 900, pause_ms: int = 300):
    page.evaluate(
        """async ({max_steps, step_px, pause_ms}) => {
            const sleep = (ms) => new Promise(r => setTimeout(r, ms));
            let lastY = -1;
            for (let i = 0; i < max_steps; i++) {
                window.scrollBy(0, step_px);
                await sleep(pause_ms);
                const y = window.scrollY;
                if (y === lastY || (window.innerHeight + window.scrollY) >= document.body.scrollHeight) break;
                lastY = y;
            }
            window.scrollTo(0, 0);
        }""",
        {"max_steps": max_steps, "step_px": step_px, "pause_ms": pause_ms},
    )

# turn the whole webpage into a pdf using headless browser
def render_html_to_pdf_playwright(
    url: str,
    out_dir: Path,
    user_agent: str,
    skip_existing: bool,
    pdf_format: str = "Letter",
    timeout_ms: int = 45000,
    wait_until: str = "networkidle",
    wait_selector: Optional[str] = None,
    wait_text: Optional[str] = None,
    extra_wait_ms: int = 1500,
    auto_scroll: bool = True,
    max_scrolls: int = 10,
    screenshot_fallback: bool = True,
) -> Optional[Path]:

    # Helper function to create a filename from the page title or URL
    def build_filename(title: str, url: str) -> str:

        # use page title if we can for name
        if title:
            base_name = sanitize_filename(title)
        else:
            # make a name from URL
            url_parts = urlparse(url)
            website_name = url_parts.netloc  # e.g., "example.com"
            page_path = url_parts.path      # e.g., "/docs/page1"
            combined = f"{website_name}{page_path}".replace("/", "_").strip("_")
            base_name = sanitize_filename(combined) or "page"
        # Make sure it ends with .pdf
        if not base_name.lower().endswith(".pdf"):
            base_name += ".pdf"
        
        return base_name

    try:
        # start Playwright and open browser
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            
            # create a browser context/window
            context = browser.new_context(
                user_agent=user_agent, # browser
                java_script_enabled=True,
                ignore_https_errors=True,
                viewport={"width": 1366, "height": 2400},  # big enough screen size to be readable
            )
            
            # new tab
            page = context.new_page()
            # go to url
            page.goto(url, timeout=timeout_ms, wait_until=wait_until)
            page.emulate_media(media="screen")
            page.add_style_tag(content="*{-webkit-print-color-adjust:exact;print-color-adjust:exact;}")

            # wait to load
            try:
                page.wait_for_function("document.fonts ? document.fonts.ready : true", timeout=timeout_ms)
            except Exception:
                pass  # not end of world if the font color stuf doesnt work
            
            # Wait for network to be idle
            try:
                page.wait_for_load_state("networkidle", timeout=timeout_ms)
            except Exception:
                pass

            # Wait for specific selector if provided
            if wait_selector:
                try:
                    page.wait_for_selector(wait_selector, state="visible", timeout=timeout_ms)
                except Exception:
                    logging.debug("wait_selector not found: %s", wait_selector)
            
            # Wait for specific text if provided
            if wait_text:
                try:
                    page.wait_for_selector(f"text={wait_text}", state="visible", timeout=timeout_ms)
                except Exception:
                    logging.debug("wait_text not found: %s", wait_text)

            # Wait for loading indicators to disappear
            loading_selectors = [
                "text=/loading/i",    # Text containing "loading"
                "#loading",           # Element with id="loading"
                ".loading",           # Element with class="loading"
                ".loader",            # Element with class="loader"
                "#preloader",         # Element with id="preloader"
                ".preloader",         # Element with class="preloader"
                "[data-loading]",     # Element with data-loading attribute
            ]
            
            for selector in loading_selectors:
                try:
                    page.wait_for_selector(selector, state="detached", timeout=2000)
                except Exception:
                    pass

            # scroll the page to trigger lazy loading
            if auto_scroll:
                _auto_scroll(page, max_steps=max_scrolls)
            
            # extra wait time if given
            if extra_wait_ms > 0:
                page.wait_for_timeout(extra_wait_ms)

            # make  filename & check if it already exists
            title = page.title()
            output_file = out_dir / build_filename(title, url)
            
            if skip_existing and output_file.exists():
                logging.info("Exists, skipping render: %s", output_file)
                context.close()
                browser.close()
                return output_file

            # save as pdf
            try:
                page.pdf(
                    path=str(output_file),
                    print_background=True, 
                    format=pdf_format,
                    landscape=False 
                )
                logging.info("Rendered page to PDF: %s", output_file)
                context.close()
                browser.close()
                return output_file
                
            except Exception as pdf_error:
                logging.debug("PDF generation failed: %s", pdf_error)
                
                # If PDF failed & no ss; give up
                if not screenshot_fallback:
                    context.close()
                    browser.close()
                    return None

            # take a screenshot and convert to PDF
            try:
                # temp name
                png_temp_file = str(output_file.with_suffix(".png"))
                
                # full-page screenshot
                page.screenshot(path=png_temp_file, full_page=True)
                
                # convert PNG to PDF
                try:
                    image = Image.open(png_temp_file)
                    image_rgb = image.convert("RGB")
                    image_rgb.save(str(output_file))
                    os.remove(png_temp_file)
                    
                    context.close()
                    browser.close()
                    return output_file
                    
                except Exception as image_error:
                    logging.warning("Screenshot->PDF conversion failed: %s. Keeping PNG at %s", 
                                  image_error, png_temp_file)
                    context.close()
                    browser.close()
                    return None
                    
            except Exception as screenshot_error:
                logging.warning("Screenshot capture failed: %s", screenshot_error)
                context.close()
                browser.close()
                return None
    
    except Exception as general_error:
        logging.warning("Failed to render %s to PDF: %s", url, general_error)
        return None
    
# ================================ Orchestration ================================

# processes the links and does the recursive crawling
def process_link(
    session: requests.Session,
    url: str,
    out_dir: Path,
    delay: float,
    max_from_page: Optional[int],
    render_pages: bool,
    user_agent: str,
    pdf_format: str,
    render_timeout_ms: int,
    wait_until: str,
    skip_existing: bool,
    wait_selector: Optional[str],
    wait_text: Optional[str],
    extra_wait_ms: int,
    auto_scroll: bool,
    max_scrolls: int,
    screenshot_fallback: bool,
    current_depth: int = 0,
    max_depth: int = 1, # max depth of recursion
    visited_urls: Optional[Set[str]] = None,
) -> Tuple[int, int, int]:
    
    # visited URLs
    if visited_urls is None:
        visited_urls = set()
    
    # skip if already visited
    if url in visited_urls:
        return (0, 0, 0)
    
    # skip if too deep
    if current_depth > max_depth:
        return (0, 0, 0)
    
    # Mark this URL as visited
    visited_urls.add(url)
    
    # Initialize counters for tracking
    attempted_downloads = 0
    successful_downloads = 0
    rendered_pages = 0

    # download if pdf link
    download_result = stream_download_pdf(session, url, out_dir, skip_existing=skip_existing)
    
    if download_result:
        # if still not at max depth, get the links for this one too
        if current_depth < max_depth:
            try:
                pdf_document = fitz.open(download_result)
                
                clickable_links = extract_annotation_links(pdf_document)
                text_links = extract_text_links(pdf_document)
                all_pdf_links = clickable_links | text_links
                
                pdf_document.close()
                
                for sub_url in all_pdf_links:
                    # Only HTTP/HTTPS links
                    if sub_url.lower().startswith(("http://", "https://")):
                        # Recursive call
                        sub_attempted, sub_succeeded, sub_rendered = process_link(
                            session=session,
                            url=sub_url,
                            out_dir=out_dir,
                            delay=delay,
                            max_from_page=max_from_page,
                            render_pages=render_pages,
                            user_agent=user_agent,
                            pdf_format=pdf_format,
                            render_timeout_ms=render_timeout_ms,
                            wait_until=wait_until,
                            skip_existing=skip_existing,
                            wait_selector=wait_selector,
                            wait_text=wait_text,
                            extra_wait_ms=extra_wait_ms,
                            auto_scroll=auto_scroll,
                            max_scrolls=max_scrolls,
                            screenshot_fallback=screenshot_fallback,
                            current_depth=current_depth + 1,  # Go one level deeper
                            max_depth=max_depth,
                            visited_urls=visited_urls  # Share visited URLs
                        )
                        
                        attempted_downloads += sub_attempted
                        successful_downloads += sub_succeeded
                        rendered_pages += sub_rendered
                        # wait
                        time.sleep(delay)
                        
            except Exception as error:
                logging.debug(f"Could not extract links from downloaded PDF: {error}")
        
        # Return our results (1 for this PDF + any from recursion)
        return (1 + attempted_downloads, 1 + successful_downloads, rendered_pages)

    # Google Drive/Docs
    parsed_url = urlparse(url)
    website_host = parsed_url.netloc
    
    is_google_drive = website_host.endswith("drive.google.com")
    is_google_docs = website_host.endswith("docs.google.com")
    
    if is_google_drive or is_google_docs:
        # convert to a direct download URL
        transformed_url = google_direct_download_url(url)
        
        if transformed_url:
            logging.debug("Google transform -> %s", transformed_url)
            attempted_downloads += 1
            
            if stream_download_pdf(session, transformed_url, out_dir, skip_existing=skip_existing):
                successful_downloads += 1
                time.sleep(delay)
                return (attempted_downloads, successful_downloads, 0)
            
            # confirmation page handling
            google_file = google_drive_fetch_with_confirm(session, transformed_url, out_dir, skip_existing)
            if google_file:
                successful_downloads += 1
                time.sleep(delay)
                return (attempted_downloads + 1, successful_downloads, 0)
        
        # use Playwright to click download button
        playwright_file = playwright_download_from_drive(url, out_dir, user_agent, skip_existing)
        if playwright_file:
            return (attempted_downloads + 1, successful_downloads + 1, 0)
        
        return (attempted_downloads if attempted_downloads else 1, successful_downloads, 0)

    # regular webpages
    pdf_links_on_page = collect_pdf_links_from_page(session, url)
    
    # Limit number of PDFs if param was given
    if max_from_page is not None:
        pdf_links_on_page = pdf_links_on_page[:max_from_page]
    
    # download pdfs
    for pdf_url in pdf_links_on_page:
        attempted_downloads += 1
        if stream_download_pdf(session, pdf_url, out_dir, skip_existing=skip_existing):
            successful_downloads += 1
        time.sleep(delay)

    # render the webpage itself to PDF if param given
    rendered_pdf_path = None
    
    if render_pages:
        rendered_file = render_html_to_pdf_playwright(
            url=url,
            out_dir=out_dir,
            user_agent=user_agent,
            skip_existing=skip_existing,
            pdf_format=pdf_format,
            timeout_ms=render_timeout_ms,
            wait_until=wait_until,
            wait_selector=wait_selector,
            wait_text=wait_text,
            extra_wait_ms=extra_wait_ms,
            auto_scroll=auto_scroll,
            max_scrolls=max_scrolls,
            screenshot_fallback=screenshot_fallback,
        )
        
        if rendered_file:
            rendered_pages += 1
            rendered_pdf_path = rendered_file

    # recursively process links from this page (if not at max depth)
    if current_depth < max_depth:
        page_links = set()
        
        # get links from the rendered PDF
        if rendered_pdf_path and rendered_pdf_path.exists():
            try:
                pdf_document = fitz.open(rendered_pdf_path)
                page_links = extract_annotation_links(pdf_document) | extract_text_links(pdf_document)
                pdf_document.close()
            except Exception as error:
                logging.debug(f"Could not extract links from rendered PDF: {error}")
                page_links = set()
        
        # get links from the HTML
        else:
            try:
                # Download the webpage HTML
                response = session.get(url, timeout=30, allow_redirects=True)
                response.raise_for_status()
                
                # Parse the HTML
                soup = BeautifulSoup(response.text, "html.parser")
                page_links = set()
                
                # Find all anchor tags with href attributes
                all_anchor_tags = soup.find_all("a", href=True)
                
                for anchor_tag in all_anchor_tags:
                    # Get the URL and convert to absolute
                    link_url = anchor_tag["href"].strip()
                    absolute_url = urljoin(response.url, link_url)
                    
                    # Only keep HTTP/HTTPS links
                    if absolute_url.lower().startswith(("http://", "https://")):
                        page_links.add(absolute_url)
                        
            except Exception as error:
                logging.debug(f"Could not extract links from webpage: {error}")
                page_links = set()
        
        for sub_url in page_links:
            if sub_url not in visited_urls:
                sub_attempted, sub_succeeded, sub_rendered = process_link(
                    session=session,
                    url=sub_url,
                    out_dir=out_dir,
                    delay=delay,
                    max_from_page=max_from_page,
                    render_pages=render_pages,
                    user_agent=user_agent,
                    pdf_format=pdf_format,
                    render_timeout_ms=render_timeout_ms,
                    wait_until=wait_until,
                    skip_existing=skip_existing,
                    wait_selector=wait_selector,
                    wait_text=wait_text,
                    extra_wait_ms=extra_wait_ms,
                    auto_scroll=auto_scroll,
                    max_scrolls=max_scrolls,
                    screenshot_fallback=screenshot_fallback,
                    current_depth=current_depth + 1,  # Go one level deeper
                    max_depth=max_depth,
                    visited_urls=visited_urls  # Share visited URLs
                )
                
                attempted_downloads += sub_attempted
                successful_downloads += sub_succeeded
                rendered_pages += sub_rendered
                
                time.sleep(delay)

    # Return final counts with min 1 attempts
    return (
        attempted_downloads if attempted_downloads else 1,
        successful_downloads,
        rendered_pages
    )

# process input PDF file
def process_input_pdf(
    pdf_path: Path,
    out_dir: Path,
    session: requests.Session,
    args: argparse.Namespace,
) -> Tuple[int, int, int, int]:
    print(f"\nScanning PDF for links: {pdf_path}")
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.warning("Cannot open %s (%s)", pdf_path, e)
        return (0, 0, 0, 0)

    try:
        ann_links = extract_annotation_links(doc)
        txt_links = extract_text_links(doc)
    finally:
        doc.close()

    links: List[str] = []
    seen = set()
    for u in [*sorted(ann_links), *sorted(txt_links)]:
        if u not in seen:
            seen.add(u); links.append(u)

    print(f"Found {len(links)} unique link(s).")
    if not links:
        return (0, 0, 0, 0)

    total_attempted = total_succeeded = total_rendered = 0
    visited_urls = set()  # Track visited URLs across all links from this PDF
    
    for url in tqdm(links, desc="Processing links"):
        if not url.lower().startswith(("http://", "https://")):
            continue
        attempted, succeeded, rendered = process_link(
            session=session,
            url=url,
            out_dir=out_dir,
            delay=args.delay,
            max_from_page=args.max_from_page,
            render_pages=args.render_pages,
            user_agent=args.user_agent,
            pdf_format=args.pdf_format,
            render_timeout_ms=args.render_timeout_ms,
            wait_until=args.wait_until,
            skip_existing=args.skip_existing,
            wait_selector=args.wait_selector,
            wait_text=args.wait_text,
            extra_wait_ms=args.extra_wait_ms,
            auto_scroll=args.auto_scroll,
            max_scrolls=args.max_scrolls,
            screenshot_fallback=args.screenshot_fallback,
            current_depth=0,  # Start at depth 0
            max_depth=args.depth,  # Use the depth from arguments
            visited_urls=visited_urls,  # Share visited URLs across links
        )
        total_attempted += attempted
        total_succeeded += succeeded
        total_rendered += rendered
        time.sleep(args.delay)

    print(f"Done: {pdf_path.name}")
    return (len(links), total_attempted, total_succeeded, total_rendered)

def gather_pdfs(input_path: Path, recursive: bool) -> List[Path]:
    """Return a list of PDF files from a file or directory (optionally recursive)."""
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".pdf" else []
    if not input_path.is_dir():
        return []
    if recursive:
        files = [p for p in input_path.rglob("*.pdf") if p.is_file()]
    else:
        files = [p for p in input_path.glob("*.pdf")]
    return sorted(files)

def main() -> None:
    p = argparse.ArgumentParser(description="Extract links from PDF(s), download PDFs, and render webpages to PDF.")
    p.add_argument("input", type=str, help="Path to a single PDF or a directory containing PDFs")
    p.add_argument("--out", type=str, default="downloads", help="Output directory (all files go here by default)")
    p.add_argument("--group-by-input", action="store_true", help="Create a subfolder per input PDF under --out")
    p.add_argument("--recursive", action="store_true", help="When input is a directory, also process subdirectories")
    p.add_argument("--delay", type=float, default=0.5, help="Delay (seconds) between downloads")
    p.add_argument("--user-agent", type=str, default="Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
    p.add_argument("--max-from-page", type=int, default=None, help="Limit PDFs scraped from each webpage")
    p.add_argument("--depth", type=int, default=3, help="Maximum crawl depth for recursive link following (default: 3)")

    # Rendering controls
    p.add_argument("--render-pages", dest="render_pages", action="store_true", default=True, help="Render non-Google webpages to PDF (default on)")
    p.add_argument("--no-render-pages", dest="render_pages", action="store_false")
    p.add_argument("--wait-selector", type=str, default=None, help="CSS selector that must appear before rendering")
    p.add_argument("--wait-text", type=str, default=None, help="Text that must appear before rendering")
    p.add_argument("--extra-wait-ms", type=int, default=1500, help="Extra ms to wait just before printing")
    p.add_argument("--auto-scroll", action="store_true", default=True, help="Auto-scroll to trigger lazy loading (default on)")
    p.add_argument("--no-auto-scroll", dest="auto_scroll", action="store_false")
    p.add_argument("--max-scrolls", type=int, default=10, help="Maximum auto-scroll steps")
    p.add_argument("--screenshot-fallback", action="store_true", default=True, help="If print fails, save full-page screenshot as PDF (default on)")
    p.add_argument("--no-screenshot-fallback", dest="screenshot_fallback", action="store_false")
    p.add_argument("--pdf-format", type=str, default="Letter", choices=["Letter", "A4"], help="Paper size")
    p.add_argument("--render-timeout-ms", type=int, default=45000)
    p.add_argument("--wait-until", type=str, default="networkidle", choices=["load", "domcontentloaded", "networkidle"])

    p.add_argument("--skip-existing", action="store_true", default=True, help="Skip if target file already exists (default on)")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("-v", "--verbose", action="count", default=0)

    args = p.parse_args()
    setup_logging(args.verbose)

    in_path = Path(args.input).expanduser().resolve()
    if not in_path.exists():
        print(f"Input path not found: {in_path}", file=sys.stderr); sys.exit(1)

    out_root = Path(args.out).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect PDFs to process
    pdfs = gather_pdfs(in_path, recursive=args.recursive)
    if not pdfs:
        print("No PDFs found to process.", file=sys.stderr); sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) to process.")
    session = requests.Session()
    session.headers.update({"User-Agent": args.user_agent})

    grand_links = grand_attempted = grand_succeeded = grand_rendered = 0

    for pdf_path in tqdm(pdfs, desc="Input PDFs"):
        out_dir = out_root / pdf_path.stem if args.group_by_input else out_root
        out_dir.mkdir(parents=True, exist_ok=True)

        links, attempted, succeeded, rendered = process_input_pdf(
            pdf_path=pdf_path,
            out_dir=out_dir,
            session=session,
            args=args,
        )
        grand_links += links
        grand_attempted += attempted
        grand_succeeded += succeeded
        grand_rendered += rendered

    print("\n=== Summary ===")
    print(f"Input PDFs processed: {len(pdfs)}")
    print(f"Total links found: {grand_links}")
    print(f"PDF downloads attempted: {grand_attempted}")
    print(f"PDF downloads succeeded: {grand_succeeded}")
    if args.render_pages:
        print(f"Webpages rendered to PDF: {grand_rendered}")
    print(f"Saved to: {out_root}")

if __name__ == "__main__":
    main()
