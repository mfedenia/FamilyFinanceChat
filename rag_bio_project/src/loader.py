# -*- coding: utf-8 -*-
from pathlib import Path
from typing import List, Optional, Iterable, Dict
import hashlib, re, time
from urllib.parse import urlsplit, urlunsplit
import json

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, WebBaseLoader

def _normalize_text(s: str) -> str:
    if not s: return ""
    s = s.replace("\ufeff", "")
    s = s.replace("\xa0", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _clean_url(u: str) -> str:
    u = u.strip().replace(" ", "")
    parts = list(urlsplit(u))
    if not parts[0]:
        parts[0] = "https"
    return urlunsplit(parts)

def _doc_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8", errors="ignore")).hexdigest()[:16]

def _filter_content(text: str, min_chars: int, max_chars: Optional[int]) -> bool:
    n = len(text)
    if n < min_chars: return False
    if max_chars is not None and n > max_chars: return False
    return True

def load_sources(
    pdf_dir: str = "data_pdfs",
    txt_dir: Optional[str] = "data_txt",
    urls: Optional[List[str]] = None,
    recursive: bool = True,
    txt_extensions: Iterable[str] = (".txt", ".md"),
    txt_encoding: str = "utf-8",
    pages: Optional[str] = None,
    min_chars: int = 50,
    max_chars: Optional[int] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 15,
    max_retries: int = 2,
) -> List[Document]:
    docs: List[Document] = []
    total_raw = 0

    # PDFs
    pdir = Path(pdf_dir)
    if pdir.exists():
        pdf_iter = pdir.rglob("*.pdf") if recursive else pdir.glob("*.pdf")
        for p in pdf_iter:
            try:
                loader = PyPDFLoader(str(p))
                loaded = loader.load()
                if pages:
                    parts = [None if x in ("", "None", None) else int(x) for x in pages.split(":")]
                    start = parts[0] if len(parts) > 0 else None
                    stop  = parts[1] if len(parts) > 1 else None
                    step  = parts[2] if len(parts) > 2 else None
                    loaded = [d for d in loaded[slice(start, stop, step)]]
                for d in loaded:
                    d.page_content = _normalize_text(d.page_content)
                    d.metadata["source"] = str(p)
                    d.metadata["source_type"] = "pdf"
                    d.metadata["loader_info"] = {"type":"PyPDFLoader","pages":pages}
                    if _filter_content(d.page_content, min_chars, max_chars):
                        docs.append(d)
                total_raw += len(loaded)
            except Exception as e:
                print(f"[WARN] Skip PDF {p}: {e}")

    # TXT/MD
    if txt_dir:
        tdir = Path(txt_dir)
        if tdir.exists():
            exts = {e.lower() for e in txt_extensions}
            txt_iter = tdir.rglob("*") if recursive else tdir.glob("*")
            for p in txt_iter:
                if p.is_file() and p.suffix.lower() in exts:
                    try:
                        loader = TextLoader(str(p), encoding=txt_encoding)
                        loaded = loader.load()
                        for d in loaded:
                            d.page_content = _normalize_text(d.page_content)
                            d.metadata["source"] = str(p)
                            d.metadata["source_type"] = "txt"
                            d.metadata["loader_info"] = {"type":"TextLoader","encoding":txt_encoding}
                            if _filter_content(d.page_content, min_chars, max_chars):
                                docs.append(d)
                        total_raw += len(loaded)
                    except Exception as e:
                        print(f"[WARN] Skip TXT {p}: {e}")

    # Web
    if urls:
        urls = [u for u in (urls or []) if isinstance(u, str) and u.strip()]
        for u in urls:
            url = _clean_url(u)
            tries = 0
            while True:
                try:
                    loader = WebBaseLoader(
                        url,
                        requests_kwargs={"headers": headers or {"User-Agent":"Mozilla/5.0 (RAG-Loader/1.0)"}, "timeout": timeout}
                    )
                    loaded = loader.load()
                    for d in loaded:
                        d.page_content = _normalize_text(d.page_content)
                        d.metadata["source"] = url
                        d.metadata["source_type"] = "web"
                        d.metadata["loader_info"] = {"type":"WebBaseLoader","timeout":timeout,"headers":bool(headers)}
                        if _filter_content(d.page_content, min_chars, max_chars):
                            docs.append(d)
                    total_raw += len(loaded)
                    break
                except Exception as e:
                    tries += 1
                    if tries > max_retries:
                        print(f"[WARN] Skip URL {url} after {max_retries} retries: {e}")
                        break
                    print(f"[INFO] Retry {tries}/{max_retries} for URL {url} due to: {e}")
                    time.sleep(1.0)

    # Dedup
    seen = set()
    uniq: List[Document] = []
    for d in docs:
        h = _doc_hash(d.page_content)
        key = (d.metadata.get("source"), h)
        if key not in seen:
            seen.add(key)
            uniq.append(d)

    print(f"[INFO] Loader complete. raw={total_raw}, kept={len(uniq)}, pdf_dir={pdf_dir}, txt_dir={txt_dir}, urls={len(urls or [])}")
    return uniq
