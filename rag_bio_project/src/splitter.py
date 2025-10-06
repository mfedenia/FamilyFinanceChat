# -*- coding: utf-8 -*-
from typing import List, Dict
from dataclasses import dataclass
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class SplitterProfile:
    chunk_size: int
    chunk_overlap: int
    separators: List[str]

PDF_PROFILE = SplitterProfile(1200, 200, ["\n\n","\n","。","！","？",".","!","?"," ",""])
TXT_PROFILE = SplitterProfile(1000, 150, ["\n\n","\n",".","?","!","。","？","！"," ",""])
WEB_PROFILE = SplitterProfile(900, 150, ["\n\n","\n","。","！","？",".","!","?"," ",""])

PROFILE_MAP: Dict[str, SplitterProfile] = {"pdf":PDF_PROFILE,"txt":TXT_PROFILE,"web":WEB_PROFILE}

def _build_splitter(p: SplitterProfile) -> RecursiveCharacterTextSplitter:
    overlap = p.chunk_overlap if p.chunk_overlap < p.chunk_size else max(0, min(p.chunk_size//5, 200))
    if overlap != p.chunk_overlap:
        print(f"[WARN] overlap >= size; fallback to {overlap}")
    return RecursiveCharacterTextSplitter(chunk_size=p.chunk_size, chunk_overlap=overlap, separators=p.separators)

def split_documents_type_aware(docs: List[Document], default_type: str = "pdf", verbose: bool=True) -> List[Document]:
    if verbose:
        print(f"[INFO] Type-aware splitting started. total_docs={len(docs)}, default_type={default_type}")
    splitter_cache: Dict[str, RecursiveCharacterTextSplitter] = {}
    def get_splitter(kind: str):
        typ = (kind or "").lower()
        if typ not in PROFILE_MAP:
            typ = default_type
        if typ not in splitter_cache:
            splitter_cache[typ] = _build_splitter(PROFILE_MAP[typ])
            if verbose:
                p = PROFILE_MAP[typ]
                print(f"[INFO] Splitter ready for type='{typ}' (size={p.chunk_size}, overlap={p.chunk_overlap})")
        return splitter_cache[typ]

    out: List[Document] = []
    per_source_index: Dict[str, int] = {}
    for d in docs:
        stype = (d.metadata.get("source_type") or default_type).lower()
        splitter = get_splitter(stype)
        chunks = splitter.split_documents([d])
        source_key = str(d.metadata.get("source","unknown"))
        start_idx = per_source_index.get(source_key, 0)
        for i, c in enumerate(chunks):
            c.metadata = dict(d.metadata) | {"chunk_index": start_idx + i, "splitter_profile": stype}
            out.append(c)
        per_source_index[source_key] = start_idx + len(chunks)
        if verbose:
            print(f"[INFO] Split {stype} source={source_key} -> {len(chunks)} chunks (acc={len(out)})")

    if verbose:
        print(f"[INFO] Type-aware splitting done. total_chunks={len(out)}")
    return out
