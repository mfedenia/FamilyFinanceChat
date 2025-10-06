# -*- coding: utf-8 -*-
import os, re, time, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings

from chromadb import PersistentClient
import json


@dataclass
class EmbeddingConfig:
    provider: str = "hf"                     # 'hf' | 'openai' | 'dashscope'
    model: str = "BAAI/bge-m3"
    normalize: bool = True

def _sanitize_name(s: str, fallback_prefix: str = "user") -> str:
    s2 = re.sub(r"[^a-zA-Z0-9_]+", "_", str(s or "")).strip("_")
    if not s2:
        s2 = f"{fallback_prefix}_{int(time.time())}"
    return s2

def _doc_id(meta: Dict, idx: int) -> str:
    base = f"{meta.get('source','unknown')}|{meta.get('chunk_index', idx)}"
    return hashlib.md5(base.encode('utf-8', errors='ignore')).hexdigest()[:12] + f"_{idx}"

def build_embeddings(cfg: EmbeddingConfig):
    prov = cfg.provider.lower()
    if prov == "hf":
        print(f"[INFO] Using HF embeddings: {cfg.model} (normalize={cfg.normalize})")
        return HuggingFaceBgeEmbeddings(model_name=cfg.model, encode_kwargs={"normalize_embeddings": cfg.normalize})
    elif prov == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set.")
        model_name = cfg.model or "text-embedding-3-small"
        print(f"[INFO] Using OpenAI embeddings: {model_name}")
        return OpenAIEmbeddings(model=model_name)
    elif prov == "dashscope":
        if not os.getenv("DASHSCOPE_API_KEY"):
            raise RuntimeError("DASHSCOPE_API_KEY is not set.")
        model_name = cfg.model or "text-embedding-v1"
        print(f"[INFO] Using DashScope embeddings: {model_name}")
        return DashScopeEmbeddings(model=model_name)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")

def _sanitize_meta_for_chroma(m: dict) -> dict:
    out = {}
    for k, v in (m or {}).items():
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = json.dumps(v, ensure_ascii=False)
    return out

def compute_vectors_once(chunks: List[Document], embedder):
    texts = [c.page_content for c in chunks]
    metas = [dict(c.metadata or {}) for c in chunks]
    ids = [_doc_id(m, i) for i, m in enumerate(metas)]
    print(f"[INFO] Computing embeddings for {len(texts)} chunks...")
    vectors = embedder.embed_documents(texts)
    if not vectors or not vectors[0]:
        raise RuntimeError("Empty embeddings returned.")
    dim = len(vectors[0])
    emb_info = {"provider": type(embedder).__name__, "dim": dim}
    print(f"[INFO] Embedding dim={dim}")
    return ids, texts, metas, vectors, emb_info

def persist_vectorstores_for_characters(ids, texts, metas, vectors, persist_dir, username, character_names, emb_meta, collection_prefix: Optional[str]=None):
    user_tag = _sanitize_name(username, "user")
    created = []
    client: PersistentClient = PersistentClient(path=persist_dir)
    for cname in character_names:
        char_tag = _sanitize_name(cname, "char")
        base_name = f"{user_tag}_{char_tag}"
        coll_name = _sanitize_name(f"{collection_prefix}_{base_name}", "coll") if collection_prefix else base_name
        print(f"[INFO] Creating/updating collection: {coll_name}")
        meta_for_coll = {"embedding": json.dumps(emb_meta, ensure_ascii=False)}
        coll = client.get_or_create_collection(
            name=coll_name,
            metadata=meta_for_coll
        )

        B = 256
        for i in range(0, len(ids), B):
            batch_metas = [ _sanitize_meta_for_chroma(m) for m in metas[i:i+B] ]
            coll.add(ids=ids[i:i+B], documents=texts[i:i+B], metadatas=batch_metas, embeddings=vectors[i:i+B])
        print(f"[INFO] Collection '{coll_name}' upserted with {len(ids)} items.")
        created.append(coll_name)
    print(f"[INFO] Done. Created/updated {len(created)} collections.")
    return created

def build_embeddings_and_vectorstores(chunks: List[Document], username: str, character_names: List[str], persist_dir: str = "index", emb_cfg: EmbeddingConfig = EmbeddingConfig()):
    if not character_names:
        raise ValueError("character_names must not be empty. Provide at least one name.")
    embedder = build_embeddings(emb_cfg)
    ids, texts, metas, vectors, emb_info = compute_vectors_once(chunks, embedder)
    return persist_vectorstores_for_characters(ids, texts, metas, vectors, persist_dir, username, character_names, {"provider": emb_cfg.provider, "model": emb_cfg.model, "dim": emb_info["dim"], "normalize": emb_cfg.normalize})
