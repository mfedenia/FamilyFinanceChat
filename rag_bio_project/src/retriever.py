# -*- coding: utf-8 -*-
"""
Retriever for ChromaDB collections with:
- Auto routing across collections named as {username}_{character}
- Space-aware distance→similarity conversion (cosine / l2 / ip)
- High-threshold filtering (strict/medium/loose) with explicit override
- Similarity or MMR strategies
- Result grading: hit_keyword / related / unrelated
- Optional metadata `where` filter
- Optional `embedding_override` to force a particular embedder

Printing is kept in English to ease debugging.
"""

from __future__ import annotations
from typing import List, Dict, Optional, Literal, Any, Tuple
from dataclasses import dataclass
import json
import numpy as np

from chromadb import PersistentClient

RetrievalStrategy = Literal["similarity", "mmr"]
Strictness = Literal["strict", "medium", "loose"]

STRICT_PRESETS = {
    "strict": 0.70,
    "medium": 0.60,
    "loose": 0.50,
}

# ---------------------------- Utilities ---------------------------- #

def _parse_user_character(col_name: str) -> Tuple[str, str]:
    if "_" in col_name:
        a, b = col_name.split("_", 1)
        return a, b
    return col_name, ""

def _infer_target_collection(persist_dir: str, query_text: str, preferred_username: Optional[str]=None) -> List[str]:
    """Try to pick collection(s) by character keyword in the question.
    Fallback: all collections (or username-prefixed ones if provided)."""
    client = PersistentClient(path=persist_dir)
    q = (query_text or "").lower()
    best, best_len = None, -1
    cands = []
    for c in client.list_collections():
        name = c.name
        cands.append(name)
        u, ch = _parse_user_character(name)
        if ch and ch.lower() in q and len(ch) > best_len:
            best, best_len = name, len(ch)
    if best:
        return [best]
    if preferred_username:
        xs = [n for n in cands if n.lower().startswith(preferred_username.lower()+"_")]
        if xs:
            return xs
    if len(cands) == 1:
        return [cands[0]]
    return cands

def _embedder_from_coll_meta(meta: dict):
    """Construct an embedder from collection metadata 'embedding' object.
    Fallback to a sensible zh model if missing."""
    info = meta.get("embedding")
    if isinstance(info, str):
        try:
            info = json.loads(info)
        except Exception:
            info = {}
    info = info or {}
    provider  = (info.get("provider")  or "hf")
    model     = (info.get("model")     or "BAAI/bge-small-zh-v1.5")
    normalize = bool(info.get("normalize", True))
    try:
        if provider == "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=model)
        else:
            from langchain_community.embeddings import HuggingFaceBgeEmbeddings
            return HuggingFaceBgeEmbeddings(
                model_name=model,
                encode_kwargs={"normalize_embeddings": normalize}
            )
    except Exception as e:
        raise RuntimeError(f"Cannot construct embedder for provider={provider}, model={model}: {e}")

def _cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T

def _mmr_select(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float = 0.5) -> List[int]:
    """Maximal Marginal Relevance selection (indexes)."""
    N = doc_vecs.shape[0]
    if N == 0:
        return []
    k = min(k, N)
    q = query_vec.reshape(1, -1)
    sim_q = _cosine_sim_matrix(doc_vecs, q).reshape(-1)
    sim_dd = _cosine_sim_matrix(doc_vecs, doc_vecs)
    selected: List[int] = []
    candidates = set(range(N))
    while len(selected) < k and candidates:
        if not selected:
            idx = int(np.argmax(sim_q))
            selected.append(idx); candidates.remove(idx)
            continue
        max_sim_to_sel = np.max(sim_dd[:, selected], axis=1)
        best_idx, best_score = -1, -1e9
        for j in list(candidates):
            score = lambda_mult * sim_q[j] - (1 - lambda_mult) * max_sim_to_sel[j]
            if score > best_score:
                best_score, best_idx = score, j
        selected.append(best_idx); candidates.remove(best_idx)
    return selected

# --- Keyword helpers for grading --- #

KEYWORD_SYNS = {
    "收入": ["收入","年收入","工资","薪资","薪水","年薪","收益","报酬"],
    "婚姻": ["婚姻","已婚","未婚","结婚","配偶","伴侣","离异"],
    "财政": ["财政","财务","资产","负债","理财","投资","基金","风险","现金流"],
    "房租": ["房租","租金","租房","住房","房贷"],
    "工作": ["工作","职位","职业","岗位","就职","履历","经历","项目"],
}
ALL_SYNS = set([w for v in KEYWORD_SYNS.values() for w in v])

def _hit_keywords(text: str, query_text: str) -> bool:
    t = (text or "").lower()
    for kw in ALL_SYNS:
        if kw.lower() in t:
            return True
    if any(s in t for s in ["收入","年薪","工资"]) and any(ch.isdigit() for ch in t):
        return True
    return False

def _grade(score: float, hit_kw: bool, mid: float=0.60):
    if hit_kw: return "hit_keyword"
    if score >= mid: return "related"
    return "unrelated"

# --- Distance → similarity conversion, aware of space --- #

def _scores_from_dists(dists, space: str):
    """
    Convert Chroma distances to a unified 'similarity' score in [~0,1].
    - cosine:   dist = 1 - cosine_sim  -> sim = 1 - dist
    - l2:       for unit vectors, cosine ≈ 1 - (l2^2)/2
    - ip/dot:   larger better; Chroma may return negative dist -> sim = -dist
    """
    d = np.array(dists, dtype="float32")
    s = (space or "").lower()
    if s in ("cosine", "cos"):
        return 1.0 - d
    if s in ("l2", "euclidean"):
        return 1.0 - (d * d) / 2.0
    if s in ("ip", "dot", "inner"):
        return -d
    # default fallback: treat like cosine
    return 1.0 - d

# ---------------------------- Public API ---------------------------- #

def retrieve(
    persist_dir: str,
    query_text: str,
    k: int = 5,
    strategy: str = "mmr",
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    strictness: str = "strict",
    score_threshold: Optional[float] = None,
    preferred_username: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
    embedding_override: Any = None,   # Optional embedder to force usage
) -> Dict[str, Any]:
    """
    Retrieve relevant chunks across collections in a Chroma DB folder.

    Returns:
        {
          "route": {"collection": "...", "username": "...", "character": "..."} or None,
          "items": [ {collection, document, metadata, score, grade}, ... ],
          "params": {...}
        }
    """
    th = float(score_threshold if score_threshold is not None else STRICT_PRESETS.get(strictness, 0.99))
    client = PersistentClient(path=persist_dir)
    coll_names = _infer_target_collection(persist_dir, query_text, preferred_username)

    all_items = []
    for cname in coll_names:
        coll = client.get_collection(cname)
        meta = coll.metadata or {}
        # Detect HNSW space; common keys: "hnsw:space" or "space"
        space = meta.get("hnsw:space") or meta.get("space") or "l2"

        # Build embedder (override takes precedence)
        if embedding_override is not None:
            embedder = embedding_override
        else:
            embedder = _embedder_from_coll_meta(meta)

        # Encode query
        if hasattr(embedder, "embed_query"):
            qvec = np.array(embedder.embed_query(query_text), dtype="float32")
        else:
            qvec = np.array(embedder.embed_documents([query_text])[0], dtype="float32")

        n_cand = fetch_k if strategy == "mmr" else max(k, fetch_k)
        qres = coll.query(
            query_embeddings=[qvec.tolist()],
            n_results=n_cand,
            include=["documents", "metadatas", "distances", "embeddings"],
            where=where,
        )

        docs  = qres.get("documents", [[]])[0]
        metas = qres.get("metadatas", [[]])[0]
        dists = qres.get("distances", [[]])[0]
        embs  = qres.get("embeddings", [[]])[0]
        if not docs:
            continue

        sims = _scores_from_dists(dists, space)           # <<< correct, space-aware
        vecs = np.array(embs, dtype="float32")

        if strategy == "mmr":
            order = _mmr_select(qvec, vecs, k=k, lambda_mult=lambda_mult)
        else:
            order = list(np.argsort(-sims))[:k]

        for i in order:
            score = float(sims[i])
            if score < th:
                continue
            doc  = docs[i]
            meta_i = metas[i] or {}
            all_items.append({
                "collection": cname,
                "document": doc,
                "metadata": meta_i,
                "score": score,
                "grade": _grade(score, _hit_keywords(doc, query_text)),
            })

    # Re-rank: grade first, then score
    grade_rank = {"hit_keyword": 2, "related": 1, "unrelated": 0}
    all_items.sort(key=lambda r: (grade_rank.get(r["grade"], 0), r["score"]), reverse=True)

    route = None
    if all_items:
        top = all_items[0]["collection"]
        u, ch = _parse_user_character(top)
        route = {"collection": top, "username": u, "character": ch}

    return {
        "route": route,
        "items": all_items,
        "params": {
            "k": k, "strategy": strategy, "strictness": strictness,
            "score_threshold": th, "fetch_k": fetch_k, "lambda_mult": lambda_mult,
            "where": where
        },
    }
