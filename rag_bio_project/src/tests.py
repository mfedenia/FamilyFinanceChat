# src/tests.py
# Utilities for testing retriever → middleware → prompt → LLM → pipeline.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Iterable
from pathlib import Path
import time, csv, json

from chromadb import PersistentClient

from retriever import retrieve
from prompting import build_prompt_messages_auto
from middleware import detect_characters_from_question, VerificationConfig, verify_answer_against_context
from llm import LLMConfig, build_chat_model
from pipeline import PipelineConfig, run_pipeline

# ---------- 基础工具 ----------

def print_collections(persist_dir: str | Path) -> List[str]:
    client = PersistentClient(path=str(persist_dir))
    names = [c.name for c in client.list_collections()]
    print("[Collections]", names)
    for n in names:
        coll = client.get_collection(n)
        meta = coll.metadata or {}
        emb = meta.get("embedding")
        if isinstance(emb, str):
            try: emb = json.loads(emb)
            except: pass
        print(f"  - {n}: count={coll.count()} | embedding={emb}")
    return names

def pretty_print_items(items: List[Dict[str,Any]], n: int = 5) -> None:
    for i, it in enumerate(items[:n], 1):
        m = it.get("metadata") or {}
        src = m.get("source", "unknown")
        print(f"[{i}] grade={it.get('grade')} score={it.get('score'):.3f} source={src}")

def show_citations_map(retrieval: Dict[str,Any], numbered: bool=True, limit: int=10) -> None:
    items = retrieval.get("items", [])
    for i, it in enumerate(items[:limit], 1):
        m = it.get("metadata") or {}
        if numbered:
            print(f"[{i}] coll={it.get('collection')} | source={m.get('source')} | page={m.get('page')}")
        else:
            print(f"- coll={it.get('collection')} | source={m.get('source')} | page={m.get('page')}")

# ---------- 单步/端到端演示 ----------

def sanity_query(
    persist_dir: str | Path,
    question: str,
    *,
    k: int = 5,
    strategy: str = "mmr",
    strictness: str = "strict",
    where: Optional[Dict[str,Any]] = None,
) -> Dict[str,Any]:
    print(f"\n[Sanity] Q: {question}")
    res = retrieve(str(persist_dir), question, k=k, strategy=strategy, strictness=strictness, where=where)
    print("  Route:", res.get("route"))
    pretty_print_items(res.get("items", []), n=5)
    return res

def demo_full_pipeline(
    persist_dir: str | Path,
    question: str,
    *,
    provider: str = "openai",
    model: Optional[str] = "gpt-4o-mini",
    strictness: str = "strict",
    do_role_detection: bool = True,
    prompt_mode: Optional[str] = None,
    where: Optional[Dict[str,Any]] = None,
) -> Dict[str,Any]:
    print(f"\n[Pipeline Demo] Q: {question}")
    cfg = PipelineConfig(
        persist_dir=str(persist_dir),
        strictness=strictness,
        where=where,
        provider=provider, model=model,
        prompt_mode=prompt_mode,
        do_role_detection=do_role_detection,
        # 下列中间层目前默认不改变行为；需要时可打开：
        # suspicion=dict(enabled=True, level=0.0),
        # deception=dict(enabled=False, mode="hide", scope=["资产","财务"]),
        # persona=dict(enabled=False, type="mbti", traits={"MBTI":"INTJ"}),
        # verification=dict(enabled=True, mode="presence", min_hits=1),
    )
    t0 = time.time()
    out = run_pipeline(question, cfg)
    t1 = time.time()
    print("  Route:", out.get("route"))
    print("  Mode:", out.get("prompt_mode"))
    if out.get("references"):
        print("\n— References —")
        print(out["references"])
    print("\n— Answer —")
    print(out.get("answer","")[:1000])
    print(f"\n  Latency: {(t1-t0):.2f}s")
    return out

# ---------- 对比/扫参 ----------

def sweep_strictness(persist_dir: str | Path, question: str, levels: Iterable[str] = ("strict","medium","loose")):
    rows = []
    for lv in levels:
        res = retrieve(str(persist_dir), question, k=5, strictness=lv)
        cnt = len(res.get("items", []))
        top = res.get("items", [None])[0]
        best = (top or {}).get("score")
        print(f"[strictness={lv}] hits={cnt} best={best}")
        rows.append({"strictness": lv, "hits": cnt, "best_score": best})
    return rows

def ablation_strategy(persist_dir: str | Path, question: str, k: int = 5):
    for strat in ("mmr", "similarity"):
        res = retrieve(str(persist_dir), question, k=k, strategy=strat)
        top = res.get("items", [])[:3]
        print(f"\n[strategy={strat}]")
        pretty_print_items(top, n=3)

def compare_roles(
    persist_dir: str | Path,
    question: str,
    character_names: List[str],
    *,
    provider: str = "openai",
    model: Optional[str] = "gpt-4o-mini",
    strictness: str = "strict",
) -> Dict[str,Any]:
    """
    针对多个角色（collection 命名 {username}_{character}）逐一问同一问题。
    """
    client = PersistentClient(path=str(persist_dir))
    out = {}
    for c in client.list_collections():
        name = c.name
        if not any(name.endswith("_"+ch) or name.split("_",1)[-1] == ch for ch in character_names):
            continue
        print(f"\n=== {name} ===")
        cfg = PipelineConfig(
            persist_dir=str(persist_dir),
            strictness=strictness,
            provider=provider, model=model,
            do_role_detection=False,  # 已指定角色，不需再识别
        )
        res = run_pipeline(question, cfg)
        print(res.get("answer","")[:600])
        out[name] = res
    return out

# ---------- 验证/诊断 ----------

def verification_check(
    persist_dir: str | Path,
    question: str,
    *,
    enable: bool = True,
    provider: str = "openai",
    model: Optional[str] = "gpt-4o-mini",
) -> Dict[str,Any]:
    cfg = PipelineConfig(
        persist_dir=str(persist_dir),
        provider=provider, model=model,
        verification=dict(enabled=enable, mode="presence", min_hits=1),
    )
    res = run_pipeline(question, cfg)
    print("Verification:", res.get("verification"))
    return res

def role_detection_debug(persist_dir: str | Path, question: str):
    det = detect_characters_from_question(question, str(persist_dir))
    print("[Role detection]", det)
    return det

# ---------- 批量评测 / 导出 ----------

def batch_questions(
    persist_dir: str | Path,
    questions: List[str],
    *,
    provider: str = "openai",
    model: Optional[str] = "gpt-4o-mini",
    strictness: str = "strict",
    csv_path: Optional[str | Path] = None,
) -> List[Dict[str,Any]]:
    results = []
    for q in questions:
        out = demo_full_pipeline(
            persist_dir=persist_dir,
            question=q,
            provider=provider, model=model,
            strictness=strictness,
        )
        results.append({
            "question": q,
            "route": out.get("route"),
            "mode": out.get("prompt_mode"),
            "answer": out.get("answer",""),
            "references": out.get("references",""),
            "hit_count": len(out.get("retrieval",{}).get("items",[])),
        })
    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print("[Saved]", csv_path)
    return results
