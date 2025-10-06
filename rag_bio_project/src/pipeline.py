# -*- coding: utf-8 -*-
"""
Minimal end-to-end pipeline (no middleware):
retriever -> prompt -> LLM -> answer (+references)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from retriever import retrieve
from prompting import build_prompt_messages_auto
from llm import LLMConfig, build_chat_model


# ----------------------------- Config ----------------------------- #

@dataclass
class PipelineConfig:
    # Vector store
    persist_dir: str
    strictness: str = "strict"
    where: Optional[Dict[str, Any]] = None

    # LLM
    provider: str = "openai"
    model: Optional[str] = "gpt-4o-mini"
    temperature: float = 0.2

    # Routing & prompt
    do_role_detection: bool = True       # 现在不会真的做识别，只是保留这个开关
    prompt_mode: Optional[str] = None    # "concise"/"balanced"/"detailed"/"qa_strict"/"compare"/None(auto)

    # Optional: force a specific embedder for queries
    embedding_override: Any = None


# ----------------------------- Helpers ---------------------------- #

def _format_references(items: List[Dict[str, Any]], top_k: int = 5) -> str:
    refs = []
    for i, it in enumerate(items[:top_k], 1):
        m = (it.get("metadata") or {})
        src = m.get("source", "unknown")
        page = m.get("page")
        refs.append(f"[{i}] {src}" + (f" · p.{page}" if page is not None else ""))
    return "\n".join(refs)
    
def _to_text(resp) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    # LangChain ChatMessage / AIMessage
    if hasattr(resp, "content"):
        return resp.content
    # OpenAI SDK / 其它对象
    if isinstance(resp, dict) and "content" in resp:
        return str(resp["content"])
    return str(resp)



# ----------------------------- Pipeline --------------------------- #

def run_pipeline(question: str, cfg: PipelineConfig) -> Dict[str, Any]:
    """Run the full pipeline (no middleware)."""

    # 1) Role detection placeholder (disabled); keep variable for future
    preferred_username = None
    role_det = {}
    if cfg.do_role_detection:
        # 如果未来需要恢复角色识别，这里填充 preferred_username
        # 例如：
        # role_det = detect_characters_from_question(question, cfg.persist_dir) or {}
        # preferred_username = role_det.get("username")
        pass

    # 2) Retrieve
    retrieval = retrieve(
        persist_dir=cfg.persist_dir,
        query_text=question,
        k=5, strategy="mmr", fetch_k=50,
        strictness=cfg.strictness,
        where=cfg.where,
        preferred_username=preferred_username,
        embedding_override=cfg.embedding_override,
    )

    # 3) Prompt messages
    messages, pinfo = build_prompt_messages_auto(
        question=question,
        retrieved=retrieval,
        mode=cfg.prompt_mode,
    )


    # 4) LLM
    llm = build_chat_model(LLMConfig(
        provider=cfg.provider,
        model=cfg.model,
        temperature=cfg.temperature,
    ))
    raw_answer = llm.invoke(messages)
    answer = _to_text(raw_answer)   # << 统一转成字符串


    # 5) References
    references = _format_references(retrieval.get("items", []), top_k=5)

    return {
        "answer": answer,
        "references": references,
        "prompt_mode": pinfo.get("mode"),
        "retrieval": retrieval,
        "route": retrieval.get("route"),
        "role_detection": role_det,
        "params": {
            "strictness": cfg.strictness,
            "provider": cfg.provider,
            "model": cfg.model,
            "prompt_mode": cfg.prompt_mode,
        },
    }
