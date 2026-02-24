# src/llm.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

from prompting import build_prompt_messages_auto
from retriever import retrieve

import os

@dataclass
class LLMConfig:
    provider: str = "openai"
    model: Optional[str] = None
    temperature: float = 0.2
    max_tokens: Optional[int] = 512
    base_url: Optional[str] = None
    timeout: int = 60

def build_chat_model(cfg: LLMConfig):
    prov = (cfg.provider or "openai").lower()
    if prov == "openai":
        from langchain_openai import ChatOpenAI
        model = cfg.model or "gpt-4o-mini"
        return ChatOpenAI(model=model, temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.timeout)
    elif prov == "dashscope":
        from langchain_openai import ChatOpenAI
        base = cfg.base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = cfg.model or "qwen-plus"
        return ChatOpenAI(model=model, temperature=cfg.temperature, max_tokens=cfg.max_tokens, timeout=cfg.timeout,
                          base_url=base, api_key=os.getenv("DASHSCOPE_API_KEY"))
    elif prov == "ollama":
        from langchain_ollama import ChatOllama
        model = cfg.model or "qwen2:7b-instruct-q4_0"
        base = cfg.base_url or "http://localhost:11434"
        return ChatOllama(model=model, temperature=cfg.temperature, base_url=base, timeout=cfg.timeout)
    else:
        raise ValueError(f"Unknown provider: {cfg.provider}")

def answer(question: str, *, persist_dir: str, strictness: str="strict",
           retriever_kwargs: Optional[Dict[str, Any]]=None, prompt_mode: Optional[str]=None,
           chat_history: Optional[str]=None, llm_cfg: Optional[LLMConfig]=None) -> Dict[str, Any]:

    rk = {"k":5, "strategy":"mmr", "strictness":strictness}
    if retriever_kwargs:
        rk.update(retriever_kwargs or {})
    retrieval = retrieve(persist_dir=persist_dir, query_text=question, **rk)
    msgs, info = build_prompt_messages_auto(question, retrieval, mode=prompt_mode, chat_history=chat_history)
    cfg = llm_cfg or LLMConfig(provider="openai")
    chat = build_chat_model(cfg)
    resp = chat.invoke(msgs)
    text = getattr(resp, "content", str(resp))
    numbered = (info.get("mode") == "rag_with_citations")
    refs = []
    for i, it in enumerate(retrieval.get("items", []), 1):
        src = (it.get("metadata") or {}).get("source", "unknown")
        coll = it.get("collection","unknown")
        if numbered:
            refs.append(f"[{i}] source={src} | coll={coll}")
        else:
            refs.append(f"- source={src} | coll={coll}")
    return {"route": retrieval.get("route"), "prompt_mode": info.get("mode"),
            "messages": msgs, "answer": text, "references": "\n".join(refs),
            "retrieval": retrieval}
