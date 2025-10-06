# src/prompting.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import re
from langchain_core.prompts import ChatPromptTemplate

PROMPT_MODES: Dict[str, ChatPromptTemplate] = {}

PROMPT_MODES["rag_concise"] = ChatPromptTemplate.from_messages([
    ("system",
     "You are an assistant for question-answering tasks. "
     "Use the following pieces of retrieved context to answer the question. "
     "If you don't know the answer, say you don't know. "
     "Use at most three sentences and keep the answer concise."),
    ("human", "Question: {input}\n\nContext:\n{context}\n\nAnswer:")
])

PROMPT_MODES["rag_with_citations"] = ChatPromptTemplate.from_messages([
    ("system",
     "You are a careful assistant. Answer strictly using the provided context. "
     "Cite sources in square brackets like [1], [2]. If the answer is not in the context, say you don't know."),
    ("human",
     "Question: {input}\n\n"
     "Context (each item is prefixed with [n]):\n{context}\n\n"
     "Requirements:\n"
     "- Keep it concise.\n"
     "- Include bracketed citations [n] for key claims.\n\n"
     "Answer:")
])

PROMPT_MODES["contextualize_question"] = ChatPromptTemplate.from_messages([
    ("system",
     "Given a conversation and a follow-up question, rewrite the question to be a standalone query. "
     "Do not answer the question, only rewrite it if needed."),
    ("human",
     "Chat History:\n{chat_history}\n\n"
     "Follow-up Question: {input}\n\n"
     "Standalone Question:")
])

PROMPT_MODES["rag_extraction"] = ChatPromptTemplate.from_messages([
    ("system",
     "Extract structured information from the context. If a field is not present, use null. "
     "Output JSON only, no extra text."),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {input}\n\n"
     "Extract these fields (JSON keys): "
     "['name','birthplace','birth_year','annual_income','marital_status','assets','work_experience']\n\n"
     "JSON:")
])

PROMPT_MODES["rag_timeline"] = ChatPromptTemplate.from_messages([
    ("system",
     "Build a concise, chronological timeline strictly from the context. "
     "If dates are unclear, use approximate ranges. If nothing relevant, say you don't know."),
    ("human",
     "Question: {input}\n\n"
     "Context:\n{context}\n\n"
     "Return a bullet list: [YYYY or YYYY-MM] — event.\n\n"
     "Timeline:")
])

PROMPT_MODES["rag_compare"] = ChatPromptTemplate.from_messages([
    ("system",
     "Compare the two entities strictly using the provided context. "
     "Be neutral, concise, and show differences clearly."),
    ("human",
     "Question: {input}\n\n"
     "Context:\n{context}\n\n"
     "Return a short table-like bullet list with dimensions such as background, income, marriage, assets, and experience. "
     "Conclude with a 1-2 sentence summary.\n\n"
     "Comparison:")
])

def get_prompt(mode: str = "rag_concise") -> ChatPromptTemplate:
    if mode not in PROMPT_MODES:
        raise ValueError(f"Unknown prompt mode: {mode}. Available: {list(PROMPT_MODES.keys())}")
    return PROMPT_MODES[mode]

_CIT = re.compile(r"\b(cite|citation|sources?|reference[s]?)\b", re.I)
_CMP = re.compile(r"\b(compare|vs\.?|versus|difference|pros\s+and\s+cons)\b", re.I)
_TML = re.compile(r"\b(timeline|chronolog(y|ical)|when|year(s)?|history of)\b", re.I)
_EXT = re.compile(r"\b(extract|structured|json|fields?)\b", re.I)

def auto_select_mode(question: str, *, has_chat_history: bool=False, fallback: str="rag_concise") -> str:
    q = (question or "").strip()
    if not q: return fallback
    if _CIT.search(q): return "rag_with_citations"
    if _CMP.search(q): return "rag_compare"
    if _TML.search(q): return "rag_timeline"
    if _EXT.search(q): return "rag_extraction"
    return fallback

def _format_context_plain(items: List[Dict[str, Any]], limit: int=12) -> str:
    out = []
    for it in items[:limit]:
        out.append(it.get("document",""))
    return "\n".join(out).strip() or "(no context)"

def _format_context_with_indices(items: List[Dict[str, Any]], limit: int=12) -> str:
    lines = []
    for i, it in enumerate(items[:limit], 1):
        lines.append(f"[{{i}}] {{it.get('document','')}}")
    return "\n".join(lines).strip() or "(no context)"

def build_prompt_messages_auto(question: str, retrieved: Dict[str, Any], *, mode: Optional[str]=None, chat_history: Optional[str]=None):
    items = retrieved.get("items", []) if isinstance(retrieved, dict) else []
    selected = mode or auto_select_mode(question, has_chat_history=bool(chat_history and chat_history.strip()), fallback="rag_concise")
    tpl = get_prompt(selected)
    if selected == "rag_with_citations":
        ctx = _format_context_with_indices(items, 12)
    else:
        ctx = _format_context_plain(items, 12)
    variables = {"input": question, "context": ctx}
    if selected == "contextualize_question":
        variables["chat_history"] = chat_history or ""
    msgs = tpl.format_messages(**variables)
    print(f"[INFO] Prompt mode={selected}, context_len={{len(ctx)}}")
    return msgs, {"mode": selected, **variables}
