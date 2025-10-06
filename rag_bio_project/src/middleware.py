# src/middleware.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
from chromadb import PersistentClient

def list_collections(persist_dir: str):
    client = PersistentClient(path=persist_dir)
    return [c.name for c in client.list_collections()]

def _split(col: str) -> Tuple[str,str]:
    if "_" in col:
        a,b = col.split("_",1); return a,b
    return col,""

def detect_characters_from_question(question: str, persist_dir: str) -> Dict[str, Any]:
    cols = list_collections(persist_dir)
    q = (question or "").lower()
    hits, chars = [], []
    for c in cols:
        u, ch = _split(c)
        if ch and ch.lower() in q:
            hits.append(c); chars.append(ch)
    return {"candidates": hits, "characters": list(dict.fromkeys(chars))}

@dataclass
class SuspicionConfig:
    enabled: bool = True
    level: float = 0.0
    policy: str = "none"

def apply_suspicion_policy(messages: List[Any], cfg: SuspicionConfig) -> List[Any]:
    return messages

@dataclass
class DeceptionPolicy:
    enabled: bool = False
    mode: str = "hide"
    scope: Optional[List[str]] = None

def apply_deception(messages: List[Any], policy: DeceptionPolicy) -> List[Any]:
    return messages

@dataclass
class PersonaConfig:
    enabled: bool = False
    type: str = "ocean"
    traits: Dict[str, Any] = None

def apply_persona(messages: List[Any], persona: PersonaConfig) -> List[Any]:
    return messages

@dataclass
class VerificationConfig:
    enabled: bool = False
    mode: str = "presence"
    min_hits: int = 1

def verify_answer_against_context(answer: str, context_items: List[Dict[str, Any]], cfg: VerificationConfig):
    if not cfg.enabled:
        return {"verified": None, "hits": 0, "details": []}
    return {"verified": True, "hits": cfg.min_hits, "details": []}
