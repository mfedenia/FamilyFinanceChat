import re
import time
from contextlib import asynccontextmanager
from fastapi import Request
from prometheus_client import Histogram, Counter, Gauge

# ── Metric Definitions ────────────────────────────────────────────────────────

# RETRIEVED_CHUNKS = Histogram(
#     "openwebui_rag_chunks_retrieved",
#     "Number of chunks retrieved per query",
#     buckets=(1, 2, 3, 5, 8, 10, 15, 20)
# )

# RETRIEVAL_SCORE = Histogram(
#     "openwebui_rag_retrieval_score",
#     "Similarity scores of retrieved chunks",
#     buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
# )

STAGE_LATENCY = Histogram(
    "openwebui_stage_latency_seconds",
    "Time spent in each named processing stage per request",
    ["stage"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 15, 20, 30)
)

EMBEDDING_LATENCY = Histogram(
    "openwebui_embedding_latency_seconds",
    "Time to generate query embeddings",
    ["model"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5)
)

QDRANT_SEARCH_LATENCY = Histogram(
    "openwebui_qdrant_search_latency_seconds",
    "Time for Qdrant vector similarity search",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5)
)

RERANKER_LATENCY = Histogram(
    "openwebui_reranker_latency_seconds",
    "Time spent in reranking stage",
    buckets=(0.1, 0.5, 1, 2, 5, 10, 15, 20)
)

OPENAI_QUEUE_LATENCY = Histogram(
    "openwebui_openai_queue_latency_seconds",
    "Time waiting for OpenAI API to return first token",
    ["model"],
    buckets=(0.5, 1, 2, 5, 10, 20, 30, 45, 60)
)
CONTEXT_TOKENS = Histogram(
    "openwebui_context_tokens_total",
    "Total tokens sent to LLM including system prompt and RAG chunks",
    ["model"],
    buckets=(500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000)
)

API_LATENCY = Histogram(
    "openwebui_api_latency_seconds",
    "Per-route API latency",
    ["method", "route", "status"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)
)
API_ERRORS = Counter(
    "openwebui_api_errors_total",
    "API error count by route and status",
    ["method", "route", "status"]
)
REQUESTS_IN_FLIGHT = Gauge(
    "openwebui_requests_in_flight",
    "Requests currently being processed"
)

# RAG — retrieval API routes only (/api/v1/retrieval/*)
RAG_REQUEST_LATENCY = Histogram(
    "openwebui_rag_request_latency_seconds",
    "HTTP-level latency of retrieval API routes",
    ["route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10)
)
RAG_ERRORS = Counter(
    "openwebui_rag_errors_total",
    "RAG retrieval errors by route and status",
    ["route", "status"]
)
RAG_REQUESTS = Counter(
    "openwebui_rag_requests_total",
    "Total RAG retrieval requests by route",
    ["route"]
)

# Chat payload — embedding + retrieval + reranking inside process_chat_payload
CHAT_PAYLOAD_LATENCY = Histogram(
    "openwebui_chat_payload_processing_seconds",
    "Time in process_chat_payload: embedding + retrieval + reranking",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20)
)

# LLM — split into two distinct phases
LLM_FIRST_TOKEN = Histogram(
    "openwebui_llm_time_to_first_token_seconds",
    "Time until first token is generated",
    ["model"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20)
)
LLM_COMPLETION = Histogram(
    "openwebui_llm_completion_latency_seconds",
    "Total time for full LLM completion",
    ["model"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60)
)

# Tokens — split prompt vs completion
PROMPT_TOKENS = Counter(
    "openwebui_llm_prompt_tokens_total",
    "Prompt tokens consumed per model",
    ["model"]
)
COMPLETION_TOKENS = Counter(
    "openwebui_llm_completion_tokens_total",
    "Completion tokens generated per model",
    ["model"]
)

# Chat context
CHAT_CONTEXT_LENGTH = Histogram(
    "openwebui_chat_context_length",
    "Number of messages in context window per request",
    ["model"],
    buckets=(1, 2, 5, 10, 20, 50)
)

# ── Path filtering ────────────────────────────────────────────────────────────

EXCLUDED_PATHS = {
    "/metrics", "/health", "/manifest.json",
    "/opensearch.xml", "/favicon.ico"
}
EXCLUDED_PREFIXES = (
    "/static/", "/cache/", "/_app/",
    "/ws", "/favicon"
)

# ── Route normalization ───────────────────────────────────────────────────────

_UUID_RE = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.IGNORECASE
)

def normalize_route(request: Request) -> str:
    # Use FastAPI matched route template — prevents UUID cardinality explosion
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return route.path
    # Fallback: replace raw UUIDs with {id}
    return _UUID_RE.sub("{id}", request.url.path)

def should_exclude(path: str) -> bool:
    if path in EXCLUDED_PATHS:
        return True
    return any(path.startswith(p) for p in EXCLUDED_PREFIXES)

# ── Async timing context manager ──────────────────────────────────────────────

@asynccontextmanager
async def observe_latency(metric, **labels):
    start = time.perf_counter()
    try:
        yield
    finally:
        metric.labels(**labels).observe(time.perf_counter() - start)