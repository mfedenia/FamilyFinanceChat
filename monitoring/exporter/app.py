import time
import httpx
from fastapi import FastAPI, Response
from prometheus_client import Histogram, Counter, Gauge, generate_latest, CONTENT_TYPE_LATEST

OPENWEBUI_BASE = "http://open-webui:8080"

app = FastAPI()

REQ_LAT = Histogram(
    "openwebui_probe_latency_seconds",
    "Latency probing OpenWebUI endpoints (sidecar)",
    ["endpoint", "status"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40),
)
REQ_CNT = Counter(
    "openwebui_probe_requests_total",
    "Probe count to OpenWebUI endpoints (sidecar)",
    ["endpoint", "status"],
)
UP = Gauge("openwebui_up", "Whether OpenWebUI is reachable (1=yes/0=no)")

PROBES = [
    ("/health", "GET"),
    ("/api/v1/chats/?page=1", "GET"),
]

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/probe")
async def probe():
    ok_any = False
    timeout = httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for path, method in PROBES:
            endpoint = path.split("?")[0]
            start = time.perf_counter()
            status = "error"
            try:
                r = await client.request(method, f"{OPENWEBUI_BASE}{path}")
                status = str(r.status_code)
                ok_any = ok_any or (200 <= r.status_code < 500)
            except Exception:
                status = "exception"
            dur = time.perf_counter() - start
            REQ_LAT.labels(endpoint=endpoint, status=status).observe(dur)
            REQ_CNT.labels(endpoint=endpoint, status=status).inc()
    UP.set(1 if ok_any else 0)
    return {"ok": ok_any}

@app.on_event("startup")
async def start_loop():
    import asyncio
    async def loop():
        while True:
            try:
                await probe()
            except Exception:
                pass
            await asyncio.sleep(10)
    asyncio.create_task(loop())
