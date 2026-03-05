import asyncio
import logging
import time

import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
from prometheus_client import (
    Counter, Gauge, Histogram,
    generate_latest, CONTENT_TYPE_LATEST,
)

log = logging.getLogger("exporter")
logging.basicConfig(level=logging.INFO)

OPENWEBUI_BASE = "http://open-webui:8080"

# Align with prometheus.yml scrape_interval: 15s
PROBE_INTERVAL = 15

PROBES = [
    ("/health", "GET"),
   
]

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


async def _probe_once() -> bool:
    """Run all probes and update metrics. Returns True if any probe succeeded."""
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
                # Only 2xx = healthy. 401/403/5xx all mean something is wrong.
                ok_any = ok_any or (200 <= r.status_code < 300)
            except Exception as exc:
                status = "exception"
                log.warning("Probe %s failed: %s", path, exc)
            finally:
                dur = time.perf_counter() - start
                REQ_LAT.labels(endpoint=endpoint, status=status).observe(dur)
                REQ_CNT.labels(endpoint=endpoint, status=status).inc()
    UP.set(1 if ok_any else 0)
    return ok_any


async def _probe_loop():
    """Background loop — keeps metrics fresh between Prometheus scrapes."""
    while True:
        try:
            ok = await _probe_once()
            log.info("Probe result: ok=%s", ok)
        except Exception as exc:
            # Log but never let the loop die silently
            log.error("Unexpected probe loop error: %s", exc)
        await asyncio.sleep(PROBE_INTERVAL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Replaces deprecated @app.on_event("startup")
    task = asyncio.create_task(_probe_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/probe")
async def probe():
    """On-demand probe — useful for manual health checks."""
    ok = await _probe_once()
    return {"ok": ok}