# Pushgateway metrics pipeline (Filter Functions → Prometheus → Grafana)

This document explains how the pieces fit together, how to run and verify them, how to push from an Open WebUI Filter Function, and how chat metrics map from the **old** (forked `main.py` / `observability.py`) path to the **new** path.

---

## 1. How it works (simple flow)

1. **Open WebUI** runs a **Filter Function** (Python you install in the admin UI). On each chat request/response it measures time, counts tokens, etc.

2. The Filter Function **does not** need to expose `/metrics` on the WebUI process. Instead it **HTTP POSTs** text in Prometheus format to **Pushgateway**.

3. **Prometheus** is configured to **scrape** Pushgateway on a schedule (every 15s in this repo). It **pulls** whatever was last pushed and stores it in its time-series database.

4. **Grafana** queries Prometheus and draws panels. You either keep the **same metric names** as before (easiest) or update panel queries using the mapping table below.

```
Chat → Filter Function → POST /metrics/job/... → Pushgateway
                                                    ↑
Prometheus ───────── scrape /metrics ──────────────┘
     ↑
Grafana ──── PromQL queries
```

**Why Pushgateway?** Filter code often runs inside the app without a good place to run a scrape-only HTTP metrics server. Pushgateway is the standard bridge: **short-lived or embedded producers push; Prometheus still pulls.**

---

## 2. What we added in this repo

| File | Change |
|------|--------|
| `docker-compose.yml` | Service `pushgateway` on `ai-net`, host port **9091** → container **9091**. |
| `docker-compose.staging.yml` | Service `pushgateway-staging` on `ai-net-staging` + external `ai-net`, host port **`${STAGING_PUSHGATEWAY_PORT:-9092}`**. |
| `prometheus.yml` | Jobs `pushgateway` and `pushgateway-staging` with **`honor_labels: true`**. |

`honor_labels: true` matters so labels you send with the push (for example `job`, `instance`) are **not** overwritten by Prometheus’ scrape job name.

---

## 3. Run and verify (deliverable: end-to-end)

### Production stack (`docker-compose.yml`)

1. Reload config and start Pushgateway:

   ```bash
   docker compose up -d pushgateway
   docker compose restart prometheus
   ```

2. Confirm Pushgateway UI: open `http://<host>:9091` — you should see the status page.

3. **Fake a push** (no Filter Function required yet):

   ```bash
   echo 'owui_filter_test_metric 1' | curl --data-binary @- \
     http://localhost:9091/metrics/job/openwebui_filter/instance/test
   ```

4. Confirm Prometheus sees it: **Status → Targets** — `pushgateway` should be **UP**.  
   **Graph** → query: `owui_filter_test_metric` — should return `1`.

5. **Grafana** (this repo: port **3001** → container 3000): add Prometheus data source `http://prometheus:9090` if not already set, then a stat panel with query `owui_filter_test_metric`.

### Staging

1. Ensure network `ai-net` exists (e.g. production compose has been brought up once, or create the external network).

2. Start staging Pushgateway:

   ```bash
   docker compose -f docker-compose.staging.yml up -d pushgateway-staging
   ```

3. Push to staging (example host port **9092**):

   ```bash
   echo 'owui_filter_test_metric 1' | curl --data-binary @- \
     "http://localhost:${STAGING_PUSHGATEWAY_PORT:-9092}/metrics/job/openwebui_filter/instance/staging_test"
   ```

4. Prometheus (on `ai-net`) scrapes `pushgateway-staging:9091` — query with a label if you duplicate metrics: e.g. use different `instance` or `environment` in the pushed labels (see §5).

**Note:** If `pushgateway-staging` is not running, the `pushgateway-staging` target in Prometheus will show **DOWN** until you start it. That is expected.

---

## 4. What your Filter Function should POST

Pushgateway expects **Prometheus text exposition format** (lines like `metric_name{label="a"} value`).

Typical URL shape:

```http
POST http://pushgateway:9091/metrics/job/<JOB_NAME>/instance/<INSTANCE_ID>
```

From **inside** the **production** Open WebUI container (`open-webui` on `ai-net`), POST to **`http://pushgateway:9091`**. From **inside** **`open-webui-staging`**, POST to **`http://pushgateway-staging:9091`** (that container is on the same networks as staging WebUI). From the **host** machine, use **`http://localhost:9091`** (prod) or **`http://localhost:${STAGING_PUSHGATEWAY_PORT:-9092}`** (staging).

Use a **stable** `instance` per deployment (e.g. `openwebui-prod-1`) so new pushes **replace** the same group instead of creating unbounded cardinality.

**Do not** put unbounded values in labels (no raw `user_id`, `chat_id`, or full URLs). Keep the same cardinality rules as normal Prometheus metrics.

---

## 5. Old metric name → new metric name mapping

Today, chat- and RAG-related metrics are defined in `custom-code/observability.py` and observed from forked `main.py`. **Recommended approach:** in the Filter Function, expose **the same metric names and label names** where possible. Then Grafana panels only need to change the **metric source** (query `job="pushgateway"` or combine with `or` for migration).

If you **must** rename (e.g. to mark filter-only semantics), use this mapping as the contract.

| Old metric (from `observability.py`) | Labels | Suggested new / pushed equivalent |
|-------------------------------------|--------|-----------------------------------|
| `openwebui_stage_latency_seconds` | `stage` | Same name; push histogram buckets or use **summary**-style gauges updated per request (see note below) |
| `openwebui_embedding_latency_seconds` | `model` | Same |
| `openwebui_qdrant_search_latency_seconds` | (none) | Same |
| `openwebui_reranker_latency_seconds` | (none) | Same |
| `openwebui_openai_queue_latency_seconds` | `model` | Same |
| `openwebui_context_tokens_total` | `model` | Same (histogram) |
| `openwebui_chat_payload_processing_seconds` | `model` | Same |
| `openwebui_llm_time_to_first_token_seconds` | `model` | Same |
| `openwebui_llm_completion_latency_seconds` | `model` | Same |
| `openwebui_llm_prompt_tokens_total` | `model` | Same (counter) |
| `openwebui_llm_completion_tokens_total` | `model` | Same (counter) |
| `openwebui_chat_context_length` | `model` | Same |
| `openwebui_api_latency_seconds` | `method`, `route`, `status` | **Not** from Filter Function — still scraped from Open WebUI `/metrics` if you keep HTTP middleware; otherwise omit or push from a separate hook if you add one |
| `openwebui_api_errors_total` | `method`, `route`, `status` | Same as above |
| `openwebui_requests_in_flight` | — | Gauge; optional push or keep from app scrape |
| `openwebui_rag_request_latency_seconds` | `route` | RAG **HTTP** routes only; Filter may not see these unless you instrument inside retrieval — keep scrape from `open-webui` or push from custom code path |
| `openwebui_rag_errors_total` | `route`, `status` | Same |
| `openwebui_rag_requests_total` | `route` | Same |

**Note on histograms from a Filter Function:** Native Prometheus client histograms **increment `_bucket`, `_sum`, `_count`**. Pushgateway accepts those lines. If pushing from a minimal environment is awkward, common patterns are: (a) push **pre-aggregated** metrics with a small set of labels, or (b) push a **summary** implemented as a few gauges (e.g. last latency / moving average) — document any deviation from the table above in your Filter Function README.

---

## 6. Grafana: update or duplicate panels

1. **Identify** panels that use metrics from **forked** chat instrumentation (`openwebui_stage_latency_seconds`, `openwebui_llm_*`, `openwebui_chat_payload_processing_seconds`, etc.).

2. **During migration**, you can duplicate a panel and set query to the **pushgateway** job, for example:

   ```promql
   histogram_quantile(0.95, sum by (le, stage) (rate(openwebui_stage_latency_seconds_bucket{job="pushgateway"}[5m])))
   ```

   After cutover, either remove the old `job="openwebui"` series or use:

   ```promql
   histogram_quantile(0.95, sum by (le, stage) (rate(openwebui_stage_latency_seconds_bucket[5m])))
   ```

   if only one source remains.

3. If you use **Recording rules** or **dashboard variables** for `job`, add `pushgateway` (and `pushgateway-staging` if you split by environment).

4. **Staging:** pushed metrics often use a distinct `instance` or custom label `environment="staging"` — filter panels with that label so prod and staging do not mix.

---

## 7. Deliverable checklist

| Item | Check |
|------|--------|
| Pushgateway running | `docker ps` shows `pushgateway` (and optionally `pushgateway-staging`) |
| Prometheus scraping | Targets UI: `pushgateway` UP; manual push visible in Graph |
| Grafana | Panel shows pushed test metric or real Filter metrics |
| Mapping | This file §5; adjust if Filter Function renames metrics |

---

## 8. References

- [Pushgateway README](https://github.com/prometheus/pushgateway/blob/master/README.md) — when to use push, label hygiene, `PUT` vs `POST`.
- Open WebUI docs for **Filter Functions** (install path, Python entrypoints, request/response hooks).
