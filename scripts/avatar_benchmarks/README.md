# B4 — AI Conversation Latency Benchmark

## Overview

This benchmark is part of B4 (Stage Latency / Grafana) for the FamilyFinanceChat capstone project.

The goal is to measure AI response latency and provide reproducible performance data for evaluating different models and the future voice/avatar conversation pipeline.

The current implementation focuses on **text-based AI responses** through the Open WebUI API. Audio, avatar, Qdrant retrieval, and Grafana integration will be completed later.

## 1. Setup

### Requirements

- Python 3
- Access to the FamilyFinanceChat Open WebUI website
- Open WebUI API key
- `requests` and `python-dotenv`

Install dependencies:

```bash
pip install requests python-dotenv
```

Configure the repository's root `.env` file:

```dotenv
OPENWEBUI_URL=https://your-openwebui-url
OPENWEBUI_API_KEY=your-api-key
OPENWEBUI_MODEL=your-model-id
```

The API key must be kept private. Do not commit `.env` to GitHub.

### Run the benchmark

From the repository root:

```bash
python scripts/avatar_benchmarks/benchmark.py
```

The script sends five questions to the selected family model and records response latency.

## 2. Testing Method

The benchmark sends these five questions:

1. What are your family's main financial goals?
2. What is your family's monthly income?
3. What are your biggest monthly expenses?
4. Do you currently have any debts?
5. How much do you save every month?

Each question is sent as an independent request through the Open WebUI `/api/chat/completions` endpoint with streaming enabled.

The script measures:

- **Time to First Text (TTFT):** Time from sending the API request until the first nonempty text chunk is received.
- **Total Response Time:** Time until the HTTP response finishes.
- **Streaming Duration:** Time between the first and last received text chunks.
- **p50:** Median first-text latency.
- **p95:** Estimated 95th-percentile first-text latency.

Results are exported to `b4_results.csv`.

These measurements include client-to-server network latency and Open WebUI processing. They are not isolated model inference measurements.

## 3. Benchmark Results

Two preliminary benchmark runs were completed in November 2026.

### Run 1 — Initial Model

Underlying model observed in the API response: `gpt-5-mini-2025-08-07`

| Question | First Text (s) | Total (s) |
|---|---:|---:|
| Q1 | 4.232 | 4.234 |
| Q2 | 6.181 | 6.182 |
| Q3 | 4.444 | 4.445 |
| Q4 | 6.247 | 6.252 |
| Q5 | 7.267 | 7.269 |

**Results:**
- p50: 6.181 seconds
- p95: 7.063 seconds
- Streaming duration: approximately 0–0.001 seconds

### Run 2 — New Model Configuration

The second run used the updated model configuration selected for testing. The underlying model ID still needs to be verified from the API response before labeling this run as GPT-5.5.

| Question | First Text (s) | Total (s) |
|---|---:|---:|
| Q1 | 4.000 | 4.005 |
| Q2 | 2.117 | 2.120 |
| Q3 | 2.577 | 2.579 |
| Q4 | 1.735 | 1.738 |
| Q5 | 1.820 | 1.821 |

**Results:**
- p50: 2.117 seconds
- p95: 3.716 seconds
- Streaming duration: approximately 0–0.002 seconds

### Comparison

| Measurement | Run 1 | Run 2 |
|---|---:|---:|
| p50 first text | 6.181s | 2.117s |
| p95 first text | 7.063s | 3.716s |
| Requests tested | 5 | 5 |

The second run showed approximately 66% lower median first-text latency than the first run.

These results are preliminary. Five requests are not sufficient for a reliable p95 estimate, and differences may also reflect network conditions, server load, caching, or model configuration.

## 4. Streaming Observations

Although the API returns multiple streaming chunks, almost all text chunks arrive at the Python client within 0–2 milliseconds.

Possible explanations include:

- HTTP client buffering
- Reverse-proxy buffering
- Open WebUI response buffering
- Model-side response generation behavior

The precise cause has not been confirmed. Further testing should use smaller HTTP read chunks and, when available, server-side timing measurements.

## 5. Current Progress

Completed:

- Open WebUI API authentication
- Automated text-based questions
- First-text latency measurement
- Total response-time measurement
- Streaming-duration measurement
- p50 and p95 calculations
- CSV results export
- Two preliminary benchmark runs

## 6. Remaining B4 Work

The following tasks will be completed when the required infrastructure and voice/avatar components are available.

### Qdrant Retrieval

- Confirm whether the model's API requests trigger Qdrant retrieval.
- Measure retrieval duration using server-side instrumentation.
- Compare retrieval latency across test questions.

### Audio and Avatar Latency

Coordinate with the voice and avatar team members to measure:

- Speech endpoint detection
- Speech-to-text latency
- Model response latency
- Text-to-speech latency
- Time to first audible speech
- Avatar rendering latency
- Interruption response time

### Prometheus and Grafana

Reuse the existing monitoring infrastructure in the repository:

- `monitoring/chat_metrics_filter.py`
- Prometheus
- Pushgateway, if enabled
- Grafana dashboards

Add stage-level latency measurements and ensure individual observations are preserved for reliable p50/p95 calculations.

### Final Validation

- Benchmark the final selected model.
- Increase the number of test samples.
- Run a complete spoken conversation.
- Validate latency measurements in Grafana.
- Compare performance against the project targets.
- Document findings and limitations.
- Submit the completed B4 changes through a reviewed pull request.

## 7. Current Status

**B4 Status: Partially Complete — Text Baseline Implemented**

The initial text benchmarking infrastructure is working and reusable for other models.

The full B4 task remains open until voice/avatar latency, retrieval timing, and Grafana integration have been completed and verified.

**Last updated:** November 17, 2026.