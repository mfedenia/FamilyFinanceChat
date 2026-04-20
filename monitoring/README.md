# Monitoring & Chat Metrics

This directory contains the tools for monitoring and observability of the FamilyFinanceChat OpenWebUI instance.

## Metrics Overview
We use a **Decoupled Monitoring Architecture**. Instead of modifying the OpenWebUI source code, we use a **Filter Function** to capture metrics and push them to a **Prometheus Pushgateway**.

1. **Filter Function**: Captures chat timing and token usage.
2. **Pushgateway**: Receives metrics via HTTP POST.
3. **Prometheus**: Scrapes the Pushgateway.
4. **Grafana**: Visualizes the metrics.

---

## How to Install the Chat Metrics Filter

Whenever you deploy a new instance or need to restore metrics, follow these steps:

### 1. Copy the Filter Code
Locate the code in [monitoring/chat_metrics_filter.py](chat_metrics_filter.py) and copy the entire file.

### 2. Add to OpenWebUI
1. Log in as an **Admin**.
2. Navigate to **Workspace** > **Functions**.
3. Click the **+ (Plus)** button to create a new function.
4. **Name**: `Chat Metrics`.
5. **Code**: Paste the code you copied.
6. Click **Save** (bottom right).

### 3. Enable Globally
1. In the Functions list, find your new "Chat Metrics" filter.
2. Toggle the **Global** switch to **ON** (green). This ensures metrics are collected for all models and all users.

### 4. Configure Valves (Targets)
1. Click the **Gear (Settings)** icon next to the Chat Metrics function.
2. Look for the **Valves** section.
3. Set **`pushgateway_url`**:
   - For **Production**: `http://pushgateway:9091`
   - For **Test Stack**: `http://pushgateway-test:9091`
4. Click **Save**.

---

## Troubleshooting Metrics

If metrics aren't appearing in Prometheus/Grafana:

### Check the Logs
Run this on the VM to see if the filter is outputting debug prints or errors:
```bash
sudo docker logs -f open-webui
```

### Manual Metric Check
Verify if the Pushgateway is receiving data by running this on the VM:
```bash
curl -s http://localhost:9091/metrics | grep openwebui_chat
```
(Use port `9093` for the Test Stack).

### Verify Connectivity
Ensure the OpenWebUI container can reach the Pushgateway:
```bash
sudo docker exec open-webui curl -s http://pushgateway:9091/metrics
```
