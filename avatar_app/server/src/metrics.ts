/**
 * AV5a — Prometheus metrics for the bridge.
 *
 * Served on a SEPARATE port (AVATAR_METRICS_PORT, default 9464) by its own
 * tiny HTTP server, never on the main port: Caddy routes /practice/* to the
 * main port, so metrics are internal by construction rather than by a proxy
 * rule someone could forget. Prometheus scrapes `avatar-app:9464` over ai-net
 * (wired up in AV6).
 *
 * Hand-written exposition format instead of prom-client: five metrics do not
 * justify a dependency, and the upstream code is deliberately dependency-light.
 */

import { createServer, type Server } from "node:http";

type Labels = Record<string, string>;

const labelKey = (labels: Labels): string =>
  Object.entries(labels)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([k, v]) => `${k}="${v.replace(/\\/g, "\\\\").replace(/"/g, '\\"').replace(/\n/g, "\\n")}"`)
    .join(",");

class Counter {
  private readonly values = new Map<string, number>();
  constructor(
    readonly name: string,
    readonly help: string,
  ) {}
  inc(labels: Labels = {}, by = 1): void {
    const key = labelKey(labels);
    this.values.set(key, (this.values.get(key) ?? 0) + by);
  }
  render(): string {
    const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} counter`];
    if (this.values.size === 0) lines.push(`${this.name} 0`);
    for (const [key, value] of this.values) {
      lines.push(`${this.name}${key ? `{${key}}` : ""} ${value}`);
    }
    return lines.join("\n");
  }
}

class Histogram {
  private readonly counts: number[];
  private sum = 0;
  private count = 0;
  constructor(
    readonly name: string,
    readonly help: string,
    private readonly buckets: number[],
  ) {
    this.counts = buckets.map(() => 0);
  }
  observe(value: number): void {
    this.sum += value;
    this.count += 1;
    this.buckets.forEach((le, i) => {
      if (value <= le) this.counts[i] = (this.counts[i] ?? 0) + 1;
    });
  }
  render(): string {
    const lines = [`# HELP ${this.name} ${this.help}`, `# TYPE ${this.name} histogram`];
    this.buckets.forEach((le, i) => lines.push(`${this.name}_bucket{le="${le}"} ${this.counts[i] ?? 0}`));
    lines.push(`${this.name}_bucket{le="+Inf"} ${this.count}`);
    lines.push(`${this.name}_sum ${this.sum}`, `${this.name}_count ${this.count}`);
    return lines.join("\n");
  }
}

const started = new Counter(
  "avatar_sessions_started_total",
  "Avatar sessions started (billing begins at start).",
);
const startFailures = new Counter(
  "avatar_session_start_failures_total",
  "Session starts refused or failed, by reason.",
);
const ended = new Counter("avatar_sessions_ended_total", "Avatar sessions ended, by reason.");
const sessionSeconds = new Counter(
  "avatar_session_seconds_total",
  "Total billed session time in seconds (cost ~ this x per-second price).",
);
const readySeconds = new Histogram(
  "avatar_session_ready_seconds",
  "Time from the student pressing start until the voice model is ready.",
  [1, 2, 3, 5, 8, 13, 20, 30],
);

export type StartFailureReason =
  | "disabled"
  | "not_configured"
  | "bad_request"
  | "persona_not_found"
  | "persona_invalid"
  | "persona_unavailable"
  | "upstream";

export const metrics = {
  sessionStarted(): void {
    started.inc();
  },
  startFailed(reason: StartFailureReason): void {
    startFailures.inc({ reason });
  },
  sessionEnded(reason: string, durationSec: number): void {
    ended.inc({ reason });
    sessionSeconds.inc({}, Math.max(0, durationSec));
  },
  sessionReady(secondsSinceRequest: number): void {
    readySeconds.observe(secondsSinceRequest);
  },
};

/** The full exposition, with the live-session gauge read at scrape time. */
export function renderMetrics(activeSessions: number): string {
  return (
    [
      "# HELP avatar_sessions_active Avatar sessions currently running (each one is billing).",
      "# TYPE avatar_sessions_active gauge",
      `avatar_sessions_active ${activeSessions}`,
      started.render(),
      startFailures.render(),
      ended.render(),
      sessionSeconds.render(),
      readySeconds.render(),
    ].join("\n") + "\n"
  );
}

export function startMetricsServer(port: number, activeSessions: () => number): Server | null {
  if (!port) return null;
  const server = createServer((req, res) => {
    if (new URL(req.url ?? "/", "http://x").pathname !== "/metrics") {
      res.writeHead(404).end();
      return;
    }
    res.writeHead(200, { "Content-Type": "text/plain; version=0.0.4; charset=utf-8" });
    res.end(renderMetrics(activeSessions()));
  });
  server.on("error", (err) => console.warn(`[metrics] server error: ${err.message}`));
  server.listen(port, () => console.log(`[metrics] Prometheus metrics on :${port}/metrics`));
  return server;
}
