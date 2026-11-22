/**
 * HTTP + WebSocket entrypoint.
 *
 *   GET  /api/personas       → [{id, name}] the clients a student can pick (AV2)
 *   POST /api/session/start  → {persona_id}: mint + start a LITE session, spin up the legs
 *   POST /api/session/stop   → tear one down
 *   GET  /healthz
 *   WS   /ws/:sessionId      → the browser leg: mic audio up; transcripts,
 *                              status, and errors down
 *
 * Metrics are NOT on this port: see metrics.ts (AVATAR_METRICS_PORT).
 *
 * In dev the web app runs on Vite (:5173) and proxies /api and /ws here, so
 * the browser sees one origin. In production this server also serves the built
 * frontend from web/dist.
 *
 * ⚠ No auth yet — local development only. AV3 gates /api/session/start on an
 * Open WebUI login and the websocket upgrade on a short-lived signed ticket.
 * Do not expose this before it lands. Task list: capstone_fall2026/AVATAR_TRACK.md §16.
 */

import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile, stat } from "node:fs/promises";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";
import { WebSocketServer } from "ws";
import type { ClientMessage } from "../../shared/messages";
import { config, missingConfig } from "./config";
import { startSession as startUpstream } from "./liveavatar";
import { metrics, startMetricsServer, type StartFailureReason } from "./metrics";
import { assemble, personaSource, PersonaError, type AssembledPersona } from "./personas";
import { addSession, getSession, liveCount, stopAll, stopSession } from "./registry";
import { Session } from "./session";

const WEB_DIST = fileURLToPath(new URL("../../web/dist", import.meta.url));
const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript",
  ".css": "text/css",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
  ".json": "application/json",
};

// Request bodies here are tiny ({persona_id} / {session_id}); anything bigger
// is not a legitimate client.
const MAX_BODY_BYTES = 16 * 1024;

// What a student sees when a start is refused. Plain words only: the real
// reason goes to the server log (AV5a — raw upstream errors can name
// providers, models, credit balances).
const STUDENT_MESSAGES = {
  disabled: "Practice sessions are turned off right now.",
  unavailable: "The practice client is unavailable right now. Please try again later.",
  choosePersona: "Please choose a practice client first.",
  personaNotFound: "That practice client was not found. Please choose another.",
} as const;

const server = createServer((req, res) => {
  void route(req, res).catch((err: unknown) => {
    console.error("[http] unhandled:", err);
    if (!res.headersSent) json(res, 500, { error: STUDENT_MESSAGES.unavailable });
  });
});

async function route(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const path = new URL(req.url ?? "/", "http://localhost").pathname;

  if (path === "/healthz") {
    res.writeHead(200).end("ok");
    return;
  }
  if (req.method === "GET" && path === "/api/personas") return handlePersonas(res);
  if (req.method === "POST" && path === "/api/session/start") return handleStart(req, res);
  if (req.method === "POST" && path === "/api/session/stop") return handleStop(req, res);
  return serveStatic(res, path);
}

async function handlePersonas(res: ServerResponse): Promise<void> {
  try {
    json(res, 200, { personas: await personaSource().list() });
  } catch (err) {
    console.error(`[http] persona list failed: ${errorText(err)}`);
    json(res, 503, { error: STUDENT_MESSAGES.unavailable });
  }
}

/** Refuse a start: count it, log the real reason, tell the student plain words. */
function refuse(
  res: ServerResponse,
  status: number,
  reason: StartFailureReason,
  studentMessage: string,
  detail?: string,
): void {
  metrics.startFailed(reason);
  console.warn(`[http] session start refused (${reason})${detail ? `: ${detail}` : ""}`);
  json(res, status, { error: studentMessage });
}

async function handleStart(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const requestStartedAt = Date.now();

  // AV5a kill switch — checked first, before anything that costs money.
  if (!config.limits.enabled) {
    return refuse(res, 503, "disabled", STUDENT_MESSAGES.disabled, "AVATAR_ENABLED=false");
  }
  const missing = missingConfig();
  if (missing.length > 0) {
    return refuse(
      res,
      503,
      "not_configured",
      STUDENT_MESSAGES.unavailable,
      `missing env vars: ${missing.join(", ")}`,
    );
  }

  // AV2 — which client to play.
  const body = (await readJson(req)) as { persona_id?: unknown };
  const personaId = typeof body.persona_id === "string" ? body.persona_id.trim() : "";
  if (!personaId || personaId.length > 200) {
    return refuse(res, 400, "bad_request", STUDENT_MESSAGES.choosePersona, "no persona_id");
  }
  let persona: AssembledPersona;
  try {
    const found = await personaSource().get(personaId);
    if (!found) {
      return refuse(
        res,
        404,
        "persona_not_found",
        STUDENT_MESSAGES.personaNotFound,
        `unknown persona "${personaId}"`,
      );
    }
    persona = assemble(found);
  } catch (err) {
    const kind = err instanceof PersonaError ? err.kind : "unavailable";
    return refuse(
      res,
      kind === "invalid" ? 500 : 503,
      kind === "invalid" ? "persona_invalid" : "persona_unavailable",
      STUDENT_MESSAGES.unavailable,
      errorText(err),
    );
  }

  try {
    const upstream = await startUpstream();
    const session = new Session(
      upstream.sessionId,
      upstream.wsUrl,
      {
        persona,
        maxSessionMs: config.limits.maxSessionMin * 60_000,
        requestStartedAt,
      },
      (id, reason) => void stopSession(id, reason),
    );
    addSession(session);
    session.start();
    metrics.sessionStarted();
    console.log(
      `[http] session started: ${upstream.sessionId} (persona ${persona.id}, ` +
        `~${persona.instructionTokens} instruction tokens, voice ${persona.voice})`,
    );
    json(res, 201, {
      session_id: upstream.sessionId,
      livekit_url: upstream.livekitUrl,
      livekit_client_token: upstream.livekitClientToken,
      // Relative on purpose: the browser dials the same origin it loaded from
      // (the Vite proxy in dev, this server in prod).
      ws_path: `/ws/${upstream.sessionId}`,
      persona: { id: persona.id, name: persona.name },
      max_session_seconds: Math.round(config.limits.maxSessionMin * 60),
    });
  } catch (err) {
    return refuse(res, 502, "upstream", STUDENT_MESSAGES.unavailable, errorText(err));
  }
}

async function handleStop(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = (await readJson(req)) as { session_id?: string };
  if (typeof body.session_id === "string") await stopSession(body.session_id, "client_stop");
  // Already-gone is success: the caller wanted it gone and it is gone.
  res.writeHead(204).end();
}

async function serveStatic(res: ServerResponse, pathname: string): Promise<void> {
  let rel: string;
  try {
    rel = decodeURIComponent(pathname);
  } catch {
    res.writeHead(400).end();
    return;
  }
  if (rel === "/" || rel === "") rel = "/index.html";
  const filePath = normalize(join(WEB_DIST, rel));
  if (!filePath.startsWith(normalize(WEB_DIST))) {
    res.writeHead(403).end();
    return;
  }
  try {
    if ((await stat(filePath)).isDirectory()) throw new Error("dir");
    res.writeHead(200, { "Content-Type": MIME[extname(filePath)] ?? "application/octet-stream" });
    res.end(await readFile(filePath));
  } catch {
    try {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(await readFile(join(WEB_DIST, "index.html")));
    } catch {
      res.writeHead(404).end("not found — in dev, use the Vite server (pnpm dev)");
    }
  }
}

// ── browser websocket ─────────────────────────────────────────────────────────

const wss = new WebSocketServer({ noServer: true, maxPayload: 1024 * 1024 });

server.on("upgrade", (req, socket, head) => {
  const match = /^\/ws\/([A-Za-z0-9_-]+)$/.exec(new URL(req.url ?? "", "http://x").pathname);
  const session = match?.[1] ? getSession(match[1]) : undefined;
  if (!session) {
    socket.write("HTTP/1.1 404 Not Found\r\n\r\n");
    socket.destroy();
    return;
  }
  wss.handleUpgrade(req, socket, head, (ws) => {
    // One browser per session. The loser of a race (second tab, double click)
    // closes without entering the teardown below, so it cannot stop the
    // winner's session.
    if (!session.tryAttachFrontend(ws)) {
      ws.close(1008, "already connected");
      return;
    }
    console.log(`[ws] browser attached: ${session.sessionId}`);

    ws.on("message", (raw) => {
      let msg: ClientMessage;
      try {
        msg = JSON.parse(raw.toString()) as ClientMessage;
      } catch {
        return;
      }
      if (msg.type === "mic_audio" && typeof msg.audio === "string" && msg.audio) {
        session.sendMicAudio(msg.audio);
      } else if (msg.type === "stop") {
        ws.close(1000);
      }
    });

    ws.on("close", () => {
      session.detachFrontend(ws);
      // The browser leaving ends the session — there is no reconnect path, and
      // a session nobody is watching should not keep billing.
      void stopSession(session.sessionId, "frontend_disconnect");
    });
  });
});

// ── boot ──────────────────────────────────────────────────────────────────────

server.listen(config.port, () => {
  console.log(`[server] listening on http://localhost:${config.port}`);
  console.log(
    `[server] sessions ${config.limits.enabled ? "ENABLED" : "DISABLED (AVATAR_ENABLED=false)"}, ` +
      `limit ${config.limits.maxSessionMin} min, personas from ${config.personas.source}` +
      (config.personas.source === "file"
        ? ` (${config.personas.dir})`
        : ` (models tagged "${config.personas.tag}")`),
  );
  const missing = missingConfig();
  if (missing.length > 0) {
    console.warn(`[server] ⚠ missing env vars: ${missing.join(", ")} — sessions will not start`);
    console.warn("[server]   copy .env.example to .env and fill it in");
  }
});
startMetricsServer(config.metricsPort, liveCount);

// Sessions live in this process; a shutdown is the end of them. Stop them
// upstream on the way out rather than leaving them to the billing clock.
for (const signal of ["SIGINT", "SIGTERM"] as const) {
  process.on(signal, () => {
    void stopAll().finally(() => process.exit(0));
  });
}

function errorText(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { "Content-Type": "application/json" });
  res.end(JSON.stringify(body));
}

function readJson(req: IncomingMessage): Promise<unknown> {
  return new Promise((resolve) => {
    let data = "";
    let tooBig = false;
    req.on("data", (chunk: Buffer) => {
      if (tooBig) return;
      data += chunk.toString();
      if (data.length > MAX_BODY_BYTES) {
        tooBig = true;
        data = "";
      }
    });
    req.on("end", () => {
      if (tooBig) return resolve({});
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch {
        resolve({});
      }
    });
    req.on("error", () => resolve({}));
  });
}
