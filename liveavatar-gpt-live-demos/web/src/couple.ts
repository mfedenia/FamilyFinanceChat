/**
 * The couple meeting page: two avatars (two LiveKit rooms), one mic, one
 * session websocket. The server decides who speaks (the "floor"); this page
 * shows it, plays both avatars, and streams the mic.
 */
import type { Speaker, Turn } from "../../shared/messages";
import { joinAvatarRoom, type AvatarRoom } from "./livekitRoom";
import { startMicCapture, type MicCapture } from "./micCapture";
import { openSessionSocket, type SessionSocket } from "./socket";

const WHO: Speaker[] = ["tony", "antonella"];
const NAMES: Record<Speaker, string> = { tony: "Tony", antonella: "Antonella" };

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const startBtn = $<HTMLButtonElement>("start");
const stopBtn = $<HTMLButtonElement>("stop");
const micBtn = $<HTMLButtonElement>("mic");
const statusEl = $<HTMLSpanElement>("status");
const transcriptEl = $<HTMLDivElement>("transcript");
const tiles = Object.fromEntries(
  WHO.map((w) => [
    w,
    {
      tile: $<HTMLDivElement>(`tile-${w}`),
      video: $<HTMLVideoElement>(`video-${w}`),
      audio: $<HTMLAudioElement>(`audio-${w}`),
    },
  ]),
) as Record<Speaker, { tile: HTMLDivElement; video: HTMLVideoElement; audio: HTMLAudioElement }>;

type StatusKind = "idle" | "busy" | "live" | "error";
function setStatus(msg: string, kind: StatusKind = "idle"): void {
  statusEl.textContent = msg;
  statusEl.dataset.kind = kind;
}

// ── transcript ──────────────────────────────────────────────────────────────

const lines = new Map<string, HTMLSpanElement>();
let lastSpouseLine: HTMLDivElement | null = null;
/** Each spouse's latest line, so a cut-off marks the right person. */
const lastLineOf: Partial<Record<Speaker, HTMLDivElement>> = {};

function upsertTurn(turn: Turn): void {
  let text = lines.get(turn.id);
  if (!text) {
    const line = document.createElement("div");
    const who = turn.role === "user" ? "you" : (turn.speaker ?? "tony");
    line.className = `turn ${who}`;
    const label = document.createElement("span");
    label.className = "who";
    label.textContent = who === "you" ? "You" : NAMES[who as Speaker];
    text = document.createElement("span");
    text.className = "text";
    line.append(label, text);
    transcriptEl.appendChild(line);
    lines.set(turn.id, text);
    if (turn.role === "assistant") {
      lastSpouseLine = line;
      if (turn.speaker) lastLineOf[turn.speaker] = line;
    }
  }
  text.textContent = turn.text;
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// ── floor + speaking indicators ─────────────────────────────────────────────

const decisionEl = $<HTMLDivElement>("decision");

/** The server's reason for a floor change, in plain words. */
function describeWhy(why: string | undefined, speaker: Speaker): string {
  if (!why) return "";
  if (why === "named") return "a name was said";
  if (why === "greeting") return "opens the meeting";
  if (why === "chime-in") return "adds a view (chime-in)";
  if (why === "handoff") return `${NAMES[speaker === "tony" ? "antonella" : "tony"]} asked them directly`;
  if (why === "backchannel") return "keeps talking (you said a short acknowledgement)";
  if (why === "continuing") return "was still talking";
  if (why.startsWith("decision too slow")) return "answered first (decision too slow)";
  const m = /^(\S+) in (\d+)ms/.exec(why);
  if (m) return `decision model, ${m[2]} ms${why.includes("chime-in") ? ", the other may chime in" : ""}`;
  return why;
}

function showFloor(speaker: Speaker | null, why?: string): void {
  for (const w of WHO) {
    if (w === speaker) tiles[w].tile.dataset.floor = "";
    else delete tiles[w].tile.dataset.floor;
  }
  if (!speaker) {
    decisionEl.textContent = "";
    return;
  }
  const reason = describeWhy(why, speaker);
  const who = document.createElement("b");
  who.textContent = `→ ${NAMES[speaker]}`;
  decisionEl.replaceChildren(who, reason ? ` · ${reason}` : "");
}

// A speaking glow per tile, measured off each avatar's LiveKit audio.
const meters: Partial<Record<Speaker, { ctx: AudioContext; analyser: AnalyserNode; buf: Uint8Array<ArrayBuffer>; quietSince: number; stream: MediaStream }>> = {};
let raf = 0;

for (const w of WHO) {
  tiles[w].audio.addEventListener("playing", () => {
    const stream = tiles[w].audio.srcObject;
    if (!(stream instanceof MediaStream) || meters[w]?.stream === stream) return;
    void meters[w]?.ctx.close();
    const ctx = new AudioContext();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    ctx.createMediaStreamSource(stream).connect(analyser);
    meters[w] = { ctx, analyser, buf: new Uint8Array(analyser.fftSize), quietSince: 0, stream };
  });
  tiles[w].video.addEventListener("playing", () => {
    tiles[w].tile.dataset.live = "";
  });
}

function tick(now: number): void {
  for (const w of WHO) {
    const m = meters[w];
    if (!m) continue;
    m.analyser.getByteTimeDomainData(m.buf);
    let sum = 0;
    for (let i = 0; i < m.buf.length; i++) {
      const v = ((m.buf[i] ?? 128) - 128) / 128;
      sum += v * v;
    }
    if (Math.sqrt(sum / m.buf.length) > 0.03) {
      m.quietSince = 0;
      tiles[w].tile.dataset.speaking = "";
    } else {
      m.quietSince ||= now;
      if (now - m.quietSince > 500) delete tiles[w].tile.dataset.speaking;
    }
  }
  raf = requestAnimationFrame(tick);
}

// ── session lifecycle ───────────────────────────────────────────────────────

interface StartResponse {
  session_id: string;
  ws_path: string;
  avatars: Record<Speaker, { name: string; livekit_url: string; livekit_client_token: string }>;
  error?: string;
}

let sessionId: string | null = null;
let socket: SessionSocket | null = null;
let rooms: AvatarRoom[] = [];
let mic: MicCapture | null = null;
let micMuted = false;
let generation = 0;
let stopping = false;

async function start(): Promise<void> {
  const myGeneration = ++generation;
  const superseded = () => generation !== myGeneration;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus("starting both avatars…", "busy");
  raf ||= requestAnimationFrame(tick);

  let started: StartResponse;
  try {
    const response = await fetch("/api/couple/start", { method: "POST" });
    started = (await response.json()) as StartResponse;
    if (!response.ok) throw new Error(started.error ?? "could not start the meeting");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : "could not start the meeting", "error");
    startBtn.disabled = false;
    stopBtn.disabled = true;
    return;
  }
  if (superseded()) {
    void endOnServer(started.session_id);
    return;
  }
  sessionId = started.session_id;

  setStatus("connecting to Tony and Antonella…", "busy");
  try {
    rooms = await Promise.all(
      WHO.map((w) =>
        joinAvatarRoom(started.avatars[w].livekit_url, started.avatars[w].livekit_client_token, {
          video: tiles[w].video,
          audio: tiles[w].audio,
        }),
      ),
    );
  } catch (err) {
    await stop(err instanceof Error ? err.message : "could not reach the avatars", "error");
    return;
  }
  if (superseded()) return;

  socket = openSessionSocket(started.ws_path, {
    onReady: () => {
      if (superseded()) return;
      setStatus("live — Tony will say hello, then just talk", "live");
      void startMicCapture((b64) => socket?.sendMicAudio(b64))
        .then((capture) => {
          if (superseded()) {
            capture.stop();
            return;
          }
          mic = capture;
          micBtn.disabled = false;
        })
        .catch((err: unknown) => {
          setStatus(
            err instanceof Error ? `microphone unavailable: ${err.message}` : "microphone unavailable",
            "error",
          );
        });
    },
    onTurn: upsertTurn,
    onUi: () => {},
    onFloor: showFloor,
    onInterrupted: (speaker) => (speaker ? lastLineOf[speaker] : lastSpouseLine)?.classList.add("interrupted"),
    onError: (message) => setStatus(message, "error"),
    onClose: () => {
      if (!stopping) void stop("the meeting ended");
    },
  });
}

async function stop(reason?: string, kind: StatusKind = "idle"): Promise<void> {
  generation += 1;
  if (stopping) return;
  stopping = true;
  stopBtn.disabled = true;
  micBtn.disabled = true;

  mic?.stop();
  mic = null;
  setMuted(false);
  socket?.close();
  socket = null;
  await Promise.all(rooms.map((r) => r.disconnect().catch(() => {})));
  rooms = [];

  cancelAnimationFrame(raf);
  raf = 0;
  for (const w of WHO) {
    void meters[w]?.ctx.close();
    delete meters[w];
    tiles[w].video.srcObject = null;
    delete tiles[w].tile.dataset.live;
    delete tiles[w].tile.dataset.speaking;
  }
  showFloor(null);

  const id = sessionId;
  sessionId = null;
  if (id) void endOnServer(id);

  setStatus(reason ?? "meeting ended", kind);
  startBtn.disabled = false;
  stopping = false;
}

function setMuted(muted: boolean): void {
  micMuted = muted;
  mic?.setMuted(muted);
  micBtn.setAttribute("aria-pressed", String(muted));
  micBtn.textContent = muted ? "Unmute mic" : "Mute mic";
}

async function endOnServer(id: string): Promise<void> {
  try {
    await fetch("/api/session/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: id }),
    });
  } catch {
    // The server's idle watchdog reaps it anyway.
  }
}

window.addEventListener("pagehide", () => {
  if (!sessionId) return;
  navigator.sendBeacon?.(
    "/api/session/stop",
    new Blob([JSON.stringify({ session_id: sessionId })], { type: "application/json" }),
  );
});

startBtn.addEventListener("click", () => {
  transcriptEl.textContent = "";
  for (const w of WHO) delete lastLineOf[w];
  lines.clear();
  void start();
});
stopBtn.addEventListener("click", () => void stop());
micBtn.addEventListener("click", () => setMuted(!micMuted));
