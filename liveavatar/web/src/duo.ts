/**
 * The duo page: two avatars (two LiveKit rooms), one mic, one transcript.
 *
 * The browser does nothing clever — the server's director decides who speaks
 * (`floor`), and each avatar's voice arrives over its own room. This page
 * only shows it: who holds the floor, who is audible, how the decision split.
 */

import type {
  DuoPersonaInfo,
  DuoServerMessage,
  DuoStartResponse,
  ClientMessage,
  Turn,
} from "../../shared/messages";
import { toOpenWebUIExport, type SpokenSession, type SpokenTurn } from "../../shared/openwebui";
import { joinAvatarRoom, type AvatarRoom } from "./livekitRoom";
import { startMicCapture, type MicCapture } from "./micCapture";

const $ = <T extends Element>(sel: string, root: ParentNode = document) => root.querySelector(sel) as T;

const layout = $<HTMLDivElement>("#layout");
const startBtn = $<HTMLButtonElement>("#start");
const stopBtn = $<HTMLButtonElement>("#stop");
const micBtn = $<HTMLButtonElement>("#mic");
const exportBtn = $<HTMLButtonElement>("#export");
const statusEl = $<HTMLDivElement>("#status");
const transcriptEl = $<HTMLDivElement>("#transcript");
const decisionBar = { a: $<HTMLSpanElement>("#decision .a"), b: $<HTMLSpanElement>("#decision .b") };
const decisionWhy = $<HTMLSpanElement>("#decision .why");
const titleEl = $<HTMLHeadingElement>("header h1");
const subEl = $<HTMLSpanElement>("header .sub");
/** What the human is called in the transcript ("You (Adviser)"). */
let userLabel = "You";

interface Seat {
  el: HTMLDivElement;
  video: HTMLVideoElement;
  audio: HTMLAudioElement;
  persona: DuoPersonaInfo | null;
  room: AvatarRoom | null;
  analyser: AnalyserNode | null;
  buf: Uint8Array<ArrayBuffer> | null;
  quietSince: number;
}

const seats: Seat[] = [...document.querySelectorAll<HTMLDivElement>(".seat")].map((el) => ({
  el,
  video: $<HTMLVideoElement>("video", el),
  audio: $<HTMLAudioElement>("audio", el),
  persona: null,
  room: null,
  analyser: null,
  buf: null,
  quietSince: 0,
}));
const seatOf = (personaId: string | undefined) => seats.findIndex((s) => s.persona?.id === personaId);

type StatusKind = "idle" | "busy" | "live" | "error";
const setStatus = (msg: string, kind: StatusKind = "idle") => {
  statusEl.textContent = msg;
  statusEl.dataset.kind = kind;
};

// ── speaking glow: one analyser per avatar's audio ───────────────────────────

let audioCtx: AudioContext | null = null;
let raf = 0;

for (const seat of seats) {
  seat.video.addEventListener("playing", () => seat.video.classList.add("playing"));
  seat.audio.addEventListener("playing", () => {
    const stream = seat.audio.srcObject;
    if (!(stream instanceof MediaStream) || !audioCtx) return;
    seat.analyser = audioCtx.createAnalyser();
    seat.analyser.fftSize = 512;
    audioCtx.createMediaStreamSource(stream).connect(seat.analyser);
    seat.buf = new Uint8Array(seat.analyser.fftSize);
  });
}

function tick(now: number): void {
  for (const seat of seats) {
    if (!seat.analyser || !seat.buf) continue;
    seat.analyser.getByteTimeDomainData(seat.buf);
    let sum = 0;
    for (const v of seat.buf) sum += ((v - 128) / 128) ** 2;
    if (Math.sqrt(sum / seat.buf.length) > 0.03) {
      seat.quietSince = 0;
      seat.el.dataset.speaking = "";
    } else {
      seat.quietSince ||= now;
      if (now - seat.quietSince > 450) delete seat.el.dataset.speaking;
    }
  }
  raf = requestAnimationFrame(tick);
}

// ── transcript ───────────────────────────────────────────────────────────────

const lines = new Map<string, HTMLDivElement>();
const lastLineOf = new Map<string, HTMLDivElement>();

// The record the Open WebUI export is built from: every line by turn id, with
// the time it started. Kept after End so it can be exported.
const record = new Map<string, SpokenTurn>();
const lastTurnOf = new Map<string, SpokenTurn>();
let recordMeta: { title: string; scenario: string; personas: string[]; startedAt: number; endedAt: number } | null =
  null;

function recordTurn(turn: Turn): void {
  let r = record.get(turn.id);
  if (!r) {
    const speaker = turn.role === "assistant" ? seats[seatOf(turn.speaker)]?.persona?.name : undefined;
    r = { role: turn.role, speaker, text: "", at: Date.now() };
    record.set(turn.id, r);
  }
  r.text = turn.text;
  if (turn.speaker) lastTurnOf.set(turn.speaker, r);
}

function exportForOpenWebUI(): void {
  if (!recordMeta || record.size === 0) return;
  const session: SpokenSession = { ...recordMeta, turns: [...record.values()] };
  const blob = new Blob([JSON.stringify(toOpenWebUIExport(session), null, 2)], { type: "application/json" });
  const stamp = new Date(recordMeta.startedAt).toISOString().slice(0, 16).replace(/[-:]/g, "").replace("T", "-");
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${recordMeta.scenario}-session-${stamp}.openwebui.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
}

function upsertTurn(turn: Turn): void {
  recordTurn(turn);
  let line = lines.get(turn.id);
  if (!line) {
    line = document.createElement("div");
    line.className = `turn ${turn.role}`;
    const who = document.createElement("span");
    who.className = "who";
    const i = seatOf(turn.speaker);
    if (turn.role === "user") who.textContent = userLabel;
    else {
      who.textContent = seats[i]?.persona?.name ?? "Host";
      line.dataset.seat = String(i);
    }
    const text = document.createElement("span");
    text.className = "text";
    line.append(who, text);
    transcriptEl.append(line);
    lines.set(turn.id, line);
  }
  ($<HTMLSpanElement>(".text", line)).textContent = turn.text;
  line.classList.toggle("pending", !turn.done);
  if (turn.speaker) lastLineOf.set(turn.speaker, line);
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function showFloor(msg: Extract<DuoServerMessage, { type: "floor" }>): void {
  seats.forEach((s) => {
    if (s.persona?.id === msg.persona) s.el.dataset.floor = "";
    else delete s.el.dataset.floor;
  });
  const name = seats[seatOf(msg.persona)]?.persona?.name ?? msg.persona;
  if (msg.reason === "greeting") {
    decisionWhy.textContent = `${name} opens`;
    return;
  }
  const [a, b] = seats.map((s) => msg.scores[s.persona?.id ?? ""] ?? 0);
  if (msg.reason === "user") {
    decisionBar.a.style.flexGrow = String(a ?? 0.5);
    decisionBar.b.style.flexGrow = String(b ?? 0.5);
    const pct = (v: number | undefined) => `${Math.round((v ?? 0) * 100)}%`;
    decisionWhy.textContent = `→ ${name} · ${seats[0]!.persona?.name} ${pct(a)} / ${seats[1]!.persona?.name} ${pct(b)} · ${msg.ms}ms`;
  } else {
    decisionWhy.textContent = `→ ${name} · handed the floor (p=${(msg.scores[msg.persona] ?? 0).toFixed(2)})`;
  }
}

// ── session lifecycle ────────────────────────────────────────────────────────

let sessionId: string | null = null;
let ws: WebSocket | null = null;
let mic: MicCapture | null = null;
let generation = 0;
let muted = false;

const send = (msg: ClientMessage) => {
  if (ws?.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
};

async function start(): Promise<void> {
  const mine = ++generation;
  const stale = () => generation !== mine;
  startBtn.disabled = true;
  setStatus("starting two avatars…", "busy");
  // Created inside the click so it is allowed to run.
  audioCtx = new AudioContext();
  raf ||= requestAnimationFrame(tick);

  let started: DuoStartResponse & { error?: string };
  try {
    const res = await fetch("/api/duo/start", { method: "POST" });
    started = (await res.json()) as typeof started;
    if (!res.ok) throw new Error(started.error ?? "could not start");
  } catch (err) {
    setStatus(err instanceof Error ? err.message : "could not start", "error");
    startBtn.disabled = false;
    return;
  }
  sessionId = started.session_id;
  if (stale()) return void endOnServer(started.session_id);

  // A new call starts a new record (the old one was exportable until now).
  record.clear();
  lastTurnOf.clear();
  lines.clear();
  lastLineOf.clear();
  transcriptEl.textContent = "";
  exportBtn.disabled = true;
  recordMeta = {
    title: `${started.scenario.title} — spoken session ${new Date().toLocaleString()}`,
    scenario: started.scenario.id,
    personas: started.personas.map((p) => p.name),
    startedAt: Date.now(),
    endedAt: Date.now(),
  };
  titleEl.textContent = started.scenario.title;
  subEl.textContent = started.scenario.subtitle;
  userLabel = started.scenario.user_label === "User" ? "You" : `You (${started.scenario.user_label})`;
  started.personas.forEach((p, i) => {
    const seat = seats[i]!;
    seat.persona = p;
    $<HTMLSpanElement>(".name", seat.el).textContent = p.name;
    $<HTMLSpanElement>(".tagline", seat.el).textContent = p.tagline;
  });

  setStatus("connecting to both avatars…", "busy");
  try {
    await Promise.all(
      seats.map(async (seat) => {
        seat.room = await joinAvatarRoom(seat.persona!.livekit_url, seat.persona!.livekit_client_token, {
          video: seat.video,
          audio: seat.audio,
        });
      }),
    );
  } catch (err) {
    return void stop(err instanceof Error ? err.message : "could not reach the avatars", "error");
  }
  if (stale()) return;

  setStatus("warming up the hosts…", "busy");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}${started.ws_path}`);
  ws.onmessage = (event: MessageEvent<string>) => {
    let msg: DuoServerMessage;
    try {
      msg = JSON.parse(event.data) as DuoServerMessage;
    } catch {
      return;
    }
    switch (msg.type) {
      case "ready":
        onReady(stale);
        break;
      case "turn":
        upsertTurn(msg);
        break;
      case "floor":
        showFloor(msg);
        break;
      case "interrupted":
        lastLineOf.get(msg.persona)?.classList.add("interrupted");
        {
          const r = lastTurnOf.get(msg.persona);
          if (r) r.interrupted = true;
        }
        break;
      case "error":
        setStatus(msg.message, "error");
        break;
    }
  };
  ws.onclose = () => {
    if (!stale()) void stop("the session ended");
  };
}

function onReady(stale: () => boolean): void {
  if (stale() || mic) return;
  layout.dataset.live = "";
  stopBtn.disabled = false;
  setStatus(`live — talk to ${seats.map((s) => s.persona?.name).join(" or ")}`, "live");
  void startMicCapture((b64) => send({ type: "mic_audio", audio: b64 }))
    .then((capture) => {
      if (stale()) return capture.stop();
      mic = capture;
      micBtn.disabled = false;
    })
    .catch((err: unknown) =>
      setStatus(err instanceof Error ? `microphone unavailable: ${err.message}` : "microphone unavailable", "error"),
    );
}

async function stop(reason = "idle", kind: StatusKind = "idle"): Promise<void> {
  generation += 1;
  stopBtn.disabled = true;
  micBtn.disabled = true;
  mic?.stop();
  mic = null;
  const sock = ws;
  ws = null;
  if (sock) {
    sock.onclose = null;
    if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "stop" } satisfies ClientMessage));
    sock.close();
  }
  await Promise.all(seats.map((s) => s.room?.disconnect()));
  for (const s of seats) {
    s.room = null;
    s.analyser = null;
    s.buf = null;
    s.video.classList.remove("playing");
    delete s.el.dataset.speaking;
    delete s.el.dataset.floor;
  }
  cancelAnimationFrame(raf);
  raf = 0;
  void audioCtx?.close();
  audioCtx = null;
  delete layout.dataset.live;
  muted = false;
  micBtn.setAttribute("aria-pressed", "false");
  micBtn.textContent = "Mute";
  const id = sessionId;
  sessionId = null;
  if (id) void endOnServer(id);
  if (recordMeta) recordMeta.endedAt = Date.now();
  exportBtn.disabled = record.size === 0;
  setStatus(reason, kind);
  startBtn.disabled = false;
}

async function endOnServer(id: string): Promise<void> {
  await fetch("/api/session/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: id }),
  }).catch(() => {});
}

// Two billable avatars: a closing tab must release both.
window.addEventListener("pagehide", () => {
  if (!sessionId) return;
  navigator.sendBeacon?.(
    "/api/session/stop",
    new Blob([JSON.stringify({ session_id: sessionId })], { type: "application/json" }),
  );
});

startBtn.addEventListener("click", () => void start());
exportBtn.addEventListener("click", exportForOpenWebUI);
stopBtn.addEventListener("click", () => void stop());
micBtn.addEventListener("click", () => {
  if (!mic) return;
  muted = !muted;
  mic.setMuted(muted);
  micBtn.setAttribute("aria-pressed", String(muted));
  micBtn.textContent = muted ? "Unmute" : "Mute";
});
