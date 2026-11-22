/**
 * The wire protocol between the web client and the orchestrator.
 *
 * This file IS the contract. The server imports it to type what it emits, the
 * client imports it to type its message switch — so a widget the server can
 * send that the client cannot render is a compile error, not a runtime mystery.
 *
 * Deliberately absent: avatar audio and video. Those reach the browser over
 * LiveKit (the server threads GPT-Live's audio into the avatar directly), so
 * this socket only carries what LiveKit cannot — mic audio up, transcripts and
 * tool-driven visuals down.
 */

/**
 * Props for the `term_card` widget. A pure overlay: a transparent lower-third
 * over the full-frame avatar, which never moves or resizes for it.
 */
export interface TermCardProps {
  /** The word or phrase in its own script: こんにちは, お茶. */
  term: string;
  /** How it is pronounced: "kon-ni-chi-wa". */
  reading?: string;
  /** Short English gloss: "hello". */
  meaning?: string;
}

/**
 * Props for the `learned_words` widget — the recap panel. This one owns the
 * stage: the client shrinks the avatar to the bottom-right corner while it is
 * up. The word list comes from the SERVER's per-session store (every term card
 * shown is recorded), never from the model — the model only asks for the recap
 * and supplies a heading.
 */
export interface LearnedWordsProps {
  title: string;
  words: TermCardProps[];
}

/**
 * One on-screen visual instruction. Every tool call the model makes lands in
 * the browser as one of these, so the client renders visuals with a single
 * switch over `widget` (see web/src/overlays/). A new tool that reuses an
 * existing widget costs zero client code.
 *
 * Staging is per-widget, decided by the client — not a model argument: a
 * term_card overlays the full-frame avatar; learned_words puts the avatar in
 * a corner while the panel holds the stage.
 */
export type UiMessage =
  | { widget: "term_card"; props: TermCardProps }
  | { widget: "learned_words"; props: LearnedWordsProps }
  | { widget: "hide"; props: Record<string, never> };

export type Widget = UiMessage["widget"];

/**
 * One conversational turn, projected from GPT-Live's transcript fragments.
 * Streaming updates re-send the same `id` with longer `text`; the client
 * rewrites that line in place and only breaks to a new line on a new id.
 */
export interface Turn {
  id: string;
  role: "user" | "assistant";
  text: string;
  done: boolean;
  /** Duo demo only: which persona an assistant turn belongs to. */
  speaker?: string;
}

/** Server → browser. */
export type ServerMessage =
  | { type: "ready" }
  | ({ type: "turn" } & Turn)
  | ({ type: "ui" } & UiMessage)
  /** The avatar was cut off mid-sentence; what it was saying is never coming. */
  | { type: "interrupted" }
  | { type: "error"; message: string };

/** Browser → server. */
export type ClientMessage =
  /** Base64 PCM16 mono @ 24kHz, produced by the AudioWorklet in micCapture.ts. */
  | { type: "mic_audio"; audio: string }
  | { type: "stop" };

// ── Duo demo (/duo.html) ─────────────────────────────────────────────────────
// Two avatars, one conversation: a listener transcribes the mic, the decision
// model picks which persona answers, and only that persona's avatar speaks.

/** One persona as the browser sees it — plus its own LiveKit room. */
export interface DuoPersonaInfo {
  id: string;
  name: string;
  /** The Open WebUI model preset this persona comes from, if any. */
  preset_id: string | null;
  tagline: string;
  livekit_url: string;
  livekit_client_token: string;
}

/** POST /api/duo/start → */
export interface DuoStartResponse {
  session_id: string;
  ws_path: string;
  scenario: { id: string; title: string; subtitle: string; user_label: string };
  personas: DuoPersonaInfo[];
}

/** Why a persona got the floor. */
export type DuoFloorReason = "greeting" | "user" | "handoff";

/** Server → browser, duo socket. Mic audio goes up as a ClientMessage. */
export type DuoServerMessage =
  | { type: "ready" }
  | ({ type: "turn" } & Turn)
  /**
   * Who speaks next and why. `scores` are the decision model's per-persona
   * probabilities (empty for greeting); `ms` is how long the decision took.
   */
  | {
      type: "floor";
      persona: string;
      reason: DuoFloorReason;
      scores: Record<string, number>;
      ms: number;
    }
  /** That persona was cut off by the user; the rest of its line is gone. */
  | { type: "interrupted"; persona: string }
  | { type: "error"; message: string };
