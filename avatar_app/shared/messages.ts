/**
 * The wire protocol between the web client and the orchestrator.
 *
 * This file IS the contract. The server imports it to type what it emits, the
 * client imports it to type its message switch — so a message the server can
 * send that the client cannot handle is a compile error, not a runtime mystery.
 *
 * Deliberately absent: avatar audio and video. Those reach the browser over
 * LiveKit (the server threads GPT-Live's audio into the avatar directly), so
 * this socket only carries what LiveKit cannot — mic audio up, transcripts and
 * status down.
 *
 * FamilyFinanceChat fork: the upstream demo's on-screen widgets (`ui`
 * messages — term cards, the recap panel) were removed in AV1. A role-played
 * client has nothing to put on screen.
 */

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
}

/** Server → browser. */
export type ServerMessage =
  | { type: "ready" }
  | ({ type: "turn" } & Turn)
  /** The avatar was cut off mid-sentence; what it was saying is never coming. */
  | { type: "interrupted" }
  /**
   * The server ended the session on purpose (AV5a time limit). `message` is
   * plain words for the student; the socket closes right after.
   */
  | { type: "ended"; reason: "time_limit"; message: string }
  | { type: "error"; message: string };

/** Browser → server. */
export type ClientMessage =
  /** Base64 PCM16 mono @ 24kHz, produced by the AudioWorklet in micCapture.ts. */
  | { type: "mic_audio"; audio: string }
  | { type: "stop" };
