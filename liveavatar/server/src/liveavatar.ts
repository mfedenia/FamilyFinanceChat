/**
 * HTTP wrapper over the LiveAvatar API session endpoints. One function per
 * endpoint, no product logic.
 *
 * LITE mode is the whole trick: it gives back a `ws_url` — a direct websocket
 * into the avatar's media server — so this server can thread GPT-Live's audio
 * straight into the avatar without the browser in the loop. The browser only
 * receives the LiveKit credentials and watches.
 */

import { config } from "./config";

export interface StartedSession {
  sessionId: string;
  livekitUrl: string;
  livekitClientToken: string;
  /** The media-server websocket this server threads audio into. */
  wsUrl: string;
}

export class LiveAvatarApiError extends Error {
  constructor(
    readonly status: number,
    readonly body: string,
  ) {
    super(`LiveAvatar API returned ${status}: ${body}`);
  }
}

async function post(
  path: string,
  body: Record<string, unknown>,
  headers: Record<string, string>,
): Promise<Record<string, unknown>> {
  const response = await fetch(`${config.liveavatar.apiUrl}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: JSON.stringify(body),
  });
  const text = await response.text();
  if (!response.ok) throw new LiveAvatarApiError(response.status, text.slice(0, 500));
  const parsed = JSON.parse(text) as { data?: Record<string, unknown> };
  return parsed.data ?? {};
}

// Resolved once per process: which avatar to drive when LIVEAVATAR_AVATAR_ID
// is not set. `avatar_id` is required by the API with no server-side default,
// but `GET /v1/avatars/public` needs no auth — so the zero-config path is
// "the first active public avatar", named in the log so it is not a mystery.
let fallbackAvatarId: string | null = null;

async function resolveAvatarId(): Promise<string> {
  if (config.liveavatar.avatarId) return config.liveavatar.avatarId;
  if (fallbackAvatarId) return fallbackAvatarId;

  const response = await fetch(`${config.liveavatar.apiUrl}/v1/avatars/public?page_size=20`);
  if (!response.ok) throw new Error(`could not list public avatars (${response.status})`);
  const body = (await response.json()) as {
    data?: { results?: { id?: string; name?: string; status?: string; type?: string }[] };
  };
  const avatars = body.data?.results ?? [];
  const pick =
    avatars.find((a) => a.status === "ACTIVE" && a.type === "VIDEO") ??
    avatars.find((a) => a.status === "ACTIVE");
  if (!pick?.id) {
    throw new Error("no public avatar available — set LIVEAVATAR_AVATAR_ID in .env");
  }
  console.log(`[liveavatar] no LIVEAVATAR_AVATAR_ID set — using public avatar "${pick.name}" (${pick.id})`);
  fallbackAvatarId = pick.id;
  return pick.id;
}

export interface PublicAvatar {
  id: string;
  name: string;
}

/**
 * `count` distinct active public video avatars, chosen at random — the duo
 * demo's zero-config cast. One page of 100 is plenty of variety; no auth.
 */
/** Every active public video avatar (all pages — ~300 today). */
async function listPublicVideoAvatars(): Promise<PublicAvatar[]> {
  const out: PublicAvatar[] = [];
  for (let page = 1; page <= 10; page++) {
    const response = await fetch(`${config.liveavatar.apiUrl}/v1/avatars/public?page_size=100&page=${page}`);
    if (!response.ok) throw new Error(`could not list public avatars (${response.status})`);
    const body = (await response.json()) as {
      data?: { next?: string | null; results?: { id?: string; name?: string; status?: string; type?: string }[] };
    };
    for (const a of body.data?.results ?? []) {
      if (a.id && a.name && a.status === "ACTIVE" && a.type === "VIDEO") out.push({ id: a.id, name: a.name });
    }
    if (!body.data?.next) break;
  }
  return out;
}

/** Pinned avatar ids, named from the public list where possible. */
export async function describeAvatars(ids: string[]): Promise<PublicAvatar[]> {
  const pool = await listPublicVideoAvatars().catch(() => [] as PublicAvatar[]);
  return ids.map((id, i) => pool.find((a) => a.id === id) ?? { id, name: `Host ${i + 1}` });
}

export async function pickRandomAvatars(count: number): Promise<PublicAvatar[]> {
  const pool = await listPublicVideoAvatars();
  // Fisher–Yates, then keep one avatar per first name: "Judy Lawyer" and
  // "Judy Teacher" are the same face, which would make a strange duo.
  for (let i = pool.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [pool[i], pool[j]] = [pool[j]!, pool[i]!];
  }
  const picked: PublicAvatar[] = [];
  const seen = new Set<string>();
  for (const a of pool) {
    const first = a.name.split(/\s+/)[0]!.toLowerCase();
    if (seen.has(first)) continue;
    seen.add(first);
    picked.push({ id: a.id, name: a.name });
    if (picked.length === count) return picked;
  }
  throw new Error(`only ${picked.length} distinct public avatars available, need ${count}`);
}

/**
 * Mint + start one LITE session.
 *
 * The token payload deliberately carries no `*_config` block: `livekit_config`
 * would mean bring-your-own-LiveKit and would null out the `livekit_url` /
 * `livekit_client_token` the browser needs back from start. A bare LITE
 * payload returns those plus `ws_url`.
 */
export async function startSession(avatarId?: string): Promise<StartedSession> {
  const token = await post(
    "/v1/sessions/token",
    {
      mode: "LITE",
      avatar_id: avatarId ?? (await resolveAvatarId()),
    },
    { "X-API-KEY": config.liveavatar.apiKey },
  );
  const sessionId = String(token.session_id ?? "");
  const sessionToken = String(token.session_token ?? "");
  if (!sessionId || !sessionToken) {
    throw new Error("LiveAvatar token mint returned no session_id/session_token");
  }

  const started = await post(
    "/v1/sessions/start",
    {},
    { Authorization: `Bearer ${sessionToken}` },
  );
  const livekitUrl = String(started.livekit_url ?? "");
  const livekitClientToken = String(started.livekit_client_token ?? "");
  const wsUrl = String(started.ws_url ?? "");
  if (!livekitUrl || !livekitClientToken || !wsUrl) {
    // A LITE start without an agent config returns all three; missing any
    // means the payload drifted. Fail loudly rather than hand out a session
    // the legs cannot work with — but release the upstream slot first.
    await stopSession(sessionId).catch(() => {});
    throw new Error(`session ${sessionId} started without livekit_url/livekit_client_token/ws_url`);
  }

  return { sessionId, livekitUrl, livekitClientToken, wsUrl };
}

/**
 * Uses the API-key form rather than the session JWT: teardown can happen well
 * after mint, and this way it does not depend on the JWT still being valid.
 */
export async function stopSession(sessionId: string): Promise<void> {
  await post(
    "/v1/sessions/stop",
    { session_id: sessionId },
    { "X-API-KEY": config.liveavatar.apiKey },
  );
}
