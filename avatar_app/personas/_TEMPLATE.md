---
name: Display name shown in the persona picker
voice: marin
greeting: How the client opens the meeting, in one line. Leave this line out to use the default greeting.
---

Everything below the front matter is the persona: who the client is, what they
know and will share if asked well, and how they behave. It is sent to the voice
model once, at the start of every session, and cannot change mid-session.

Rules of thumb:

- Write it for listening, not reading. Ask for short spoken replies, no lists.
- Keep it under roughly 40,000 characters. The bridge refuses longer personas
  rather than cutting them off (GPT-Live's limit is 16,384 tokens, and the
  role-play rules are added on top).
- Voices: marin, cedar, alloy, ash, ballad, coral, echo, sage, shimmer, verse.

Files whose names start with "_" (like this one) are ignored. Save a real
persona as personas/<id>.md, where <id> uses only letters, digits, - and _.

In production, personas come from Open WebUI instead: set
AVATAR_PERSONA_SOURCE=openwebui, and tag the professor's role-play models
"avatar" in Workspace → Models. Add a tag such as "voice:cedar" to pick a voice.
