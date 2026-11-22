# Workstream B — Embodied Advising

### Turning FamilyFinanceChat from a chatbot into a conversation

**Status:** research and prototyping track with defined gates and a defined kill switch
**Owner:** realtime / avatar engineer, with frontend and data/assessment support
**Term:** Fall 2026 · meetings **M2–M11** (22 September – 1 December). Task numbers below
(B1…B18) match [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §7; meeting numbers match §8.
**Governed by:** [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) — no exceptions
**Implementation (November 2026):** provider chosen (LiveAvatar + GPT-Live). The bridge is
being built as tasks **AV0–AV7**; see [§16](#16-implementation-the-gpt-live-avatar-bridge-av0av7).

---

## 1. The problem with typing

FIN 602 exists to prepare students for a room with a real family in it. What that room
actually demands is: listening while someone is still talking, hearing hesitation, holding
eye contact through an uncomfortable question about money, and asking the follow-up out
loud without a backspace key.

The current platform teaches the *content* of good advising well. It cannot teach the
*performance* of it, because typing removes every one of those pressures. A student can
draft, delete, and reword for ninety seconds before asking "what happens to the business if
something happens to you?" — a question they will have to ask, in real time, to a person
whose face they can see.

**The hypothesis for this semester:** a spoken, face-to-face AI client produces meaningfully
better preparation than a text chatbot — and we can measure whether that is true using the
scoring pipeline the platform already has.

That is a hypothesis, not a conclusion. This track is designed to test it and to be
allowed to answer "no."

---

## 2. Target experience

A student opens a link, sees a face, and says out loud:

> "Hi — I'm going to ask you a few questions about your family's finances. Is that alright?"

The client on screen looks at them, nods, and answers in their own voice, in character, with
their own financial situation, drawn from the same course documents the text bot uses. If the
student interrupts, the client stops talking. If the student goes quiet, the client waits — and
may eventually prompt them, the way a real, slightly impatient client would.

At the end, the transcript lands in the same grading dashboard the instructor already uses,
scored on the same seven dimensions and the same ABI trust rubric.

**What we are not building:** a photorealistic digital human, a likeness of any real person,
or a replacement for the text interface. Text stays. Voice and face are an additional mode.

---

## 3. The constraint that determines the architecture

The instinct is to put the avatar inside the Open WebUI chat window. **That is forbidden**
(policy F3/F4): Open WebUI's frontend is a Svelte application we do not own, and embedding a
WebRTC video pane in it means forking and re-merging that frontend forever. That is precisely
the mistake this project spent a semester undoing.

So the constraint dictates the design, and the design it dictates is better:

> **The avatar is a companion web application in its own container.** Open WebUI remains the
> brain (model routing, RAG, knowledge base) and the system of record (transcripts). The
> avatar app talks to it exclusively over the public REST API.

Consequences, all favourable:

- it can be deployed, rolled back, and load-tested independently of the chat platform;
- it survives every upstream Open WebUI release by construction;
- if the track is killed in December, nothing has to be un-merged — one container is deleted;
- students, professors, and graders keep the interface they already know.

---

## 4. Reference architecture

```
                          ┌─────────────────────────────────────────────┐
   Student browser        │            GCP VM · docker compose          │
   ┌───────────────┐      │                                             │
   │  mic + camera │      │   Nginx / Caddy  (TLS, auth, path routing)  │
   │  <video> pane │◄────►│      /            → open-webui  :8080       │
   └───────┬───────┘      │      /grading     → grading dashboard       │
           │ WebRTC       │      /practice    → avatar-app   :7000  ◄── NEW
           │              │                                             │
           │              │   avatar-app (Node or Python)               │
           │              │     · session + persona management          │
           │              │     · turn taking, barge-in, VAD            │
           │              │     · transcript assembly + write-back      │
           │              │            │            │            │      │
           └──────────────┼────────────┘            │            │      │
                          │      ┌──────────────────┘            │      │
                          │      ▼                               ▼      │
                          │   open-webui  ────► Qdrant (course docs)    │
                          │   POST /api/chat/completions                │
                          │   POST /api/v1/chats/…  (transcript)        │
                          └──────────┬──────────────────┬───────────────┘
                                     │                  │
                          ┌──────────▼───────┐  ┌───────▼──────────────┐
                          │  STT + TTS       │  │  Avatar renderer     │
                          │  (Whisper /      │  │  (streaming provider │
                          │   Voxtral;       │  │   or self-hosted)    │
                          │   ElevenLabs /   │  │  audio ──► lip-synced│
                          │   Kokoro /       │  │  video, published    │
                          │   Chatterbox)    │  │  into the session    │
                          └──────────────────┘  └──────────────────────┘
```

**The pattern that matters** (and the one the industry converged on during 2026): the avatar
renderer is not a post-processing step the app waits on. It **joins the session as its own
participant.** The orchestrator streams the TTS audio to it over a fast data channel, and the
renderer publishes finished, synchronised audio and video directly to the student. This
removes a full round trip through our server and is the difference between "video call" and
"laggy video attachment."

**Orchestration:** do not hand-roll the WebRTC and turn-taking layer. Use an agent framework
built for this — **LiveKit Agents** (which now lists a dozen-plus avatar provider plugins) or
**Pipecat** (which integrates Simli, HeyGen, Tavus, and fully local models). Either gives us
voice activity detection, endpointing, barge-in, and provider swapping for free. Hand-rolling
this is a semester by itself, and it is not the semester's research question.

---

## 5. Latency: the number that decides everything

Conversation tolerates about **a second**. Past roughly two seconds of silence, a person
stops feeling heard and starts feeling processed — and the pedagogical value collapses,
because the student is no longer rehearsing a real interaction.

Budget from end of student speech to first audible syllable from the avatar:

| Stage | Realistic range | Notes and levers |
|---|---:|---|
| Voice activity detection / endpointing | 150–300 ms | tunable; aggressive settings cut latency but clip the student |
| Speech to text | 150–300 ms | streaming STT overlaps with speech; batch STT does not |
| Retrieval (RAG) | 100–300 ms | **the hidden cost.** Consider pre-loading the persona's facts into the system prompt instead of retrieving per turn |
| LLM time-to-first-token | 300–600 ms | streaming is mandatory; model choice dominates |
| Text to speech, first byte | 150–300 ms | must be streaming and sentence-chunked |
| Avatar render + network | 200–500 ms | provider-dependent; the largest single variable |
| **Total (first syllable)** | **≈ 1.0–2.3 s** | |

**Design targets for this semester:** p50 under **1.2 s**, p95 under **2.0 s**, with
barge-in interrupting playback within **300 ms**.

**Measure this by M4 (6 October), before choosing a provider.** Instrument every stage separately —
a single end-to-end number tells you that it is slow but not what to fix. Log per-turn stage
timings into the existing Prometheus stack; they belong on the same Grafana dashboard as the
chat metrics.

**The RAG trap, stated plainly:** the text platform retrieves from Qdrant on every turn. In a
spoken loop that retrieval sits directly in the critical path. Strongly consider loading the
family scenario into the system prompt once at session start — the personas are small, and a
2,000-token persona brief costs less latency than a vector search per turn, every turn.

---

## 6. Provider landscape (as of August 2026 — verify before committing)

Real-time avatar streaming became a commodity during 2026. There are three viable strategies.

### 6.1 Hosted avatar streaming (recommended starting point)

| Provider | Notable characteristic | Integration |
|---|---|---|
| **HeyGen LiveAvatar** | mature stock-avatar library; WebRTC streaming with natural lip-sync and gesture; typically low-seconds latency | LiveKit and Pipecat plugins; direct SDK |
| **Tavus (Phoenix-4)** | advertises sub-600 ms end-to-end and full-duplex behaviour — it can listen while speaking | Pipecat integration, open-sourced examples |
| **Simli** | developer-first, designed to plug into agent frameworks | LiveKit / Pipecat plugin |
| **Anam** | positions on conversational realism and latency | LiveKit plugin |
| **bitHuman, Beyond Presence, D-ID, others** | LiveKit's avatar plugin ecosystem lists a dozen-plus providers | LiveKit plugins |

**Do not take these characterisations on faith — that is the assignment.** Vendor latency
claims are measured under vendor conditions. Measure ours.

### 6.2 Self-hosted open source

Whisper (STT) → an open TTS (Kokoro, Chatterbox — Open WebUI already documents both) → an
open lip-sync renderer (MuseTalk, LivePortrait, SadTalker class of models).

- **For:** no per-minute cost, no student audio leaving our infrastructure, full control —
  a genuinely stronger privacy story for an education setting.
- **Against:** needs a GPU the current VM does not have; quality and latency are materially
  behind hosted options; it becomes an infrastructure project rather than a pedagogy project.

**Do the arithmetic before advocating for it.** An L4-class GPU VM runs on the order of
$0.70/hour on demand — roughly $500/month if left running. At this course's usage volume
(§8), that is *more* expensive than metered streaming, not less. Self-hosting wins on
privacy and at scale, not on cost at forty hours a semester. Say so in the report either way.

### 6.3 Voice only, no face (B0 — ships first, ships regardless)

Open WebUI **already has** hands-free voice: a call mode where the student speaks, the model
answers, the reply is read back sentence-by-sentence as it streams, and the microphone
re-arms. STT is configurable (browser Web Speech API, Whisper, Voxtral); TTS accepts any
OpenAI-compatible `/audio/speech` endpoint, plus ElevenLabs, Kokoro, and Chatterbox natively.

This is configuration, not engineering — surface A1 of the compatibility policy. It captures a
large share of the pedagogical benefit (speaking out loud, no backspace, real-time pressure)
for close to zero build cost and zero compatibility risk.

**Build B0 by M3 (29 September) and put it in front of students immediately.** It de-risks the whole
track: if avatars prove too slow, too costly, or too fragile, the course still gains spoken
practice this semester.

---

## 7. Integration with grading — the part that is easy to forget

An avatar session that never reaches the grading dashboard is a demo, not a feature. The
instructor's workflow — extract, score on seven dimensions, ABI, per-student feedback — must
work for spoken sessions exactly as it does for typed ones.

**Open technical question the team must answer by M7 (27 October):** does a chat created through
`POST /api/chat/completions` appear in Open WebUI's chat history — and therefore in
`extract_chats.py` — or does it need to be written back explicitly via the chat endpoints?
Test it; do not assume. The two viable designs:

- **Preferred — write back into Open WebUI.** The avatar app creates a chat and appends
  each turn through the public API. One system of record; grading needs no changes at all.
- **Fallback — the avatar app owns its own transcript store**, and `extract_chats.py` grows a
  second source that merges by student and timestamp. More code, more drift, but no
  dependency on write-back behaviour.

**Also worth capturing, because spoken conversation exposes things typing hides:**
speech disfluencies and filler rate, response latency *by the student* (hesitation before
asking hard questions), interruption counts, and talk-time ratio. Whether a student lets the
client finish talking is, arguably, a more direct measure of advising skill than anything the
current rubric captures. That is a genuinely novel contribution this team could make.

---

## 8. Cost model — build it by M4 (6 October) with real numbers

Worked example to be replaced with measured figures:

```
40 students × 4 sessions × 15 minutes = 2,400 minutes = 40 hours per semester
```

| Line item | Basis | Semester estimate |
|---|---|---|
| Avatar streaming | ~$0.10–0.20 per minute (mode-dependent) | **$240–480** |
| STT | streaming transcription, student speech only | $10–30 |
| TTS | if not bundled with the avatar provider | $20–60 |
| LLM | short spoken turns; cheaper than long typed exchanges | $15–40 |
| **Total** | | **≈ $285–610 per semester** |

Prices move quarterly and tiers are renamed frequently — re-derive from the provider's live
pricing page at signup, and record the date you checked.

**Non-negotiable controls from the first prototype, not after the first surprise:**
per-session hard time cap (15 minutes, enforced server-side); per-student session quota;
a billing alert at 50% of the semester budget; sessions that terminate on disconnect rather
than idling; and a kill switch that disables the feature without redeploying.

---

## 9. Evaluation design

The platform already produces the outcome measure — this is a real advantage, so use it.

**Design:** within-subjects, counterbalanced. Each participating student completes advising
sessions in two or three modalities (text / voice-only / avatar) with matched but different
family scenarios, in varied order to control for practice effects.

| Measure | Source | Question it answers |
|---|---|---|
| 7-dimension question quality (0–100) | existing scoring service | do students ask better questions when speaking? |
| ABI trust scores | existing ABI pipeline | does modality change how trustworthy the interaction reads? |
| Questions asked per session; session duration | transcripts | does speaking increase or suppress engagement? |
| Interruptions; talk-time ratio | avatar session logs | are students learning to listen? |
| Student hesitation before sensitive questions | turn-level timing | does rehearsal reduce discomfort over sessions? |
| Perceived realism, social presence, anxiety | short post-session survey | is the experience believable and productive, or uncanny? |
| Preference and willingness to reuse | survey | would students actually choose this? |

**Sample size is small and the semester is short.** Do not over-claim. "Twelve students,
counterbalanced, with these effect sizes and these confidence intervals" is a credible
result; "avatars improve learning" is not. A well-designed study with an honest negative
result is a better deliverable than an over-claimed positive one, and it will be graded that
way.

---

## 10. Phased plan, with gates

### B0 — Voice, no face (M1–M3 · to 29 September)

Configure Open WebUI STT and TTS; select a voice per family persona; run a full advising
session hands-free; measure baseline round-trip latency; demo to the instructor at M3;
document the setup so a professor can change voices without a developer.

**Exit:** spoken practice is available to students. *This ships no matter what happens next.*

### B1 — Provider bake-off (M3–M6 · 29 September – 20 October)

Stand up an orchestration harness (LiveKit Agents or Pipecat) behind a provider-agnostic
interface. Integrate **two** providers. Measure, per provider, on identical scripted
conversations: stage-by-stage latency (p50/p95), barge-in responsiveness, lip-sync quality,
audio quality, cost per minute measured not quoted, SDK maturity, failure behaviour on a
dropped connection, and whether persona voice and appearance can be configured per family.

**Gate 1 (M6, 20 October): choose a provider, or recommend stopping.** Both are acceptable outcomes;
neither is acceptable without data.

### B2 — Companion prototype (M6–M10 · 20 October – 17 November)

Build `avatar-app` as a container: session creation and persona selection; a browser client
with the video pane, a live captions track, and a visible end-session control; turn-taking
with barge-in; the family scenario loaded as a system prompt at session start; transcript
write-back into Open WebUI; behind the reverse proxy with the same authentication as the rest
of the platform; hard session caps enforced server-side.

**Gate 2 (M10, 17 November): five students complete a full advising session end to end, and the
transcripts appear in the grading dashboard, scored.** No transcript, no gate.

### B3 — Evaluate and recommend (M8–M11 · 3 November – 1 December)

Run the counterbalanced sessions; analyse against the existing rubric; survey participants;
write the report; deliver a recommendation with cost, latency, and outcome evidence, plus a
concrete sizing estimate for what production deployment to a full cohort would require.

**Exit:** a decision a professor can act on without re-doing the analysis.

---

## 11. Ethics, privacy, and consent

Voice and video raise obligations that text does not. Settle these at **M2 (22 September)**,
before any recording — not in late November, when the data already exists.

- **Student chat transcripts are education records** under FERPA. Adding audio does not
  change that; it adds a biometric-adjacent identifier to it.
- **Written informed consent** before any session is recorded: what is captured, where it is
  stored, how long it is kept, who can see it, and how a student opts out. Opting out must
  not affect a grade — provide the text path as an equivalent alternative.
- **No likeness of any real person** is used to build an avatar. Use provider stock avatars
  under their licence. Do not clone a professor's face or voice "as a joke" — that is a
  deepfake of a colleague and it will end this project.
- **Disclose the AI.** The client is obviously synthetic, and the interface says so. We are
  teaching advising, not testing whether students can be fooled.
- **Data minimisation.** Prefer keeping the transcript and discarding the audio. If audio is
  retained for research, keep it separately with a stated deletion date.
- **Vendor data handling.** Before sending a single student's voice to a provider, read the
  data-processing terms: retention, training use, sub-processors, region. Record the finding
  in the bake-off report — for an education platform this can outweigh a latency win.
- **Institutional review.** Confirm with the instructor whether the learning-outcome study
  needs IRB review. Ask at M2; approval takes longer than students expect.

---

## 12. Risks specific to this track

| # | Risk | Mitigation |
|---|---|---|
| B-R1 | Latency lands at 3+ seconds and the experience feels dead | measure by M4; B0 ships regardless; consider dropping per-turn RAG |
| B-R2 | The uncanny valley — students find the avatar off-putting rather than engaging | measure it in the survey; stylised or clearly-synthetic avatars often outperform near-photoreal ones |
| B-R3 | Costs run away during testing | hard caps, quotas, and a billing alarm from day one, not after the first bill |
| B-R4 | The provider deprecates or reprices mid-semester | provider-agnostic interface first; keep the runner-up integrated and warm |
| B-R5 | Avatar sessions never reach the grading dashboard | make the write-back the *first* thing B2 builds, not the last |
| B-R6 | Someone proposes embedding the video pane in the Open WebUI chat page | policy F3/F4 — refuse; the companion app exists for this reason |
| B-R7 | The track absorbs the whole semester and Workstream A slips | Gate 1 and Gate 2 are real; if a gate fails, stop and reinforce A |

---

## 13. Deliverables

| # | Deliverable | Due |
|---|---|---|
| D1 | Requirements note: what a spoken advising session must do | M3 · 29 Sep |
| D2 | Working voice-only mode, documented for a non-developer | M3 · 29 Sep |
| D3 | Latency instrumentation on the existing Grafana stack | M4 · 6 Oct |
| D4 | Provider bake-off report with measured data and a recommendation | M6 · 20 Oct |
| D5 | Cost model with live pricing, usage assumptions, and enforced caps | M4 · 6 Oct |
| D6 | `avatar-app` container, source, and deployment documentation | M9 · 10 Nov |
| D7 | Transcript write-back verified end to end into the grading dashboard | M9 · 10 Nov |
| D8 | Evaluation report: methodology, data, findings, limitations | M11 · 1 Dec |
| D9 | Go/no-go recommendation with a production sizing estimate | M11 · 1 Dec |
| D10 | Handoff section in `HANDOFF.md` for the next team | M11 · 1 Dec |

---

## 14. Open questions to answer, not assume

1. Does a chat created via `POST /api/chat/completions` become visible to `extract_chats.py`,
   or is explicit write-back required?
2. Does per-turn Qdrant retrieval fit inside the latency budget, or must the persona be
   pre-loaded into the system prompt?
3. Can each family persona have a distinct voice *and* appearance without a separate paid
   avatar per persona?
4. What is barge-in latency in practice, per provider — not in the marketing material?
5. Does an Open WebUI **Action Function** (surface A4) provide an acceptable "start a spoken
   session about this chat" button, carrying context across to the companion app?
6. What are the provider's data-retention and training-use terms for student audio?
7. At what cohort size does self-hosting on a GPU become cheaper than metered streaming?
8. Do spoken sessions actually score differently under the existing rubric — and if they do,
   is the rubric measuring the right thing for speech?

---

## 15. Sources

- Open WebUI — [Extensibility: Tools & Functions](https://docs.openwebui.com/features/extensibility/plugin/), [Pipelines](https://docs.openwebui.com/features/extensibility/pipelines/), [API Endpoints](https://docs.openwebui.com/reference/api-endpoints/), [Speech-to-Text & Text-to-Speech](https://docs.openwebui.com/features/chat-conversations/audio/), [Plugin migration to 0.9.0](https://docs.openwebui.com/features/extensibility/plugin/migration/to-0.9.0/), [Updating Open WebUI](https://docs.openwebui.com/getting-started/updating/)
- [Open WebUI v0.11.0 release notes](https://openwebui.com/blog/v0-11-0-the-interface-reorganized) · [Releases](https://github.com/open-webui/open-webui/releases)
- [HeyGen — LiveAvatar](https://help.heygen.com/en/articles/12758516-introducing-liveavatar) · [HeyGen API pricing](https://www.heygen.com/api-pricing) · [Developer docs](https://developers.heygen.com/)
- [Tavus — building real-time AI interactions with Pipecat](https://www.tavus.io/post/open-sourcing-ai-innovation-building-real-time-ai-interactions-with-pipecat-and-tavus)
- [Real-time avatars and lip-sync in a video call](https://www.forasoft.com/learn/ai-for-video-engineering/articles-ai/real-time-avatar-lipsync-in-call) · [Virtual avatar solutions compared, 2026](https://www.toughtongueai.com/blog/best-virtual-avatar-solutions-2026)

*Pricing, latency claims, and provider lists in this document were checked in August 2026 and
move quickly. Re-verify before making a commitment, and record the date you checked.*

---

## 16. Implementation: the GPT-Live avatar bridge (AV0–AV7)

**Status:** in progress on branch `feat/avatar-bridge` · code in [`avatar_app/`](../avatar_app/)
· started 2026-11-22.
**Numbering:** **AV** tasks are separate from the B1–B18 research tasks in
[`PROJECT_PLAN.md`](PROJECT_PLAN.md) §7 and from the M1–M11 meetings in §8.

### 16.1 What was decided, and what it replaces

- **Provider and voice model are chosen:** HeyGen **LiveAvatar** in **LITE** mode (we
  supply the voice; it renders the face) driven by OpenAI **GPT-Live** (`gpt-live-1`,
  full-duplex speech-to-speech). This settles the bake-off question of B5–B9 for the
  prototype.
- **Why we need our own bridge.** LiveAvatar's built-in OpenAI connector speaks the older
  Realtime protocol. GPT-Live is reachable only at `v1/live/sessions`, so something has to
  carry audio between the two. HeyGen publishes exactly that program as an MIT-licensed
  reference: [`heygen-com/liveavatar-gpt-live-demos`](https://github.com/heygen-com/liveavatar-gpt-live-demos).
  We **fork it rather than write one**. Its interruption handling, turn stitching, greeting
  gate and teardown are measured and working. This replaces the "stand up LiveKit Agents or
  Pipecat" route of §4 and B6–B8: the reference bridge already gives us what those
  frameworks were for.
- **Stack: TypeScript/Node**, kept from upstream even though the rest of the repo is
  Python. A rewrite would cost about a week and gain nothing.
- **Scope of this track is the bridge (the backend).** The student-facing page, the consent
  screen (B2/B11), the entry button inside Open WebUI (§14 Q5) and grading changes are
  separate work. `avatar_app/web/` is a developer test harness only.

```
Student browser ──mic audio──▶ BRIDGE ──▶ GPT-Live (listens, thinks, talks back)
       ▲                          │
       │                          └──voice audio──▶ LiveAvatar (moves the face)
       └────────── video of the face (LiveKit) ◀──────────────┘
BRIDGE ◀──▶ Open WebUI REST API (who is this student? which family? save the transcript)
```

**Compatibility statement (for every AV pull request):**
- Surfaces used: **A8** sidecar container, **A9** Caddy route at `/practice/`, **A7** public
  REST API. Open WebUI is not modified.
- On the next upstream release, only the Open WebUI API calls can break. They live in one
  module (`openwebui.ts`, AV3) and are exercised by AV7's tests, so the upgrade runbook
  catches a break.

**Cost:** about **$0.14 per student-minute** (LiveAvatar LITE ≈ $0.09 + GPT-Live $0.05). The
§8 working figure of 40 h a semester comes to about **$340**. Development fits on the
LiveAvatar free plan (2-minute sessions, watermark). A classroom test needs the Essential
plan ($99/month).

### 16.2 Task list

Ordered as they should be built. AV5 is split so the cost guards land **before** persona
work: every test session costs money.

| ID | Task | Depends on | Estimate | Status |
|---|---|---|---|---|
| AV0 | Import the HeyGen demo, pinned | — | ½ day | **Done** |
| AV1 | Remove the tutor logic | AV0 | ½–1 day | **Done** (a human-spoken conversation is still to be run) |
| AV5a | Guardrails that need no identity | AV1 | ~6 h | **Done** (Prometheus scrape job comes with AV6) |
| AV2 | Persona loader | AV1 | 1 day | **Done** (Open WebUI source tested against a stand-in, not the real stack) |
| AV3 | Login check + WebSocket ticket | AV1 | 1 day | Not started |
| AV5b | Per-student daily minutes | AV3 | ~2 h | Not started |
| AV4 | Transcript write-back to Open WebUI | AV3 | 1–1½ days | Not started |
| AV6 | Container, compose service, Caddy route | AV3, AV5a | 1 day | Not started |
| AV7 | Tests and CI | all | 1 day | Not started |

Total: about 7–8 working days. After AV1, AV2, AV3 and AV5a are independent and can be
split between two people.

**Relation to the B tasks:** AV3 + AV5 + AV6 are B10. AV2 is B12, which AV2 decides by
pre-loading the persona. AV4 is B13. B14's spoken-session metrics build on AV4's
timestamps. B2 (consent) must still be signed off **before any real student session**.

### 16.3 Task specifications

#### AV0 · Import the demo, pinned — ½ day · **Done**
**Why.** Every later change should be a reviewable diff against upstream, not a mystery.
**Steps.** Copy the demo verbatim into `avatar_app/` at commit
`daa8cc92ab34c75c0bbed201c94fb98360da2179`, keep its MIT `LICENSE`, install with the
pinned pnpm (`corepack pnpm install --frozen-lockfile`), typecheck.
**Acceptance.** Installs and typechecks unmodified ✅ (commit `4ad6924`). Live sessions
against both services ran on 2026-11-22 (see AV5a) ✅.

#### AV1 · Remove the tutor logic — ½–1 day · **Done**
**Why.** Editing the persona file alone gives "a financial client who starts teaching
Japanese". The tutor behaviour lives in code (upstream `docs/REPURPOSING.md`).
**Steps.**
- Remove the lesson script, the review-break and recap machinery, and the three on-screen
  tools.
- Remove the overlay player, its compositions and the vendored GSAP file, plus the
  `@hyperframes/player` dependency.
- Rewrite the mechanics prompts for role-play: stay in character, never coach, answer
  honestly if sincerely asked "are you an AI?", and check in after silence the way a client
  would.
- Replace the persona with a clearly-placeholder family client.
- Keep the domain-independent parts unchanged: audio path, turn projection, barge-in,
  greeting gate, silence watchdog, teardown.
**Acceptance.**
- Typecheck, web build, and a key-less boot smoke test all pass ✅: `/healthz` answers, and
  `/api/session/start` refuses with a message naming the missing keys.
- GPT-Live accepts the tool-less `session.start` ✅. In live 30-second sessions
  (2026-11-22) the client greeted in character, e.g. *"Hi, I'm Dana. My accountant
  suggested I come in and talk with you."*, with no tutor behaviour ✅.
- **Still to do:** a full conversation with a person speaking. The automated runs streamed
  silence, so no questions were asked.
- The placeholder persona is about 692 tokens against the 16,384-token limit ✅.

#### AV5a · Guardrails that need no identity — ~6 h · **Done**
**Why.** Every session bills from `/api/session/start`. The demo has a 60-second idle
reaper and **no session length cap**: it relies on the LiveAvatar plan's limit.
**Decision (2026-11-22): no limit on simultaneous sessions.** Total spend is bounded by
the per-session cap, the AV5b daily quota and the kill switch. A whole class starting at
once is real practice, not waste. If LiveAvatar's own plan limits simultaneous sessions,
the extra starts fail with the generic message.
**What was built.**
- **Kill switch:** `AVATAR_ENABLED=false` refuses new sessions (HTTP 503, *"Practice
  sessions are turned off right now."*). It is checked before anything that costs money.
- **Hard cap:** `AVATAR_MAX_SESSION_MIN` (default 15, fractions allowed for tests).
  - About one minute before the cap (half the cap when it is very short), an instructions
    append asks the client to wrap up in character.
  - At the cap the server sends the page an `ended` message with a plain-words reason,
    then stops the session and closes the socket.
- **Errors:** students only ever see plain messages; the detail goes to the server log.
  Errors arriving during teardown are logged but not shown, so they cannot overwrite
  "time limit reached".
- **Metrics:** `metrics.ts`, served on a **separate port** (`AVATAR_METRICS_PORT`, default
  9464). Caddy routes `/practice/*` to the main port, so metrics cannot be exposed by
  accident.
  - Metrics reported: `avatar_sessions_active`, `avatar_sessions_started_total`,
    `avatar_session_start_failures_total{reason}`, `avatar_sessions_ended_total{reason}`,
    `avatar_session_seconds_total`, and the `avatar_session_ready_seconds` histogram.
  - The format is written by hand, with no new dependency.
**Acceptance.**
- Kill switch, bad requests and metric counts were checked against a running server ✅.
- Live 30-second session on 2026-11-22 ✅:
  - ready 1.3–2.5 s after start;
  - the client said *"Hey, sorry to jump in, I actually need to head out soon. Do you have
    one last question for me before I go?"*;
  - the session ended at exactly 30 s, with `time_limit` counted and 30 billed seconds
    recorded.
- **Moved to AV6:** the Prometheus scrape job (the container does not exist yet).
**Measured lessons, now in `avatar_app/AGENTS.md`.**
1. A wrap-up worded *"at the next natural pause"* was accepted but **never spoken** while
   the advisor was quiet. GPT-Live speaks unprompted only when told to speak now, as the
   upstream team found with the greeting.
2. Appends take effect only while mic audio streams. With no input audio, GPT-Live rejects
   them with `context_injection_incomplete`. The browser streams silence for this reason.

#### AV2 · Persona loader — 1 day · **Done**
**Why.** The persona must be the same family the text bot plays. GPT-Live's startup
instructions are **fixed once the session starts**, so the whole persona goes in at
`session.start`. Mid-session appends are capped at 500 tokens and are for nudges only. This
is the §5 "RAG trap" answer: no per-turn retrieval in the speech loop.
**What was verified first** (Open WebUI v0.11.4 source, 2026-11-22):
- Every non-admin model endpoint **strips the system prompt** (`params`) from callers
  without write access, students included. Only an admin can read it.
- `GET /api/v1/models/all` (admin only) returns every model with `params.system`,
  `meta.tags` and `meta.knowledge`.
- That endpoint is in the auto-generated API reference, not the hand-written docs. That is
  the same A7 standing as the grader's `chats/all/db`. It is called only from
  `openwebui.ts`.
**What was built** (`personas.ts`, `openwebui.ts`):
- A `PersonaSource` interface with two implementations, chosen by `AVATAR_PERSONA_SOURCE`:
  - **file** (default): `avatar_app/personas/<id>.md`, with front matter (`name`, `voice`,
    `greeting`) and the persona as the body. Lookups go by directory listing, never by
    path, so a crafted id cannot read other files. `_TEMPLATE.md` explains the format.
  - **openwebui:** every active model tagged **`avatar`** (`AVATAR_PERSONA_TAG`) with a
    non-empty system prompt. The professor opts a model in from Workspace → Models; no
    redeploy is needed.
    - An optional `voice:<name>` tag picks the voice.
    - It is read with an **admin** `OPENWEBUI_API_KEY` (the grader's variable names), and
      the list is cached for 30 s.
    - `{{CURRENT_DATE}}`-style variables are filled in. `{{USER_NAME}}` becomes "the
      advisor" until AV3. Unknown variables are removed with a warning.
- **Routes:** `GET /api/personas` lists `{id, name}`, and `/api/session/start` requires a
  `persona_id`.
- **Token budget:** the assembled instructions (persona + role-play directive) are checked
  against 16,384 tokens, and the greeting against 500. Personas over the limit are
  **refused, never truncated**. The estimate is deliberately pessimistic (characters ÷ 3).
- **Voice:** each persona has its own voice. An unknown name falls back to `GPT_LIVE_VOICE`
  with a warning.
- **Test page:** the harness has a persona picker.
**Known limit: attached documents are not loaded.** Only the system prompt is. The log
warns for each persona model that has knowledge attached. If a family's facts live mainly
in its documents, either fold a summary into the system prompt or add a follow-up task
(load small documents into the instructions, or delegate to `/api/chat/completions` for
retrieval).
**Acceptance.**
- 8 unit checks pass ✅: parsing, budget refusal, voice fallback, variable filling, and
  path-trick rejection.
- Against a stand-in Open WebUI server ✅: only tagged, active, prompted models are listed;
  the admin key is sent; the knowledge warning fires.
- File-source personas ran live ✅.
- **Still to do:**
  - point at the real Open WebUI test stack with a tagged model;
  - measure every real course persona against the token limit and write the numbers down;
  - check in a spoken session that the client knows its own facts.

#### AV3 · Login check + WebSocket ticket — 1 day
**Why.** Without it, anyone who can reach `/practice/` can spend our credits. Saving the
transcript also needs to know who the student is.
**Steps.**
- `/api/session/start` requires the student's Open WebUI token and validates it with
  `GET /api/v1/auths/` (the token belongs to Open WebUI; we do not mint our own logins).
- Issue a short-lived HMAC-signed ticket (`AVATAR_TICKET_SECRET`) that binds the student ID
  to the session ID. The WebSocket upgrade rejects a missing, expired or mismatched ticket.
- Keep the student's token in memory only, for the session's lifetime, for AV4.
- All Open WebUI calls go in `openwebui.ts`.
**Acceptance.** Logged-out requests, forged tickets and another student's ticket are all
rejected. The session carries the student ID.
**Verify first.**
- Does the Open WebUI login reach `/practice/` on the same hostname as a cookie, or only in
  localStorage? If localStorage only, the page must pass it explicitly.
- Can the token expire inside a 15-minute session?

#### AV5b · Per-student daily minutes — ~2 h
**Steps.** Count each student's minutes per day (`AVATAR_DAILY_MIN_PER_STUDENT`) in
**Valkey**, which already runs on `ai-net`. An in-memory counter would reset on every
redeploy. Refuse a start over quota with a clear message.
**Acceptance.** Over-quota starts are refused. The count survives a bridge restart.

#### AV4 · Transcript write-back to Open WebUI — 1–1½ days
**Why.** A session that never reaches the grading dashboard is a demo, not a feature (B13).
**What is already verified** (Open WebUI v0.11.4 source, read 2026-11-22):
- `POST /api/v1/chats/new` takes `{chat: {...}}` and stores the chat under the **calling
  user**. It must therefore be called with the **student's** token: an admin key would file
  every transcript under the admin.
- `POST /api/v1/chats/{id}` updates a chat, merging messages by ID. That makes repeated
  incremental saves safe.
- The grader (`extract_chats.py`) reads the flat `chat.messages` list. It pairs strictly
  *user → assistant*: an assistant message with no open question is **skipped**, and a
  second consecutive assistant message is **dropped**. Open WebUI's own UI renders the
  `chat.history` tree.
**Steps.**
- Build the chat document from AV1's projected turns, with **both** `messages` (flat list
  with `timestamp`) and `history` (`messages` map with `parentId` and `currentId`). Title it
  `[Avatar] <family>`.
- **Merge back-to-back same-speaker turns** before saving, and mark cut-off assistant turns
  `[interrupted]`.
- Create the chat after the first full exchange, then save again on each finished turn and
  at stop, so a crash loses at most one turn. Retry on failure, and log loudly if the student
  is not allowed to create chats.
- Keep per-turn timestamps for B14 (hesitation, interruptions, talk-time ratio).
**Acceptance.**
- After a session, the chat appears in the student's Open WebUI sidebar.
- The chat appears in a grading-dashboard extraction with the right number of
  question/answer pairs.
- A few real transcripts are run through the grader to check that spoken questions are
  detected (the detector relies on "?" and keywords).

#### AV6 · Container, compose service, Caddy route — 1 day
**Steps.**
- **Dockerfile:** multi-stage, Node 20, `pnpm build` for the web harness, non-root user.
- **Compose:** an `avatar-app` service on `ai-net` behind a compose **profile** (`avatar`),
  so it is off unless asked for. The image tag is pinned (the CI pin guard rejects
  `:latest`).
- **Caddy:** add `handle_path /practice/*`, and serve the harness and API under that base
  path. Do **not** route the metrics port.
- **Prometheus:** add a scrape job for `avatar-app:9464`. The service is off unless the
  profile is on, so make sure the existing target-down alerting (A20) doesn't fire when the
  avatar isn't deployed.
- **SETUP.md:** document the new environment variables (all in the stack's `.env`, never in
  the repo).
**Acceptance.**
- `docker compose --profile avatar up` serves `https://<host>/practice/`.
- The existing CI compatibility guards still pass.
- A stack without the profile behaves exactly as before.

#### AV7 · Tests and CI — 1 day
**Steps.**
- **Unit tests:** turn stitching, same-speaker merging, Q/A pairing, ticket signing and
  expiry, each limit.
- **Fake upstream servers:** a fake GPT-Live and a fake LiveAvatar, so the whole flow runs in
  CI without paying.
- **Contract test:** a test against the real Open WebUI test stack for the AV4 chat
  document.
- **CI job:** a typecheck-and-test job for `avatar_app/`.
**Acceptance.** CI is green with no real API keys. Bumping the Open WebUI test stack to a
new tag re-runs the contract test.

### 16.4 Settings introduced (environment only, never committed)

| Variable | Task | Purpose |
|---|---|---|
| `LIVEAVATAR_API_KEY`, `LIVEAVATAR_AVATAR_ID` | AV0 | Avatar service |
| `OPENAI_API_KEY`, `GPT_LIVE_MODEL`, `GPT_LIVE_VOICE` | AV0 | Voice model (`GPT_LIVE_MODEL` defaults to `gpt-live-1`) |
| `AVATAR_ENABLED`, `AVATAR_MAX_SESSION_MIN` | AV5a | Kill switch and per-session time limit |
| `AVATAR_METRICS_PORT` (9464) | AV5a | Internal Prometheus port, never routed by Caddy |
| `AVATAR_PERSONA_SOURCE`, `AVATAR_PERSONA_DIR`, `AVATAR_PERSONA_TAG` | AV2 | Where personas come from |
| `OPENWEBUI_BASE_URL` (`http://open-webui:8080`), `OPENWEBUI_API_KEY` (admin) | AV2 | Reading persona models. Same names as the grading service |
| `AVATAR_TICKET_SECRET` | AV3 | WebSocket ticket signing |
| `AVATAR_DAILY_MIN_PER_STUDENT`, `REDIS_URL` | AV5b | Per-student quota in Valkey |

### 16.5 Open items that could change the plan

1. **Are the Open WebUI endpoints we use documented?** We use `auths/`, `chats/new`,
   `chats/{id}` and `models/all`. They are in the auto-generated API reference, not the
   hand-written docs. That is the same standing as the grader's `chats/all/db`, which the
   policy accepts under A7. All of them are confined to `openwebui.ts`.
   - `models/all` needs an **admin** key held by the bridge. Ask the instructor whether
     that is acceptable, or whether personas should stay in files.
2. **Does the Open WebUI login reach `/practice/`, and is it stable for 15 minutes?** (AV3)
3. **Does the grader recognise spoken questions?** (AV4)
4. **Latency through the extra hop on the VM.** The audio path is browser → VM → OpenAI and
   VM → LiveAvatar. Measure it against the §5 budget once AV6 is deployed.
5. **LiveAvatar data terms.** Retention, training use and region for student audio (§14 Q6)
   must be read before B2 sign-off.
