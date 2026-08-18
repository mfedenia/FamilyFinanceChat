# Workstream B — Embodied Advising

### Turning FamilyFinanceChat from a chatbot into a conversation

**Status:** research and prototyping track with defined gates and a defined kill switch
**Owner:** realtime / avatar engineer, with frontend and data/assessment support
**Term:** Fall 2026, Weeks 3–15
**Governed by:** [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) — no exceptions

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

**Measure this in Week 5, before choosing a provider.** Instrument every stage separately —
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

**Build B0 in Week 3 and put it in front of students immediately.** It de-risks the whole
track: if avatars prove too slow, too costly, or too fragile, the course still gains spoken
practice this semester.

---

## 7. Integration with grading — the part that is easy to forget

An avatar session that never reaches the grading dashboard is a demo, not a feature. The
instructor's workflow — extract, score on seven dimensions, ABI, per-student feedback — must
work for spoken sessions exactly as it does for typed ones.

**Open technical question the team must answer in Week 9:** does a chat created through
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

## 8. Cost model — build it in Week 4 with real numbers

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

### B0 — Voice, no face (Weeks 3–5)

Configure Open WebUI STT and TTS; select a voice per family persona; run a full advising
session hands-free; measure baseline round-trip latency; demo to the instructor in Week 5;
document the setup so a professor can change voices without a developer.

**Exit:** spoken practice is available to students. *This ships no matter what happens next.*

### B1 — Provider bake-off (Weeks 4–7)

Stand up an orchestration harness (LiveKit Agents or Pipecat) behind a provider-agnostic
interface. Integrate **two** providers. Measure, per provider, on identical scripted
conversations: stage-by-stage latency (p50/p95), barge-in responsiveness, lip-sync quality,
audio quality, cost per minute measured not quoted, SDK maturity, failure behaviour on a
dropped connection, and whether persona voice and appearance can be configured per family.

**Gate 1 (Week 8): choose a provider, or recommend stopping.** Both are acceptable outcomes;
neither is acceptable without data.

### B2 — Companion prototype (Weeks 7–12)

Build `avatar-app` as a container: session creation and persona selection; a browser client
with the video pane, a live captions track, and a visible end-session control; turn-taking
with barge-in; the family scenario loaded as a system prompt at session start; transcript
write-back into Open WebUI; behind the reverse proxy with the same authentication as the rest
of the platform; hard session caps enforced server-side.

**Gate 2 (Week 12): five students complete a full advising session end to end, and the
transcripts appear in the grading dashboard, scored.** No transcript, no gate.

### B3 — Evaluate and recommend (Weeks 11–15)

Run the counterbalanced sessions; analyse against the existing rubric; survey participants;
write the report; deliver a recommendation with cost, latency, and outcome evidence, plus a
concrete sizing estimate for what production deployment to a full cohort would require.

**Exit:** a decision a professor can act on without re-doing the analysis.

---

## 11. Ethics, privacy, and consent

Voice and video raise obligations that text does not. Settle these in **Week 2**, before any
recording, not in Week 13 when the data already exists.

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
  needs IRB review. Ask in Week 2; approval takes longer than students expect.

---

## 12. Risks specific to this track

| # | Risk | Mitigation |
|---|---|---|
| B-R1 | Latency lands at 3+ seconds and the experience feels dead | measure in Week 5; B0 ships regardless; consider dropping per-turn RAG |
| B-R2 | The uncanny valley — students find the avatar off-putting rather than engaging | measure it in the survey; stylised or clearly-synthetic avatars often outperform near-photoreal ones |
| B-R3 | Costs run away during testing | hard caps, quotas, and a billing alarm from day one, not after the first bill |
| B-R4 | The provider deprecates or reprices mid-semester | provider-agnostic interface first; keep the runner-up integrated and warm |
| B-R5 | Avatar sessions never reach the grading dashboard | make the write-back the *first* thing B2 builds, not the last |
| B-R6 | Someone proposes embedding the video pane in the Open WebUI chat page | policy F3/F4 — refuse; the companion app exists for this reason |
| B-R7 | The track absorbs the whole semester and Workstream A slips | Gate 1 and Gate 2 are real; if a gate fails, stop and reinforce A |

---

## 13. Deliverables

| # | Deliverable | Week |
|---|---|---|
| D1 | Requirements note: what a spoken advising session must do | 3 |
| D2 | Working voice-only mode, documented for a non-developer | 5 |
| D3 | Latency instrumentation on the existing Grafana stack | 5 |
| D4 | Provider bake-off report with measured data and a recommendation | 8 |
| D5 | Cost model with live pricing, usage assumptions, and enforced caps | 8 |
| D6 | `avatar-app` container, source, and deployment documentation | 12 |
| D7 | Transcript write-back verified end to end into the grading dashboard | 12 |
| D8 | Evaluation report: methodology, data, findings, limitations | 15 |
| D9 | Go/no-go recommendation with a production sizing estimate | 15 |
| D10 | Handoff section in `HANDOFF.md` for the next team | 15 |

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
