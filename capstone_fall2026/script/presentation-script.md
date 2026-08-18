# Presentation Script — FamilyFinanceChat, Fall 2026 Kickoff

**Deck:** [`../slides/familyfinancechat-fall2026.pdf`](../slides/) — 33 slides
**Runtime:** 5,780 words — about **34 minutes** at a brisk presentation pace (~170 words per
minute) and about **40 minutes** at a slow, measured one (~145 wpm). Budget 35–40 minutes plus
Q&A, and use the short cut below if you have a hard 30-minute slot.
**Intended use:** verbatim narration for an AI presenter (HeyGen-class avatar or TTS voice-over),
and equally usable by a human presenter reading aloud.

---

## How to use this script

- **One block per slide, in order.** The `Slide N` heading matches the page number printed in
  the top-right corner of each slide.
- **The narration is meant to be spoken exactly as written.** It contains no stage directions,
  no bracketed asides, and no bullet fragments — everything between the rules is speech.
- **Numbers and version strings are written the way they should be said**, not the way they are
  written on the slide. "v0.8.12" appears here as *version zero point eight point twelve*, so a
  text-to-speech engine does not read it as a date or a decimal.
- **Acronyms** that should be spelled out letter by letter are hyphenated: *A-B-I*, *C-I*,
  *F-E-R-P-A* is spoken as a word ("ferpa"), *R-A-G* is spoken as a word ("rag").
- `../slides/familyfinancechat-fall2026-notes.pdf` contains short **presenter cues** for a live
  human speaker. Those cues are deliberately *not* the same text as this script — they are
  reminders, this is the speech.
- A machine-readable version of this script, with per-slide timings, is in
  [`presentation-script.json`](presentation-script.json).

**If you need a shorter cut:** dropping slides 9, 16, 30, and 31 removes roughly four minutes
and does not break the argument. Dropping 11, 24, and 29 as well brings it near 25 minutes.
Do not drop slides 8, 20, or 26 — they carry the reasoning the rest of the deck depends on.

---

## Slide 1 — Title

*Approx. 0:25*

Good afternoon. This is FamilyFinanceChat, and this presentation is about what the next
development team is going to build.

There are two halves to it. The first is finishing the job of making this a professional,
maintainable platform. The second is a genuinely new question: whether we can turn a chatbot
into a conversation — with a face, and a voice, that a student can actually talk to.

Let me start with where things stand.

---

## Slide 2 — What we will cover

*Approx. 0:45*

The first part of this presentation is about the platform itself. Where the system stands
today, the architectural lesson that shaped it, the compatibility rules we work under, what we
still have to remove, and how we close the gap to the current version of the software we depend
on.

The second part is the new bet. Why typing is not enough for what this course is trying to
teach, what an avatar-based practice session would actually look like, and how we would build,
cost, and honestly evaluate one.

Before any of that, one rule, because it governs every decision in this deck. We never modify
Open WebUI's core. If a feature cannot be built on a published extension surface, then the
feature changes — not the core. I will come back to why that rule exists, and what it has
already cost this project to learn.

---

## Slide 3 — The problem this platform solves

*Approx. 0:55*

Start with the purpose, because the architecture only makes sense in light of it.

FIN 602 students have to practise client-facing financial advising before they sit across from
a real family. Human role-players do not scale. You get scheduling bottlenecks, you get
inconsistent scenarios, and the quality of feedback depends entirely on who happened to be in
the room.

Instructors have the mirror-image problem. Reading every transcript by hand does not scale
either.

So what we built is unlimited, on-demand A-I role-play against realistic family scenarios,
grounded in the actual course documents. And on top of that, automated scoring of student
questions across seven quality dimensions, plus an Ability, Benevolence, and Integrity trust
rubric — A-B-I. A grading dashboard turns those transcripts into per-student feedback.

That scoring pipeline matters more than it might sound. It is the thing that will let us
measure whether the avatar work in the second half of this presentation is actually worth
doing.

---

## Slide 4 — The system today

*Approx. 1:15*

Here is what actually runs.

Eight containers on a single Google Cloud VM. Open WebUI in the centre handles the chat
interface, routes model calls, and stores chat history. Qdrant holds the vector index over the
course documents, so answers stay grounded in real material instead of being invented. Valkey
handles WebSocket session state. The model itself is served through the OpenAI A-P-I, and file
uploads land in a Google Cloud Storage bucket mounted on the machine.

Across the top is the monitoring stack. A plugin inside Open WebUI pushes per-chat metrics to a
Pushgateway, Prometheus scrapes that alongside container metrics from cAdvisor, and Grafana
draws the dashboards.

Now look at the box on the lower left, in amber. That is the grading dashboard — the tool the
professor actually uses. It is not on the VM. It runs on a developer's laptop.

That single fact is the headline problem for the first half of this semester. We built a
grading system that requires a terminal and an S-S-H session to use. If a professor cannot open
it in a browser, it does not matter how good the scoring is.

---

## Slide 5 — The lesson that shaped this project

*Approx. 1:30*

This slide is the most important piece of history in the deck, and it explains almost every
rule that follows.

On the left is how this project used to work. Open WebUI is open-source software that we run
but do not own. To add two features, the previous architecture replaced three of its internal
source files by mounting our own versions over them inside the container. A twenty-five-hundred
line replacement for the application entry point. A thirty-eight-hundred line replacement for
the configuration module, which existed only so that the first file's imports would resolve. A
custom metrics module. And a custom router injected directly into the vendor's package.

That is roughly six and a half thousand lines of forked code, to get two features.

The cost was severe. The platform was frozen on version zero point six point forty-one, because
every upgrade meant hand-merging someone else's application code. Mysterious startup failures
kept tracing back to fork drift. And users reached one of those custom features through a
browser bookmarklet — a snippet of JavaScript saved as a bookmark, which they had to click to
inject code into the page.

The team before us removed all of it. On the right is what we inherited. Every customisation
now rides a documented plugin or A-P-I surface, and the Dockerfile is a single line.

That is the real achievement of the last team, and it is the reason this semester is possible
at all.

---

## Slide 6 — Where it still hurts

*Approx. 1:10*

The fork is gone. Plenty of other debt is not.

We are pinned to version zero point eight point twelve. Upstream is now on zero point eleven.
That is three minor versions of fixes and features we do not have — and, more importantly,
three sets of database migrations we have deferred rather than avoided.

The metrics plugin has to be pasted into an admin form by hand after every single deployment.
That means our deployment is not reproducible, and when someone forgets, the metrics silently
disappear. Worse, that plugin is written synchronously, with a blocking network call, on a
platform whose backend went fully asynchronous two versions ago.

The grading dashboard is not hosted. There is no continuous integration and no automated smoke
test — past upgrades broke things quietly, and nobody found out until a student did. There is
no alerting, so an outage was discovered passively, from students. There are two retrieval
systems and two copies of the scoring logic. And Grafana is still shipping with the default
username and password.

None of this is exotic. This is the ordinary gap between a system that works and a system you
can actually hand to somebody else.

---

## Slide 7 — Workstream A

*Approx. 0:12*

So, the first workstream. Harden and modernise. This is the part that must not slip.

---

## Slide 8 — The compatibility contract

*Approx. 1:15*

Here is the rule, stated properly.

Open WebUI is a dependency, not a codebase we own. We must be able to pull any published
release of it and have our platform keep working, with no merge, no patch, and no re-fork.

That means every feature we build has to ride a surface that upstream publishes, documents, and
versions. And if a feature cannot be built that way, then the feature gets redesigned or
dropped. The core is never modified. Not temporarily, not read-only, not just this once.

There is a simple test for whether you are following this, and I would like the team to apply
it in every code review this semester. Ask: if upstream cut a release tomorrow, and we ran it
unchanged, would this still work?

If the honest answer needs the words "well, as long as they don't change" — then it is
forbidden. That is the whole test.

This is not bureaucracy. In a few slides you will see that this constraint is what produces the
right architecture for the avatar work, rather than getting in its way.

---

## Slide 9 — Where we are allowed to build

*Approx. 0:55*

This is the menu of legitimate places to build. I am not going to read all ten.

The pattern to internalise is this. If your feature needs to see or change something inside a
request that Open WebUI is already handling, you use a plugin surface — a Filter, a Pipe, or an
Action. If your feature needs its own runtime, its own dependencies, or its own G-P-U, then it
becomes a separate container and it talks to Open WebUI over the public A-P-I.

Two of these matter most for what comes later. A-seven, the public R-E-S-T A-P-I, which is how
the grading tools already read chat data, and how the avatar application will get its answers.
And A-eight, sidecar containers, which is where the avatar application itself will live.

Everything on this list survives an upgrade. That is the point of the list.

---

## Slide 10 — What is forbidden, permanently

*Approx. 1:00*

And this is the other side of it.

Do not mount or copy anything into Open WebUI's package directory — that is the fork, by
definition. Do not add modules to it. Do not build a custom image from modified source. Do not
inject JavaScript into the frontend. Do not run S-Q-L directly against its database — that
schema is private, and it genuinely did change underneath this project. Do not monkeypatch
internals from inside a plugin, which is a fork wearing a plugin costume. Do not depend on
undocumented endpoints.

And the last one is the one I want to underline. Do not pin to an old release just to avoid
doing the migration work. That is exactly how this project ended up frozen on version zero
point six point forty-one in the first place.

Every one of these violations starts the same way, and it always sounds reasonable in the
moment. It's just one file. We'll mount it read-only. We'll clean it up next sprint.

---

## Slide 11 — Audit: what we still have to strip out

*Approx. 1:15*

Now, the good news first. The active production stack is already compliant. No compose file
mounts anything into Open WebUI's source tree. That part of the work is genuinely done.

What remains is peripheral, but some of it is what I would call a loaded gun left in the tree.

Item C-one is the clearest example. The entire vendor fork is still sitting in the repository
under the legacy directory — including a README that walks you step by step through
bind-mounting the application entry point. It is a working recipe for re-creating the exact
mistake we spent a semester undoing. Delete it. Git history preserves it; leave a one-page
note saying what it was and which commit to recover it from.

C-two is a command-line flag in the grading extractor advertised as a direct-database
fallback, whose body is an empty to-do. Delete the flag.

C-four is the metrics plugin I mentioned — synchronous, blocking, and sharing mutable state
across concurrent users. C-seven is Grafana with default credentials. C-ten is an A-P-I key
that was committed to this repository, scrubbed from the files, but not yet rotated at the
provider. Rotate it, and add automated secret scanning so it cannot happen again.

None of these are large jobs. Together they close the door for good.

---

## Slide 12 — Version debt: what three releases cost us

*Approx. 1:20*

Let me be specific about the version gap, because "we should upgrade" is not an argument.

Version zero point nine moved the entire backend from synchronous to asynchronous. Every
database-backed method became something you have to await. That directly affects us: our
metrics plugin is synchronous and makes a blocking network call, and on an asynchronous core
that behaviour does not just fail — it degrades performance for every concurrent user.

Version zero point ten replaced the configuration table with a per-key table. That is a one-way
migration. Once it runs, an older instance pointed at that database fails immediately. There is
no rolling back without restoring a backup.

Version zero point eleven is gentler — additive columns only — but it substantially
reorganised the user interface. Low migration risk, high documentation risk. Every screenshot
and every click-path in our setup guide is now wrong.

We are on zero point eight point twelve. And here is the part I want to land: the longer this
waits, the more expensive it becomes. That compounding is precisely the dynamic that produced
the version zero point six point forty-one freeze. We are watching the same movie start again.

---

## Slide 13 — How we will do the upgrade

*Approx. 1:15*

So here is how we do it, and I want to argue for two habits in particular.

First: snapshot production, and then restore that snapshot into the test stack. Not just take
the backup — restore it. A backup you have never restored is not a backup, it is a hope. And
restoring it gives us the second benefit of upgrading against real production data instead of
an empty database.

Then move one minor version at a time. Zero point eight, to zero point nine, to zero point ten,
to zero point eleven. The migrations run in sequence anyway, and jumping straight to the end
means that when something breaks, you cannot tell which change broke it.

Run the smoke suite at every step, and write the runbook as you go, not afterwards from memory.

Decide explicitly whether to move from SQLite to PostgreSQL during this window, and write down
the decision either way, including if the answer is no.

And second habit: write the rollback procedure before you start the production window, not
during it. When something goes wrong at nine at night, you want to be reading a plan, not
writing one.

We are done when production runs the current release, the runbook exists and was actually
followed, the smoke suite passes, the documentation matches what a user really sees, and the
rollback has been proven rather than assumed.

---

## Slide 14 — CI/CD, and the guard that keeps the fork dead

*Approx. 1:10*

There is no C-I on this project at all today, so we start with the highest value per hour.

Number one is a smoke test on every push. Bring the stack up, wait for the health endpoint,
then the readiness endpoint, send a chat completion, upload a document, and assert that
retrieval actually returns it. That one test would have caught most of what broke silently in
past upgrades.

Number two is the fork guard, on the right. It fails the build if any compose file mounts into
Open WebUI's package, if the image is built from modified source, if Python code opens the
platform's database directly, if a credential scanner finds a secret, or if any image reference
ends in "latest" instead of a pinned version.

Then secret scanning, then linting and unit tests — and I would prioritise unit tests on the
scoring service specifically, because it produces numbers that become student grades. Then
deployment on tag, with an automatic rollback if the smoke test fails afterwards.

Here is why the fork guard matters more than it looks. Everything I said two slides ago about
compatibility is, right now, a document. A policy nobody enforces is a wish. About forty lines
of grep and one workflow file turn that wish into a constraint. Build it in week two.

---

## Slide 15 — Two fixes that change who can use this

*Approx. 1:10*

Two items that change not how well the system works, but who is able to use it.

The first is hosting the grading dashboard. Today a professor has to S-S-H into a cloud VM and
run a shell script. We containerise the backend, build the frontend into static assets, put a
proxy in front with T-L-S and real authentication, and give them a U-R-L. One thing to watch:
the refresh action scores every question through a paid A-P-I, so make it asynchronous, show
progress, and cache the results. A professor clicking a button should not trigger an
unbounded wait or an unbounded bill.

We are done when a professor grades with no terminal, no V-P-N, and no help from a student.

The second is observability. Right now our chat metrics depend on a human pasting Python into
an admin form after every deployment. The fix is not to remember to do it — the fix is to
remove the requirement. Open WebUI can export OpenTelemetry metrics natively. Turn that on,
collect it, and retire the plugin entirely. Then add a readiness check that alerts a person,
and a memory alert as a leading indicator, because that container has approached its ten
gigabyte limit before.

Done means a clean deployment produces metrics with zero manual steps, and an induced outage
pages somebody within five minutes.

---

## Slide 16 — Remove the ambiguity

*Approx. 1:00*

Three places where this codebase currently gives two different answers to the same question.

Scoring is implemented twice — once in JavaScript, once in Python — kept in sync by hand. Pick
Python as canonical, retire the prototype, and add tests that pin the rubric arithmetic, so a
refactor cannot quietly move somebody's grade.

Retrieval is implemented twice. There is the native pipeline students actually use, and a
separate standalone pipeline that nothing in production touches. Either retire it or promote it
behind a Pipelines server so it is reachable properly. Just decide, in writing. And note this:
the citation bug we fixed recently lived in the stack nobody runs. Dead code with live bugs is
a tax you pay in confusion.

The third one is the sleeper. Student questions are currently identified by punctuation and
keyword heuristics. Nobody has measured how accurate that is. If it is seventy percent
accurate, then every grade built on top of it inherits that error invisibly. Measure it against
a hand-labelled sample before trusting anything downstream.

Each of these will eventually produce a wrong answer that nobody can explain.

---

## Slide 17 — Workstream B

*Approx. 0:12*

Now the second workstream, and the new research question. Embodied advising.

---

## Slide 18 — Why typing is not enough

*Approx. 1:10*

FIN 602 exists to prepare students for a room with a real family in it.

Think about what that room actually demands. Listening to someone while they are still
talking. Hearing hesitation in a voice. Holding eye contact through an uncomfortable question
about money. And asking the follow-up out loud, in real time, without a backspace key.

The platform we have teaches the content of good advising well. What it cannot teach is the
performance of it — because typing removes every single one of those pressures. A student can
draft, delete, and reword for ninety seconds before asking, "what happens to the business if
something happens to you?" In a real meeting, they get one attempt, out loud, while somebody is
looking at them.

So here is the hypothesis for this semester. A spoken, face-to-face A-I client produces
meaningfully better preparation than a text chatbot — and we can actually measure whether that
is true, using the scoring pipeline this platform already has.

I want to be precise about the word hypothesis. This track is designed so that it is allowed to
answer no.

---

## Slide 19 — What we are actually building

*Approx. 1:10*

Concretely, here is the experience.

A student opens a link, sees a face, and says out loud: "Hi, I'm going to ask you a few
questions about your family's finances. Is that alright?"

The client on screen looks at them, nods, and answers — in their own voice, in character, with
their own financial situation, drawn from exactly the same course documents the text version
already uses.

If the student interrupts, the client stops talking. That detail is not decoration; it is the
difference between a real prototype and a demo video, and it is one of the things we will
measure providers on. If the student goes quiet, the client waits, and may eventually prompt
them — the way a real, slightly impatient client would.

At the end, the transcript lands in the same grading dashboard the instructor already uses,
scored on the same seven dimensions and the same A-B-I rubric.

And I want to be equally clear about what we are not building. Not a photorealistic digital
human. Not a likeness of any real person. And not a replacement for the text interface. Text
stays. Voice and face are an additional mode, for students who want to rehearse the harder
thing.

---

## Slide 20 — The constraint dictates the architecture

*Approx. 1:05*

Now, the obvious instinct is to put the avatar inside the Open WebUI chat window, right where
the conversation already happens.

That is forbidden. Open WebUI's frontend is an application we do not own. Embedding a video
pane in it means forking that frontend and re-merging it forever — which is precisely the
mistake this project spent an entire semester undoing.

So the constraint decides the design for us. And the design it produces is better than the
instinctive one.

The avatar becomes a companion web application, in its own container. Open WebUI stays the
brain — the model, the retrieval, the knowledge base — and it stays the system of record for
transcripts. Our application talks to it over the public A-P-I.

Look at what that buys us. We can deploy, roll back, and load-test the avatar independently of
the platform students rely on for coursework. It survives every upstream release by
construction. If we decide in December that this is not worth shipping, we delete one container
— there is nothing to un-merge. And students and professors keep the interface they already
know.

This is my case that the compatibility policy is not red tape. It just did our architecture
work for us.

---

## Slide 21 — Reference architecture

*Approx. 1:20*

Here is how the pieces fit.

The student's browser connects over WebRTC. Everything goes through the same reverse proxy that
already fronts the platform, so there is one hostname, one certificate, and one place
authentication happens.

The avatar application manages sessions, turn-taking, and interruption handling, and assembles
the transcript. For every turn it calls Open WebUI's chat completions endpoint — the same
model, the same retrieval, the same course documents. Speech-to-text and text-to-speech sit
alongside it.

Now the part in teal, which is the pattern that actually matters, and it is the thing the
industry converged on during this past year. The avatar renderer is not a processing step we
sit and wait on. It joins the session as its own participant. We stream it the audio, and it
publishes finished, lip-synced audio and video directly to the student. That removes an entire
round trip back through our server, and it is the difference between something that feels like
a video call and something that feels like a laggy video attachment.

One piece of engineering advice attached to this slide. Do not hand-roll the WebRTC layer, the
voice activity detection, or the turn-taking. Use an agent framework built for it — LiveKit
Agents or Pipecat. Both already have avatar provider plugins. Building that layer yourself is
an entire semester, and it is not the question we are trying to answer.

---

## Slide 22 — Latency is the number that decides everything

*Approx. 1:25*

If this project fails, this is the slide that will explain why.

Conversation tolerates about a second. Past roughly two seconds of silence, a person stops
feeling heard and starts feeling processed — and at that point the pedagogical value collapses,
because the student is no longer rehearsing a real interaction.

Here is the budget from the moment the student stops speaking to the first sound coming back.
Detecting that they finished, a couple hundred milliseconds. Speech-to-text, a couple hundred
more. Retrieval. The model's time to first token. Text-to-speech first byte. Then the avatar
render and the network. Add it up and you are somewhere between one second and two point three
seconds, depending almost entirely on choices we control.

Our targets: median under one point two seconds, ninety-fifth percentile under two seconds, and
interruption handled within three hundred milliseconds.

Now the item in red, because it is the counterintuitive one. Our text platform runs a vector
search on every single turn. In a spoken loop, that search sits directly in the critical path,
every turn, forever. The family scenarios are small. Loading the whole persona into the system
prompt once, at the start of the session, will almost certainly cost less total latency than
retrieving on every turn.

And measure each stage separately, starting in week five. A single end-to-end number tells you
that it is slow. It does not tell you what to fix.

---

## Slide 23 — Three strategies, and the one we start with

*Approx. 1:15*

There are three ways to do this.

The first is hosted avatar streaming. There are now a dozen-plus providers — HeyGen, Tavus,
Simli, Anam, and others — and most of them already have plugins for the agent frameworks I
mentioned, so we can put them behind one interface and swap between them. What I would stress
to the team: every latency number those companies publish was measured under their conditions,
on their network. Measuring it under ours is literally the assignment.

The second is self-hosting open source. Whisper for transcription, an open text-to-speech
model, and an open lip-sync renderer. In its favour: no per-minute cost, and no student audio
ever leaving our infrastructure, which is a genuinely strong privacy story for a university.
Against it: it needs a G-P-U we do not currently have, quality and latency lag the hosted
options, and it turns a pedagogy project into an infrastructure project.

And the third is voice only, with no face at all. Here is the thing — Open WebUI already has a
hands-free call mode built in. Speech-to-text and text-to-speech are configuration, not
engineering. That captures a large share of the benefit — speaking out loud, no backspace key,
real-time pressure — at almost no build cost.

So we start with the third one, in week three. If avatars turn out to be too slow, too
expensive, or too fragile, the course still gains spoken practice this semester. That is how
you make an ambitious research track safe to attempt.

---

## Slide 24 — What it costs, and how we stop it running away

*Approx. 1:10*

Money, briefly, because this is the question that gets asked first in every budget meeting.

Take forty students, four sessions each, fifteen minutes a session. That is forty hours of
streaming across a semester. At current rates for real-time avatar streaming, that is somewhere
between two hundred and forty and four hundred and eighty dollars. Add speech services and
model calls and you land in the range of roughly three hundred to six hundred dollars for a
semester. Those prices were checked in August and they move quarterly, so re-derive them at
signup and record the date you checked.

The controls on the right go in from day one, not after the first surprising invoice. A hard
fifteen-minute cap enforced on the server, not in the browser. A per-student session quota. A
billing alert at half the budget. Sessions that terminate when someone closes the tab. And a
kill switch that turns the feature off without a redeployment.

One more thing, on the bottom right, because students consistently get this backwards. Do the
arithmetic before you advocate self-hosting. A G-P-U instance capable of running these models
costs on the order of five hundred dollars a month if you leave it running. At forty hours a
semester, buying beats building — by a lot. Self-hosting wins on privacy, and it wins at scale.
It does not win on cost, here.

---

## Slide 25 — How we will evaluate it honestly

*Approx. 1:20*

This is where this project has an unusual advantage. We already own the outcome measure.

The design is within-subjects and counterbalanced. Each student practises in two or three
modalities — text, voice-only, and avatar — with matched but different family scenarios, in
varying order so we are not just measuring practice effects.

Then we ask the questions in this table. Do students ask better questions when they are
speaking, measured on the same seven-dimension rubric we already run? Does the modality change
the A-B-I trust scores? Do they ask more questions, or fewer? Are the sessions longer?

The row in teal is the one I find most interesting, and it may be the most novel contribution
this team makes. Interruptions and talk-time ratio. Whether a student can let a client finish
talking is arguably a more direct measure of advising skill than anything our current rubric
captures — and it is a measurement that only becomes possible once the conversation is spoken.

Then a short survey on realism, social presence, and anxiety, because an avatar that students
find unsettling is a finding, not a bug to hide.

And a warning. The sample is small and the semester is short. "Twelve students, counterbalanced,
with these effect sizes and these confidence intervals" is a credible result. "Avatars improve
learning" is not. A well-designed study with an honest negative result is the better
deliverable here — and it will be graded that way.

---

## Slide 26 — Phases and gates

*Approx. 1:00*

Four phases, each with a gate.

B-zero, weeks three to five: hands-free spoken practice using the call mode that already ships
in the platform. That ships regardless of everything that follows.

B-one, weeks four to seven: two providers integrated behind one interface, measured on latency,
cost, and interruption behaviour. The gate is to choose one — or to recommend stopping.

B-two, weeks seven to twelve: the companion application, with transcripts written back into
grading. The gate is five students completing a full advising session end to end.

B-three, weeks eleven to fifteen: the comparison study, and a recommendation.

Two things about the gates. First, they are real. "Stop" is an acceptable outcome at every one
of them. "We didn't measure" is not. A research track that cannot fail is not a research track.

Second, in B-two, build the transcript write-back first, not last. An avatar session that never
reaches the grading dashboard is a demo. It is not a feature.

---

## Slide 27 — Ethics, privacy, and consent

*Approx. 1:15*

This has to be settled in week two, before any data exists — not in week thirteen when it
already does.

Student transcripts are education records under F-E-R-P-A. Adding audio does not change that
obligation; it adds a biometric-adjacent identifier on top of it.

So: written informed consent before anything is recorded. What is captured, where it is stored,
how long it is kept, who can see it, and how to opt out — and opting out must not affect a
grade, which means the text path has to remain a genuinely equivalent alternative.

No likeness of any real person. Use the providers' stock avatars under licence. And I will say
this plainly because it comes up every time: do not clone a professor's face or voice as a
joke. That is a deepfake of a colleague, and it will end this project.

Disclose the A-I. We are teaching advising. We are not testing whether students can be fooled.

Minimise the data — prefer keeping the transcript and discarding the audio. Read the vendor's
terms before sending a single student's voice: retention, training use, sub-processors, and
region. For an education platform, that can outweigh a latency advantage.

And ask about institutional review in week two. Approval always takes longer than students
expect.

---

## Slide 28 — Execution

*Approx. 0:12*

Which brings us to execution. Timeline, roles, and risk.

---

## Slide 29 — The semester

*Approx. 1:05*

Sixteen weeks, with both tracks running in parallel — deliberately interleaved so that neither
one starves the other.

Three things to point out. Week two: the C-I skeleton and the fork guard. That is early on
purpose, because everything else is safer once a broken change gets caught automatically.

Week seven: the production upgrade window. Notice that it lands before the avatar work gets
heavy. That ordering is not an accident. If the upgrade slips past the point where the avatar
prototype is consuming everyone's attention, it will not happen at all.

Week eight: the first avatar gate — pick a provider, or stop.

Week twelve is the second gate, five real student sessions. And weeks thirteen through fifteen
are evaluation, documentation, and handoff, with the final presentation in week sixteen.

Note that week fourteen is a short week for Thanksgiving. Plan documentation there, not a
production change.

---

## Slide 30 — Who owns what

*Approx. 0:55*

Five roles.

A platform and DevOps lead who owns the container stack, the upgrade, C-I, the proxy, secrets,
and backups. A backend engineer on the grading service, extraction, and scoring. A realtime
engineer who owns the whole speech-to-model-to-speech-to-avatar chain and, critically, the
latency budget. A frontend and user-experience owner, who has work in both tracks. And a data
and assessment owner, who designs the evaluation and owns the transcript merge.

If the team is smaller than five, one person holds two roles. The roles still exist, and so
does the accountability for them.

And there is one duty that every single member carries regardless of role: review every pull
request against the compatibility policy. The question is fixed, and it is the same question
every time. Which extension surface — and what happens on the next upstream release?

That shared duty is what keeps the policy alive in week thirteen, when there is deadline
pressure and mounting one file read-only starts to sound reasonable.

---

## Slide 31 — What could go wrong

*Approx. 1:05*

The risk register, and I will highlight two rows.

The first is the configuration migration corrupting production data. That is low likelihood and
severe impact, and the mitigation is entirely procedural: rehearse it on a restored snapshot,
verify the restore first, and run it outside a week when students have deadlines.

The second row is the one I actually expect to happen, and I want to name it in advance. The
avatar track absorbs the semester, and the platform work slips.

This is predictable, because the avatar work is exciting and the upgrade work is not. But the
course depends on the platform work, and the avatar work is speculative by design. That is why
the timeline front-loads C-I and the upgrade into the first eight weeks, and it is why gate one
exists in week eight. If a gate fails, we stop and reinforce workstream A. That is a success,
not an embarrassment.

Further down: latency, cost, privacy, provider risk, and knowledge loss at handoff. Note the
second-to-last row — if someone proposes embedding the video pane into the chat page, the
answer is no, and the companion application exists precisely so that the answer can be no
without losing the feature.

---

## Slide 32 — Done means done

*Approx. 0:55*

Here are the acceptance criteria for the semester. Not aspirations — criteria. Every one of
them is verifiable by somebody who was not in the room.

Production runs a current release, upgraded through a written runbook that was actually
followed. A push runs C-I, a tag deploys, and a red build blocks the merge. The fork guard
exists, passes, and there are no re-forking instructions left anywhere in the repository. A
professor grades from an authenticated U-R-L with no terminal. Metrics and alerting work from a
clean deployment with zero manual steps. Scoring and retrieval each have exactly one canonical
implementation, with tests.

A working avatar prototype exists, real students have used it, and a written evaluation backs a
go or no-go call — and I will say again, no-go is a legitimate outcome.

And the last one. The handoff document is good enough that the next team starts in week one
instead of week four. This team benefited enormously from the handoff it received. Pay that
forward.

---

## Slide 33 — Closing

*Approx. 0:45*

I want to end on the point of all this.

The goal is not to finish FamilyFinanceChat. It is to leave a system where the interesting work
is the pedagogy, not the plumbing.

Every hour spent this semester on upgrades, on continuous integration, and on hosting the
grading tool is an hour that a future team does not spend re-learning why the fork was a
mistake. And every hour spent on the avatar track is an attempt to answer a question that
actually matters for this course: whether practising out loud, with a face looking back at you,
makes a better advisor than practising with a keyboard.

We will have a real answer to that by December. It might be no. We will know either way, and we
will know why.

Thank you. I am happy to take questions.
