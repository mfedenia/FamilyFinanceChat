# Presentation Script — FamilyFinanceChat, Fall 2026

**Deck:** [`../slides/familyfinancechat-fall2026.pdf`](../slides/) — 10 slides
**Runtime:** 783 words — **about 5 minutes**: 4:54 at a brisk 160 words per minute, 5:13 at a
measured 150. Add a beat between slides and it lands on five.
**Intended use:** verbatim narration for an AI presenter (HeyGen-class avatar or TTS voice-over),
and equally usable by a human presenter reading aloud.

---

## How to use this script

- **One block per slide, in order.** The `Slide N` heading matches the page number printed in
  the top-right corner of each slide.
- **The narration is meant to be spoken exactly as written.** No stage directions, no bracketed
  asides — everything between the rules is speech.
- **Numbers are written the way they should be said**, not the way they appear on the slide, so
  a speech engine does not read them as dates or decimals.
- `../slides/familyfinancechat-fall2026-notes.pdf` has short **presenter cues** for a live human
  speaker. Those cues are deliberately different text — they are reminders, this is the speech.
- A machine-readable version with per-slide timings is in
  [`presentation-script.json`](presentation-script.json). Regenerate it with
  `python3 build-script-json.py` after any edit here — this file is the source of truth.

**This is a five-minute overview, and it is meant to stay that way.** Every detail someone asks
about lives in the reports — [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md),
[`../COMPATIBILITY_POLICY.md`](../COMPATIBILITY_POLICY.md),
[`../AVATAR_TRACK.md`](../AVATAR_TRACK.md) — with a Q&A prep sheet in
[`../reference/system-facts.md`](../reference/system-facts.md). If a slide starts growing, move
the material into a report instead.

---

## Slide 1 — Title

*Approx. 0:14*

This is FamilyFinanceChat. In five minutes: what it is, what is wrong with it, and what this
team is going to do about it.

Two jobs this semester. Make the platform solid, and give the client a face.

---

## Slide 2 — What we have

*Approx. 0:25*

Here is the whole system.

A student practises financial advising against an A-I client — a realistic family, grounded in
the actual course documents so it does not invent its facts. Every question the student asks is
scored automatically, and the instructor sees the results in a dashboard.

The point is scale. Unlimited practice, at any hour, without booking a human role-player.

---

## Slide 3 — It works. Three things hurt.

*Approx. 0:32*

It works today. Students use it. But three things hurt.

We are three versions behind the open-source software this is built on, and every month we
wait, catching up gets harder.

The professor's grading tool runs on a laptop. Using it takes a terminal and a developer, which
is the biggest barrier to anyone adopting this.

And every deployment needs manual steps that people forget — and things break quietly.

That is the gap between a system that works and a system you can hand to someone else.

---

## Slide 4 — The one rule

*Approx. 0:35*

One rule governs everything, because this project learned it the hard way.

We never modify the core of Open WebUI — the open-source platform underneath us.

The team before us inherited six and a half thousand lines of that project's code, copied and
modified locally. It froze us on one old version, because every upgrade meant merging someone
else's software by hand.

They removed all of it. Today our version is one line long. We just run their release.

So the test for everything we build is simple. Could we take their next release tomorrow,
unchanged, and still work?

---

## Slide 5 — First job: make it solid

*Approx. 0:28*

The first job is unglamorous, and it is what the course depends on.

Upgrade to the current version — carefully, one step at a time, because these upgrades change
the database and you cannot undo that.

Give the professor a web address instead of a laptop.

Automate deployment and testing, so a bad change is caught before it reaches students.

And delete the last of that old copied code, so nobody can bring it back by accident.

---

## Slide 6 — Second job: give the client a face

*Approx. 0:31*

Now the new part.

Today the student types to the client. We want them to talk to it — to a face that listens and
answers out loud, in character.

Here is why that matters. This course prepares students for a room with a real family in it. In
that room you ask the hard question about money out loud, once, while somebody is looking at you.

You cannot rehearse that with a backspace key.

---

## Slide 7 — How, without breaking the rule

*Approx. 0:31*

So how do we add a talking face without breaking that rule?

We do not put it inside the platform. We build it beside the platform — a separate application
in its own container. The student speaks to it, it asks Open WebUI for the answer, and Open
WebUI stays untouched.

Same brain, same documents, same grading. And if this turns out not to be worth shipping, we
delete one container.

The rule did not get in the way of the design. It chose the better design for us.

---

## Slide 8 — Three numbers decide whether it works

*Approx. 0:33*

Three numbers will decide whether this is real.

Speed. The client has to start answering in about one point two seconds. Slower, and it stops
feeling like a conversation.

Cost. Roughly four hundred dollars a semester for a class of forty, with hard caps so it cannot
run away from us.

Permission. Voice and video of students are education records. Consent and deletion get settled
in week two, before anything is recorded.

If any one of those fails, the answer is no — and no is an acceptable answer.

---

## Slide 9 — The semester

*Approx. 0:28*

Four checkpoints.

By week five, students can talk to it — voice only, no face yet. By week seven, the platform is
upgraded. By week eight, we pick an avatar provider or decide to stop. By week twelve, five
students complete full spoken sessions, graded like any other.

The voice-only version ships in week five either way. That is what makes the ambitious part
safe to attempt.

---

## Slide 10 — What finished looks like

*Approx. 0:33*

Here is what finished looks like.

A professor grades from a web address, with no laptop and no help from a student. Deployments
are automatic, and a bad change is caught before it ships. And we know — with evidence from the
scoring system we already have — whether talking to an avatar makes better advisors.

That last one might come back negative. We publish it either way.

Because the goal is not to finish this system. It is to leave one where the interesting work is
the teaching, not the plumbing.

Thank you — happy to take questions.
