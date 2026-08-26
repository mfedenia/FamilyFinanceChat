# Presentation Script — FamilyFinanceChat, Fall 2026

**Deck:** [`../slides/familyfinancechat-fall2026.pdf`](../slides/) — 10 slides
**Runtime:** 703 words — **about four and a half minutes**: 4:24 at a brisk 160 words
per minute, 4:41 at a measured 150.
**Intended use:** a live human presenter reading aloud, or a plain voice-over of the ten-slide
deck.
**Not the video script.** The HeyGen video runs on a shorter, separately maintained script — 27
build steps, 573 words — in the `NARRATION` table of
[`../slides/build-heygen-pptx.py`](../slides/build-heygen-pptx.py). Do not feed this file to
HeyGen: it is written for ten slides, not for the build steps the video advances through. Both
tell the same story; this one has room to breathe.

---

## How to use this script

- **One block per slide, in order.** The `Slide N` heading matches the slide's position in the
  deck.
- **The narration is meant to be spoken exactly as written.** No stage directions, no bracketed
  asides — everything between the rules is speech.
- **It is written in the first person, as Mark.** If a student or a team member is presenting,
  change "I" to "the sponsor" on slide 1 and slide 10; nothing else needs touching.
- **Numbers are written the way they should be said**, not the way they appear on the slide, so
  a speech engine does not read them as dates or decimals.
- `../slides/familyfinancechat-fall2026-notes.pdf` has short **presenter cues** for a live
  speaker. Those cues are deliberately different text — they are reminders, this is the speech.
- A machine-readable version with per-slide timings is in
  [`presentation-script.json`](presentation-script.json). Regenerate it with
  `python3 build-script-json.py` after any edit here — this file is the source of truth.

**This is a five-minute pitch, and it is meant to stay that way.** Every detail someone asks
about lives in the reports — [`../PROJECT_PLAN.md`](../PROJECT_PLAN.md),
[`../COMPATIBILITY_POLICY.md`](../COMPATIBILITY_POLICY.md),
[`../AVATAR_TRACK.md`](../AVATAR_TRACK.md) — with a Q&A prep sheet in
[`../reference/system-facts.md`](../reference/system-facts.md), and the one-page version students
see is [`../CS620_Project_fall2026.pdf`](../CS620_Project_fall2026.pdf).

---

## Slide 1 — Title

*Approx. 0:23*

Hi. I'm Mark Fedenia. I teach finance here at Wisconsin, and I want to show you something my
students use every week, and where I would like to take it next.

It is called Family Finance Chat. Right now it is a chat box. By the end of this project, I want
it to be a face you talk to.

---

## Slide 2 — What it is

*Approx. 0:42*

Here is the problem I am trying to solve. My students are learning to advise families about
money, and the hard part of that job is not the arithmetic. It is sitting across from people and
asking uncomfortable questions well.

You cannot learn that from a textbook, and I cannot hire forty families to practise on.

So we built them somebody to practise on. An A-I family, with a real financial situation taken
from the course material, so it does not invent its own facts. And because every question a
student asks is scored automatically, I can see how the whole class is doing without reading
forty transcripts.

---

## Slide 3 — How it works today

*Approx. 0:32*

Today the whole thing lives in a browser, on a cloud machine, built on an open source platform
called Open Web U-I. Underneath there are about eight moving parts, and one language model doing
the talking.

I want to be clear that it works. Students use it every week, and it is a real part of the
course.

That is exactly why the next step is worth taking seriously. You are not rescuing something
broken. You are improving something people depend on.

---

## Slide 4 — Two ways to make it better

*Approx. 0:20*

There are two things I want to fix.

The first is not glamorous. We have drifted behind the software we are built on, and every
update takes somebody who knows where all the pieces are.

The second is the one I am excited about. I want the student to stop typing.

---

## Slide 5 — One: solid ground

*Approx. 0:23*

So, the groundwork first.

Get current with the platform, and stay current, so that upgrading stops being frightening and
becomes routine.

Put the grading on the web, where any instructor can open it, instead of on one laptop in my
office.

And make deployment automatic, so that improving this system does not depend on one person
remembering the steps.

---

## Slide 6 — Two: the leap

*Approx. 0:21*

Now the interesting part.

Right now a student types a question, waits, and reads an answer. That is nothing like the job
they are training for.

I want them to say it out loud. To a face that listens, waits, and answers back in character —
as the client, not as a chatbot.

---

## Slide 7 — Why that matters

*Approx. 0:20*

Here is why I care about that difference.

In a real meeting you get one shot. You ask the hard question out loud, once, while a family
watches your face and decides whether they trust you.

You cannot practise that with a keyboard. There is no backspace key in that room.

---

## Slide 8 — The project: evaluate, then build

*Approx. 0:39*

So here is the actual project.

There are a dozen companies now selling real time avatars, and the field changes every month.
None of them have been tested for anything like this. Somebody has to find out which one holds
up, honestly, with numbers.

Speed matters most. If it takes longer than about a second to answer, it stops feeling like a
conversation and starts feeling like a machine.

Then cost, per student, per semester, with a ceiling we can live with.

And permission. A student's voice and face are protected records, and that gets settled before
anyone records anything.

---

## Slide 9 — How it bolts on

*Approx. 0:24*

Whatever wins has to bolt on beside what we already have, not replace it.

A separate service that handles the talking, and asks the existing system for the answer. Same
grounding in the course material, same scoring, same everything the course already depends on.

And if it turns out not to be worth it, we switch it off and lose nothing.

---

## Slide 10 — What we are after

*Approx. 0:30*

What I want by the end of the term is students actually talking to this thing. Not a demo. A
working part of the course.

And I want to know whether it makes them better at the job — measured against the scoring we
already have, not guessed at.

It might turn out that it does not, and that is a finding worth publishing too.

If that sounds like a semester well spent, come build it with us.
