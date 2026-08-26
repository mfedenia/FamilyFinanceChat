# Capstone Fall 2026 — Project Briefing Package

This folder is **not part of the FamilyFinanceChat application.** Nothing in here is built,
deployed, imported, or tested by the rest of the repository. It lives here because it is
entirely *about* this repository — it is the briefing package that tells the next development
team what to build and why.

It contains three deliverables and their supporting material: a detailed work plan written for a
four-person development team, a Beamer slide deck, and a verbatim presentation script intended
to drive an AI presenter.

**The plan is the detailed document; the deck is the five-minute summary.** `PROJECT_PLAN.md`
carries the task specifications, per-person assignments, and a dated agenda for every weekly
check-in (Tuesdays, 15 September – 1 December 2026).

**The deck is deliberately thin — 10 slides, five minutes, one idea per slide.** It exists to
explain the project, not to document it. All the detail lives in the reports, and the deck
should stay that way: if a slide starts growing a table, the material belongs in a report
instead.

---

## What is here

```
capstone_fall2026/
├── README.md                      you are here
├── PROJECT_PLAN.md                the work plan: 43 task specs, 4 roles, 11 dated meetings
├── COMPATIBILITY_POLICY.md        the binding "never fork Open WebUI" constraint + repo audit
├── AVATAR_TRACK.md                the new research track: spoken, embodied advising
├── slides/
│   ├── familyfinancechat-fall2026.tex     Beamer source (self-contained theme)
│   ├── familyfinancechat-fall2026.pdf      10 slides / 5 minutes
│   ├── familyfinancechat-fall2026-notes.pdf  slides + presenter cue cards
│   ├── familyfinancechat-fall2026-build.tex  the same deck, 36 cumulative build steps
│   ├── familyfinancechat-fall2026-build.pdf  one page per build step
│   ├── build-heygen-pptx.py                PDF pages -> PowerPoint + speaker notes
│   ├── familyfinancechat-fall2026-heygen.pptx  the HeyGen deliverable, 36 slides / 4 min
│   └── build.sh                            build helper
├── script/
│   ├── presentation-script.md      verbatim narration, one block per slide  ← source of truth
│   ├── presentation-script.json    same script, machine-readable, for an AI presenter
│   └── build-script-json.py        regenerates the JSON from the Markdown
└── reference/
    └── system-facts.md             verified facts, sources, and Q&A prep
```

---

## Read in this order

| If you are… | Start with |
|---|---|
| a developer joining the team | [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §1–5, then your own tasks in §6–7 |
| working on the avatar track | [`AVATAR_TRACK.md`](AVATAR_TRACK.md), plus tasks B1–B18 in the plan |
| reviewing a pull request | [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) §3–4 — the allowed and forbidden surfaces |
| presenting this | [`script/presentation-script.md`](script/presentation-script.md) alongside the deck PDF |
| preparing for questions | [`reference/system-facts.md`](reference/system-facts.md) |
| running a weekly check-in | [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §8 — agendas for all 11 meetings |

The three documents are meant to be read together. `PROJECT_PLAN.md` is the plan;
`COMPATIBILITY_POLICY.md` is the constraint that shapes it; `AVATAR_TRACK.md` is the new work
that constraint turned out to improve.

---

## Building the slides

Requires `latexmk` and `pdflatex` with `beamer`, `tikz`, `xcolor`, `helvet`, and `microtype`.
No external Beamer theme is needed — the theme is defined inside the `.tex` file, so it
compiles on a stock TeX Live or a minimal TinyTeX install.

```bash
cd slides && ./build.sh          # slides only
cd slides && ./build.sh notes    # slides + presenter-notes PDF
cd slides && ./build.sh clean    # remove build artefacts
```

Both PDFs are committed, so you do not need LaTeX just to present.

---

## Using the script with an AI presenter

`script/presentation-script.md` is the source of truth. `presentation-script.json` is generated
from it — **edit the Markdown, then regenerate**, never the other way round:

```bash
cd script && python3 build-script-json.py
```

The JSON gives one narration string per slide:

```json
{
  "slide": 8,
  "title": "Three numbers decide whether it works",
  "estimated_seconds": 33,
  "word_count": 89,
  "narration": "Three numbers will decide whether this is real. Speed. …"
}
```

Feed each `narration` to the avatar or TTS engine and advance the deck on slide boundaries.
Numbers and acronyms in the narration are already written the way they should be pronounced
(`one point two seconds`, `A-I`), so a speech engine does not read them as dates or decimals.

**Note the division of labour:** the `\note{}` blocks inside the `.tex` are short cue cards for
a *live human* presenter, and are deliberately different text. The script folder holds the
speech.

---

## Making the HeyGen video

HeyGen takes a **PowerPoint file**, renders one video segment per slide, and reads each slide's
**speaker notes** as the script for that segment. So the deck it receives is not the ten-slide
deck — it is a build-step version where each revealed element is its own slide, and the notes on
that slide say only what belongs to what has just appeared.

```bash
cd slides && ./build.sh video
```

That produces **`slides/familyfinancechat-fall2026-heygen.pptx`** — 36 slides, 596 words of
narration, just under **four minutes**. The pipeline is:

`familyfinancechat-fall2026-build.tex` → `latexmk` → a 36-page PDF (one page per overlay step) →
`pdftoppm` → one 1920×1080 PNG per page → `python-pptx` → a 16:9 PPTX whose slides are full-bleed
images with the narration in the notes.

**Three files, three jobs — keep them straight:**

| File | For | Length |
|---|---|---|
| `familyfinancechat-fall2026.tex` | a live human presenting | 10 slides / 5 min |
| `script/presentation-script.md` | that human's speech, or a plain voice-over | 801 words |
| `familyfinancechat-fall2026-build.tex` + `build-heygen-pptx.py` | the HeyGen video | 36 steps / 596 words / 4 min |

The video narration is a **separate, shorter script** and lives in the `NARRATION` table inside
`build-heygen-pptx.py`. Edit it there. The build refuses to run if the number of narration
entries does not equal the number of PDF pages, which is what stops the speech and the slides
drifting apart — if you add an overlay step, you must add its line.

**Rules that keep the video from looking broken:**

- **Cumulative, never re-flowing.** Each step adds; nothing already on screen moves. Every frame
  fixes its own bounding box (`\useasboundingbox`, `\onslide` rather than `\only`, `\item<n->`
  rather than `\item<only@n>`). A line that shifts a few pixels between steps reads as a glitch.
- **`\setbeamercovered{invisible}`** — no ghosted preview of the next bullet.
- **Written for the ear, not the eye.** An avatar reads this aloud with no idea what any of it
  means, so the narration is spelled the way it should *sound*: numbers as words
  (`one point two seconds`); acronyms hyphenated so the letters come out separately (`A-I`,
  `U-I`, and `S-Q-L` or `A-P-I` for anything added later); product names respelled as spoken
  (`Family Finance Chat`, `Open Web U-I` — run-together capitals get read as one invented word);
  dates said in full (`the twentieth of October`, `the middle of November`); and no em dashes or
  semicolons, because a full stop is an unambiguous pause and an em dash is not. Only the speech
  is respelled — the slides still show the real spelling. `check_tts()` enforces the mechanical
  half of this on every build and **fails** rather than warns, because a mispronounced word is
  not something you notice until the video is rendered and paid for.
- **Contractions on purpose.** "We're" and "doesn't" are what a person says; "we are" and
  "does not" are what a document says, and an avatar reading the document version sounds like a
  kiosk.
- **Under four minutes.** The build prints the estimate at 150 words per minute and warns if it
  goes over. If you add narration, cut some elsewhere.
- **Under fifty slides.** HeyGen charges per rendered segment, and the deck gets unwieldy past
  that. 36 is the current count.

**The title slide names the director**, in both decks — they are the same deck in two renderings,
so a change to one belongs in the other. In the video it is a two-step build: the byline appears
first, while the narration introduces Professor Mark Fedenia and the spring Wealth Management &
Financial Planning class, and the `make it solid, give it a face` line lands on the second step,
exactly as it is spoken. The narration introduces him in the **third person**, on the assumption
that the avatar is a presenter rather than a stand-in for him; if you pick an avatar meant to be
him, rewrite entry `(1, 1)` in the first person.

Upload the `.pptx` to HeyGen, pick the avatar and voice, and check two things on the first
render: that the notes came through as the script for the right slide, and that the 16:9 frames
are not letterboxed.

---

## Keeping this current

This package makes claims about the state of the repository and about third-party products.
Two categories go stale at different speeds:

- **Repository claims** (versions in `docker-compose.yml`, the audit findings in
  `COMPATIBILITY_POLICY.md` §5) — re-verify against the tree before presenting, and tick off
  audit items as they are closed.
- **External claims** (Open WebUI's current release, avatar provider pricing and latency) —
  checked August 2026, and moving quickly. `reference/system-facts.md` lists every source.

Every number in the deck traces to one of those two places. If you change a fact, change it in
the document first, then in the deck, then in the script.
