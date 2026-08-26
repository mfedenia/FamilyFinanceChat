# Capstone Fall 2026 — Project Briefing Package

This folder is **not part of the FamilyFinanceChat application.** Nothing in here is built,
deployed, imported, or tested by the rest of the repository. It lives here because it is
entirely *about* this repository — it is the briefing package that tells the next development
team what to build and why.

It contains the one-page project description students choose from, a work plan for the team that
picks it, a slide deck, and the narration that turns that deck into a video.

---

## What is here

```
capstone_fall2026/
├── README.md                      you are here
├── CS620_Project_fall2026.docx    the one-pager students choose from  ← the pitch
├── CS620_Project_fall2026.pdf     the same, for handing out
├── PROJECT_PLAN.md                the work plan: 43 task specs, 4 roles, 11 dated meetings
├── COMPATIBILITY_POLICY.md        the binding "never fork Open WebUI" constraint + repo audit
├── AVATAR_TRACK.md                the research track: spoken, embodied advising
├── slides/
│   ├── familyfinancechat-fall2026.tex        ONE source, three outputs
│   ├── familyfinancechat-fall2026.pdf         10 slides, for a live presenter
│   ├── familyfinancechat-fall2026-notes.pdf   the same, plus presenter cue cards
│   ├── familyfinancechat-fall2026-build.pdf   27 pages, one per build step
│   ├── familyfinancechat-fall2026-heygen.pptx the video deliverable  ← upload this
│   ├── build-heygen-pptx.py                   narration + PDF pages → PowerPoint
│   └── build.sh                               build helper
├── script/
│   ├── presentation-script.md      the five-minute script a human reads from
│   ├── presentation-script.json    same, machine-readable
│   └── build-script-json.py        regenerates the JSON from the Markdown
└── reference/
    └── system-facts.md             verified facts, sources, and Q&A prep
```

---

## Read in this order

| If you are… | Start with |
|---|---|
| a student deciding whether to pick this | [`CS620_Project_fall2026.pdf`](CS620_Project_fall2026.pdf) |
| a developer joining the team | [`PROJECT_PLAN.md`](PROJECT_PLAN.md) §1–5, then your own tasks in §6–7 |
| working on the avatar track | [`AVATAR_TRACK.md`](AVATAR_TRACK.md), plus tasks B1–B18 in the plan |
| reviewing a pull request | [`COMPATIBILITY_POLICY.md`](COMPATIBILITY_POLICY.md) §3–4 |
| presenting this live | [`script/presentation-script.md`](script/presentation-script.md) with the deck PDF |
| making the video | **Making the HeyGen video**, below |
| preparing for questions | [`reference/system-facts.md`](reference/system-facts.md) |

---

## The deck: one source, three outputs

`slides/familyfinancechat-fall2026.tex` is the only slide source. It used to have a twin that
had to be kept in sync by hand; it does not any more.

```bash
cd slides
./build.sh          # handout mode: overlays collapse → 10 slides for a live presenter
./build.sh notes    # the same, plus the presenter cue-card PDF
./build.sh video    # \VIDEO: one page per build step, avatar column, then the .pptx
./build.sh clean
```

Requires `latexmk`/`pdflatex` with beamer, tikz, xcolor, helvet and microtype. No external
Beamer theme — the theme is inside the `.tex`, so it compiles on a stock TeX Live or TinyTeX.
The video target additionally needs `pdftoppm` (poppler) and python3; it creates a throwaway
virtualenv in `.venv-pptx` for python-pptx. All PDFs and the `.pptx` are committed, so you do
not need LaTeX just to present or to upload.

---

## Making the HeyGen video

HeyGen takes a **PowerPoint**, renders one video segment per slide, and reads each slide's
**speaker notes** as the script for that segment. So the deck it receives is not the ten-slide
deck — it is a build-step version where each revealed element is its own slide, with notes that
say only what belongs to what has just appeared.

`./build.sh video` produces **`slides/familyfinancechat-fall2026-heygen.pptx`** — 27 slides,
573 words, about **3:49**. The pipeline:

`familyfinancechat-fall2026.tex` + `\VIDEO` → `pdflatex` → a 27-page PDF (one page per overlay
step) → `pdftoppm` → one 1920×1080 PNG per page → `python-pptx` → a 16:9 PPTX whose slides are
full-bleed images with the narration in the notes.

### The avatar column

**In `\VIDEO` mode the text block stops at 10.5 cm of the 16 cm page**, leaving a clear portrait
column down the right-hand side for HeyGen to composite the presenter into. Nothing — title,
body, diagram, page number — may cross that line, or the avatar will sit on top of the content.
Use the `\contentwidth` length rather than hard-coding a width, and check a rendered page in
`build-png/` after any layout change.

### Rules that keep the video from looking broken

- **Cumulative, never re-flowing.** Each step adds; nothing already on screen moves. `\onslide`
  (which reserves space) rather than `\only` (which does not), `\item<n->` rather than
  `\item<only@n>`, and a fixed `\useasboundingbox` on every tikzpicture. A line that shifts a few
  pixels between steps reads as a glitch.
- **`\setbeamercovered{invisible}`** — no ghosted preview of the next bullet.
- **Keep every `\\` at the top level of a tikz node's text.** A line break nested inside a brace
  group breaks TikZ's path parser with a `\tikzscope@linewidth` error that points at
  `\end{frame}`, nowhere near the actual cause.
- **The narration is not a reading of the slides.** The slides show; the speech explains. If a
  line could be deleted and nothing lost because the slide already says it, it is the wrong line.
- **Written for the ear, not the eye.** An avatar reads this aloud with no idea what any of it
  means, so the narration is spelled the way it should *sound*: numbers as words
  (`about a second`); acronyms hyphenated so the letters come out separately (`A-I`, `U-I`, and
  `S-Q-L` or `A-P-I` for anything added later); product names respelled as spoken
  (`Family Finance Chat`, `Open Web U-I` — run-together capitals get read as one invented word);
  and no em dashes or semicolons, because a full stop is an unambiguous pause and an em dash is
  not. Only the speech is respelled — the slides still show the real spelling. `check_tts()`
  enforces the mechanical half of this on every build and **fails** rather than warns, because a
  mispronounced word is not something you notice until the video is rendered and paid for.
- **Contractions on purpose.** "We're" and "doesn't" are what a person says; "we are" and
  "does not" are what a document says, and an avatar reading the document version sounds like a
  kiosk.
- **Under four minutes, under fifty slides.** The build prints the estimate at 150 words per
  minute and warns if it runs over. HeyGen charges per rendered segment; 27 is the current count.

The narration lives in the `NARRATION` table in `build-heygen-pptx.py`, one entry per build step.
**The build fails if the number of entries does not equal the number of PDF pages** — that is
what stops the speech and the slides drifting apart when someone adds an overlay step later.

Upload the `.pptx`, pick the avatar and voice, place the avatar in the right-hand column, and
check two things on the first render: that the notes came through as the script for the right
slide, and that the frames are not letterboxed.

---

## Three scripts, three jobs

| File | For | Length |
|---|---|---|
| `slides/…-notes.pdf` (`\note{}` in the `.tex`) | cue cards for a live presenter | a line per slide |
| `script/presentation-script.md` | that presenter's full speech, or a voice-over | 703 words / ~4:34 |
| `slides/build-heygen-pptx.py` → `NARRATION` | the HeyGen video | 573 words / ~3:49 |

They tell the same story at three levels of detail. If you change the story, change all three.

---

## The student-facing one-pager

`CS620_Project_fall2026.docx` is the document students read when choosing a capstone. It is an
updated version of the Spring '26 description, and deliberately keeps that document's structure
and length — overview, what last semester built, what this semester does, deliverables, sponsors,
who should pick it.

The PDF is generated from the `.docx`:

```bash
pandoc CS620_Project_fall2026.docx -o CS620_Project_fall2026.pdf --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=11pt
```

Edit the `.docx` in Word, then regenerate the PDF. Do not edit them independently.

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
the document first, then in the deck, then in the narration.
