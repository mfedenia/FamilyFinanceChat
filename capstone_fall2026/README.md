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
