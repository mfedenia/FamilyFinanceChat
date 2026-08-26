#!/usr/bin/env python3
"""
Build the HeyGen-ready PowerPoint from the build-step Beamer deck.

Pipeline
--------
    familyfinancechat-fall2026-build.tex   35 cumulative overlay steps
        -> latexmk                          familyfinancechat-fall2026-build.pdf (35 pages)
        -> pdftoppm                         one PNG per page, 1920x1080
        -> python-pptx                      familyfinancechat-fall2026-heygen.pptx

Each PDF page becomes one 16:9 PowerPoint slide holding that page as a
full-bleed image, with its narration line in the speaker notes.  HeyGen reads
the speaker notes as the script for that slide, so the spoken words always match
what has just appeared on screen.

NARRATION is the single source of truth for the video script.  It must have
exactly one entry per PDF page, in order -- the script refuses to build if the
counts disagree, which is what stops the deck and the speech drifting apart.

This is a SHORTER script than ../script/presentation-script.md (which is the
five-minute version for a live human presenter).  Target here is under four
minutes.

Requires: latexmk + pdflatex, pdftoppm (poppler), python-pptx.
    python3 -m venv venv && venv/bin/pip install python-pptx
    venv/bin/python build-heygen-pptx.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "familyfinancechat-fall2026-build.tex"
PDF = HERE / "familyfinancechat-fall2026-build.pdf"
PNG_DIR = HERE / "build-png"
PPTX = HERE / "familyfinancechat-fall2026-heygen.pptx"

# Words per minute a HeyGen avatar speaks at, for the runtime estimate below.
# Deliberately the slow end of the plausible range, so the estimate errs long.
WPM = 150

# ---------------------------------------------------------------------------
# One entry per build step, in page order.  (deck_slide, step, narration)
# deck_slide/step are documentation only -- the order of this list is what maps
# narration onto pages.
# ---------------------------------------------------------------------------
NARRATION = [
    (1, 1, "This is FamilyFinanceChat. Two jobs this semester: make the platform "
           "solid, and give the client a face."),

    (2, 1, "A student practises financial advising against an A-I client — a realistic "
           "family, grounded in the course documents, so it does not invent facts."),
    (2, 2, "Every question is scored automatically, and the instructor sees the results "
           "in a dashboard."),
    (2, 3, "The point is scale: unlimited practice at any hour, without booking a human "
           "role-player."),

    (3, 1, "It works today. Students use it. But three things hurt. We are three versions "
           "behind the software we run on, and catching up gets harder."),
    (3, 2, "The professor's grading tool runs on a laptop. Using it takes a terminal and a "
           "developer."),
    (3, 3, "And every deployment needs manual steps people forget, so things break quietly."),
    (3, 4, "That is the gap between a system that works and one you can hand over."),

    (4, 1, "One rule governs everything. We never modify the core of Open WebUI, the "
           "platform underneath."),
    (4, 2, "The last team inherited six and a half thousand lines of that code, forked here. "
           "It froze us on one old version. They removed all of it. Our version is one line."),
    (4, 3, "So the test for anything we build: could we take their next release tomorrow, "
           "unchanged?"),

    (5, 1, "The first job is unglamorous, and the course depends on it. Upgrade to the current "
           "version one step at a time — these upgrades change the database, and you cannot "
           "undo it."),
    (5, 2, "Give the professor a web address, not a laptop."),
    (5, 3, "Automate deployment and testing, so a bad change is caught before it reaches students."),
    (5, 4, "And delete the last of the old fork, so nobody brings it back."),

    (6, 1, "Today the student types to the client."),
    (6, 2, "We want them to talk to it — a face that listens and answers out loud, in character."),
    (6, 3, "This course prepares them for a room with a real family, where you ask the hard "
           "question out loud, once, while somebody watches. You cannot rehearse that with a "
           "backspace key."),

    (7, 1, "So how do we add a face without breaking that rule?"),
    (7, 2, "We build it beside the platform — a separate app in its own container."),
    (7, 3, "The student speaks to it, it asks Open WebUI for the answer, and Open WebUI stays "
           "untouched. Same brain, same grading."),
    (7, 4, "If it does not work out, we delete one container. The rule chose the better design for us."),

    (8, 1, "Three numbers decide whether this is real. Speed — the client must answer in about one "
           "point two seconds, or it stops feeling like a conversation."),
    (8, 2, "Cost — about four hundred dollars a semester for a class of forty, hard-capped."),
    (8, 3, "Permission — student voice and video are education records. Consent is settled in week "
           "two, before anything is recorded."),
    (8, 4, "If any one fails, the answer is no. And no is an acceptable answer."),

    (9, 1, "By the end of September, students can talk to it — voice only, no face yet."),
    (9, 2, "By the twentieth of October we pick an avatar provider, or decide to stop."),
    (9, 3, "A week later, the platform is upgraded."),
    (9, 4, "By mid-November, five students complete full spoken sessions, graded like any other."),
    (9, 5, "Voice-only ships in September either way. That is what makes the ambitious part safe."),

    (10, 1, "Here is what finished looks like. A professor grades from a web address — no laptop."),
    (10, 2, "Deployments are automatic; a bad change is caught before it ships."),
    (10, 3, "And we know, from the scoring system we already have, whether talking to an avatar "
            "makes better advisors. That may come back negative. We publish either way."),
    (10, 4, "The goal is not to finish this system. It is to leave one where the interesting work "
            "is the teaching, not the plumbing."),
]


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


def build_pdf():
    print("1/4  compiling the build-step deck")
    run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", TEX.name], cwd=HERE)


def render_pngs():
    print("2/4  rendering pages to 1920x1080 PNG")
    if PNG_DIR.exists():
        shutil.rmtree(PNG_DIR)
    PNG_DIR.mkdir()
    # -scale-to-x/-scale-to-y forces exactly 1920x1080; the deck is already 16:9,
    # so nothing is distorted, and HeyGen gets a clean full-HD frame.
    run(["pdftoppm", "-png", "-r", "300", "-scale-to-x", "1920", "-scale-to-y", "1080",
         str(PDF), str(PNG_DIR / "slide")])
    return sorted(PNG_DIR.glob("slide-*.png"))


def build_pptx(pngs):
    print("3/4  assembling the PowerPoint")
    from pptx import Presentation
    from pptx.util import Emu, Inches

    if len(pngs) != len(NARRATION):
        sys.exit(f"MISMATCH: {len(pngs)} PDF pages but {len(NARRATION)} narration entries.\n"
                 f"Every build step needs exactly one line of narration — fix NARRATION "
                 f"in this file, or the overlay specs in {TEX.name}.")

    prs = Presentation()
    prs.slide_width = Inches(13.333)   # 16:9
    prs.slide_height = Inches(7.5)
    blank = prs.slide_layouts[6]

    for png, (deck_slide, step, narration) in zip(pngs, NARRATION):
        slide = prs.slides.add_slide(blank)
        slide.shapes.add_picture(str(png), Emu(0), Emu(0),
                                 width=prs.slide_width, height=prs.slide_height)
        # HeyGen reads the speaker notes as the script for this slide.
        slide.notes_slide.notes_text_frame.text = narration

    prs.save(PPTX)


def report():
    print("4/4  done")
    words = sum(len(n.split()) for _, _, n in NARRATION)
    seconds = words / WPM * 60
    print(f"\n     {PPTX.name}")
    print(f"     {len(NARRATION)} slides · {words} words · about "
          f"{int(seconds // 60)}:{int(seconds % 60):02d} at {WPM} words per minute")
    if seconds > 240:
        print("     WARNING: over the four-minute target — trim NARRATION.")


if __name__ == "__main__":
    build_pdf()
    pngs = render_pngs()
    build_pptx(pngs)
    report()
