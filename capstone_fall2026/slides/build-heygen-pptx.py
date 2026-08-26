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
TEX = HERE / "familyfinancechat-fall2026.tex"
PDF = HERE / "familyfinancechat-fall2026-build.pdf"
PNG_DIR = HERE / "build-png"
PPTX = HERE / "familyfinancechat-fall2026-heygen.pptx"

# Words per minute a HeyGen avatar speaks at, for the runtime estimate below.
# Deliberately the slow end of the plausible range, so the estimate errs long.
WPM = 150

# ---------------------------------------------------------------------------
# One entry per build step, in page order.  (deck_slide, step, narration)
#
# VOICE: Mark talking to a room of computer science students about a project he
# wants them to take on.  Not a reading of the slides -- the slides show, the
# speech explains.  If a line could be deleted and nothing would be lost because
# the slide already says it, it is the wrong line.  Whole sentences, contractions
# where a person would use them, and the reason to care said out loud.
#
# WRITTEN FOR THE EAR, NOT THE EYE.  A text-to-speech engine reads this with no
# idea what any of it means, so it is spelled the way it should sound:
#
#   numbers as words        "about a second", "forty students" -- never "1s"
#   acronyms letter by letter   "A-I", "U-I"; hyphens force the letters apart.
#                           Same for any acronym added later: S-Q-L, A-P-I.
#   product names as spoken     "Family Finance Chat", "Open Web U-I" -- solid
#                           capitals get read as one invented word.  The SLIDES
#                           still show the real spelling; only speech is respelled.
#   names spelled to sound right    "Fedeenia" -- the surname is Fedenia, and a
#                           speech engine says FED-en-ya unless the long e is
#                           written in.  Getting somebody's own name wrong is the
#                           one mistake an audience is guaranteed to notice.
#   no em dashes, no semicolons, no hyphenated words     a full stop is an
#                           unambiguous pause; an em dash is engine-dependent.
#
# check_tts() enforces the mechanical half of this on every build.
# ---------------------------------------------------------------------------
NARRATION = [
    # Scene zero is the thumbnail: a duplicate of the title slide, held for about
    # a second. Two words, so HeyGen renders a real segment rather than a
    # zero-length one, and the video opens on a clean frame instead of a face
    # already mid-sentence.
    (0, 1, "Hi there."),

    (1, 1, "I'm Mark Fedeenia. I teach finance at Wisconsin, and I want to show you "
           "something my students use, and where I'd like to take it next."),
    (1, 2, "It's called Family Finance Chat, and right now it's a chat box. By the end of "
           "this project, I want it to be a face you talk to."),

    (2, 1, "Here's the problem I'm trying to solve. My students are learning to advise "
           "families about money, and the hard part isn't the arithmetic."),
    (2, 2, "It's sitting across from people and asking uncomfortable questions well. So we "
           "built them somebody to practise on. An A-I family with a real financial "
           "situation, taken from the course material, so it doesn't make things up."),
    (2, 3, "And because every question gets scored automatically, I can see how forty "
           "students are doing without reading forty transcripts."),

    (3, 1, "Today the whole thing lives in a browser, on a cloud machine, built on an open "
           "source platform called Open Web U-I."),
    (3, 2, "Underneath there are about eight moving parts, and one language model doing "
           "the talking."),
    (3, 3, "It works. Students use it every week, and that's exactly why the next step is "
           "worth taking seriously."),

    (4, 1, "There are two things I want to fix. The first isn't glamorous. We've drifted "
           "behind the software underneath us, and every update takes somebody who knows "
           "where everything is."),
    (4, 2, "The second is the one I'm excited about. I want the student to stop typing."),

    (5, 1, "So, the groundwork. Get current with the platform, and stay current, so "
           "upgrades stop being frightening."),
    (5, 2, "Put the grading on the web where any instructor can open it, instead of on one "
           "laptop in my office."),
    (5, 3, "And make deployment automatic, so improving this thing doesn't depend on one "
           "person remembering the steps."),

    (6, 1, "Now the interesting part. Right now a student types a question and reads an "
           "answer, which is nothing like the job."),
    (6, 2, "I want them to say it out loud, to a face that listens, waits, and answers "
           "back in character."),

    (7, 1, "Because in a real meeting you get one shot. You ask the hard question out "
           "loud, once, while a family watches your face."),
    (7, 2, "You can't practise that with a keyboard. There's no backspace key in that room."),

    (8, 1, "So here's the actual project. There are a dozen companies now doing real time "
           "avatars, and none of them have been tested for something like this. Somebody "
           "has to find out which one holds up."),
    (8, 2, "Speed matters most. If it takes longer than about a second to answer, it stops "
           "feeling like a conversation and starts feeling like a machine."),
    (8, 3, "Then cost, per student, per semester, with a ceiling we can actually live with."),
    (8, 4, "And permission, because a student's voice and face are protected records, and "
           "that gets settled before anyone records anything."),

    (9, 1, "Whatever wins has to bolt on beside what we already have, not replace it."),
    (9, 2, "A separate service that handles the talking."),
    (9, 3, "It asks the existing system for the answer, so the grading and the course "
           "material keep working exactly as they do now. And if it turns out not to be "
           "worth it, we switch it off and lose nothing."),

    (10, 1, "What I want by the end of the term is students actually talking to this thing."),
    (10, 2, "And I want to know whether it makes them better at the job. Measured, not guessed."),
    (10, 3, "It might turn out that it doesn't, and that's a finding worth publishing too. "
            "If that sounds like a semester well spent, come build it with us."),
]


def check_tts():
    """Catch the spellings a speech engine reliably gets wrong.

    Mechanical only -- it cannot tell you the writing is stiff, just that a digit
    or a run-together acronym slipped in.  Fatal, not a warning: a mispronounced
    word is not something you notice until the video is rendered and paid for.
    """
    import re
    rules = [
        (r"\d",                 "digits — write the number in words"),
        (r"[A-Z][a-z]+[A-Z]",   "run-together capitals — respell as separate spoken words"),
        (r"\b[A-Z]{2,}\b",      "solid acronym — hyphenate it, like A-I or S-Q-L"),
        (r"—",                  "em dash — use a full stop or a comma"),
        (r";",                  "semicolon — use a full stop"),
        (r"\b\w+-(?![A-Z]\b)\w",  "hyphenated word — say it in full, like 'the middle of November'"),
    ]
    problems = [f"     slide {d}.{s}: {why}\n       {n}"
                for d, s, n in NARRATION
                for pat, why in rules if re.search(pat, n)]
    if problems:
        sys.exit("NARRATION is not safe for text-to-speech:\n" + "\n".join(problems))


def run(cmd, **kw):
    proc = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if proc.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    return proc


def build_pdf():
    print("1/4  compiling the build-step deck")
    # \VIDEO turns on the avatar column and keeps every overlay step a separate
    # page.  Without it the same file builds the ten-slide presenter handout.
    (HERE / "familyfinancechat-fall2026-build.aux").unlink(missing_ok=True)
    # Twice, always.  The title art is positioned with tikz "remember picture",
    # whose coordinates are only known once they have been written to the .aux --
    # a single pass silently produces a title slide with everything in the wrong
    # place, which is exactly the frame HeyGen uses as the thumbnail.
    for _ in range(2):
        run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-jobname=familyfinancechat-fall2026-build",
             "\\def\\VIDEO{}\\input{%s}" % TEX.stem], cwd=HERE)


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
    check_tts()
    build_pdf()
    pngs = render_pngs()
    build_pptx(pngs)
    report()
