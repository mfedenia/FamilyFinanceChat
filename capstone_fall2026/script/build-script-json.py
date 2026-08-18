#!/usr/bin/env python3
"""Generate presentation-script.json from presentation-script.md.

The Markdown file is the single source of truth for the narration. This script
parses it into a machine-readable form for an AI presenter (HeyGen-class avatar,
TTS voice-over, or any pipeline that wants one narration string per slide).

Usage:  python3 build-script-json.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "presentation-script.md"
OUT = HERE / "presentation-script.json"

SLIDE_RE = re.compile(r"^##\s+Slide\s+(\d+)\s+[—-]\s+(.+?)\s*$")
TIME_RE = re.compile(r"^\*Approx\.\s+(\d+):(\d+)\*\s*$")
WORDS_PER_MINUTE = 145


def parse(md: str) -> list[dict]:
    slides: list[dict] = []
    current: dict | None = None
    lines: list[str] = []

    def flush() -> None:
        if current is None:
            return
        # Join wrapped lines into paragraphs, drop horizontal rules.
        text = "\n".join(lines)
        paragraphs = [
            " ".join(p.split())
            for p in text.split("\n\n")
            if p.strip() and p.strip() != "---"
        ]
        narration = "\n\n".join(paragraphs)
        current["narration"] = narration
        current["word_count"] = len(narration.split())
        slides.append(current)

    for line in md.splitlines():
        m = SLIDE_RE.match(line)
        if m:
            flush()
            current = {
                "slide": int(m.group(1)),
                "title": m.group(2),
                "estimated_seconds": None,
            }
            lines = []
            continue
        if current is None:
            continue
        t = TIME_RE.match(line.strip())
        if t:
            current["estimated_seconds"] = int(t.group(1)) * 60 + int(t.group(2))
            continue
        if line.strip() == "---":
            continue
        lines.append(line)

    flush()
    return slides


def main() -> None:
    slides = parse(SRC.read_text(encoding="utf-8"))
    if not slides:
        raise SystemExit(f"no slides parsed from {SRC}")

    numbers = [s["slide"] for s in slides]
    expected = list(range(1, len(slides) + 1))
    if numbers != expected:
        raise SystemExit(f"slide numbers are not 1..N in order: {numbers}")

    total_words = sum(s["word_count"] for s in slides)
    doc = {
        "title": "FamilyFinanceChat - Fall 2026 Project Kickoff",
        "deck": "../slides/familyfinancechat-fall2026.pdf",
        "source": "presentation-script.md",
        "slide_count": len(slides),
        "total_words": total_words,
        "estimated_runtime_seconds": sum(s["estimated_seconds"] or 0 for s in slides),
        "runtime_from_word_count_seconds": round(total_words / WORDS_PER_MINUTE * 60),
        "speaking_rate_wpm": WORDS_PER_MINUTE,
        "notes": (
            "One narration string per slide, spoken verbatim. Version numbers and "
            "acronyms are already written the way they should be pronounced."
        ),
        "slides": slides,
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"wrote {OUT.name}: {len(slides)} slides, {total_words} words, "
        f"~{doc['runtime_from_word_count_seconds'] // 60} min at {WORDS_PER_MINUTE} wpm"
    )


if __name__ == "__main__":
    main()
