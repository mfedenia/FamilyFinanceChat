#!/usr/bin/env bash
# Build the Fall 2026 CS620 capstone deck.  ONE SOURCE, THREE OUTPUTS.
#
#   ./build.sh          handout mode: overlays collapse, 10 slides -- the deck a
#                       human presents from
#   ./build.sh notes    the same, plus a presenter-notes PDF
#   ./build.sh video    \VIDEO: one page per build step, and the right fifth of
#                       the page left clear for the HeyGen avatar; then the .pptx
#   ./build.sh clean    remove build artefacts
#
# Requires: latexmk + pdflatex with beamer, tikz, xcolor, helvet, microtype.
# The video target also needs pdftoppm (poppler) and python3; it creates a
# throwaway virtualenv in .venv-pptx for python-pptx.

set -euo pipefail
cd "$(dirname "$0")"

DOC=familyfinancechat-fall2026
HANDOUT='\PassOptionsToClass{handout}{beamer}\input{'"$DOC"'}'

case "${1:-slides}" in
  clean)
    latexmk -C "$DOC.tex" || true
    rm -f "$DOC"{,-notes,-build}.{aux,log,out,nav,snm,toc,fls,fdb_latexmk}
    rm -f "$DOC-build.pdf" "$DOC-notes.pdf"
    rm -rf build-png .venv-pptx
    echo "cleaned"
    ;;

  notes)
    # twice each: tikz "remember picture" needs the .aux from a previous pass
    pdflatex -interaction=nonstopmode -halt-on-error "$HANDOUT"
    pdflatex -interaction=nonstopmode -halt-on-error "$HANDOUT"
    for _ in 1 2; do
      pdflatex -interaction=nonstopmode -halt-on-error -jobname="$DOC-notes" \
        '\PassOptionsToClass{handout}{beamer}\def\SHOWNOTES{}\input{'"$DOC"'}'
    done
    echo "built: $DOC.pdf and $DOC-notes.pdf"
    ;;

  video)
    # build-heygen-pptx.py runs pdflatex itself with \VIDEO, renders every step
    # to PNG, and writes the .pptx with one narration line per slide in the notes.
    if [ ! -x .venv-pptx/bin/python ]; then
      python3 -m venv .venv-pptx
      .venv-pptx/bin/pip install --quiet python-pptx
    fi
    .venv-pptx/bin/python build-heygen-pptx.py
    ;;

  slides|*)
    pdflatex -interaction=nonstopmode -halt-on-error "$HANDOUT"
    pdflatex -interaction=nonstopmode -halt-on-error "$HANDOUT"
    echo "built: $DOC.pdf (10 slides)"
    ;;
esac
