#!/usr/bin/env bash
# Build the Fall 2026 kickoff deck.
#
#   ./build.sh          slides only
#   ./build.sh notes    slides + a presenter-notes PDF (each slide followed by its cue card)
#   ./build.sh video    the HeyGen PowerPoint (35 build steps, narration in speaker notes)
#   ./build.sh clean    remove build artefacts
#
# Requires: latexmk + pdflatex, with beamer, tikz, booktabs, xcolor, helvet, microtype.
# No external Beamer theme is needed -- the theme is defined inside the .tex file.
# The video target additionally needs pdftoppm (poppler) and python3; it creates a
# throwaway virtualenv in .venv-pptx for python-pptx.

set -euo pipefail
cd "$(dirname "$0")"

DOC=familyfinancechat-fall2026

case "${1:-slides}" in
  clean)
    latexmk -C "$DOC.tex" || true
    latexmk -C "$DOC-build.tex" || true
    rm -f "$DOC-notes".{aux,log,out,nav,snm,toc,fls,fdb_latexmk,pdf}
    rm -rf build-png .venv-pptx
    echo "cleaned"
    ;;

  video)
    # build-heygen-pptx.py runs latexmk itself, renders every overlay step to PNG,
    # and writes the .pptx with one narration line per slide in the speaker notes.
    if [ ! -x .venv-pptx/bin/python ]; then
      python3 -m venv .venv-pptx
      .venv-pptx/bin/pip install --quiet python-pptx
    fi
    .venv-pptx/bin/python build-heygen-pptx.py
    ;;
  notes)
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$DOC.tex"
    latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname="$DOC-notes" \
      -pdflatex='pdflatex %O "\def\SHOWNOTES{}\input{%S}"' "$DOC.tex"
    echo "built: $DOC.pdf and $DOC-notes.pdf"
    ;;
  slides|*)
    latexmk -pdf -interaction=nonstopmode -halt-on-error "$DOC.tex"
    echo "built: $DOC.pdf"
    ;;
esac
