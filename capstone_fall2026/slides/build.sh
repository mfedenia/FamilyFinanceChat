#!/usr/bin/env bash
# Build the Fall 2026 kickoff deck.
#
#   ./build.sh          slides only
#   ./build.sh notes    slides + a presenter-notes PDF (each slide followed by its cue card)
#   ./build.sh clean    remove build artefacts
#
# Requires: latexmk + pdflatex, with beamer, tikz, booktabs, xcolor, helvet, microtype.
# No external Beamer theme is needed -- the theme is defined inside the .tex file.

set -euo pipefail
cd "$(dirname "$0")"

DOC=familyfinancechat-fall2026

case "${1:-slides}" in
  clean)
    latexmk -C "$DOC.tex" || true
    rm -f "$DOC-notes".{aux,log,out,nav,snm,toc,fls,fdb_latexmk,pdf}
    echo "cleaned"
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
