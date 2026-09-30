#!/usr/bin/env bash
# Assemble the project page into _site/ from files already in the repository (no duplicated media).
#   bash website/build.sh && python3 -m http.server -d _site 8000
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf _site && mkdir -p _site/media _site/figures
cp -r website/index.html website/assets _site/
cp -r media/previews _site/media/
cp docs/figures/*.png _site/figures/
cp paper/main.pdf _site/paper.pdf
touch _site/.nojekyll
echo "built _site ($(du -sh _site | cut -f1))"
