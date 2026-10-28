#!/usr/bin/env bash
# Downloads the 24 Kodak test images (~15 MB) from http://r0k.us/graphics/kodak/ into ./data/kodak
set -euo pipefail
ROOT="${1:-data}"
mkdir -p "$ROOT/kodak"
for i in $(seq -w 1 24); do
  curl -sSL -o "$ROOT/kodak/kodim$i.png" "http://r0k.us/graphics/kodak/kodak/kodim$i.png"
done
echo "Kodak: $(ls "$ROOT/kodak" | wc -l) files"
