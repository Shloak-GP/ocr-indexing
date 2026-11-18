#!/usr/bin/env bash
# One-step Claude OCR run outside Kaggle (a laptop or this cloud box).
#   ANTHROPIC_API_KEY=... ./run_claude_ocr.sh "/path/to/GP Petroleums Ltd" [workdir]
set -euo pipefail
DATA="$1"; W="${2:-$PWD/claude_bk}"; HERE="$(cd "$(dirname "$0")" && pwd)"
: "${ANTHROPIC_API_KEY:?set ANTHROPIC_API_KEY first}"
pip install -q 'anthropic>=0.70' pymupdf pandas tabulate pillow
python3 "$HERE/bakeoff.py" render --data "$DATA" --work "$W"
python3 "$HERE/engines.py" --engine claude_sonnet55 --pages "$W/pages.csv" --out "$W/runs/claude_sonnet55.jsonl"
python3 "$HERE/bakeoff.py" score --work "$W" --gt "$HERE/gt_fields.csv"
