#!/usr/bin/env bash
# Copy the REAL pipeline outputs into docs/data/ and remove the placeholder marker.
# Run from the repo root AFTER: python main.py --tune, --train, --evaluate.
set -euo pipefail
SRC=results
DST=docs/data
for f in "$SRC/evaluation_pooled_summary.csv" "$SRC/prediction_timing_deltas.csv" \
         "$SRC/site-data/race-explorer-2020.json" \
         "$SRC/site-data/validation-threshold-metrics-2019.json"; do
  [ -f "$f" ] || { echo "Missing $f - run the pipeline first." >&2; exit 1; }
done
cp "$SRC/evaluation_pooled_summary.csv" "$SRC/prediction_timing_deltas.csv" "$DST/"
cp "$SRC/site-data/validation-threshold-metrics-2019.json" "$DST/"
# The explorer JSON is written with indent=2 (~14 MB). Write a compact copy with
# rounded numbers (~6 MB) so it loads faster on phones. Values are not otherwise changed.
python3 - "$SRC/site-data/race-explorer-2020.json" "$DST/race-explorer-2020.json" <<'PY'
import json, sys
def rnd(x):
    if isinstance(x, float): return round(x, 4)
    if isinstance(x, list): return [rnd(v) for v in x]
    if isinstance(x, dict): return {k: rnd(v) for k, v in x.items()}
    return x
data = json.load(open(sys.argv[1], encoding='utf-8'))
json.dump(rnd(data), open(sys.argv[2], 'w', encoding='utf-8'), separators=(',', ':'))
PY
rm -f "$DST/placeholder.json"
echo "Real results copied into $DST. The placeholder notice will disappear."
echo "Now search docs/index.html for 'class=\"draft' and fill in each draft note."
