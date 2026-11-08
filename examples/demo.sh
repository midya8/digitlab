#!/usr/bin/env bash
set -eu
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
python -m digitlab demo > "$TMP/demo.json"
cmp "$TMP/demo.json" digitlab/artifacts/report.json
python -m digitlab train --artifacts "$TMP/run" > "$TMP/train.json"
python -m digitlab evaluate --artifacts "$TMP/run" > "$TMP/evaluate.json"
python - "$TMP/evaluate.json" <<'PY'
import json,sys
r=json.load(open(sys.argv[1]));assert r['clean']['1nn']['correct']==1760
assert r['clean']['mlp']['correct']==1734
assert r['clean']['linear']['correct']==1698
assert r['clean']['1nn']['correct']>r['clean']['mlp']['correct']
a=r['attacks']['mlp'];fg=next(x for x in a if x['method']=='fgsm'and x['epsilon_normalized']==.1);pg=next(x for x in a if x['method']=='pgd_pool'and x['epsilon_normalized']==.1)
assert pg['correct']==768 and pg['clean_correct_fooled']==966
assert fg['correct']==898 and fg['clean_correct_fooled']==836
for rows in r['attacks'].values():
 for x in rows:assert x['max_actual_linf']<=x['epsilon_normalized']+1e-12
print('Real-data clean baselines, bounded attacks, nonzero attack success and frozen-protocol refit: OK')
PY
python examples/render_audit.py "$TMP/figure.svg"
cmp "$TMP/figure.svg" assets/digit-audit.svg
