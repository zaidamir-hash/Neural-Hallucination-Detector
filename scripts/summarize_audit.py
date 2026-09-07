import json
from pathlib import Path
import statistics
p=Path('results/metrics/halueval_text_audit.json')
if not p.exists():
    print('Missing audit file')
    raise SystemExit(1)
a=json.loads(p.read_text(encoding='utf-8'))
for k in ['grounded','hallucinated']:
    toks=a[k]['token_counts']
    lex=a[k]['lexical_coverages']
    print(k)
    print(' count', a[k]['count'])
    print(' token mean', statistics.mean(toks), 'median', statistics.median(toks))
    print(' lexical_coverage mean', round(statistics.mean(lex),3))
