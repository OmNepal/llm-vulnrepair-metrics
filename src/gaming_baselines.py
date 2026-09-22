"""diff_F1 on degenerate ("gaming") patches.

Two outputs, CPU-only, no model calls:
  1. SYNTHETIC baselines applied to every test function: no-op, whole-function `#if 0`
     wrap, empty body with a placeholder comment, and a `return 0;` body.
  2. REAL degenerate patches: compiling round 1-3 patches from the compiler-feedback loop
     that match a degenerate signature (`#if 0`, empty body, placeholder comment, or a
     very short output). Each is re-checked on the local harness before being scored.

Signature matching is a heuristic (see `degenerate_kind`); it is meant to surface
candidates for manual reading, not to be an exhaustive classifier.

Run from project root:  python src/gaming_baselines.py
Writes results/gaming_baselines.csv
"""
import csv
import json
import os
import re
import statistics as st
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diff_aware_score import diff_f1  # noqa: E402
from validation.compile_check import is_compilable  # noqa: E402

TEST_CSV = os.path.join('data', 'test_functions_200.csv')

# Manual exclusions after reading the heuristic's candidates. The keyword match flags
# instruct/CWE-125_04 because of a stray `// ...` comment, but the patch is a full
# repair attempt (whole function present, ~1.1x the input length), not a degenerate one.
MANUAL_EXCLUDE = {('instruct', 'CWE-125_04')}
FEEDBACK = [('base', 'results/patches_feedback_deepseek_1.3b_200.jsonl'),
            ('instruct', 'results/patches_feedback_deepseek_1.3b_instruct_200.jsonl')]


def wrap_if0(fb):
    return '#if 0\n' + fb + '\n#endif'


def _stub(fb, body):
    i = fb.find('{')
    return fb[:i + 1] + body + '}' if i >= 0 else fb


SYNTHETIC = {
    'no-op (copy of input)': lambda fb: fb,
    '#if 0 around whole function': wrap_if0,
    'empty body + placeholder comment': lambda fb: _stub(fb, '\n    // your code here\n'),
    'body = return 0;': lambda fb: _stub(fb, '\n    return 0;\n'),
}


def degenerate_kind(patch, before):
    s = patch.strip()
    if '#if 0' in s:
        return 'if0'
    body = re.sub(r'//.*|/\*.*?\*/', '', s, flags=re.S)
    m = re.search(r'\{(.*)\}\s*;?\s*$', body, re.S)
    if m and len(m.group(1).strip()) < 12:
        return 'empty-body'
    if re.search(r'your code|rest of|your fields|// \.\.\.|other functions|add here', s, re.I):
        return 'placeholder'
    if len(s) / max(1, len(before)) < 0.25:
        return 'short'
    return None


def load(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    df = pd.read_csv(TEST_CSV, encoding='utf-8')
    rows = []

    print(f'SYNTHETIC baselines over {len(df)} test functions')
    for name, fn in SYNTHETIC.items():
        vals = [diff_f1(r.func_before, r.func_after, fn(r.func_before))
                for r in df.itertuples()]
        vals = [v for v in vals if v is not None]
        print(f'  {name:34s} mean={st.mean(vals):.3f} median={st.median(vals):.3f}')
        rows.append(('synthetic', name, len(vals), round(st.mean(vals), 4), round(st.median(vals), 4)))

    print('\nREAL degenerate compiling patches (feedback loop, rounds 1-3, distinct patches)')
    seen = {}
    for alias, path in FEEDBACK:
        for r in load(path):
            if r['round'] == 0 or not r.get('compilable'):
                continue
            kind = degenerate_kind(r['patch'], r['func_before'])
            if not kind or (alias, r['func_id']) in MANUAL_EXCLUDE:
                continue
            key = (alias, r['func_id'], r['patch'])
            if key in seen or not is_compilable(r['patch'])['compilable']:
                continue
            v = diff_f1(r['func_before'], r['func_after'], r['patch'])
            if v is not None:
                seen[key] = (v, kind)
    for (alias, fid, _), (v, kind) in sorted(seen.items(), key=lambda kv: kv[1][0]):
        print(f'  {v:.3f}  {alias:8s} {fid:11s} {kind}')
        rows.append(('real', f'{alias}:{fid}:{kind}', 1, round(v, 4), round(v, 4)))
    scores = [v for v, _ in seen.values()]
    print(f'\n  {len(scores)} distinct patches; min={min(scores):.3f} max={max(scores):.3f} '
          f'mean={st.mean(scores):.3f}')

    os.makedirs('results', exist_ok=True)
    with open('results/gaming_baselines.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'name', 'n', 'mean_diff_f1', 'median_diff_f1'])
        w.writerows(rows)
    print('\nSaved results/gaming_baselines.csv')


if __name__ == '__main__':
    main()
