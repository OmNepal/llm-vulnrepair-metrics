"""Change-aware (diff-level) scoring — the positive contribution.

Whole-function CodeBLEU/string-sim mostly measure the copied unchanged context, so
they (a) reward copying and (b) are fooled by degenerate "fixes" that compile but
delete the code. This script scores the *edit* the model made against the *edit* the
developer made, and proves the point with a copy-input baseline.

Three outputs, all CPU-only, from existing patches_similarity_*.jsonl:
  1. COPY-INPUT baseline — score the UNCHANGED vulnerable function (func_before) as if
     it were the model's answer. High whole-function CodeBLEU here == the metric mostly
     measures copying; its diff score is 0 by construction.
  2. DIFF-F1 — token-level F1 between the developer's edit (func_before->func_after) and
     the model's edit (func_before->patch). A no-op / #if 0 / empty-body "fix" scores ~0.
  3. Comparison table — whole-function vs copy-input vs diff-F1 across all runs.

Run from project root:  python src/diff_aware_score.py [--inputs f1.jsonl f2.jsonl ...]
"""
import argparse
import csv
import difflib
import glob
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_similarity import score_pair  # noqa: E402

_TOK = re.compile(r'\w+|[^\s\w]')


def load_records(path):
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def line_changes(before, after):
    """Lines added and removed going from `before` to `after`."""
    b, a = before.splitlines(), after.splitlines()
    sm = difflib.SequenceMatcher(None, b, a, autojunk=False)
    added, removed = [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ('replace', 'delete'):
            removed += b[i1:i2]
        if tag in ('replace', 'insert'):
            added += a[j1:j2]
    return added, removed


def _toks(lines):
    return Counter(_TOK.findall(' '.join(lines)))


def _f1(gold, pred):
    if not gold or not pred:
        return 0.0
    overlap = sum((gold & pred).values())
    if overlap == 0:
        return 0.0
    p = overlap / sum(pred.values())
    r = overlap / sum(gold.values())
    return 2 * p * r / (p + r)


def diff_f1(before, after, patch):
    """Token-level F1 of the model's edit vs the developer's edit.
    Returns None if the developer made no line change (can't define a target edit)."""
    ga, gr = line_changes(before, after)          # developer's added / removed lines
    ma, mr = line_changes(before, patch)           # model's added / removed lines
    parts = []
    if ga:
        parts.append(_f1(_toks(ga), _toks(ma)))    # did the model add the right lines?
    if gr:
        parts.append(_f1(_toks(gr), _toks(mr)))    # did the model remove the right lines?
    return sum(parts) / len(parts) if parts else None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def score_file(path):
    recs = load_records(path)
    wf_cb, wf_ss, df = [], [], []
    for r in recs:
        patch, after, before = (r.get('patch', '') or '', r.get('func_after', '') or '',
                                r.get('func_before', '') or '')
        # whole-function: use precomputed if present, else compute
        if 'codebleu' in r and 'string_similarity' in r:
            wf_cb.append(r['codebleu']); wf_ss.append(r['string_similarity'])
        else:
            sc = score_pair(patch, after)
            wf_cb.append(sc['codebleu']); wf_ss.append(sc['string_similarity'])
        df.append(diff_f1(before, after, patch))
    return {'n': len(recs), 'whole_codebleu': mean(wf_cb),
            'whole_string_sim': mean(wf_ss), 'diff_f1': mean(df)}


def copy_input_baseline(path):
    """Score the UNCHANGED vulnerable function as if it were the answer (one per func)."""
    seen = {}
    for r in load_records(path):
        fid = r['func_id']
        if fid not in seen:
            seen[fid] = (r.get('func_before', '') or '', r.get('func_after', '') or '')
    cb, ss, df = [], [], []
    for before, after in seen.values():
        sc = score_pair(before, after)             # patch := func_before (no-op)
        cb.append(sc['codebleu']); ss.append(sc['string_similarity'])
        df.append(diff_f1(before, after, before))  # == 0 by construction
    return {'n': len(seen), 'whole_codebleu': mean(cb),
            'whole_string_sim': mean(ss), 'diff_f1': mean(df)}


def label_of(path):
    return os.path.splitext(os.path.basename(path))[0].replace('patches_similarity_', '')


def main():
    ap = argparse.ArgumentParser(description='Change-aware / diff-level scoring')
    ap.add_argument('--inputs', '-i', nargs='*', default=None,
                    help='similarity JSONL files (default: results/patches_similarity_*.jsonl)')
    args = ap.parse_args()
    os.makedirs('results', exist_ok=True)

    files = args.inputs or sorted(glob.glob('results/patches_similarity_*.jsonl'))
    if not files:
        print('No patches_similarity_*.jsonl found.')
        return

    rows = []
    base = copy_input_baseline(files[0])   # same 70 functions across all files
    rows.append(('COPY-INPUT (no-op)', base))
    for path in files:
        rows.append((label_of(path), score_file(path)))

    # ── Print ─────────────────────────────────────────────────────────────────
    print('\n' + '=' * 84)
    print('CHANGE-AWARE SCORING - whole-function metrics reward copying; diff-F1 does not')
    print('=' * 84)
    print(f'  {"run":<34}{"n":>5}{"whole_CodeBLEU":>16}{"whole_strSim":>14}{"diff_F1":>10}')
    print('  ' + '-' * 80)
    for name, m in rows:
        print(f'  {name:<34}{m["n"]:>5}{m["whole_codebleu"]:>16.4f}'
              f'{m["whole_string_sim"]:>14.4f}{m["diff_f1"]:>10.4f}')

    # gap vs copy baseline (how much signal does whole-func CodeBLEU add over copying?)
    print('\n  Whole-function CodeBLEU vs the copy-input baseline (negative = worse than'
          ' returning the vulnerable code unchanged):')
    for name, m in rows[1:]:
        delta = m['whole_codebleu'] - base['whole_codebleu']
        print(f'    {name:<32} {delta:+.4f}   '
              f'(diff_F1 {m["diff_f1"]:.4f} vs baseline {base["diff_f1"]:.4f})')

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = 'results/diff_aware_scores.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['run', 'n', 'whole_codebleu', 'whole_string_sim', 'diff_f1'])
        for name, m in rows:
            w.writerow([name, m['n'], round(m['whole_codebleu'], 4),
                        round(m['whole_string_sim'], 4), round(m['diff_f1'], 4)])
    print(f'\nSaved {csv_path}')
    print('\nDone.')


if __name__ == '__main__':
    main()
