"""Compile-only re-validation.

Re-runs ONLY the compile check (with the current, fixed stub-injection harness)
over an existing validated JSONL, updating `compilable` and `compile_error` in
place while preserving every other field (Semgrep results, patch text, etc.).
Backs up the original to <input>.oldstub.bak before overwriting.

Use after changing compile_check.py so downstream taxonomy/metrics reflect the
fixed harness, without re-running Semgrep.

Run from project root:  python src/revalidate_compile.py -i results/patches_validated_<model>_512.jsonl
"""
import argparse
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation.compile_check import is_compilable  # noqa: E402

WORKERS = 12


def load_records(path):
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def recompile(rec):
    res = is_compilable(rec.get('patch', '') or '')
    return res['compilable'], res.get('error')


def main():
    ap = argparse.ArgumentParser(description='Compile-only re-validation (fixed harness)')
    ap.add_argument('--input', '-i', required=True, help='Validated JSONL to update in place')
    args = ap.parse_args()
    path = args.input

    recs = load_records(path)
    old_ok = sum(1 for r in recs if r.get('compilable'))

    results = [None] * len(recs)
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, out in enumerate(ex.map(recompile, recs)):
            results[i] = out

    for r, (ok, err) in zip(recs, results):
        r['compilable'] = ok
        r['compile_error'] = err
    new_ok = sum(1 for r in recs if r['compilable'])

    bak = path + '.oldstub.bak'
    if not os.path.exists(bak):
        shutil.copy2(path, bak)
        print(f'Backed up original -> {bak}')

    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    n = len(recs)
    print(f'{os.path.basename(path)}: compile {old_ok}/{n} ({old_ok/n*100:.2f}%) '
          f'-> {new_ok}/{n} ({new_ok/n*100:.2f}%)   [{new_ok-old_ok:+d}]')


if __name__ == '__main__':
    main()
