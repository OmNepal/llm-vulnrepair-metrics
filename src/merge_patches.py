"""Merge JSONL patch files (raw, validated, or feedback-loop) into one, with
consistency checks.

Used to (a) concatenate the deepseek-6.7b batch raw files, (b) combine the
existing 70-function validated file with the new ~130 validated file into the
final ~200-function file, and (c) combine the original 70-function feedback-loop
output with a new ~130-function feedback-loop run. Verifies there are no duplicate
records under the given dedup key and that model_alias is consistent, so a bad
merge fails loudly instead of silently double-counting.

Dedup key defaults to (func_id, prompt_type, sample_idx) for raw/validated files.
Feedback-loop files use `round` instead of `sample_idx` — pass
`--dedup-key func_id,round` for those (each file has multiple rounds per function,
so the default key would misreport every record as a duplicate).

Run from project root:
  python src/merge_patches.py -i A.jsonl B.jsonl -o combined.jsonl
  python src/merge_patches.py -i feedback_A.jsonl feedback_B.jsonl -o combined.jsonl --dedup-key func_id,round
"""
import argparse
import json
from collections import Counter


def load(path):
    recs = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def main():
    ap = argparse.ArgumentParser(description='Merge patch JSONL files with checks')
    ap.add_argument('--inputs', '-i', nargs='+', required=True)
    ap.add_argument('--output', '-o', required=True)
    ap.add_argument('--dedup-key', default='func_id,prompt_type,sample_idx',
                     help='comma-separated record fields identifying a unique row '
                          '(default matches raw/validated files; use '
                          '"func_id,round" for feedback-loop files)')
    args = ap.parse_args()
    key_fields = args.dedup_key.split(',')

    all_recs, keys = [], Counter()
    aliases, per_func_samples = set(), Counter()
    for path in args.inputs:
        recs = load(path)
        print(f'  {path}: {len(recs)} records')
        for r in recs:
            k = tuple(r.get(f) for f in key_fields)
            keys[k] += 1
            aliases.add(r.get('model_alias'))
            per_func_samples[(r.get('func_id'), r.get('prompt_type'))] += 1
        all_recs += recs

    dups = {k: c for k, c in keys.items() if c > 1}
    if dups:
        ex = list(dups.items())[:5]
        raise SystemExit(f'ERROR: {len(dups)} duplicate {tuple(key_fields)} '
                         f'keys — refusing to merge. Examples: {ex}')
    if len(aliases) > 1:
        raise SystemExit(f'ERROR: mixed model_alias across inputs: {aliases}. '
                         f'Only merge files from the SAME model.')
    sample_counts = set(per_func_samples.values())
    n_funcs = len({k[key_fields.index('func_id')] for k in keys})
    print(f'\nMerged: {len(all_recs)} records | model {aliases} | '
          f'{n_funcs} distinct functions | records/(func,prompt_type) {sorted(sample_counts)}')
    if len(sample_counts) > 1:
        print('  WARNING: inconsistent counts per (func,prompt_type) — check your inputs.')

    with open(args.output, 'w', encoding='utf-8', newline='\n') as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
