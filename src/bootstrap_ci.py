"""Bootstrap 95% confidence intervals for the headline per-model metrics.

Resamples FUNCTIONS with replacement (cluster bootstrap — the function is the unit of
variation, not the individual sample), recomputes each metric, and reports the 2.5/97.5
percentiles. This quantifies how much each number would wobble under a different draw of
test functions — the honest answer to the small-n concern, and the evidence that the
model-agnostic finding is real (do the models' CIs overlap?). Works at any n.

Input: patches_similarity_*.jsonl (they carry compilable + codebleu + string_similarity).

Run from project root:
  python src/bootstrap_ci.py -i results/patches_similarity_deepseek_1.3b_512.jsonl ... [--iters 2000]
"""
import argparse
import glob
import json
import os
import random
from collections import defaultdict

def load(path):
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


METRICS = {
    'compile_rate_%': lambda recs: 100.0 * sum(bool(r.get('compilable')) for r in recs) / len(recs),
    'codebleu':       lambda recs: sum(r.get('codebleu', 0.0) for r in recs) / len(recs),
    'string_sim':     lambda recs: sum(r.get('string_similarity', 0.0) for r in recs) / len(recs),
}


def bootstrap(path, iters, rng):
    recs = load(path)
    by_func = defaultdict(list)
    for r in recs:
        by_func[r['func_id']].append(r)
    funcs = list(by_func)
    n = len(funcs)

    point = {name: fn(recs) for name, fn in METRICS.items()}
    dist = {name: [] for name in METRICS}
    for _ in range(iters):
        sample = [f for _ in range(n) for f in [rng.choice(funcs)]]
        pooled = [rec for f in sample for rec in by_func[f]]
        for name, fn in METRICS.items():
            dist[name].append(fn(pooled))

    ci = {}
    for name, vals in dist.items():
        vals.sort()
        lo = vals[int(0.025 * len(vals))]
        hi = vals[int(0.975 * len(vals)) - 1]
        ci[name] = (point[name], lo, hi)
    return n, ci


def main():
    ap = argparse.ArgumentParser(description='Bootstrap CIs for per-model metrics')
    ap.add_argument('--inputs', '-i', nargs='*', default=None)
    ap.add_argument('--iters', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    files = args.inputs or sorted(glob.glob('results/patches_similarity_*.jsonl'))
    if not files:
        print('No similarity files found.')
        return

    import csv
    rows = []
    print(f'\nBootstrap 95% CIs ({args.iters} iters, resampling functions):\n')
    for path in files:
        label = os.path.splitext(os.path.basename(path))[0].replace('patches_similarity_', '')
        n, ci = bootstrap(path, args.iters, rng)
        print(f'  {label}  (n={n} functions)')
        for name in METRICS:
            pt, lo, hi = ci[name]
            print(f'    {name:<16} {pt:8.3f}   95% CI [{lo:.3f}, {hi:.3f}]')
            rows.append({'model': label, 'n_functions': n, 'metric': name,
                         'point': round(pt, 4), 'ci_low': round(lo, 4), 'ci_high': round(hi, 4)})
        print()

    out = 'results/bootstrap_ci.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'n_functions', 'metric', 'point', 'ci_low', 'ci_high'])
        w.writeheader(); w.writerows(rows)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
