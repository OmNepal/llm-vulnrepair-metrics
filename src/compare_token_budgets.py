"""
Token-budget comparison: 256 vs 512 MAX_NEW_TOKENS, per model.

For each model it loads the 256-token run (validated/similarity files tagged
`_v3`) and the 512-token run (tagged `_512`) and reports, at each budget:
  - compile rate          (share of patches that pass the stub-injection harness)
  - truncation rate        (share of patches with an unclosed block: '{' > '}')
  - pass@1 / pass@3 / pass@5 (unbiased estimator over compilable outcomes)
  - CodeBLEU and string similarity vs the ground-truth fix (func_after)
...plus the 512-256 delta for each. This is the controlled before/after that
tests the token-budget decision (DECISIONS.md, Experiment 3): does giving the
models more room actually cut truncation, and does it move compile rate / the
cross-model ranking?

Usage (from project root): python src/compare_token_budgets.py
Outputs:
  results/comparison_256_vs_512.csv               (long form: one row per model x budget)
  figures/fig9_token_budget_256_vs_512.png        (compile rate + truncation, 256 vs 512)
"""
import json, math, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from collections import defaultdict

PROMPT_TYPES  = ['zero_shot', 'few_shot', 'cot']

# (alias stem, display name) — file suffix for each budget is added below.
MODELS = [
    ('codegen_350M_multi', 'CodeGen-350M'),
    ('deepseek_1.3b',      'DeepSeek-1.3B'),
    ('deepseek_6.7b',      'DeepSeek-6.7B'),
]

# (budget label, filename suffix). 256-token files were tagged _v3; 512 are _512.
BUDGETS = [('256', 'v3'), ('512', '512')]

COLORS = {'256': '#bdbdbd', '512': '#4c72b0'}


def load_records(path):
    recs = []
    if not os.path.exists(path):
        return recs
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


def pass_at_k(n, c, k):
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def is_truncated(patch: str) -> bool:
    """Heuristic: an unclosed block (more '{' than '}') => function cut off."""
    return patch.count('{') > patch.count('}')


def summarize(validated, similarity):
    recs = validated
    if not recs:
        return None
    n = len(recs)
    compile_rate = sum(1 for r in recs if r.get('compilable')) / n
    trunc_rate   = sum(1 for r in recs if is_truncated(r.get('patch', ''))) / n

    # pass@k averaged over (func_id, prompt_type) cells
    groups = defaultdict(list)
    for r in recs:
        groups[(r['func_id'], r['prompt_type'])].append(r.get('compilable', False))
    p1 = p3 = p5 = 0.0
    for outcomes in groups.values():
        nn = len(outcomes)
        cc = sum(1 for o in outcomes if o is True)
        p1 += pass_at_k(nn, cc, 1)
        p3 += pass_at_k(nn, cc, 3)
        p5 += pass_at_k(nn, cc, 5)
    m = len(groups)
    p1, p3, p5 = p1 / m, p3 / m, p5 / m

    sim = similarity
    if sim:
        ss = sum(r['string_similarity'] for r in sim) / len(sim)
        cb = sum(r['codebleu'] for r in sim) / len(sim)
    else:
        ss = cb = float('nan')

    return {
        'n': n, 'compile_rate': compile_rate, 'truncation_rate': trunc_rate,
        'pass@1': p1, 'pass@3': p3, 'pass@5': p5,
        'string_similarity': ss, 'codebleu': cb,
    }


def fmt_delta(v, decimals=1):
    return f"{v:+.{decimals}f}" if not math.isnan(v) else "  n/a"


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # data[display_name][budget_label] = summary dict
    data = {}
    rows = []
    for alias, display in MODELS:
        data[display] = {}
        for budget, suffix in BUDGETS:
            val_path = os.path.join('results', f'patches_validated_{alias}_{suffix}.jsonl')
            sim_path = os.path.join('results', f'patches_similarity_{alias}_{suffix}.jsonl')
            val = load_records(val_path)
            if not val:
                print(f"Skipping {display} @{budget} — {val_path} not found")
                continue
            sim = load_records(sim_path)
            s = summarize(val, sim)
            data[display][budget] = s
            rows.append({'model': display, 'budget': budget, **s})
            print(f"Loaded {display} @{budget}: {s['n']} records "
                  f"({len(sim)} with similarity)")

    if not rows:
        print("No files found — did you run steps 1-3 for both budgets?")
        raise SystemExit(1)

    # ── CSV (long form) ───────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    csv_path = os.path.join('results', 'comparison_256_vs_512.csv')
    df.to_csv(csv_path, index=False)

    # ── Printed before/after tables ───────────────────────────────────────────
    def print_metric(title, key, scale=100.0, unit='%', decimals=1):
        print("\n" + "=" * 64)
        print(f"{title}  (256 -> 512, delta)")
        print("=" * 64)
        print(f"{'Model':<16}{'256':>10}{'512':>10}{'delta':>10}")
        print("-" * 46)
        for _, display in MODELS:
            d = data.get(display, {})
            if '256' in d and '512' in d:
                a = d['256'][key] * scale
                b = d['512'][key] * scale
                print(f"{display:<16}{a:>9.{decimals}f}{unit}{b:>9.{decimals}f}{unit}"
                      f"{fmt_delta(b - a, decimals):>9}{unit}")

    print_metric('COMPILE RATE',     'compile_rate')
    print_metric('TRUNCATION RATE',  'truncation_rate')
    print_metric('PASS@3',           'pass@3')
    print_metric('PASS@5',           'pass@5')
    print_metric('CodeBLEU',         'codebleu', scale=1.0, unit='', decimals=3)

    print(f"\nSaved {csv_path}")

    # ── Figure 9: compile rate + truncation, 256 vs 512 ───────────────────────
    names = [d for _, d in MODELS if {'256', '512'} <= set(data.get(d, {}))]
    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (key, title, ymaxpad) in zip(
        axes,
        [('compile_rate', 'Compile Rate: 256 vs 512 tokens', 1.5),
         ('truncation_rate', 'Truncation Rate: 256 vs 512 tokens', 1.2)],
    ):
        for i, budget in enumerate(['256', '512']):
            vals = [data[n][budget][key] for n in names]
            bars = ax.bar(x + (i - 0.5) * width, vals, width,
                          label=f'{budget} tok', color=COLORS[budget], alpha=0.9)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                        f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylim(0, max(data[n][b][key] for n in names for b in ['256', '512']) * ymaxpad)
        ax.legend()

    plt.tight_layout()
    fig9_path = os.path.join('figures', 'fig9_token_budget_256_vs_512.png')
    plt.savefig(fig9_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig9_path}")
    print(f"\nDone — {len(names)} model(s) compared across 256 and 512 token budgets.")
