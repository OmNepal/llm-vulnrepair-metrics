"""Compiler -std sensitivity experiment.

Re-validates the *already generated* patches under several C-standard settings,
with no model calls and no regeneration, to show that compile rate is a function
of the measurement configuration rather than the model.

The harness (validation/compile_check.py) reads the COMPILE_STD env var and
injects `-std=<value>` (unset => no flag => the original default-standard harness).

Use --suffix to pick the input file set (512 = original n=70 files, 200 = the
n=203 scale-up files). Outputs for --suffix 512 keep the original (legacy) names;
any other suffix gets it appended so nothing is overwritten:
  results/compile_std_sweep[_<suffix>].csv        per (setting, model): compile rate + group counts
  results/compile_std_transitions[_<suffix>].csv  per (setting, model): patches gained/lost vs default
  figures/fig10_compile_std_sweep[_<suffix>].png   grouped bars: compile rate by model x setting

Run from project root:  python src/compile_std_sweep.py [--suffix 512|200]
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from validation.compile_check import is_compilable          # noqa: E402
from error_analysis import categorize, CATEGORY_GROUP, GROUP_ORDER, load_records  # noqa: E402

MODELS   = ['codegen_350M_multi', 'deepseek_1.3b', 'deepseek_6.7b']
# (label, -std value). '' => no flag => the harness default (GCC 15 => C23).
SETTINGS = [('default', ''), ('gnu17', 'gnu17'), ('gnu89', 'gnu89')]
WORKERS  = 12

# Expected default-setting compile rates — sanity check, keyed by --suffix.
# 512 = RESULTS.md §5.2 (n=70, post Exp-3d harness fix); 200 = RESULTS.md §5g.1 (n=203).
EXPECTED_DEFAULT = {
    '512': {'codegen_350M_multi': 7.43, 'deepseek_1.3b': 5.33, 'deepseek_6.7b': 4.67},
    '200': {'codegen_350M_multi': 5.45, 'deepseek_1.3b': 3.45, 'deepseek_6.7b': 3.32},
}


def eval_patch(rec):
    """Recompile one stored patch under the current COMPILE_STD; classify it."""
    patch = rec.get('patch', '') or ''
    res = is_compilable(patch)
    newrec = {'patch': patch,
              'compilable': res['compilable'],
              'compile_error': res.get('error') or ''}
    cat = categorize(newrec)
    group = CATEGORY_GROUP.get(cat['primary'], 'other')
    return res['compilable'], group


def main():
    ap = argparse.ArgumentParser(description='Compiler -std sensitivity sweep')
    ap.add_argument('--suffix', default='512',
                     help="input file suffix: 512 (n=70, default) or 200 (n=203)")
    args = ap.parse_args()
    suffix = args.suffix
    tag = '' if suffix == '512' else f'_{suffix}'
    expected = EXPECTED_DEFAULT.get(suffix, {})

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # compilable[(model, idx)][setting_label] = bool  — for transition analysis
    compilable = defaultdict(dict)
    # rows for the aggregate CSV
    agg_rows = []
    # compile_rate[model][setting] for the figure
    rate = defaultdict(dict)

    data = {m: load_records(f'results/patches_validated_{m}_{suffix}.jsonl')
            for m in MODELS}

    for label, std in SETTINGS:
        os.environ['COMPILE_STD'] = std
        print(f'\n=== setting: {label} ({"-std="+std if std else "no -std flag"}) ===')
        for m in MODELS:
            recs = data[m]
            results = [None] * len(recs)
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for i, out in enumerate(ex.map(eval_patch, recs)):
                    results[i] = out
            n = len(recs)
            n_ok = sum(1 for ok, _ in results if ok)
            grp = Counter(g for ok, g in results if not ok)
            for i, (ok, _) in enumerate(results):
                compilable[(m, i)][label] = ok

            cr = n_ok / n * 100
            rate[m][label] = cr
            # NOT-THE-MODEL groups (current taxonomy, post Exp-3d reclassification):
            # harness_artifact = toolchain/stub-fixable; dataset_artifact = Big-Vul data
            # quality (e.g. stripped return type); missing_context = needs real project code.
            fixable = grp.get('harness_artifact', 0)
            dataset = grp.get('dataset_artifact', 0)
            context = grp.get('missing_context', 0)
            row = {'setting': label, 'model': m, 'n': n, 'compilable': n_ok,
                   'compile_rate_%': round(cr, 2),
                   'setup_fixable': fixable, 'setup_dataset': dataset,
                   'setup_context': context,
                   'setup_total': fixable + dataset + context}
            for g in GROUP_ORDER:
                row[g] = grp.get(g, 0)
            agg_rows.append(row)

            note = ''
            if label == 'default' and m in expected:
                exp = expected[m]
                note = f'  (expected ~{exp}% — {"OK" if abs(cr - exp) < 0.6 else "MISMATCH!"})'
            print(f'  {m:<20} compile {n_ok:>4}/{n}  = {cr:5.2f}%{note}')

    os.environ.pop('COMPILE_STD', None)

    # ── Aggregate CSV ─────────────────────────────────────────────────────────
    agg_path = f'results/compile_std_sweep{tag}.csv'
    fields = (['setting', 'model', 'n', 'compilable', 'compile_rate_%',
               'setup_fixable', 'setup_dataset', 'setup_context', 'setup_total'] + GROUP_ORDER)
    with open(agg_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(agg_rows)
    print(f'\nSaved {agg_path}')

    # ── Transition CSV (vs default) ───────────────────────────────────────────
    trans_path = f'results/compile_std_transitions{tag}.csv'
    with open(trans_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'model', 'gained_fail_to_pass',
                    'lost_pass_to_fail', 'net_change', 'compile_rate_%'])
        for label, _ in SETTINGS:
            if label == 'default':
                continue
            for m in MODELS:
                gained = lost = 0
                for i in range(len(data[m])):
                    d = compilable[(m, i)]['default']
                    s = compilable[(m, i)][label]
                    if s and not d:
                        gained += 1
                    elif d and not s:
                        lost += 1
                w.writerow([label, m, gained, lost, gained - lost,
                            round(rate[m][label], 2)])
    print(f'Saved {trans_path}')

    # ── Figure: grouped bars ──────────────────────────────────────────────────
    labels = [s[0] for s in SETTINGS]
    x = np.arange(len(MODELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for j, lab in enumerate(labels):
        vals = [rate[m][lab] for m in MODELS]
        bars = ax.bar(x + (j - 1) * width, vals, width, label=lab)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f'{v:.1f}',
                    ha='center', va='bottom', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['CodeGen-350M', 'DeepSeek-1.3B', 'DeepSeek-6.7B'])
    ax.set_ylabel('Compile rate (%)')
    ax.set_title(f'Compile rate vs. compiler -std setting (identical patches, n={"70" if suffix=="512" else "203"})\n'
                 'GCC 15 default = C23. gnu17 fixes stub arity; gnu89 also legalizes '
                 'implicit-int -> 2-3x compile rate (0 regressions on this data)')
    ax.legend(title='-std setting')
    plt.tight_layout()
    fig_path = f'figures/fig10_compile_std_sweep{tag}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {fig_path}')

    # ── Console summary ───────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('COMPILE RATE BY SETTING (same patches, no regeneration)')
    print('=' * 60)
    print(f'  {"model":<20}' + ''.join(f'{lab:>10}' for lab in labels))
    for m in MODELS:
        print(f'  {m:<20}' + ''.join(f'{rate[m][lab]:>9.2f}%' for lab in labels))
    print('\nDone.')


if __name__ == '__main__':
    main()
