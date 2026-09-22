"""Analyze the compiler-feedback loop (Phase 2).

For each feedback file (one record per function per round), report the trajectory
over rounds 0..N of:
  - compile rate (from the stored `compilable`; or re-checked locally with --local-recompile)
  - change-in-fix-quality: CodeBLEU + string-sim of each round's patch vs func_after
    (whole-function; reuses compute_similarity.score_pair)
  - CodeBLEU split by compilable vs not — to test whether "made it compile" also means
    "closer to the human fix", the core eval-critique hypothesis.

Auto-detects results/patches_feedback_*.jsonl (base + instruct) unless --inputs given.

Outputs:
  results/feedback_analysis.csv        per (model, round) metrics
  figures/fig11_feedback_rounds.png    compile rate + CodeBLEU vs round, per model

Run from project root:  python src/analyze_feedback.py [--local-recompile]
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_similarity import score_pair  # noqa: E402

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


def model_label(path):
    """Internal run identifier (used for the CSV + console, keep stable)."""
    base = os.path.splitext(os.path.basename(path))[0]
    return base.replace('patches_feedback_', '')


# Reader-facing names for the figure legend. The internal identifiers above
# encode our file naming (e.g. the '_200' suffix marks the merged scale-up
# files, where n is actually 203), which must never reach a published figure.
DISPLAY_NAMES = {
    'deepseek_1.3b':              'DeepSeek-1.3B (base)',
    'deepseek_1.3b_200':          'DeepSeek-1.3B (base)',
    'deepseek_1.3b_instruct':     'DeepSeek-1.3B (instruct)',
    'deepseek_1.3b_instruct_200': 'DeepSeek-1.3B (instruct)',
}


def display_label(label: str) -> str:
    """Legend name for plots. Falls back to a de-underscored alias."""
    if label in DISPLAY_NAMES:
        return DISPLAY_NAMES[label]
    base = label[:-4] if label.endswith('_200') else label
    return base.replace('_', ' ')


def analyze_file(path, local_recompile):
    recs = load_records(path)
    by_round = defaultdict(list)
    for r in recs:
        by_round[int(r.get('round', 0))].append(r)

    ic = None
    if local_recompile:
        from validation.compile_check import is_compilable as ic  # noqa: N806

    rows = []
    for rnd in sorted(by_round):
        rr = by_round[rnd]
        n = len(rr)

        # compile status: stored (Kaggle gcc) or re-checked locally (GCC 15)
        if ic:
            comp = [ic(r.get('patch', '') or '')['compilable'] for r in rr]
        else:
            comp = [bool(r.get('compilable')) for r in rr]
        n_ok = sum(comp)

        # fix-quality vs func_after (whole-function CodeBLEU + string-sim)
        cb, ss, cb_ok, cb_bad = [], [], [], []
        for r, ok in zip(rr, comp):
            sc = score_pair(r.get('patch', '') or '', r.get('func_after', '') or '')
            cb.append(sc['codebleu']); ss.append(sc['string_similarity'])
            (cb_ok if ok else cb_bad).append(sc['codebleu'])

        mean = lambda x: sum(x) / len(x) if x else 0.0
        rows.append({
            'round': rnd, 'n': n, 'n_compilable': n_ok,
            'compile_rate_%': round(n_ok / n * 100, 2),
            'codebleu': round(mean(cb), 4),
            'string_sim': round(mean(ss), 4),
            'codebleu_compilable': round(mean(cb_ok), 4),
            'codebleu_noncompilable': round(mean(cb_bad), 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description='Feedback-loop trajectory analysis')
    ap.add_argument('--inputs', '-i', nargs='*', default=None,
                    help='feedback JSONL files (default: results/patches_feedback_*.jsonl)')
    ap.add_argument('--local-recompile', action='store_true',
                    help='re-check compile with the local GCC-15 harness instead of the '
                         'stored (Kaggle gcc) status')
    args = ap.parse_args()
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    files = args.inputs or sorted(glob.glob('results/patches_feedback_*.jsonl'))
    if not files:
        print('No feedback files found (results/patches_feedback_*.jsonl). '
              'Place the base/instruct feedback JSONLs there first.')
        return

    all_rows = {}
    for path in files:
        label = model_label(path)
        print(f'\n=== {label}  ({os.path.basename(path)}) ===')
        rows = analyze_file(path, args.local_recompile)
        all_rows[label] = rows
        src = 'GCC-15 local' if args.local_recompile else 'Kaggle gcc (stored)'
        print(f'  compile source: {src}')
        print(f'  {"round":>5} {"n":>4} {"compile%":>9} {"codeBLEU":>9} {"str_sim":>8} '
              f'{"CB|compiles":>12} {"CB|fails":>9}')
        for r in rows:
            print(f'  {r["round"]:>5} {r["n"]:>4} {r["compile_rate_%"]:>8.2f}% '
                  f'{r["codebleu"]:>9.4f} {r["string_sim"]:>8.4f} '
                  f'{r["codebleu_compilable"]:>12.4f} {r["codebleu_noncompilable"]:>9.4f}')
        if len(rows) > 1:
            d_c = rows[-1]['compile_rate_%'] - rows[0]['compile_rate_%']
            d_b = rows[-1]['codebleu'] - rows[0]['codebleu']
            print(f'  delta round0->{rows[-1]["round"]}: compile {d_c:+.2f}pp | '
                  f'CodeBLEU {d_b:+.4f}')

    # ── CSV ───────────────────────────────────────────────────────────────────
    import csv
    csv_path = 'results/feedback_analysis.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['model', 'round', 'n', 'n_compilable', 'compile_rate_%',
                    'codebleu', 'string_sim', 'codebleu_compilable', 'codebleu_noncompilable'])
        for label, rows in all_rows.items():
            for r in rows:
                w.writerow([label, r['round'], r['n'], r['n_compilable'],
                            r['compile_rate_%'], r['codebleu'], r['string_sim'],
                            r['codebleu_compilable'], r['codebleu_noncompilable']])
    print(f'\nSaved {csv_path}')

    # ── Figure: compile rate + CodeBLEU vs round ──────────────────────────────
    # Publication figure: sized close to an LNCS text width so it is not
    # heavily downscaled by \includegraphics[width=\linewidth], and saved as
    # vector PDF (fonttype 42 to avoid Type-3 fonts, which Springer rejects).
    plt.rcParams.update({
        'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for label, rows in all_rows.items():
        xs = [r['round'] for r in rows]
        name = display_label(label)
        ax1.plot(xs, [r['compile_rate_%'] for r in rows], marker='o', label=name)
        ax2.plot(xs, [r['codebleu'] for r in rows], marker='o', label=name)
    ax1.set_title('Compile rate')
    ax1.set_xlabel('feedback round'); ax1.set_ylabel('compile rate (%)')
    ax2.set_title('Similarity to the human fix')
    ax2.set_xlabel('feedback round'); ax2.set_ylabel('CodeBLEU vs. human fix')
    for ax in (ax1, ax2):
        ax.set_xticks(sorted({r['round'] for rows in all_rows.values() for r in rows}))
        ax.legend()
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig_path = f'figures/fig11_feedback_rounds.{ext}'
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f'Saved {fig_path}')
    plt.close(fig)
    print('\nDone.')


if __name__ == '__main__':
    main()
