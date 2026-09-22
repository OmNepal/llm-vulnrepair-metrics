import argparse
import math
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from collections import defaultdict

PROMPT_TYPES   = ['zero_shot', 'few_shot', 'cot']
PROMPT_LABELS  = {'zero_shot': 'Zero-Shot', 'few_shot': 'Few-Shot', 'cot': 'Chain-of-Thought'}


def _parse_args():
    parser = argparse.ArgumentParser(description='Compute Pass@k metrics from validated patches')
    parser.add_argument(
        '--input', '-i',
        default=os.path.join('results', 'patches_validated.jsonl'),
        help='Validated JSONL file (default: results/patches_validated.jsonl)',
    )
    args = parser.parse_args()

    # Derive a run label from the filename for output naming
    # patches_validated_codegen_350M_multi.jsonl → codegen_350M_multi
    # patches_validated.jsonl → baseline
    basename = os.path.splitext(os.path.basename(args.input))[0]
    if basename == 'patches_validated':
        run_label = 'baseline'
    else:
        run_label = basename.replace('patches_validated_', '')

    return args.input, run_label


# ── Unbiased Pass@k estimator ─────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator: probability that at least 1 of k samples is correct."""
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ── Load validated records ────────────────────────────────────────────────────

def load_records(path: str) -> list:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


# ── Compute per-function pass@k ───────────────────────────────────────────────

def compute_pass_at_k(records: list, correct_field: str = 'compilable') -> pd.DataFrame:
    groups = defaultdict(list)
    for r in records:
        key = (r['func_id'], r['cwe_id'], r['prompt_type'])
        groups[key].append(r.get(correct_field, False))

    rows = []
    for (func_id, cwe_id, prompt_type), outcomes in groups.items():
        n = len(outcomes)
        c = sum(1 for o in outcomes if o is True)
        rows.append({
            'func_id':     func_id,
            'cwe_id':      cwe_id,
            'prompt_type': prompt_type,
            'n':           n,
            'c':           c,
            'pass@1':      pass_at_k(n, c, 1),
            'pass@3':      pass_at_k(n, c, 3),
            'pass@5':      pass_at_k(n, c, 5),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    validated_path, run_label = _parse_args()

    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    print(f"Run     : {run_label}")
    print(f"Input   : {validated_path}")

    print("\nLoading validated patches...")
    all_records = load_records(validated_path)
    records = all_records
    print(f"  {len(records)} records across {len(set(r['cwe_id'] for r in records))} CWEs\n")

    df = compute_pass_at_k(records, correct_field='compilable')

    # Detect max n to know which pass@k values are meaningful
    max_n = int(df['n'].max()) if not df.empty else 1
    k_values = [k for k in [1, 3, 5] if k <= max_n]

    # ── Main results table ────────────────────────────────────────────────────
    print("=" * 65)
    print(f"MAIN RESULTS — Pass@k by Prompt Type  [{run_label}]")
    print("=" * 65)
    main_rows = []
    for pt in PROMPT_TYPES:
        sub = df[df['prompt_type'] == pt]
        row = {'prompt_type': pt, 'n_functions': len(sub)}
        parts = []
        for k in k_values:
            val = sub[f'pass@{k}'].mean()
            row[f'pass@{k}'] = val
            parts.append(f'pass@{k}={val:.4f}')
        main_rows.append(row)
        print(f"  {PROMPT_LABELS[pt]:20s}  {'  '.join(parts)}  (n={len(sub)})")

    main_df = pd.DataFrame(main_rows)
    main_csv = os.path.join('results', f'main_table_{run_label}.csv')
    main_df.to_csv(main_csv, index=False)
    print(f"\nSaved {main_csv}")

    # ── CWE breakdown table ───────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"CWE BREAKDOWN — Pass@1 by CWE and Prompt Type  [{run_label}]")
    print("=" * 65)
    cwe_rows = []
    cwes = sorted(df['cwe_id'].unique())
    header = f"{'CWE':<12}" + "".join(f"  {PROMPT_LABELS[pt]:>18}" for pt in PROMPT_TYPES)
    print(header)
    print("-" * len(header))
    for cwe in cwes:
        row = {'cwe_id': cwe}
        line = f"{cwe:<12}"
        for pt in PROMPT_TYPES:
            sub = df[(df['cwe_id'] == cwe) & (df['prompt_type'] == pt)]
            val = sub['pass@1'].mean() if len(sub) > 0 else float('nan')
            row[f'{pt}_pass@1'] = val
            line += f"  {val:>18.4f}" if not math.isnan(val) else f"  {'N/A':>18}"
        cwe_rows.append(row)
        print(line)

    cwe_df = pd.DataFrame(cwe_rows)
    cwe_csv = os.path.join('results', f'cwe_table_{run_label}.csv')
    cwe_df.to_csv(cwe_csv, index=False)
    print(f"\nSaved {cwe_csv}")

    # ── Validation rates ──────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"VALIDATION RATES by Prompt Type  [{run_label}]")
    print("=" * 65)
    rate_rows = []
    for pt in PROMPT_TYPES:
        pt_recs = [r for r in records if r['prompt_type'] == pt]
        n    = len(pt_recs)
        comp = sum(1 for r in pt_recs if r.get('compilable') is True)
        vf   = sum(1 for r in pt_recs if r.get('vuln_free') is True)
        rate_rows.append({'prompt_type': pt, 'total': n,
                          'compilable': comp, 'compile_rate': comp/n if n else 0,
                          'vuln_free': vf,   'vuln_free_rate': vf/n if n else 0})
        print(f"  {PROMPT_LABELS[pt]:20s}  compile={comp/n*100:.1f}%  "
              f"vuln_free={vf/n*100:.1f}%  (n={n})")

    rate_df = pd.DataFrame(rate_rows)

    # ── Figure 1: Pass@k grouped bar chart ───────────────────────────────────
    k_cols   = [f'pass@{k}' for k in k_values]
    k_labels = [f'Pass@{k}' for k in k_values]
    colors   = ['#4c72b0', '#dd8452', '#55a868']
    x        = np.arange(len(PROMPT_TYPES))
    width    = 0.8 / len(k_cols)

    fig, ax = plt.subplots(figsize=(9, 5))
    for idx, (col, lbl, color) in enumerate(zip(k_cols, k_labels, colors)):
        offset = (idx - len(k_cols)/2 + 0.5) * width
        vals   = [main_df[main_df['prompt_type'] == pt][col].values[0] for pt in PROMPT_TYPES]
        bars   = ax.bar(x + offset, vals, width, label=lbl, color=color, alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.001,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Pass@k (compilable)')
    ax.set_title(f'Pass@k by Prompt Strategy — {run_label} on Big-Vul (7 CWEs)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[pt] for pt in PROMPT_TYPES])
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    fig1_path = os.path.join('figures', f'fig1_pass_at_k_{run_label}.png')
    plt.savefig(fig1_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"\nSaved {fig1_path}")

    # ── Figure 2: CWE × Prompt heatmap ───────────────────────────────────────
    heat_data = np.full((len(cwes), len(PROMPT_TYPES)), np.nan)
    for i, cwe in enumerate(cwes):
        for j, pt in enumerate(PROMPT_TYPES):
            row = cwe_df[cwe_df['cwe_id'] == cwe]
            if not row.empty:
                heat_data[i, j] = row[f'{pt}_pass@1'].values[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.nan_to_num(heat_data, nan=0.0), cmap='YlOrRd', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Pass@1')
    ax.set_xticks(range(len(PROMPT_TYPES)))
    ax.set_xticklabels([PROMPT_LABELS[pt] for pt in PROMPT_TYPES], rotation=15, ha='right')
    ax.set_yticks(range(len(cwes)))
    ax.set_yticklabels(cwes)
    ax.set_title(f'Pass@1 Heatmap — {run_label}', fontsize=11)
    for i in range(len(cwes)):
        for j in range(len(PROMPT_TYPES)):
            val = heat_data[i, j]
            txt = f'{val:.3f}' if not math.isnan(val) else 'N/A'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='black' if (math.isnan(val) or val < 0.5) else 'white')
    plt.tight_layout()
    fig2_path = os.path.join('figures', f'fig2_cwe_heatmap_{run_label}.png')
    plt.savefig(fig2_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {fig2_path}")

    # ── Figure 3: Compile rate vs vuln-free rate ──────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    x     = np.arange(len(PROMPT_TYPES))
    width = 0.35
    c_vals = [rate_df[rate_df['prompt_type'] == pt]['compile_rate'].values[0] for pt in PROMPT_TYPES]
    v_vals = [rate_df[rate_df['prompt_type'] == pt]['vuln_free_rate'].values[0] for pt in PROMPT_TYPES]
    bc = ax.bar(x - width/2, c_vals, width, label='Compilable', color='#4c72b0', alpha=0.85)
    bv = ax.bar(x + width/2, v_vals, width, label='Vuln-Free (Semgrep)', color='#55a868', alpha=0.85)
    for bar in list(bc) + list(bv):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.005,
                f'{h*100:.1f}%', ha='center', va='bottom', fontsize=8)
    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Rate')
    ax.set_title(f'Compilability vs Vuln-Free Rate — {run_label}', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[pt] for pt in PROMPT_TYPES])
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    plt.tight_layout()
    fig3_path = os.path.join('figures', f'fig3_compile_vs_vulnfree_{run_label}.png')
    plt.savefig(fig3_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Saved {fig3_path}")

    print(f"\nDone — {run_label}.")
