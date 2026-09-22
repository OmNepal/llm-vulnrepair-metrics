"""
Multi-model comparison: pass@1, compile rate, and CWE breakdown across all runs.
Usage: python compare_models.py
Reads patches_validated_<model>_512.jsonl + patches_similarity_<model>_512.jsonl.
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
PROMPT_LABELS = {'zero_shot': 'Zero-Shot', 'few_shot': 'Few-Shot', 'cot': 'CoT'}

# Run tag — appended to all comparison output names so the 256-token (_v3) and
# 512-token (_512) comparison artifacts don't overwrite each other.
TAG = '_512'

# Models in display order with friendly labels. run_label must match the
# patches_validated_<run_label>.jsonl files produced by the pipeline.
MODELS = [
    ('codegen_350M_multi_512',  'CodeGen-350M'),
    ('deepseek_1.3b_512',       'DeepSeek-1.3B'),
    ('deepseek_6.7b_512',       'DeepSeek-6.7B'),
]

COLORS = ['#4c72b0', '#dd8452', '#55a868']


def load_records(path):
    recs = []
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
    if c == 0: return 0.0
    if n - c < k: return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def model_summary(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r['func_id'], r['cwe_id'], r['prompt_type'])].append(r.get('compilable', False))

    rows = []
    for (func_id, cwe_id, pt), outcomes in groups.items():
        n, c = len(outcomes), sum(1 for o in outcomes if o is True)
        rows.append({'func_id': func_id, 'cwe_id': cwe_id, 'prompt_type': pt,
                     'n': n, 'c': c,
                     'pass@1': pass_at_k(n, c, 1),
                     'pass@3': pass_at_k(n, c, 3),
                     'pass@5': pass_at_k(n, c, 5)})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)

    # ── Load available models ─────────────────────────────────────────────────
    available = []
    for run_label, display_name in MODELS:
        path = os.path.join('results', f'patches_validated_{run_label}.jsonl')
        if not os.path.exists(path):
            print(f"Skipping {display_name} — {path} not found")
            continue
        recs = load_records(path)
        df = model_summary(recs)

        # Optional: similarity scores from compute_similarity.py output
        sim_path = os.path.join('results', f'patches_similarity_{run_label}.jsonl')
        sim_recs = load_records(sim_path) if os.path.exists(sim_path) else []

        available.append({
            'label':       run_label,
            'name':        display_name,
            'records':     recs,
            'df':          df,
            'sim_records': sim_recs,
        })
        sim_status = f", {len(sim_recs)} with similarity" if sim_recs else ""
        print(f"Loaded {display_name}: {len(recs)} records{sim_status}")

    if not available:
        print("No model files found.")
        exit(1)

    # ── Table: pass@1 by model × prompt type ─────────────────────────────────
    print("\n" + "=" * 70)
    print("PASS@1 BY MODEL AND PROMPT TYPE")
    print("=" * 70)
    header = f"{'Model':<20}" + "".join(f"  {PROMPT_LABELS[pt]:>14}" for pt in PROMPT_TYPES) + "   Mean"
    print(header)
    print("-" * len(header))

    comparison_rows = []
    for m in available:
        df = m['df']
        row = {'model': m['name']}
        vals = []
        line = f"{m['name']:<20}"
        for pt in PROMPT_TYPES:
            sub = df[df['prompt_type'] == pt]
            v = sub['pass@1'].mean() if len(sub) > 0 else float('nan')
            row[f'{pt}_pass@1'] = v
            vals.append(v)
            line += f"  {v:>14.4f}"
        mean_v = np.nanmean(vals)
        row['mean_pass@1'] = mean_v
        line += f"   {mean_v:.4f}"
        comparison_rows.append(row)
        print(line)

    comp_df = pd.DataFrame(comparison_rows)
    comp_csv = os.path.join('results', f'comparison_table{TAG}.csv')
    comp_df.to_csv(comp_csv, index=False)
    print(f"\nSaved {comp_csv}")

    # ── Table: compile rate by model ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPILE RATE BY MODEL AND PROMPT TYPE")
    print("=" * 70)
    for m in available:
        recs = m['records']
        line = f"{m['name']:<20}"
        for pt in PROMPT_TYPES:
            pr = [r for r in recs if r['prompt_type'] == pt]
            c = sum(1 for r in pr if r.get('compilable'))
            n = len(pr)
            line += f"  {pt[:4]}={c/n*100:.1f}%"
        print(line)

    # ── Figure 4: Pass@1 grouped by prompt type, bars = models ───────────────
    n_models  = len(available)
    x         = np.arange(len(PROMPT_TYPES))
    width     = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(10, 5))
    for idx, m in enumerate(available):
        df   = m['df']
        offset = (idx - n_models / 2 + 0.5) * width
        vals = [df[df['prompt_type'] == pt]['pass@1'].mean() for pt in PROMPT_TYPES]
        bars = ax.bar(x + offset, vals, width, label=m['name'],
                      color=COLORS[idx % len(COLORS)], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001,
                        f'{h:.3f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Prompt Strategy')
    ax.set_ylabel('Pass@1 (compilable)')
    ax.set_title('Pass@1 by Model and Prompt Strategy — Big-Vul (7 CWEs, 70 functions)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([PROMPT_LABELS[pt] for pt in PROMPT_TYPES])
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    fig4_path = os.path.join('figures', f'fig4_model_comparison{TAG}.png')
    plt.savefig(fig4_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved {fig4_path}")

    # ── Figure 5: Pass@1 by model (mean across prompt types) ─────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    names  = [m['name'] for m in available]
    means  = [comp_df[comp_df['model'] == m['name']]['mean_pass@1'].values[0] for m in available]
    bars   = ax.bar(names, means, color=COLORS[:n_models], alpha=0.85, width=0.5)
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('Mean Pass@1 across prompt strategies')
    ax.set_title('Model Scaling — Mean Pass@1 on Big-Vul Vulnerability Repair', fontsize=11)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    plt.tight_layout()
    fig5_path = os.path.join('figures', f'fig5_model_scaling{TAG}.png')
    plt.savefig(fig5_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig5_path}")

    # ── Figure 6: CWE × Model heatmap (mean pass@1 across prompt types) ──────
    cwes = sorted(set(r['cwe_id'] for m in available for r in m['records']))
    heat = np.full((len(cwes), len(available)), np.nan)
    for j, m in enumerate(available):
        df = m['df']
        for i, cwe in enumerate(cwes):
            sub = df[df['cwe_id'] == cwe]
            if len(sub) > 0:
                heat[i, j] = sub['pass@1'].mean()

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(np.nan_to_num(heat, nan=0.0), cmap='YlOrRd', aspect='auto', vmin=0)
    plt.colorbar(im, ax=ax, label='Pass@1')
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([m['name'] for m in available], rotation=15, ha='right')
    ax.set_yticks(range(len(cwes)))
    ax.set_yticklabels(cwes)
    ax.set_title('Pass@1 by CWE and Model (mean across prompt strategies)', fontsize=11)
    for i in range(len(cwes)):
        for j in range(len(available)):
            v = heat[i, j]
            txt = f'{v:.3f}' if not math.isnan(v) else 'N/A'
            ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                    color='black' if (math.isnan(v) or v < 0.5) else 'white')
    plt.tight_layout()
    fig6_path = os.path.join('figures', f'fig6_cwe_model_heatmap{TAG}.png')
    plt.savefig(fig6_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig6_path}")

    # ── Similarity-based comparison (functional correctness proxy) ────────────
    have_sim = [m for m in available if m['sim_records']]
    if not have_sim:
        print(f"\nDone — {len(available)} model(s) compared (no similarity data).")
        exit(0)

    print("\n" + "=" * 70)
    print("FUNCTIONAL CORRECTNESS — CodeBLEU & String Similarity vs func_after")
    print("=" * 70)
    header = f"{'Model':<20}" + f"{'StringSim':>12}" + f"{'CodeBLEU':>12}" + \
             f"{'Syntax':>10}" + f"{'DataFlow':>12}"
    print(header)
    print("-" * len(header))

    sim_rows = []
    for m in have_sim:
        recs = m['sim_records']
        n = len(recs)
        ss   = sum(r['string_similarity']   for r in recs) / n
        cb   = sum(r['codebleu']            for r in recs) / n
        syn  = sum(r['codebleu_syntax']     for r in recs) / n
        df_  = sum(r['codebleu_dataflow']   for r in recs) / n
        sim_rows.append({'model': m['name'], 'n': n,
                         'string_similarity': ss, 'codebleu': cb,
                         'codebleu_syntax': syn, 'codebleu_dataflow': df_})
        print(f"{m['name']:<20}{ss:>12.4f}{cb:>12.4f}{syn:>10.4f}{df_:>12.4f}")

    sim_df = pd.DataFrame(sim_rows)
    sim_csv = os.path.join('results', f'similarity_comparison{TAG}.csv')
    sim_df.to_csv(sim_csv, index=False)
    print(f"\nSaved {sim_csv}")

    # ── Figure 7: CodeBLEU & String Similarity grouped bar chart ─────────────
    metrics = [('string_similarity', 'String Similarity'),
               ('codebleu',           'CodeBLEU'),
               ('codebleu_syntax',    'Syntax Match'),
               ('codebleu_dataflow',  'DataFlow Match')]
    n_metrics = len(metrics)
    x = np.arange(len(have_sim))
    width = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(10, 5))
    metric_colors = ['#4c72b0', '#dd8452', '#55a868', '#c44e52']
    for i, (key, label) in enumerate(metrics):
        offset = (i - n_metrics / 2 + 0.5) * width
        vals = [sim_df[sim_df['model'] == m['name']][key].values[0] for m in have_sim]
        bars = ax.bar(x + offset, vals, width, label=label,
                      color=metric_colors[i], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                    f'{h:.2f}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m['name'] for m in have_sim])
    ax.set_ylabel('Score (0–1)')
    ax.set_title('Functional Correctness Metrics — vs Ground-Truth Fix (func_after)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig7_path = os.path.join('figures', f'fig7_similarity_comparison{TAG}.png')
    plt.savefig(fig7_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig7_path}")

    # ── Figure 8: Compile rate vs CodeBLEU — shows the metrics measure
    # different things. A bar chart with both on the same x-axis (models).
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    model_names = [m['name'] for m in have_sim]
    compile_rates = []
    codebleu_means = []
    for m in have_sim:
        recs = m['records']
        comp = sum(1 for r in recs if r.get('compilable')) / len(recs)
        compile_rates.append(comp)
        codebleu_means.append(
            sim_df[sim_df['model'] == m['name']]['codebleu'].values[0]
        )

    x = np.arange(len(model_names))
    width = 0.35
    b1 = ax1.bar(x - width / 2, compile_rates, width,
                 color='#4c72b0', alpha=0.85, label='Compile Rate')
    b2 = ax2.bar(x + width / 2, codebleu_means, width,
                 color='#dd8452', alpha=0.85, label='CodeBLEU')

    for bar, v in zip(b1, compile_rates):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.002,
                 f'{v*100:.1f}%', ha='center', va='bottom', fontsize=8)
    for bar, v in zip(b2, codebleu_means):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8)

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names)
    ax1.set_ylabel('Compile Rate', color='#4c72b0')
    ax2.set_ylabel('CodeBLEU', color='#dd8452')
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax1.set_ylim(0, max(compile_rates) * 1.4)
    ax2.set_ylim(0, 1)
    ax1.set_title('Two views of model quality: Compile Rate vs CodeBLEU', fontsize=11)
    fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
    plt.tight_layout()
    fig8_path = os.path.join('figures', f'fig8_compile_vs_codebleu{TAG}.png')
    plt.savefig(fig8_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig8_path}")

    print(f"\nDone — {len(available)} model(s) compared, {len(have_sim)} with similarity.")
