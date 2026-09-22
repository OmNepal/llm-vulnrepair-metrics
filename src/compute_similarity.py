"""
Compute functional-correctness proxy metrics: CodeBLEU + string similarity vs
the ground-truth fix (func_after).

Why: compile rate alone measures syntactic validity, not whether the patch
actually fixes the vulnerability the way the project's developers fixed it.
Comparing the generated patch to func_after captures that 'is the fix close to
correct' question. This addresses the professor's evaluation criterion #6
(functional correctness).

Usage:
    python compute_similarity.py --input results/patches_validated_X_v3.jsonl

Adds these fields to each record and writes results/patches_similarity_X.jsonl:
    string_similarity   - SequenceMatcher ratio vs func_after
    codebleu            - overall CodeBLEU score
    codebleu_ngram      - n-gram overlap component
    codebleu_syntax     - AST-match component
    codebleu_dataflow   - data-flow match component

Then prints mean scores by prompt type and saves a per-model summary CSV.
"""
import argparse
import json
import os
from difflib import SequenceMatcher
from collections import defaultdict

from tqdm import tqdm
from codebleu import calc_codebleu

def _parse_args():
    p = argparse.ArgumentParser(description='CodeBLEU + similarity vs func_after')
    p.add_argument('--input', '-i', required=True, help='Validated JSONL file')
    p.add_argument('--output', '-o', default=None, help='Output JSONL (default derived)')
    args = p.parse_args()

    base = os.path.splitext(os.path.basename(args.input))[0]
    if base.startswith('patches_validated'):
        out_base = base.replace('patches_validated', 'patches_similarity')
    else:
        out_base = 'patches_similarity_' + base
    out_path = args.output or os.path.join('results', out_base + '.jsonl')

    run_label = base.replace('patches_validated_', '').replace('_v3', '')
    return args.input, out_path, run_label


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


def score_pair(patch: str, reference: str) -> dict:
    """Compute similarity scores for one (patch, reference) pair."""
    if not patch or not reference:
        return {'string_similarity': 0.0, 'codebleu': 0.0,
                'codebleu_ngram': 0.0, 'codebleu_syntax': 0.0,
                'codebleu_dataflow': 0.0}

    string_sim = SequenceMatcher(None, patch, reference).ratio()
    try:
        cb = calc_codebleu([reference], [patch], lang='c')
        return {
            'string_similarity': string_sim,
            'codebleu':          float(cb.get('codebleu', 0.0)),
            'codebleu_ngram':    float(cb.get('ngram_match_score', 0.0)),
            'codebleu_syntax':   float(cb.get('syntax_match_score', 0.0)),
            'codebleu_dataflow': float(cb.get('dataflow_match_score', 0.0)),
        }
    except Exception:
        return {'string_similarity': string_sim, 'codebleu': 0.0,
                'codebleu_ngram': 0.0, 'codebleu_syntax': 0.0,
                'codebleu_dataflow': 0.0}


if __name__ == '__main__':
    input_path, output_path, run_label = _parse_args()
    os.makedirs('results', exist_ok=True)

    print(f"Run    : {run_label}")
    print(f"Input  : {input_path}")
    print(f"Output : {output_path}\n")

    records = load_records(input_path)
    print(f"Loaded {len(records)} records\n")

    with open(output_path, 'w', encoding='utf-8', newline='\n') as out_f:
        for r in tqdm(records, desc='Scoring', ascii=True, unit='patch'):
            scores = score_pair(r.get('patch', ''), r.get('func_after', ''))
            scored = {**r, **scores}
            out_f.write(json.dumps(scored, ensure_ascii=False) + '\n')

    # Re-load with new scores for summary
    scored = load_records(output_path)

    # Mean by prompt type
    print("\n" + "=" * 70)
    print(f"SIMILARITY SCORES BY PROMPT TYPE  [{run_label}]")
    print("=" * 70)
    print(f"{'prompt':<14} {'n':>5} {'string_sim':>11} {'codeBLEU':>10} "
          f"{'syntax':>9} {'dataflow':>10}")
    print("-" * 70)

    by_pt = defaultdict(list)
    for r in scored:
        by_pt[r['prompt_type']].append(r)

    summary_rows = []
    for pt in ['zero_shot', 'few_shot', 'cot']:
        recs = by_pt.get(pt, [])
        if not recs:
            continue
        n = len(recs)
        mean = lambda key: sum(r[key] for r in recs) / n
        ss   = mean('string_similarity')
        cb   = mean('codebleu')
        syn  = mean('codebleu_syntax')
        df_  = mean('codebleu_dataflow')
        summary_rows.append({
            'prompt_type': pt, 'n': n,
            'string_similarity': ss, 'codebleu': cb,
            'codebleu_syntax': syn, 'codebleu_dataflow': df_,
        })
        print(f"{pt:<14} {n:>5} {ss:>11.4f} {cb:>10.4f} {syn:>9.4f} {df_:>10.4f}")

    # Save summary CSV
    import pandas as pd
    summary_df = pd.DataFrame(summary_rows)
    csv_path = os.path.join('results', f'similarity_table_{run_label}.csv')
    summary_df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")
    print(f"\nDone — {run_label}.")
