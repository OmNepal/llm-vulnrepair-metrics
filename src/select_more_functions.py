"""Select additional Big-Vul test functions to scale the test set from 70 -> ~200,
ADDITIVELY and safely (never touches the existing 70).

Replicates the original selection (src/phase2_preprocess.py): same filters, same
per-CWE shuffle (seed 42). The existing test used rows [3:13] per CWE; we take the
NEXT unused functions. Safety: functions are excluded by CONTENT (func_before text)
against the *deployed* test set + few-shot pool, so we can never overlap even if the
shuffle does not reproduce byte-for-byte. Also dedupes within the new set (Big-Vul
has known duplication).

Outputs (all under data/scale200/, named test_functions.csv so the Kaggle notebooks'
glob finds them; fewshot_pool.json copied alongside for few_shot prompts):
  data/scale200/extra_all/     -> codegen, 1.3b, instruct (full new set)
  data/scale200/6b_batch1/     -> deepseek-6.7b batch 1 (half)  [<12h each]
  data/scale200/6b_batch2/     -> deepseek-6.7b batch 2 (half)
  data/test_functions_200.csv  -> combined 70 + new (final test set, reference)

Run from project root:  python src/select_more_functions.py --per-cwe 19
"""
import argparse
import json
import os
import shutil

import pandas as pd

DATA_PATH = os.getenv('DATA_PATH', os.path.join('data', 'MSR_data_cleaned.csv'))
MAX_FUNC_CHARS = int(os.getenv('MAX_FUNC_CHARS', 3000))
RANDOM_STATE = 42
TOP_CWES = ['CWE-119', 'CWE-20', 'CWE-399', 'CWE-264', 'CWE-200', 'CWE-125', 'CWE-189']
COLS = ['vul', 'CWE ID', 'CVE ID', 'func_before', 'func_after', 'project']
EXISTING_PER_CWE = 13  # 3 few-shot + 10 test in the original per-CWE shuffle


def _norm(s):
    return (s or '').strip()


def main():
    ap = argparse.ArgumentParser(description='Select additional test functions (additive)')
    ap.add_argument('--per-cwe', type=int, default=19,
                    help='new functions per CWE (default 19 -> ~133 new, ~203 total)')
    args = ap.parse_args()
    out_root = os.path.join('data', 'scale200')

    # ── deployed exclusion set (by content) ───────────────────────────────────
    existing = pd.read_csv(os.path.join('data', 'test_functions.csv'), encoding='utf-8')
    exclude = {_norm(t) for t in existing['func_before']}
    fewshot = json.load(open(os.path.join('data', 'fewshot_pool.json'), encoding='utf-8'))
    for cwe, exs in fewshot.items():
        for e in exs:
            exclude.add(_norm(e['func_before']))
    print(f'Deployed exclusion set: {len(existing)} test + '
          f'{sum(len(v) for v in fewshot.values())} few-shot functions')

    # ── load Big-Vul with the SAME filters as the original ────────────────────
    print(f'Loading {DATA_PATH} (this can take a minute) ...')
    try:
        df = pd.read_csv(DATA_PATH, encoding='utf-8', usecols=COLS)
    except UnicodeDecodeError:
        df = pd.read_csv(DATA_PATH, encoding='latin-1', usecols=COLS)
    df = df[df['vul'] == 1].dropna(subset=['func_before', 'func_after', 'CWE ID'])
    df = df[df['func_before'].str.len() <= MAX_FUNC_CHARS]
    df = df[df['CWE ID'].isin(TOP_CWES)]
    print(f'Big-Vul after filters: {len(df):,} rows')

    # ── per-CWE: take next `per_cwe` unused, deduped functions ─────────────────
    new_rows, seen_new = [], set()
    print('\nPer-CWE selection:')
    for cwe in TOP_CWES:
        subset = (df[df['CWE ID'] == cwe]
                  .sample(frac=1, random_state=RANDOM_STATE)
                  .reset_index(drop=True))
        picked = 0
        for _, row in subset.iterrows():
            t = _norm(row['func_before'])
            if t in exclude or t in seen_new:
                continue
            if _norm(row['func_after']) == t:   # skip no-op "fixes" (identical before/after)
                continue
            seen_new.add(t)
            new_rows.append({
                'func_id': f'{cwe}_{10 + picked:02d}',   # existing used _00.._09
                'cwe_id': cwe,
                'func_before': row['func_before'],
                'func_after': row['func_after'],
                'cve_id': row.get('CVE ID', ''),
                'project': row.get('project', ''),
            })
            picked += 1
            if picked >= args.per_cwe:
                break
        avail = len(subset) - EXISTING_PER_CWE
        print(f'  {cwe}: +{picked}  (requested {args.per_cwe}, ~{max(avail,0)} available beyond existing)')

    extra = pd.DataFrame(new_rows)

    # ── verification (fail loudly if anything is wrong) ───────────────────────
    assert extra['func_id'].is_unique, 'duplicate func_id in new set!'
    assert not set(extra['func_id']) & set(existing['func_id']), 'func_id collides with existing!'
    overlap = {_norm(t) for t in extra['func_before']} & exclude
    assert not overlap, f'{len(overlap)} new functions overlap the deployed set!'
    assert extra['func_before'].map(_norm).is_unique, 'duplicate func_before within new set!'
    print(f'\nVerification OK: {len(extra)} new functions, no id/content overlap, no internal dupes.')

    # ── write outputs (additive; never overwrite the 70) ──────────────────────
    def _emit(folder, frame):
        d = os.path.join(out_root, folder)
        os.makedirs(d, exist_ok=True)
        frame.to_csv(os.path.join(d, 'test_functions.csv'), index=False, encoding='utf-8')
        shutil.copy(os.path.join('data', 'fewshot_pool.json'),
                    os.path.join(d, 'fewshot_pool.json'))
        print(f'  wrote {d}/test_functions.csv ({len(frame)} funcs) + fewshot_pool.json')

    print('\nWriting Kaggle-ready folders:')
    _emit('extra_all', extra)                       # codegen, 1.3b, instruct
    b1 = extra.iloc[0::2].reset_index(drop=True)    # interleaved halves -> balanced CWEs
    b2 = extra.iloc[1::2].reset_index(drop=True)
    _emit('6b_batch1', b1)
    _emit('6b_batch2', b2)
    assert len(b1) + len(b2) == len(extra), 'batch split lost rows!'

    # combined 70 + new (final test set, for reference/records)
    combined = pd.concat([existing, extra], ignore_index=True)
    combined.to_csv(os.path.join('data', 'test_functions_200.csv'), index=False, encoding='utf-8')
    print(f'\nCombined final test set: data/test_functions_200.csv '
          f'({len(existing)} existing + {len(extra)} new = {len(combined)})')
    print('\nDone. Upload each data/scale200/<folder> as a Kaggle dataset (base models use '
          'the ORIGINAL generate_kaggle.ipynb; instruct uses generate_kaggle_v2.ipynb).')


if __name__ == '__main__':
    main()
