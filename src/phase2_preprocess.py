import os
import json
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH', os.path.join('data', 'MSR_data_cleaned.csv'))
MAX_FUNC_CHARS = int(os.getenv('MAX_FUNC_CHARS', 3000))
RANDOM_STATE = 42

TOP_CWES = [
    'CWE-119', 'CWE-20',  'CWE-399', 'CWE-264', 'CWE-200', 'CWE-125', 'CWE-189',
]
FEWSHOT_PER_CWE = 3
TEST_PER_CWE = 10

COLS_NEEDED = ['vul', 'CWE ID', 'CVE ID', 'func_before', 'func_after', 'project']

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading dataset...")
try:
    df = pd.read_csv(DATA_PATH, encoding='utf-8', usecols=COLS_NEEDED)
except UnicodeDecodeError:
    df = pd.read_csv(DATA_PATH, encoding='latin-1', usecols=COLS_NEEDED)

print(f"Loaded: {df.shape[0]:,} rows")

# ── Step 1: Filter ────────────────────────────────────────────────────────────
n_before = len(df)

df = df[df['vul'] == 1]
print(f"\nAfter vul=1 filter:          {len(df):,} rows  (removed {n_before - len(df):,})")

n_after_vul = len(df)
df = df.dropna(subset=['func_before', 'func_after', 'CWE ID'])
print(f"After dropping nulls:        {len(df):,} rows  (removed {n_after_vul - len(df):,})")

n_after_null = len(df)
df = df[df['func_before'].str.len() <= MAX_FUNC_CHARS]
print(f"After len <= {MAX_FUNC_CHARS}:        {len(df):,} rows  (removed {n_after_null - len(df):,})")

# ── Step 2: Restrict to the seven evaluated CWE categories ───────────────────────────────────────────
df = df[df['CWE ID'].isin(TOP_CWES)]
print(f"\nAfter CWE filter:            {len(df):,} rows")

print("\nCWE breakdown after all filters:")
cwe_counts = df['CWE ID'].value_counts()
for cwe in TOP_CWES:
    cnt = cwe_counts.get(cwe, 0)
    needed = FEWSHOT_PER_CWE + TEST_PER_CWE
    status = "OK" if cnt >= needed else f"LOW — only {cnt} rows, need {needed}"
    print(f"  {cwe}: {cnt:,}  [{status}]")

# ── Step 3: Split per CWE ─────────────────────────────────────────────────────
fewshot_pool = {}
test_rows = []

for cwe in TOP_CWES:
    subset = (
        df[df['CWE ID'] == cwe]
        .sample(frac=1, random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )

    available = len(subset)
    if available < FEWSHOT_PER_CWE + TEST_PER_CWE:
        print(f"WARNING: {cwe} has only {available} rows — using all available")

    few = subset.iloc[:FEWSHOT_PER_CWE]
    test = subset.iloc[FEWSHOT_PER_CWE: FEWSHOT_PER_CWE + TEST_PER_CWE]

    fewshot_pool[cwe] = [
        {'func_before': row['func_before'], 'func_after': row['func_after']}
        for _, row in few.iterrows()
    ]

    for i, (_, row) in enumerate(test.iterrows()):
        test_rows.append({
            'func_id': f"{cwe}_{i:02d}",
            'cwe_id': cwe,
            'func_before': row['func_before'],
            'func_after': row['func_after'],
            'cve_id': row.get('CVE ID', ''),
            'project': row.get('project', ''),
        })

# ── Step 4: Save ──────────────────────────────────────────────────────────────
test_df = pd.DataFrame(test_rows)
test_df.to_csv(os.path.join('data', 'test_functions.csv'), index=False, encoding='utf-8')

with open(os.path.join('data', 'fewshot_pool.json'), 'w', encoding='utf-8', newline='\n') as f:
    json.dump(fewshot_pool, f, indent=2, ensure_ascii=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nSaved data/test_functions.csv  — {len(test_df)} rows")
print(f"Saved data/fewshot_pool.json   — {sum(len(v) for v in fewshot_pool.values())} examples across {len(fewshot_pool)} CWEs")

print("\nTest set breakdown by CWE:")
for cwe, cnt in test_df['cwe_id'].value_counts().sort_index().items():
    print(f"  {cwe}: {cnt} test functions")

print("\nPhase 2 preprocessing complete.")
