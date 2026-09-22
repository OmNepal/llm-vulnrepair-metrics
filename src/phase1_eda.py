import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import random

load_dotenv()

DATA_PATH = os.getenv('DATA_PATH', os.path.join('data', 'MSR_data_cleaned.csv'))
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# ── Load ──────────────────────────────────────────────────────────────────────
COLS_NEEDED = ['vul', 'CWE ID', 'CVE ID', 'func_before', 'func_after', 'project']

print("Loading dataset (key columns only to stay within RAM)...")
try:
    df_full = pd.read_csv(DATA_PATH, encoding='utf-8', usecols=COLS_NEEDED)
except UnicodeDecodeError:
    df_full = pd.read_csv(DATA_PATH, encoding='latin-1', usecols=COLS_NEEDED)

# Also read all column names separately (cheap — read 0 rows)
all_cols = pd.read_csv(DATA_PATH, encoding='utf-8', nrows=0).columns.tolist()

print(f"Full shape: {df_full.shape}")
print(f"All columns ({len(all_cols)}): {all_cols}")
print(f"Loaded columns: {list(df_full.columns)}")
print(f"dtypes:\n{df_full[COLS_NEEDED].dtypes}")
print(f"Memory (deep, loaded cols): {df_full.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

# Sample for EDA
df = df_full.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE).reset_index(drop=True)
print(f"Working sample: {df.shape[0]:,} rows\n")

# ── EDA ───────────────────────────────────────────────────────────────────────
os.makedirs('report', exist_ok=True)
os.makedirs('figures', exist_ok=True)

lines = []

def log(msg=''):
    print(msg)
    lines.append(msg)

log("=" * 60)
log("BIG-VUL EDA REPORT")
log("=" * 60)

# 1. Vulnerability balance
vul_counts = df['vul'].value_counts().sort_index()
vul_pct = df['vul'].value_counts(normalize=True).sort_index() * 100
log("\n--- Vulnerability Balance (sample) ---")
for v in vul_counts.index:
    log(f"  vul={v}: {vul_counts[v]:,} rows ({vul_pct[v]:.1f}%)")

# Also full dataset balance
vul_full = df_full['vul'].value_counts().sort_index()
log("\n--- Vulnerability Balance (full dataset) ---")
for v in vul_full.index:
    log(f"  vul={v}: {vul_full[v]:,} rows ({vul_full[v]/len(df_full)*100:.1f}%)")

# 2. CWE distribution
log("\n--- CWE ID Distribution (top 20, full dataset) ---")
cwe_counts_full = df_full['CWE ID'].dropna().value_counts()
log(f"  Unique CWE IDs (full): {cwe_counts_full.shape[0]}")
for cwe, cnt in cwe_counts_full.head(20).items():
    log(f"  {cwe}: {cnt:,}")

# 3. Function length
df['func_len'] = df['func_before'].dropna().apply(len)
log("\n--- func_before Character Length (sample) ---")
log(f"  min:    {df['func_len'].min():.0f}")
log(f"  mean:   {df['func_len'].mean():.0f}")
log(f"  median: {df['func_len'].median():.0f}")
log(f"  95th %: {df['func_len'].quantile(0.95):.0f}")
log(f"  max:    {df['func_len'].max():.0f}")

# 4. Null counts for key columns
log("\n--- Null Counts for Key Columns (sample) ---")
for col in ['func_before', 'func_after', 'CWE ID', 'CVE ID']:
    if col in df.columns:
        log(f"  {col}: {df[col].isna().sum()} nulls")

# 5. Unique projects and CVEs
log("\n--- Dataset Diversity (sample) ---")
if 'project' in df.columns:
    log(f"  Unique projects: {df['project'].nunique()}")
if 'CVE ID' in df.columns:
    log(f"  Unique CVE IDs:  {df['CVE ID'].nunique()}")

# 6. Three random vulnerable functions
log("\n--- 3 Random Vulnerable Functions ---")
vul_sample = df[df['vul'] == 1].dropna(subset=['func_before'])
random.seed(RANDOM_STATE)
sample_rows = vul_sample.sample(n=min(3, len(vul_sample)), random_state=RANDOM_STATE)
for i, (_, row) in enumerate(sample_rows.iterrows(), 1):
    log(f"\n[Sample {i}] CWE: {row.get('CWE ID','N/A')} | CVE: {row.get('CVE ID','N/A')}")
    snippet = str(row['func_before'])[:600]
    log(snippet + ("..." if len(str(row['func_before'])) > 600 else ""))

log("\n" + "=" * 60)

# Write report
with open(os.path.join('report', 'eda_report.txt'), 'w', encoding='utf-8', newline='\n') as f:
    f.write('\n'.join(lines))
print("\nReport saved to report/eda_report.txt")

# ── Figures ───────────────────────────────────────────────────────────────────

# Fig 1: Top 20 CWE distribution
top20_cwe = cwe_counts_full.head(20)
fig, ax = plt.subplots(figsize=(10, 7))
top20_cwe.sort_values().plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 20 CWE Types in Big-Vul (full dataset)', fontsize=13)
ax.set_xlabel('Count')
ax.set_ylabel('CWE ID')
plt.tight_layout()
plt.savefig(os.path.join('figures', 'eda_cwe_distribution.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/eda_cwe_distribution.png")

# Fig 2: Function length histogram
fig, ax = plt.subplots(figsize=(9, 5))
func_lens = df['func_len'].dropna()
ax.hist(func_lens, bins=60, color='teal', alpha=0.8, edgecolor='white')
for pct, color, label in [(0.50, 'orange', 'p50'), (0.95, 'red', 'p95')]:
    val = func_lens.quantile(pct)
    ax.axvline(val, color=color, linewidth=1.5, linestyle='--', label=f'{label}={val:.0f}')
ax.set_title('func_before Character Length Distribution (50K sample)', fontsize=13)
ax.set_xlabel('Characters')
ax.set_ylabel('Count')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'eda_function_length.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/eda_function_length.png")

# Fig 3: Vulnerability balance (full dataset)
fig, ax = plt.subplots(figsize=(6, 5))
labels = ['Not Vulnerable (0)', 'Vulnerable (1)']
counts = [vul_full.get(0, 0), vul_full.get(1, 0)]
colors = ['#4c72b0', '#dd8452']
bars = ax.bar(labels, counts, color=colors, edgecolor='white', width=0.5)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f'{count:,}', ha='center', va='bottom', fontsize=11)
ax.set_title('Vulnerability Label Balance (full dataset)', fontsize=13)
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(os.path.join('figures', 'eda_vuln_balance.png'), bbox_inches='tight', dpi=150)
plt.close()
print("Saved figures/eda_vuln_balance.png")

print("\nPhase 1 EDA complete.")
