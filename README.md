# Metrics Failure in LLM-Based Code Vulnerability Repair

Replication package for the paper *"Metrics Failure in LLM-Based Code Vulnerability
Repair: An Empirical Study and a Change-Aware Screen."*

The paper asks whether the metrics the field uses to judge LLM-generated vulnerability
patches are trustworthy. Across five controlled experiments on 203 vulnerable functions
from Big-Vul, three open-source code LLMs and three prompting strategies, it reports that
compile rate is unreliable in five distinct ways, that whole-function CodeBLEU is beaten by
a no-op copy of the vulnerable input, and it examines `diff_F1`, a change-aware screen that
scores only the edited region.

This repository contains the compile harness, the `diff_F1` implementation, the prompt
templates, the test-set function identifiers, and the scripts and derived result files that
reproduce every table and figure in the paper.

## What is here

| Path | Contents |
|---|---|
| `src/validation/compile_check.py` | The iterative stub-injection compile harness (Algorithm 1) |
| `src/validation/semgrep_check.py` | Semgrep wrapper |
| `src/diff_aware_score.py` | `diff_F1` and the copy-input baseline |
| `src/gaming_baselines.py` | `diff_F1` on synthetic and real degenerate patches |
| `src/*.py` | Generation, validation, scoring and analysis pipeline |
| `notebooks/*.ipynb` | Kaggle GPU generation and the compiler-feedback loop |
| `replication/PROMPTS.md` | All prompt templates, verbatim, plus generation settings |
| `data/test_set_203.csv` | The 203 evaluated functions: identifiers, CWE, CVE, project |
| `results/*.csv` | Derived results backing every table and figure |
| `figures/` | The compiler-feedback figure |

`data/test_set_203.csv` lists identifiers only. Big-Vul function bodies are not
redistributed here; see *Reproducing from scratch* below.

## Reproducing the tables and figures

Every number in the paper can be checked directly against the CSV files in `results/`,
with no GPU and no dataset download:

| Paper item | Script | Derived file |
|---|---|---|
| Table 5, headline metrics and CIs | `bootstrap_ci.py`, `compute_metrics.py`, `compute_similarity.py` | `bootstrap_ci.csv`, `main_table_*_200.csv`, `similarity_table_*_200.csv` |
| Table 6, compile rate by prompting strategy | `compute_metrics.py` | `main_table_*_200.csv` |
| Table 7, generation budget 256 vs 512 | `compare_token_budgets.py` | `comparison_256_vs_512.csv` |
| Table 8, compile-failure taxonomy | `error_analysis.py` | `failure_categories_*_200.csv` |
| Table 9, `-std` sensitivity | `compile_std_sweep.py` | `compile_std_sweep{,_200}.csv`, `compile_std_transitions{,_200}.csv` |
| Table 10, CodeBLEU versus `diff_F1` | `diff_aware_score.py` | `diff_aware_scores.csv` |
| Table 11, `diff_F1` on synthetic degenerate patches | `gaming_baselines.py` | `gaming_baselines.csv` |
| Figure 3, compiler-feedback loop | `analyze_feedback.py` | `feedback_analysis.csv` |

CodeBLEU is deterministic within a process but varies slightly between processes, because
the `codebleu` package's data-flow match depends on set iteration order. Set
`PYTHONHASHSEED=0` to reproduce CodeBLEU-based numbers byte for byte. Compile rates and
`diff_F1` are unaffected.

## Reproducing from scratch

1. **Get Big-Vul.** Download `MSR_data_cleaned.csv` from the Big-Vul release
   (<https://github.com/ZeoVan/MSR_20_Code_vulnerability_typed_dataset>) and place it at
   `data/MSR_data_cleaned.csv`.
2. **Rebuild the test set.** `python src/phase2_preprocess.py` then
   `python src/select_more_functions.py --per-cwe 19`. Both use `random_state=42`, so the
   draw is deterministic; cross-check the result against `data/test_set_203.csv`.
3. **Generate patches** on a GPU with the notebooks in `notebooks/`. The exact prompts and
   settings are in `replication/PROMPTS.md`.
4. **Validate and score.** `validate_patches.py`, then `compute_metrics.py`,
   `compute_similarity.py`, `bootstrap_ci.py`, `error_analysis.py`, `diff_aware_score.py`,
   `gaming_baselines.py`, `compile_std_sweep.py`, `analyze_feedback.py`.

Run all scripts from the repository root, for example `python src/diff_aware_score.py`.

## Requirements

Python 3.11, plus `pip install -r requirements.txt`. The compile harness needs GCC and G++
on `PATH`; the reported results use GCC 15.2.0 (MSYS2) on Windows 11, whose default
language standard is C23. Compile rate depends on the toolchain, which is one of the
paper's findings, so record the compiler version alongside any compile-based number.
Generation needs a GPU; validation, scoring and analysis are CPU-only.

## Citation

Om Nepal, Sushant Aryal, Oluseyi Olukola, and Nick Rahimi. *Metrics Failure in LLM-Based
Code Vulnerability Repair: An Empirical Study and a Change-Aware Screen.*

## Dataset

Fan, J., Li, Y., Wang, S., Nguyen, T. N. *A C/C++ Code Vulnerability Dataset with Code
Changes and CVE Summaries.* MSR 2020, pp. 508-512.

## License

MIT, see `LICENSE`.
