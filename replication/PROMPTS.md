# Prompt templates (verbatim)

This file is referenced by the "Data and code availability" statement of the paper
*Metrics Failure in LLM-Based C/C++ Vulnerability Repair*. It contains, verbatim, every
prompt used to produce every number reported in the paper.

Placeholders are written as `{...}`. At generation time they are substituted directly,
with no additional formatting:

| Placeholder | Substituted with |
|---|---|
| `{cwe_id}` | the CWE label of the target function, e.g. `CWE-119` |
| `{func_before}` | the vulnerable function verbatim, as stored in Big-Vul's `func_before` |
| `{ex_before}` / `{ex_after}` | a held-out example's `func_before` / `func_after` |
| `{prev}` | the previous round's generated patch (feedback loop only) |
| `{err}` | the compiler's stderr, truncated to the first 500 characters (feedback loop only) |

Note the trailing ```` ```c ```` fence at the end of the completion-style prompts. It is
intentional: these are base (non-instruction-tuned) models used in completion mode, so the
prompt ends mid-fence and the model continues the code block.

---

## 1. Main comparison: three prompting strategies

Used for **all three base models** (CodeGen-350M-multi, DeepSeek-Coder-1.3B,
DeepSeek-Coder-6.7B) and therefore for every headline result: the compile-rate, pass@k,
CodeBLEU, `diff_F1`, failure-taxonomy, token-budget, and `-std` numbers.

Source of record: `notebooks/generate_kaggle.ipynb`. The same templates are also
implemented in `src/prompts/prompt_builder.py`; the two were verified to produce
byte-identical strings.

### 1.1 Zero-shot

```
The following C/C++ function contains a {cwe_id} vulnerability.
Rewrite the function to fix the vulnerability.
Return ONLY the fixed function with no explanation or commentary.

Vulnerable function:
```c
{func_before}
```

Fixed function:
```c
```

### 1.2 Few-shot

Two examples of the same CWE, drawn from a held-out few-shot pool that never overlaps the
test set. If no example exists for that CWE, the builder falls back to the zero-shot
template.

```
The following examples show how to fix {cwe_id} vulnerabilities in C/C++.

### Example 1
Vulnerable:
```c
{ex1_before}
```
Fixed:
```c
{ex1_after}
```

### Example 2
Vulnerable:
```c
{ex2_before}
```
Fixed:
```c
{ex2_after}
```

### Now fix this function
Vulnerable:
```c
{func_before}
```
Fixed:
```c
```

### 1.3 Chain-of-thought

```
The C/C++ function below contains a {cwe_id} vulnerability.
Identify where the vulnerability occurs, what causes it, and what specific change is needed to fix it.
Then write the complete corrected function.

Vulnerable function:
```c
{func_before}
```

Fixed function:
```c
```

---

## 2. Chat-mode variants (instruction-tuned robustness experiment only)

The instruction-tuned DeepSeek-Coder-1.3B is a chat model, so it is prompted through its
tokenizer's chat template rather than in completion mode. The wording therefore differs
from Section 1: there is no trailing open fence, and the model is asked explicitly for a
fenced code block.

**These templates were not used for any headline number.** They apply only to the
base-versus-instruct robustness comparison and to round 0 of the compiler-feedback loop
for the instruct model. Reproducing the main results requires the Section 1 templates.

Source of record: `notebooks/generate_kaggle_v2.ipynb`.

### 2.1 Zero-shot (chat)

```
The following C/C++ function contains a {cwe_id} vulnerability.
Rewrite the function to fix the vulnerability.
Return ONLY the fixed function inside a ```c code block, no explanation.

Vulnerable function:
```c
{func_before}
```
```

### 2.2 Few-shot (chat)

```
Here are examples of fixing {cwe_id} vulnerabilities in C/C++.

### Example 1
Vulnerable:
```c
{ex1_before}
```
Fixed:
```c
{ex1_after}
```

### Example 2
Vulnerable:
```c
{ex2_before}
```
Fixed:
```c
{ex2_after}
```

### Now fix this function
Vulnerable:
```c
{func_before}
```
Return ONLY the fixed function inside a ```c code block.
```

### 2.3 Chain-of-thought (chat)

```
The C/C++ function below contains a {cwe_id} vulnerability.
Identify where the vulnerability occurs, what causes it, and what change fixes it.
Then write the complete corrected function inside a ```c code block.

Vulnerable function:
```c
{func_before}
```
```

---

## 3. Compiler-feedback loop

Used for the experiment showing that optimizing compile rate rewards non-repairs. Round 0
is an ordinary generation (zero-shot, sample 0). In each later round, every patch that
still fails to compile is regenerated with the compiler's error fed back through this
template. The same template is used for the base and the instruction-tuned model; the base
model receives it in completion mode and the instruct model through its chat template.

Source of record: `notebooks/feedback_loop_kaggle.ipynb`.

```
A previous attempt to fix a {cwe_id} vulnerability in this C/C++ function does not compile.
Vulnerable function:
```c
{func_before}
```
Previous attempt:
```c
{prev}
```
Compiler error:
{err}
Fix the compiler error and return ONLY the corrected function inside a ```c code block.
```

---

## 4. Generation settings

Identical across all models and strategies unless noted.

| Setting | Value |
|---|---|
| `max_new_tokens` | 512 |
| Samples per (function, strategy) cell | 5 (1 for the feedback loop) |
| `temperature` | 0.8 |
| `top_p` | 0.95 |
| Sampling | enabled (`do_sample=True`) |
| Max prompt tokens | 1536 for CodeGen-350M, 1600 for the DeepSeek models |
| Truncation side | left |

CodeGen-350M-multi has a 2,048-token context window, so its prompt budget is capped at
1,536 to leave room for 512 generated tokens. The DeepSeek models have a 16,384-token
context and are unaffected.

## 5. Extracting the patch from the model output

For the completion-mode templates of Section 1, the generated text is taken up to the
first closing fence, that is, the model's continuation of the open ```` ```c ```` block:

```python
match = re.search(r'^(.*?)(?:```|$)', generated_raw, re.DOTALL)
patch = match.group(1).strip() if match and match.group(1).strip() else generated_raw.strip()
```

For the chat-mode templates of Section 2, the first fenced code block in the response is
taken; if the response contains no fence, the text before the first fence is used, and
failing that the whole response. Responses that contain no code at all are retained and
scored as-is, which is why the instruction-tuned model shows a distinct prose-output
failure mode in the paper.
