import os
import sys

# Must set cache paths BEFORE transformers is imported
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_HOME'] = os.getenv('HF_HOME', 'D:/hf_cache')
os.environ['TRANSFORMERS_CACHE'] = os.getenv('TRANSFORMERS_CACHE', 'D:/hf_cache/hub')
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

import json
import re
import time
import pandas as pd
import torch
from tqdm import tqdm

from setup_model import load_model
from prompts.prompt_builder import (
    zero_shot_prompt,
    few_shot_prompt,
    cot_prompt,
    extract_cot_code,
    load_fewshot_lookup,
)

NUM_SAMPLES    = int(os.getenv('NUM_SAMPLES', 5))
TEMPERATURE    = float(os.getenv('TEMPERATURE', 0.8))
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', 512))
MAX_PROMPT_TOKENS = 1600   # leave headroom inside codegen's 2048-token context
OUTPUT_PATH = os.path.join('results', 'patches_raw.jsonl')


# ── Patch extraction ──────────────────────────────────────────────────────────

def extract_patch(generated_raw: str, prompt_type: str) -> str:
    if prompt_type == 'cot':
        return extract_cot_code(generated_raw)
    # For zero-shot / few-shot: take everything up to the first closing ```
    match = re.search(r'^(.*?)(?:```|$)', generated_raw, re.DOTALL)
    if match:
        code = match.group(1).strip()
        if code:
            return code
    return generated_raw.strip()


# ── Resume support ────────────────────────────────────────────────────────────

def load_done_set(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec['func_id'], rec['prompt_type'], rec['sample_idx']))
            except json.JSONDecodeError:
                pass
    return done


# ── Single generation call ────────────────────────────────────────────────────

def generate_patch(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    # Tokenize with left-side truncation so the target function is never cut off
    tokenizer.truncation_side = 'left'
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=MAX_PROMPT_TOKENS,
    ).to('cpu')
    input_len = inputs['input_ids'].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    df      = pd.read_csv(os.path.join('data', 'test_functions.csv'), encoding='utf-8')
    fewshot = load_fewshot_lookup()

    prompt_builders = {
        'zero_shot': lambda func, cwe: zero_shot_prompt(func, cwe),
        'few_shot':  lambda func, cwe: few_shot_prompt(func, cwe, fewshot),
        'cot':       lambda func, cwe: cot_prompt(func, cwe),
    }

    total_calls = len(df) * len(prompt_builders) * NUM_SAMPLES

    # ── Load model ────────────────────────────────────────────────────────────
    print("=" * 60)
    model, tokenizer = load_model()

    # ── Timing benchmark (1 call at reduced tokens) ───────────────────────────
    print("\nRunning timing benchmark...")
    bench_row    = df.iloc[0]
    bench_prompt = zero_shot_prompt(bench_row['func_before'], bench_row['cwe_id'])
    bench_tokens = 64
    t0 = time.time()
    generate_patch(model, tokenizer, bench_prompt, bench_tokens)
    bench_elapsed = time.time() - t0
    # Scale linearly to MAX_NEW_TOKENS
    est_per_call    = bench_elapsed * (MAX_NEW_TOKENS / bench_tokens)
    est_total_min   = (total_calls * est_per_call) / 60

    print(f"\n{'=' * 60}")
    print(f"GENERATION PLAN")
    print(f"{'=' * 60}")
    print(f"  Functions     : {len(df)}")
    print(f"  Prompt types  : {len(prompt_builders)} (zero_shot, few_shot, cot)")
    print(f"  Samples each  : {NUM_SAMPLES}")
    print(f"  Total calls   : {total_calls}")
    print(f"  Benchmark     : {bench_tokens} tokens in {bench_elapsed:.1f}s")
    print(f"  Est. per call : ~{est_per_call:.0f}s ({MAX_NEW_TOKENS} tokens)")
    print(f"  Est. total    : ~{est_total_min:.0f} min  ({est_total_min/60:.1f} hrs)")
    print(f"  Output file   : {OUTPUT_PATH}")
    print(f"{'=' * 60}")

    # ── Resume check ──────────────────────────────────────────────────────────
    done = load_done_set(OUTPUT_PATH)
    remaining = total_calls - len(done)
    if done:
        print(f"Resuming: {len(done)} records already done, {remaining} remaining.")
    else:
        print("Starting fresh run.")

    # ── Generation loop ───────────────────────────────────────────────────────
    with open(OUTPUT_PATH, 'a', encoding='utf-8', newline='\n') as out_f:
        func_iter = tqdm(
            df.iterrows(),
            total=len(df),
            desc='Functions',
            ascii=True,
            unit='fn',
        )
        for _, row in func_iter:
            func_id     = row['func_id']
            cwe_id      = row['cwe_id']
            func_before = row['func_before']
            func_after  = row['func_after']

            for prompt_type, builder in prompt_builders.items():
                prompt = builder(func_before, cwe_id)

                for sample_idx in range(NUM_SAMPLES):
                    if (func_id, prompt_type, sample_idx) in done:
                        continue

                    func_iter.set_postfix_str(
                        f"{func_id} | {prompt_type} | s{sample_idx}", refresh=False
                    )

                    try:
                        generated_raw = generate_patch(
                            model, tokenizer, prompt, MAX_NEW_TOKENS
                        )
                        patch = extract_patch(generated_raw, prompt_type)

                        record = {
                            'func_id':       func_id,
                            'cwe_id':        cwe_id,
                            'prompt_type':   prompt_type,
                            'sample_idx':    sample_idx,
                            'patch':         patch,
                            'generated_raw': generated_raw,
                            'func_before':   func_before,
                            'func_after':    func_after,
                        }
                        out_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        out_f.flush()

                    except RuntimeError as e:
                        if 'out of memory' in str(e).lower():
                            torch.cuda.empty_cache()
                            print(f"\nOOM on {func_id}/{prompt_type}/{sample_idx} — skipping")
                        else:
                            raise

    # ── Final summary ─────────────────────────────────────────────────────────
    final_done = load_done_set(OUTPUT_PATH)
    print(f"\nDone. {len(final_done)} records written to {OUTPUT_PATH}")

    counts = {}
    with open(OUTPUT_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                pt = rec['prompt_type']
                counts[pt] = counts.get(pt, 0) + 1
            except json.JSONDecodeError:
                pass
    print("Breakdown by prompt type:")
    for pt, cnt in sorted(counts.items()):
        print(f"  {pt}: {cnt}")
