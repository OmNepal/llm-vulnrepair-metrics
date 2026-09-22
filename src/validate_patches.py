import os
import sys
import json
import argparse
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

from tqdm import tqdm
from validation.compile_check import is_compilable
from validation.semgrep_check import semgrep_check


def _resolve_paths():
    parser = argparse.ArgumentParser(description='Validate generated patches')
    parser.add_argument(
        '--input', '-i',
        default=os.path.join('results', 'patches_raw.jsonl'),
        help='Input JSONL file (default: results/patches_raw.jsonl)',
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Output JSONL file (default: derived from input name)',
    )
    args = parser.parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        # e.g. patches_raw_codegen_350M_multi.jsonl → patches_validated_codegen_350M_multi.jsonl
        basename = os.path.basename(input_path).replace('patches_raw', 'patches_validated')
        output_path = os.path.join('results', basename)

    return input_path, output_path

INPUT_PATH, OUTPUT_PATH = _resolve_paths()


def load_records(path: str) -> list:
    records = []
    skipped = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1
    if skipped:
        print(f"  Skipped {skipped} malformed line(s)")
    return records


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
            except (json.JSONDecodeError, KeyError):
                pass
    return done


if __name__ == '__main__':
    os.makedirs('results', exist_ok=True)

    print("Loading patches...")
    all_records = load_records(INPUT_PATH)

    records = all_records
    print(f"Loaded {len(records)} records to validate")

    # Resume support
    done = load_done_set(OUTPUT_PATH)
    remaining = [
        r for r in records
        if (r['func_id'], r['prompt_type'], r['sample_idx']) not in done
    ]
    if done:
        print(f"Resuming: {len(done)} already validated, {len(remaining)} remaining")
    else:
        print("Starting fresh validation run")

    with open(OUTPUT_PATH, 'a', encoding='utf-8', newline='\n') as out_f:
        for rec in tqdm(remaining, desc='Validating', ascii=True, unit='patch'):
            patch = rec.get('patch', '')

            compile_result  = is_compilable(patch)
            semgrep_result  = semgrep_check(patch)

            validated = {
                **rec,
                'compilable':     compile_result['compilable'],
                'compile_error':  compile_result['error'],
                'vuln_free':      semgrep_result['vuln_free'],
                'finding_count':  semgrep_result['finding_count'],
                'findings':       semgrep_result['findings'],
            }

            out_f.write(json.dumps(validated, ensure_ascii=False) + '\n')
            out_f.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\nValidation complete. Loading results for summary...")
    validated_records = load_records(OUTPUT_PATH)
    total = len(validated_records)

    if total == 0:
        print("No records found.")
    else:
        compilable  = sum(1 for r in validated_records if r.get('compilable'))
        vuln_free   = sum(1 for r in validated_records if r.get('vuln_free') is True)
        passed_both = sum(1 for r in validated_records if r.get('compilable') and r.get('vuln_free') is True)

        print(f"\n{'='*50}")
        print(f"VALIDATION SUMMARY")
        print(f"{'='*50}")
        print(f"  Total patches validated : {total}")
        print(f"  Compilable              : {compilable}  ({compilable/total*100:.1f}%)")
        print(f"  Vuln-free (Semgrep)     : {vuln_free}  ({vuln_free/total*100:.1f}%)")
        print(f"  Passed both checks      : {passed_both}  ({passed_both/total*100:.1f}%)")
        print(f"{'='*50}")

        print("\nBreakdown by prompt type:")
        for pt in ['zero_shot', 'few_shot', 'cot']:
            pt_recs = [r for r in validated_records if r['prompt_type'] == pt]
            if not pt_recs:
                continue
            n = len(pt_recs)
            c = sum(1 for r in pt_recs if r.get('compilable'))
            print(f"  {pt:12s}: {n} patches, {c} compilable ({c/n*100:.1f}%)")
