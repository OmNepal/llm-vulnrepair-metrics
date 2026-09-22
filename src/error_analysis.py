import argparse
import json
import os
import re
import random
import csv
from collections import Counter, defaultdict

PROMPT_TYPES  = ['zero_shot', 'few_shot', 'cot']
random.seed(42)

# ── Failure taxonomy ─────────────────────────────────────────────────────────
# Categories are grouped by *kind* of problem. Truncation is deliberately NOT a
# category here — it is an orthogonal flag (see `is_truncated`), because a patch
# can be both truncated AND have, say, an unknown type. Making it a mutually
# exclusive bucket (as the old version did) let it shadow the real cause.
#
# The primary category for a failure is the highest-priority class whose
# signature is present, using CATEGORY_ORDER below (top = highest priority):
#   content problems first (it isn't even C), then structural (unparseable C),
#   then context problems (parseable C missing its surrounding project).
#
# Context-class categories (UNKNOWN_TYPE, IMPLICIT_FUNC, UNDECLARED_IDENT,
# INCOMPLETE_TYPE, BAD_MEMBER) are exactly what the stub-injection harness tries
# to synthesise away. When they appear in the *residual* compile_error, the
# injector gave up on them (e.g. a `struct`-keyword variant it doesn't match) —
# so this taxonomy, run on stub-injection output, describes what the harness
# genuinely cannot paper over, not the naive-compiler failures the old one saw.
CATEGORY_ORDER = [
    'EMPTY',            # patch empty / < 10 chars
    'NON_C_OUTPUT',     # prose / CoT reasoning / Python leaked instead of C
    'CPP_IN_C',         # C++ constructs (Class::method) — language-mode mismatch
    'IMPLICIT_INT',     # return type defaults to 'int' (model omitted return type)
    'STUB_ARTIFACT',    # harness's fabricated stub collided / wrong arity
    'SYNTAX_ERROR',     # generic parse error: expected/stray/unterminated ...
    'INCOMPLETE_TYPE',  # incomplete / undefined struct type used
    'BAD_MEMBER',       # field access on an unknown/wrong struct
    'UNKNOWN_TYPE',     # unknown type name 'X'
    'IMPLICIT_FUNC',    # implicit declaration of function 'X'
    'UNDECLARED_IDENT', # 'X' undeclared / not declared in this scope
    'OTHER',            # a compile error not matched above
    'COMPILABLE',       # (kept for completeness; excluded from failure stats)
]

# Coarser grouping for the paper: separates failures the MODEL caused (wrote
# non-C or malformed C) from failures the MEASUREMENT SETUP caused (isolated
# function missing its project context, our fabricated stubs, or modern-GCC
# strictness). The sum of the non-"model" groups is the share of "compile
# failures" that are not the model writing broken code.
CATEGORY_GROUP = {
    'EMPTY':            'non_code',          # model: no usable code
    'NON_C_OUTPUT':     'non_code',          # model: not C
    'SYNTAX_ERROR':     'malformed_c',       # model: genuinely broken C
    'CPP_IN_C':         'missing_context',   # setup: needs class/namespace defs
    'INCOMPLETE_TYPE':  'missing_context',   # setup: needs struct definitions
    'BAD_MEMBER':       'missing_context',   # setup: needs real struct layout
    'UNKNOWN_TYPE':     'missing_context',   # setup: needs project typedefs
    'IMPLICIT_FUNC':    'missing_context',   # setup: needs function decls
    'UNDECLARED_IDENT': 'missing_context',   # setup: needs surrounding decls
    'STUB_ARTIFACT':    'harness_artifact',  # setup: our stub injector's own fault
    'IMPLICIT_INT':     'dataset_artifact',  # data: Big-Vul stripped the return type
    'OTHER':            'other',
    'COMPILABLE':       'compilable',
}
GROUP_ORDER = ['non_code', 'malformed_c', 'missing_context',
               'harness_artifact', 'dataset_artifact', 'other']

CATEGORY_DESCRIPTIONS = {
    'EMPTY':            'Patch empty or < 10 chars (usually a prompt whose answer '
                        'tag the model never emitted, leaving an empty extraction).',
    'NON_C_OUTPUT':     'Model emitted prose / step-by-step reasoning / Python-style '
                        'code instead of C (def, import, print, self., numbered lists).',
    'CPP_IN_C':         "C++ constructs such as Class::method or namespaces (a '::' token, "
                        "or 'is not a class/namespace/enumeration'). The harness retries as "
                        "C++, so this failed even in C++ mode (the class/namespace is undefined).",
    'IMPLICIT_INT':     "GCC error 'return type defaults to int' — the function definition "
                        "has no visible return type. Measured to be 96-100% DATASET-origin: "
                        "Big-Vul's func_before already lacks the return type (preprocessing "
                        "stripped it), so the model faithfully echoes a type-less signature. "
                        "Not a model or harness fault; a data-quality artifact. Hard error "
                        "under C23; would compile under pre-C99.",
    'STUB_ARTIFACT':    "Failure caused by the stub injector itself: a fabricated stub "
                        "collided with a name used differently ('redeclared as different "
                        "kind', 'conflicting types'), or a stubbed function declared int f() "
                        "is called with args (modern GCC reads () as (void) -> 'too many "
                        "arguments'). These would not fail in the real project.",
    'SYNTAX_ERROR':     'Generic parse/syntax error (expected X before Y, stray token, '
                        'unterminated #if/string) not attributable to the classes above.',
    'INCOMPLETE_TYPE':  'Use of an incomplete/undefined struct type (e.g. dereferencing '
                        'struct X* whose body is unknown).',
    'BAD_MEMBER':       "Field access on a struct that has no such member — the stub "
                        "injector's generic field holder did not cover it.",
    'UNKNOWN_TYPE':     "Unknown type name — a project/kernel type the stub injector "
                        "did not synthesise (often the 'use struct keyword' variant).",
    'IMPLICIT_FUNC':    'Implicit declaration of a function the injector did not stub.',
    'UNDECLARED_IDENT': "An identifier used but never declared, not caught as a type "
                        "or function.",
    'OTHER':            'A compile error not matched by any signature above.',
    'COMPILABLE':       'Compiled successfully under the stub-injection harness.',
}

# ── Error-signature patterns (matched against the residual compile_error) ─────
_SIG = {
    'CPP_IN_C':         [re.compile(r'::'),
                         re.compile(r'is not a class, namespace, or enumeration')],
    'IMPLICIT_INT':     [re.compile(r"return type defaults to 'int'"),
                         re.compile(r'-Wimplicit-int')],
    'STUB_ARTIFACT':    [re.compile(r'redeclared as different kind'),
                         re.compile(r'conflicting types for'),
                         re.compile(r'\bredefinition of\b'),
                         re.compile(r'too (?:many|few) arguments to function')],
    'INCOMPLETE_TYPE':  [re.compile(r'incomplete type'),
                         re.compile(r'invalid use of undefined type'),
                         re.compile(r'dereferencing pointer to incomplete type')],
    'BAD_MEMBER':       [re.compile(r'has no member named'),
                         re.compile(r'request for member')],
    'UNKNOWN_TYPE':     [re.compile(r'unknown type name'),
                         re.compile(r'does not name a type')],
    'IMPLICIT_FUNC':    [re.compile(r'implicit declaration of function'),
                         re.compile(r'undefined reference')],
    'UNDECLARED_IDENT': [re.compile(r"undeclared"),
                         re.compile(r'was not declared in this scope')],
    'SYNTAX_ERROR':     [re.compile(r'\bexpected\b'), re.compile(r'\bstray\b'),
                         re.compile(r'unterminated'), re.compile(r'\bbefore\b'),
                         re.compile(r'missing terminating')],
}

_PY_SIGNALS = [
    re.compile(r'\bdef\s+\w+\s*\('),
    re.compile(r'\bimport\s+\w+'),
    re.compile(r'\bprint\s*\('),
    re.compile(r'\bself\.'),
    re.compile(r'^\s*\d+\.\s+[A-Z]'),          # numbered prose ("1. Identify ...")
    re.compile(r'^\s*(Step|The|If you)\b', re.M),  # CoT narrative leakage
]


def load_records(path):
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


def is_truncated(patch: str) -> bool:
    """Independent flag: patch cut off mid-block (more '{' than '}')."""
    return patch.count('{') > patch.count('}')


def looks_non_c(patch: str) -> bool:
    """Prose / reasoning / Python leaked instead of C."""
    return any(sig.search(patch) for sig in _PY_SIGNALS)


def matched_categories(rec: dict) -> list:
    """All failure signatures present, in CATEGORY_ORDER priority order."""
    patch = rec.get('patch', '') or ''
    err   = rec.get('compile_error', '') or ''
    hits  = []

    if not patch or len(patch.strip()) < 10 or err.strip() == 'Empty or too short':
        hits.append('EMPTY')
    if looks_non_c(patch):
        hits.append('NON_C_OUTPUT')
    for cat in ['CPP_IN_C', 'IMPLICIT_INT', 'STUB_ARTIFACT', 'SYNTAX_ERROR',
                'INCOMPLETE_TYPE', 'BAD_MEMBER', 'UNKNOWN_TYPE', 'IMPLICIT_FUNC',
                'UNDECLARED_IDENT']:
        if any(p.search(err) for p in _SIG[cat]):
            hits.append(cat)

    # Preserve CATEGORY_ORDER priority and de-duplicate.
    ordered = [c for c in CATEGORY_ORDER if c in hits]
    return ordered or (['OTHER'] if err else ['OTHER'])


def categorize(rec: dict) -> dict:
    """Return primary category, all matched categories, and truncation flag."""
    if rec.get('compilable') is True:
        return {'primary': 'COMPILABLE', 'matched': ['COMPILABLE'],
                'truncated': is_truncated(rec.get('patch', '') or '')}
    matched = matched_categories(rec)
    return {'primary': matched[0], 'matched': matched,
            'truncated': is_truncated(rec.get('patch', '') or '')}


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Failure taxonomy + error analysis from validated patches')
    parser.add_argument(
        '--input', '-i',
        default=os.path.join('results', 'patches_validated.jsonl'),
        help='Validated JSONL file (default: results/patches_validated.jsonl)',
    )
    args = parser.parse_args()

    basename = os.path.splitext(os.path.basename(args.input))[0]
    if basename == 'patches_validated':
        run_label = 'baseline'
    else:
        run_label = basename.replace('patches_validated_', '')
    return args.input, run_label


def _pct(n, d):
    return f'{n / d * 100:.1f}%' if d else '0.0%'


if __name__ == '__main__':
    input_path, run_label = _parse_args()
    os.makedirs('results', exist_ok=True)
    os.makedirs('report',  exist_ok=True)

    print(f'Loading {input_path} ...')
    all_records = load_records(input_path)
    records = all_records
    print(f'  {len(records)} records  [run label: {run_label}]\n')

    for r in records:
        info = categorize(r)
        r['primary_category']  = info['primary']
        r['matched_categories'] = info['matched']
        r['truncated']          = info['truncated']
        r['failure_group']      = CATEGORY_GROUP.get(info['primary'], 'other')

    failures   = [r for r in records if not r.get('compilable')]
    compilable = [r for r in records if r.get('compilable')]
    total_fail = len(failures)

    print(f'Compilable    : {len(compilable)} / {len(records)}  ({_pct(len(compilable), len(records))})')
    print(f'Non-compilable: {total_fail} / {len(records)}\n')

    # ── Primary-category distribution (failures only) ─────────────────────────
    cat_counts = Counter(r['primary_category'] for r in failures)
    fail_cats  = [c for c in CATEGORY_ORDER if c not in ('COMPILABLE',)]

    print('=' * 60)
    print('PRIMARY FAILURE CATEGORY (each failure counted once)')
    print('=' * 60)
    print(f"  {'Category':<18}  {'Count':>6}  {'% of fails':>10}")
    print('  ' + '-' * 40)
    for cat in fail_cats:
        n = cat_counts.get(cat, 0)
        if n:
            print(f'  {cat:<18}  {n:>6}  {_pct(n, total_fail):>10}')

    # ── Failure-group rollup (paper framing) ──────────────────────────────────
    # SETUP-CAUSED failures split two ways by who can fix them:
    #   FIXABLE       — by us, via harness/compiler config (stub arity, -std, etc.)
    #   NEEDS_CONTEXT — only by supplying the real project (we cannot fabricate it)
    grp_counts = Counter(r['failure_group'] for r in failures)
    fixable_total = grp_counts.get('harness_artifact', 0)   # fixable by us (stub injector)
    context_total = grp_counts.get('missing_context', 0)    # needs the real project
    dataset_total = grp_counts.get('dataset_artifact', 0)   # Big-Vul data quality
    setup_total   = fixable_total + context_total + dataset_total
    print('\n' + '=' * 60)
    print('FAILURE GROUP (model-caused vs NOT-the-model)')
    print('=' * 60)
    for g in GROUP_ORDER:
        n = grp_counts.get(g, 0)
        if n:
            print(f'  {g:<18}  {n:>6}  {_pct(n, total_fail):>10}')
    print('  ' + '-' * 40)
    print(f'  {"NOT-THE-MODEL total (setup/data, not bad C)":<48}  '
          f'{setup_total}  ({_pct(setup_total, total_fail)})')
    print(f'    {"- harness-FIXABLE (STUB_ARTIFACT)":<46}  '
          f'{fixable_total}  ({_pct(fixable_total, total_fail)})')
    print(f'    {"- needs-REAL-CONTEXT (missing project defs)":<46}  '
          f'{context_total}  ({_pct(context_total, total_fail)})')
    print(f'    {"- DATASET-artifact (Big-Vul stripped ret type)":<46}  '
          f'{dataset_total}  ({_pct(dataset_total, total_fail)})')

    # ── Truncation as an orthogonal dimension ─────────────────────────────────
    trunc_fail = sum(1 for r in failures if r['truncated'])
    print('\n' + '=' * 60)
    print('TRUNCATION (independent flag — NOT a category)')
    print('=' * 60)
    print(f'  Truncated failures: {trunc_fail} / {total_fail}  ({_pct(trunc_fail, total_fail)})')
    print('  Primary category among truncated vs. non-truncated failures:')
    print(f"  {'Category':<18}  {'truncated':>10}  {'complete':>10}")
    print('  ' + '-' * 42)
    for cat in fail_cats:
        t = sum(1 for r in failures if r['primary_category'] == cat and r['truncated'])
        c = sum(1 for r in failures if r['primary_category'] == cat and not r['truncated'])
        if t or c:
            print(f'  {cat:<18}  {t:>10}  {c:>10}')

    # ── Category × prompt type ────────────────────────────────────────────────
    pt_cats = defaultdict(Counter)
    for r in failures:
        pt_cats[r['prompt_type']][r['primary_category']] += 1

    # ── Category × CWE ────────────────────────────────────────────────────────
    cwe_cats = defaultdict(Counter)
    for r in failures:
        cwe_cats[r['cwe_id']][r['primary_category']] += 1

    # ── Semgrep / CoT artifact check (documented paper finding) ───────────────
    cot_vuln = [r for r in records
                if r['prompt_type'] == 'cot' and r.get('vuln_free') is False]
    rule_counts = Counter()
    for r in cot_vuln:
        for f in r.get('findings', []):
            rule = f.split(' line ')[0] if ' line ' in str(f) else str(f)
            rule_counts[rule] += 1

    # ── Save failure_categories_<label>.csv ───────────────────────────────────
    csv_path = os.path.join('results', f'failure_categories_{run_label}.csv')
    fieldnames = ['model_alias', 'func_id', 'cwe_id', 'prompt_type', 'sample_idx',
                  'compilable', 'vuln_free', 'truncated',
                  'primary_category', 'failure_group', 'matched_categories',
                  'compile_error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, '') for k in fieldnames}
            row['matched_categories'] = ';'.join(r.get('matched_categories', []))
            writer.writerow(row)
    print(f'\nSaved {csv_path}  ({len(records)} rows)')

    # ── Write report/error_analysis_<label>.txt ───────────────────────────────
    report_path = os.path.join('report', f'error_analysis_{run_label}.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('ERROR ANALYSIS / FAILURE TAXONOMY\n')
        f.write('=' * 60 + '\n\n')
        f.write(f'Input:      {input_path}\n')
        f.write(f'Run label:  {run_label}\n')
        f.write(f'Records: {len(records)}\n')
        f.write(f'Compilable:     {len(compilable)} ({_pct(len(compilable), len(records))})\n')
        f.write(f'Non-compilable: {total_fail} ({_pct(total_fail, len(records))})\n\n')

        f.write('PRIMARY FAILURE CATEGORY (each failure counted once)\n')
        f.write('-' * 50 + '\n')
        for cat in fail_cats:
            n = cat_counts.get(cat, 0)
            if n:
                f.write(f'  {cat:<18}  {n:>5}  ({_pct(n, total_fail)})\n')

        f.write('\nFAILURE GROUP (model-caused vs NOT-the-model)\n')
        f.write('-' * 50 + '\n')
        for g in GROUP_ORDER:
            n = grp_counts.get(g, 0)
            if n:
                f.write(f'  {g:<18}  {n:>5}  ({_pct(n, total_fail)})\n')
        f.write(f'  {"-"*38}\n')
        f.write(f'  NOT-THE-MODEL total (setup/data, not bad C): '
                f'{setup_total} ({_pct(setup_total, total_fail)})\n')
        f.write(f'    - harness-FIXABLE (STUB_ARTIFACT): '
                f'{fixable_total} ({_pct(fixable_total, total_fail)})\n')
        f.write(f'    - needs-REAL-CONTEXT (missing project defs): '
                f'{context_total} ({_pct(context_total, total_fail)})\n')
        f.write(f'    - DATASET-artifact (Big-Vul stripped return type): '
                f'{dataset_total} ({_pct(dataset_total, total_fail)})\n')

        f.write('\nTRUNCATION (orthogonal flag)\n')
        f.write('-' * 50 + '\n')
        f.write(f'  Truncated failures: {trunc_fail} / {total_fail} ({_pct(trunc_fail, total_fail)})\n')
        f.write(f"  {'Category':<18}  {'truncated':>10}  {'complete':>10}\n")
        for cat in fail_cats:
            t = sum(1 for r in failures if r['primary_category'] == cat and r['truncated'])
            c = sum(1 for r in failures if r['primary_category'] == cat and not r['truncated'])
            if t or c:
                f.write(f'  {cat:<18}  {t:>10}  {c:>10}\n')

        f.write('\nCATEGORY DESCRIPTIONS\n')
        f.write('-' * 50 + '\n')
        for cat in fail_cats:
            if cat_counts.get(cat, 0):
                f.write(f'\n{cat}:\n  {CATEGORY_DESCRIPTIONS[cat]}\n')

        f.write('\n\nCATEGORY x PROMPT TYPE\n')
        f.write('-' * 50 + '\n')
        f.write(f"  {'Category':<18}" + ''.join(f'  {pt:>12}' for pt in PROMPT_TYPES) + '\n')
        for cat in fail_cats:
            if cat_counts.get(cat, 0):
                f.write(f'  {cat:<18}' +
                        ''.join(f'  {pt_cats[pt].get(cat, 0):>12}' for pt in PROMPT_TYPES) + '\n')

        f.write('\n\nCATEGORY x CWE\n')
        f.write('-' * 50 + '\n')
        cwes = sorted(cwe_cats)
        f.write(f"  {'Category':<18}" + ''.join(f'  {c.replace("CWE-",""):>7}' for c in cwes) + '\n')
        for cat in fail_cats:
            if cat_counts.get(cat, 0):
                f.write(f'  {cat:<18}' +
                        ''.join(f'  {cwe_cats[c].get(cat, 0):>7}' for c in cwes) + '\n')

        f.write(f'\n\nSEMGREP FINDINGS ON CoT PATCHES ({len(cot_vuln)} vuln-flagged)\n')
        f.write('-' * 50 + '\n')
        if rule_counts:
            for rule, cnt in rule_counts.most_common(10):
                f.write(f'  {cnt:>4}x  {rule}\n')
        else:
            f.write('  (none)\n')

        f.write('\n\nREPRESENTATIVE FAILURE EXAMPLES (up to 2 per category)\n')
        f.write('-' * 50 + '\n')
        by_cat = defaultdict(list)
        for r in failures:
            by_cat[r['primary_category']].append(r)
        for cat in fail_cats:
            sample = random.sample(by_cat[cat], min(2, len(by_cat[cat])))
            for r in sample:
                flags = 'truncated' if r['truncated'] else 'complete'
                f.write(f'\n[{cat}] {r["func_id"]} | {r["prompt_type"]} | {flags}\n')
                f.write(f'  matched: {";".join(r["matched_categories"])}\n')
                f.write(f'  Patch (first 300 chars):\n    {(r.get("patch") or "")[:300]!r}\n')
                err = (r.get('compile_error') or '')[:200]
                f.write(f'  GCC error: {err}\n')

        # ── Data-driven summary (no hardcoded narrative) ──────────────────────
        f.write('\n\nSUMMARY\n')
        f.write('-' * 50 + '\n')
        if cat_counts:
            top = cat_counts.most_common(1)[0]
            f.write(f'Dominant primary failure: {top[0]} '
                    f'({top[1]} cases, {_pct(top[1], total_fail)} of failures).\n')
        f.write(f'Measurement-setup-caused failures: {setup_total} of {total_fail} '
                f'({_pct(setup_total, total_fail)}) — not the model writing broken C, '
                f'but our isolated-function compile setup rejecting it. Of these, '
                f'{fixable_total} ({_pct(fixable_total, total_fail)}) are '
                f'harness/toolchain-FIXABLE (stub arity, compiler -std) and '
                f'{context_total} ({_pct(context_total, total_fail)}) NEED-REAL-CONTEXT '
                f'(missing project definitions we cannot fabricate).\n')
        f.write(f'Truncation affects {trunc_fail} of {total_fail} failures '
                f'({_pct(trunc_fail, total_fail)}) and is reported as an orthogonal flag: '
                f'a failure can be truncated AND have a distinct primary cause.\n')

    print(f'Saved {report_path}')
    print('\nDone.')
