import json
import os
import re


def load_fewshot_lookup(path=None):
    if path is None:
        path = os.path.join('data', 'fewshot_pool.json')
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def zero_shot_prompt(func_before: str, cwe_id: str) -> str:
    return (
        f"The following C/C++ function contains a {cwe_id} vulnerability.\n"
        f"Rewrite the function to fix the vulnerability.\n"
        f"Return ONLY the fixed function with no explanation or commentary.\n\n"
        f"Vulnerable function:\n```c\n{func_before}\n```\n\n"
        f"Fixed function:\n```c\n"
    )


def few_shot_prompt(func_before: str, cwe_id: str, fewshot_lookup: dict) -> str:
    examples = fewshot_lookup.get(cwe_id, [])[:2]

    header = (
        f"The following examples show how to fix {cwe_id} vulnerabilities in C/C++.\n\n"
    )

    example_blocks = ""
    for i, ex in enumerate(examples, 1):
        example_blocks += (
            f"### Example {i}\n"
            f"Vulnerable:\n```c\n{ex['func_before']}\n```\n"
            f"Fixed:\n```c\n{ex['func_after']}\n```\n\n"
        )

    if not example_blocks:
        # Fall back to zero-shot if no examples found for this CWE
        return zero_shot_prompt(func_before, cwe_id)

    target = (
        f"### Now fix this function\n"
        f"Vulnerable:\n```c\n{func_before}\n```\n"
        f"Fixed:\n```c\n"
    )

    return header + example_blocks + target


def cot_prompt(func_before: str, cwe_id: str) -> str:
    return (
        f"The C/C++ function below contains a {cwe_id} vulnerability.\n"
        f"Identify where the vulnerability occurs, what causes it, "
        f"and what specific change is needed to fix it.\n"
        f"Then write the complete corrected function.\n\n"
        f"Vulnerable function:\n```c\n{func_before}\n```\n\n"
        f"Fixed function:\n```c\n"
    )


def extract_cot_code(response: str) -> str:
    # Try <FIXED>...</FIXED> tags first
    match = re.search(r'<FIXED>(.*?)</FIXED>', response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to last fenced code block
    blocks = re.findall(r'```(?:c|cpp|c\+\+)?\n(.*?)```', response, re.DOTALL)
    if blocks:
        return blocks[-1].strip()

    # Last resort: return the full response
    return response.strip()
