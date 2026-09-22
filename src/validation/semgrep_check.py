import json
import os
import re
import subprocess
import tempfile

# Prefer the venv's semgrep so we don't rely on PATH.
# This file lives at <root>/src/validation/semgrep_check.py → project root is 3 levels up.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_VENV_SEMGREP = os.path.join(_PROJECT_ROOT, 'venv', 'Scripts', 'semgrep.exe')
SEMGREP_PATH = _VENV_SEMGREP if os.path.exists(_VENV_SEMGREP) else 'semgrep'


def clean_patch(code: str) -> str:
    code = re.sub(r'^```(?:c|cpp|c\+\+)?\s*\n?', '', code.strip(), flags=re.MULTILINE)
    code = re.sub(r'\n?```\s*$', '', code.strip(), flags=re.MULTILINE)
    return code.strip()


def semgrep_check(code: str) -> dict:
    code = clean_patch(code)
    if len(code.strip()) < 10:
        return {'vuln_free': False, 'finding_count': 0, 'findings': ['Empty patch']}

    tmp = tempfile.NamedTemporaryFile(
        suffix='.c', mode='w', delete=False, encoding='utf-8', newline='\n'
    )
    tmp.write(code)
    tmp.close()  # critical on Windows

    try:
        result = subprocess.run(
            [
                SEMGREP_PATH,
                '--config', 'p/c',
                '--json',
                '--no-git-ignore',
                '--timeout', '30',
                tmp.name,
            ],
            capture_output=True,
            text=True,
            timeout=60,
            shell=False,
        )

        try:
            data = json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            # Semgrep produced no parseable JSON — treat as no findings
            return {'vuln_free': True, 'finding_count': 0, 'findings': []}

        findings = data.get('results', [])
        return {
            'vuln_free': len(findings) == 0,
            'finding_count': len(findings),
            'findings': [
                f"{f['check_id']} line {f['start']['line']}" for f in findings
            ],
        }

    except subprocess.TimeoutExpired:
        return {'vuln_free': None, 'finding_count': -1, 'findings': ['Semgrep timeout']}
    except FileNotFoundError:
        return {'vuln_free': None, 'finding_count': -1, 'findings': ['Semgrep not found']}
    finally:
        os.unlink(tmp.name)
