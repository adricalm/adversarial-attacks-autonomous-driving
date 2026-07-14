#!/usr/bin/env python3
"""Run DSGN++ tools/test.py with L40S compatibility patches applied first."""
import runpy
import sys
from pathlib import Path

# Apply patches before any DSGN2 model code imports torch_utils.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
import dsgn2_apply_l40s_patches  # noqa: F401,E402

if __name__ == "__main__":
    dsgn2_root = Path.cwd()
    tools_dir = dsgn2_root / "tools"
    test_py = tools_dir / "test.py"
    if not test_py.exists():
        print(f"error: run from DSGN2 root; missing {test_py}", file=sys.stderr)
        raise SystemExit(1)
    # Mirror `python tools/test.py` — script dir must be on sys.path for eval_utils.
    sys.path.insert(0, str(tools_dir))
    sys.argv[0] = str(test_py)
    runpy.run_path(str(test_py), run_name="__main__")
