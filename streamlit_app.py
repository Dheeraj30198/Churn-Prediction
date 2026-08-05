"""
Root entry point for Streamlit Community Cloud.
Streamlit Cloud main file path should be set to: app/dashboard.py
This file is kept as a fallback reference only.
"""
import sys
import runpy
from pathlib import Path

# Add both repo root and app/ to path
_repo_root = Path(__file__).resolve().parent
_app_dir   = _repo_root / "app"

for _p in [str(_repo_root), str(_app_dir)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Run dashboard as a script (not as a module import, so Streamlit can re-run it)
runpy.run_path(str(_app_dir / "dashboard.py"), run_name="__main__")
