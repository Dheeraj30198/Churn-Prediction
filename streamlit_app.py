"""
Root entry point for Streamlit Community Cloud.
This file exists so Streamlit Cloud can discover the app at the repo root.
"""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `app` package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Run the dashboard
import app.dashboard  # noqa: F401, E402
