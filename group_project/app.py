"""Streamlit entrypoint for the group chatbot.

Run:
    streamlit run group_project/app.py
"""

import sys
import runpy
from pathlib import Path

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

runpy.run_path(str(APP_DIR / "chatbot_app.py"), run_name="__main__")
