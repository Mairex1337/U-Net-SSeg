import sys
from pathlib import Path

# Add the root of the repo to sys.path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
