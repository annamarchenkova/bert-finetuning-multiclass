import os
from pathlib import Path

PROJECT_DIR = Path().resolve().parents[0]
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
CONFIG_DIR = os.path.join(PROJECT_DIR, "src", "config")
