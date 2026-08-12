#!/usr/bin/env python3
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_architecture_diagram.cli import main

if __name__ == "__main__":
    argv = ["render", *sys.argv[1:]]
    raise SystemExit(main(argv))
