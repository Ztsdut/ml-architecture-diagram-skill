#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_architecture_diagram.spec import load_spec, validate_spec
from ml_architecture_diagram.cli import _render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--formats", nargs="+", default=["svg", "png"], choices=["svg", "drawio", "pptx", "png", "pdf"])
    ap.add_argument("--outdir", default=str(ROOT / "build" / "examples"))
    args = ap.parse_args()
    outdir = Path(args.outdir)
    for spec_path in sorted((ROOT / "examples").glob("*.yaml")):
        spec = load_spec(spec_path)
        errors = validate_spec(spec)
        if errors:
            raise SystemExit(f"{spec_path}: {errors}")
        outputs = _render(spec, spec_path, outdir, args.formats)
        print(spec_path.name, "->", ", ".join(p.name for p in outputs))


if __name__ == "__main__":
    main()
