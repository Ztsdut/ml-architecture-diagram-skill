from pathlib import Path

from ml_architecture_diagram.ai_bundle import prepare_ai_bundle
from ml_architecture_diagram.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def test_prepare_ai_bundle(tmp_path):
    spec = load_spec(ROOT / "examples/transformer_encoder.yaml")
    out = prepare_ai_bundle(spec, tmp_path, stem="demo", width=800)
    assert out["svg"].exists()
    assert out["png"].exists()
    assert "NON-NEGOTIABLE" in out["prompt"].read_text(encoding="utf-8")
