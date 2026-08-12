from pathlib import Path

from ml_architecture_diagram.parsers.pytorch_ast import parse_pytorch_file
from ml_architecture_diagram.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]


def test_parse_residual_cnn():
    spec = parse_pytorch_file(ROOT / "examples/pytorch/residual_cnn.py", class_name="ResidualCNN", detail="system")
    assert validate_spec(spec) == []
    labels = {n["label"] for n in spec["nodes"]}
    assert "Output" in labels
    assert any(e["type"] == "residual" for e in spec["edges"])
    assert any(n["kind"] == "merge" and n["label"] == "+" for n in spec["nodes"])


def test_parse_modulelist_repeat():
    spec = parse_pytorch_file(ROOT / "examples/pytorch/transformer_stack.py", class_name="EncoderModel", detail="system")
    assert validate_spec(spec) == []
    repeated = [n for n in spec["nodes"] if n.get("repeat", 1) > 1]
    assert repeated
    assert any(n.get("repeat") == 6 for n in repeated)
