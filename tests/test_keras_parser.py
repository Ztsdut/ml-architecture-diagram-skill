from pathlib import Path

from ml_architecture_diagram.parsers.keras_ast import parse_keras_file
from ml_architecture_diagram.spec import validate_spec

ROOT = Path(__file__).resolve().parents[1]


def test_parse_keras_functional_fusion():
    spec = parse_keras_file(ROOT / "examples/keras/functional_fusion.py", detail="system")
    assert validate_spec(spec) == []
    assert sum(1 for n in spec["nodes"] if n.get("role") == "input") == 2
    assert any(n.get("kind") == "merge" and n.get("label") == "Concat" for n in spec["nodes"])
    assert any(n.get("role") == "output" for n in spec["nodes"])
