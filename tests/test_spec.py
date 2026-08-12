from pathlib import Path
from ml_architecture_diagram.spec import load_spec, validate_spec

ROOT = Path(__file__).resolve().parents[1]


def test_examples_validate():
    for path in (ROOT / "examples").glob("*.yaml"):
        spec = load_spec(path)
        assert validate_spec(spec) == [], path


def test_unknown_edge_is_rejected():
    spec = {
        "version": "1.0",
        "figure": {"title": "x"},
        "nodes": [{"id": "a", "label": "A"}],
        "edges": [{"from": "a", "to": "missing"}],
    }
    errors = validate_spec(spec, use_jsonschema=False)
    assert any("unknown target" in e for e in errors)
