from copy import deepcopy

from ml_architecture_diagram.scientific_illustration import (
    compile_scientific_illustrations,
    infer_illustration,
    validate_illustration,
)


def _spec():
    return {
        "figure": {"target": "paper"},
        "metadata": {},
        "nodes": [
            {"id": "drivers", "label": "72-h causal driver history", "role": "input", "kind": "data"},
            {"id": "sphere", "label": "Global latent prior on Fibonacci sphere nodes", "role": "novel", "kind": "module"},
            {"id": "graph", "label": "Local graph propagation", "role": "novel", "kind": "module"},
            {"id": "head", "label": "Prediction head", "role": "head", "kind": "module"},
        ],
        "edges": [
            {"from": "drivers", "to": "sphere", "type": "main"},
            {"from": "sphere", "to": "graph", "type": "main"},
            {"from": "graph", "to": "head", "type": "main"},
        ],
    }


def test_compile_illustrations_preserves_topology():
    spec = _spec()
    before_nodes = [n["id"] for n in spec["nodes"]]
    before_edges = deepcopy(spec["edges"])
    out = compile_scientific_illustrations(spec, max_auto=3)
    assert [n["id"] for n in out["nodes"]] == before_nodes
    assert out["edges"] == before_edges
    assert out["metadata"]["scientific_illustration_version"] == "1.0"
    assert any(n.get("illustration", {}).get("type") == "fibonacci_sphere" for n in out["nodes"])
    assert any(n.get("illustration", {}).get("type") == "graph_network" for n in out["nodes"])


def test_explicit_custom_dsl_wins_and_validates():
    node = {
        "id": "custom",
        "label": "Novel sensor geometry",
        "illustration": {
            "type": "custom_dsl",
            "composition": "illustration-top",
            "primitives": [
                {"shape": "circle", "cx": 20, "cy": 60, "r": 5, "fill": "accent"},
                {"shape": "line", "x1": 20, "y1": 60, "x2": 80, "y2": 30},
            ],
        },
    }
    assert infer_illustration(node)["type"] == "custom_dsl"
    assert validate_illustration(node["illustration"]) == []


def test_bad_custom_dsl_is_rejected():
    errors = validate_illustration({"type": "custom_dsl", "primitives": [{"shape": "path"}]})
    assert errors
