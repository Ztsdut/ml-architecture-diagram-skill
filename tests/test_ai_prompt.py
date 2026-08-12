from pathlib import Path

from ml_architecture_diagram.ai_prompt import architecture_prompt
from ml_architecture_diagram.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def test_prompt_preserves_topology_terms():
    spec = load_spec(ROOT / "examples" / "transformer_encoder.yaml")
    prompt = architecture_prompt(spec, mode="reference")
    assert "NON-NEGOTIABLE STRUCTURAL CONSTRAINTS" in prompt
    for node in spec["nodes"]:
        assert node["id"] in prompt
        assert node["label"] in prompt
    for edge in spec["edges"]:
        assert f"{edge['from']} -> {edge['to']}" in prompt


def test_reference_prompt_mentions_blueprint():
    spec = load_spec(ROOT / "examples" / "cnn_classifier.yaml")
    prompt = architecture_prompt(spec, mode="reference")
    assert "structural blueprint" in prompt


def test_generate_prompt_has_no_reference_requirement():
    spec = load_spec(ROOT / "examples" / "cnn_classifier.yaml")
    prompt = architecture_prompt(spec, mode="generate")
    assert "reference image containing the exact architecture layout" not in prompt.lower()
