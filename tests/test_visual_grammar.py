from pathlib import Path
from ml_architecture_diagram.spec import load_spec
from ml_architecture_diagram.visual_grammar import compile_visual_spec, infer_architecture_family, visual_inventory

ROOT = Path(__file__).resolve().parents[1]


def _compiled(name):
    return compile_visual_spec(load_spec(ROOT / "examples" / name))


def test_family_inference():
    assert infer_architecture_family(load_spec(ROOT / "examples" / "transformer_encoder.yaml")) == "transformer"
    assert infer_architecture_family(load_spec(ROOT / "examples" / "unet_encoder_decoder.yaml")) == "unet"
    assert infer_architecture_family(load_spec(ROOT / "examples" / "gnn_readout.yaml")) == "gnn"
    assert infer_architecture_family(load_spec(ROOT / "examples" / "moe_router.yaml")) == "moe"
    assert infer_architecture_family(load_spec(ROOT / "examples" / "lstm_forecaster.yaml")) == "rnn"
    assert infer_architecture_family(load_spec(ROOT / "examples" / "diffusion_denoiser.yaml")) == "diffusion"
    assert infer_architecture_family(load_spec(ROOT / "examples" / "neural_operator.yaml")) == "operator"


def test_transformer_visuals_are_specialized():
    spec = _compiled("transformer_encoder.yaml")
    inv = visual_inventory(spec)
    assert inv.get("transformer_stack", 0) >= 1
    assert inv.get("attention_heads", 0) >= 1
    assert inv.get("ffn_block", 0) >= 1


def test_unet_gets_u_layout():
    spec = _compiled("unet_encoder_decoder.yaml")
    assert spec["figure"]["layout_preset"] == "unet"
    assert any(n["visual"]["type"] == "bottleneck" for n in spec["nodes"])


def test_visual_compiler_does_not_change_topology():
    raw = load_spec(ROOT / "examples" / "multimodal_fusion.yaml")
    compiled = compile_visual_spec(raw)
    assert [(e["from"], e["to"], e.get("type", "main")) for e in raw["edges"]] == [
        (e["from"], e["to"], e.get("type", "main")) for e in compiled["edges"]
    ]
    assert [n["id"] for n in raw["nodes"]] == [n["id"] for n in compiled["nodes"]]
