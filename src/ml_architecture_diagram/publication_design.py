from __future__ import annotations

from copy import deepcopy
from collections import defaultdict

from .visual_grammar import compile_visual_spec
from .publication_view import compile_publication_view
from .scientific_illustration import compile_scientific_illustrations
from .detail_panels import compile_detail_panels


def _panel_nodes(spec: dict) -> dict[str, list[dict]]:
    panels = defaultdict(list)
    for node in spec.get("nodes", []):
        panels[str(node.get("panel") or "overall")].append(node)
    return panels


def _set_if_missing(mapping: dict, key: str, value):
    if key not in mapping:
        mapping[key] = value


def _decorate_unet(spec: dict) -> None:
    enc = [n for n in spec.get("nodes", []) if "encoder" in str(n.get("label", "")).lower()]
    dec = [n for n in spec.get("nodes", []) if "decoder" in str(n.get("label", "")).lower()]
    for i, n in enumerate(enc):
        v = n.setdefault("visual", {})
        v["type"] = "feature_tensor"
        v["stage"] = "encoder"
        v["depth"] = i
        v["spatial_scale"] = max(0.48, 1.0 - 0.18 * i)
        v["channel_scale"] = 1.0 + 0.22 * i
    for i, n in enumerate(dec):
        # Decoder list follows computation order (deep -> shallow) in normal IRs.
        depth = max(0, len(dec) - 1 - i)
        v = n.setdefault("visual", {})
        v["type"] = "feature_tensor"
        v["stage"] = "decoder"
        v["depth"] = depth
        v["spatial_scale"] = max(0.48, 1.0 - 0.18 * depth)
        v["channel_scale"] = 1.0 + 0.22 * depth
    for n in spec.get("nodes", []):
        label = str(n.get("label", "")).lower()
        if "bottleneck" in label:
            v = n.setdefault("visual", {})
            v["type"] = "feature_tensor"
            v["stage"] = "bottleneck"
            v["depth"] = max(len(enc), 2)
            v["spatial_scale"] = 0.42
            v["channel_scale"] = 1.6
        elif n.get("role") == "input" and n.get("kind") == "data":
            n.setdefault("visual", {})["type"] = "input_tensor"


def _decorate_transformer(spec: dict) -> None:
    panels = _panel_nodes(spec)
    for pid, nodes in panels.items():
        labels = " ".join(str(n.get("label", "")).lower() for n in nodes)
        detail = any(k in labels for k in ("multi-head", "attention", "feed-forward", "ffn")) and len(nodes) >= 5
        for n in nodes:
            text = str(n.get("label", "")).lower()
            v = n.setdefault("visual", {})
            if "transformer" in text and ("encoder" in text or "decoder" in text):
                v["type"] = "transformer_macro"
                v["representation"] = "macro"
            if detail:
                if "attention" in text or "mha" in text:
                    v["type"] = "attention_block"
                elif "feed-forward" in text or "feed forward" in text or "ffn" in text:
                    v["type"] = "ffn_block_publication"
                elif "norm" in text:
                    v["type"] = "norm_bar_publication"
                elif n.get("kind") == "merge":
                    v["type"] = "add_node"
                elif n.get("role") in {"input", "output"}:
                    v["type"] = "sequence_port"
    spec.setdefault("figure", {})["layout_preset"] = "publication_transformer"


def _decorate_multimodal(spec: dict) -> None:
    for n in spec.get("nodes", []):
        text = str(n.get("label", "")).lower()
        v = n.setdefault("visual", {})
        if n.get("role") == "input":
            if any(k in text for k in ("image", "vision", "map", "pixel")):
                v["type"] = "input_tensor"
            elif any(k in text for k in ("text", "token", "language", "sequence")):
                v["type"] = "token_strip_publication"
            else:
                v["type"] = "feature_vector"
        elif "encoder" in text:
            v["type"] = "encoder_module"
        elif n.get("role") == "fusion" or n.get("kind") == "merge":
            v["type"] = "fusion_node_publication"
        elif n.get("role") == "novel":
            v["type"] = "emphasis_module"
    spec.setdefault("figure", {})["layout_preset"] = "publication_multilane"


def _decorate_operator(spec: dict) -> None:
    for n in spec.get("nodes", []):
        text = str(n.get("label", "")).lower()
        v = n.setdefault("visual", {})
        if n.get("role") == "input" and n.get("kind") == "data":
            v["type"] = "field_tensor"
        elif "spectral" in text or "fourier" in text:
            v["type"] = "spectral_operator_publication"
        elif "lifting" in text or "projection" in text:
            v["type"] = "linear_map_publication"
        elif n.get("role") == "output":
            v["type"] = "field_tensor_output"
    spec.setdefault("figure", {})["layout_preset"] = "publication_operator"


def _decorate_cnn(spec: dict) -> None:
    depth = 0
    for n in spec.get("nodes", []):
        text = str(n.get("label", "")).lower()
        v = n.setdefault("visual", {})
        if n.get("role") == "input" and n.get("kind") == "data":
            v["type"] = "input_tensor"
            v["spatial_scale"] = 1.0
        elif any(k in text for k in ("conv", "res", "stage", "backbone", "feature")):
            v["type"] = "feature_tensor"
            v["depth"] = depth
            v["spatial_scale"] = max(0.46, 1.0 - 0.14 * depth)
            v["channel_scale"] = 1.0 + 0.20 * depth
            depth += 1
        elif "pool" in text:
            v["type"] = "pooling_bar_publication"



def _decorate_framework(spec: dict) -> None:
    """Compose complex multi-input/multi-head systems as a staged framework figure.

    This is deliberately a *publication composition* decision. It does not change
    graph topology; it assigns stage groups and visual primitives so the renderer
    can produce a Figure-1-style overview rather than a long DAG ribbon.
    """
    nodes = spec.get("nodes", [])
    for n in nodes:
        if n.get("_detail_panel"):
            continue
        text = str(n.get("label", "")).lower()
        role = n.get("role")
        kind = n.get("kind")
        v = n.setdefault("visual", {})
        if role == "input":
            n["group"] = "stage_inputs"
            v["type"] = "input_card_publication"
            if any(k in text for k in ("map", "image", "grid", "field")):
                v["input_symbol"] = "tensor"
            elif any(k in text for k in ("sequence", "path", "history", "token")):
                v["input_symbol"] = "tokens"
            else:
                v["input_symbol"] = "vector"
        elif role == "head" or role == "output" or kind == "output":
            n["group"] = "stage_outputs"
            v["type"] = "emphasis_module" if role == "head" else "output_card"
        elif "encoder" in text or role in {"preprocess", "representation"}:
            n["group"] = "stage_encoders"
            v["type"] = "encoder_module"
        else:
            n["group"] = "stage_interaction"
            if "attention" in text:
                v["type"] = "attention_block"
                v["emphasis"] = "primary"
            elif "gate" in text:
                v["type"] = "emphasis_module"
                v["emphasis"] = "primary"
            elif kind == "merge":
                if "feature fusion" in text or "feature-fusion" in text:
                    v["type"] = "fusion_bar_publication"
                else:
                    v["type"] = "add_node" if str(n.get("label")) == "+" else "fusion_node_publication"
            elif role == "novel":
                v["type"] = "emphasis_module"

    stage_labels = spec.get("figure", {}).get("stage_labels") or {}
    spec["groups"] = [
        {"id": "stage_inputs", "label": stage_labels.get("inputs", "1) Inputs"), "accent": "input"},
        {"id": "stage_encoders", "label": stage_labels.get("encoders", "2) Encoders"), "accent": "backbone"},
        {"id": "stage_interaction", "label": stage_labels.get("interaction", "3) Context interaction and fusion"), "accent": "novel"},
        {"id": "stage_outputs", "label": stage_labels.get("outputs", "4) Prediction and outputs"), "accent": "output"},
    ]
    fig = spec.setdefault("figure", {})
    fig["layout_preset"] = "publication_framework"
    fig["composition"] = "framework_overview"
    spec.setdefault("style", {})["stage_containers"] = True


def _apply_emphasis(spec: dict) -> None:
    for n in spec.get("nodes", []):
        v = n.setdefault("visual", {})
        if n.get("role") == "novel":
            _set_if_missing(v, "emphasis", "primary")
        elif n.get("role") in {"input", "output"}:
            _set_if_missing(v, "emphasis", "quiet")
        else:
            _set_if_missing(v, "emphasis", "normal")


def compile_publication_spec(spec: dict) -> dict:
    """Enrich Architecture IR with publication-design metadata only.

    The exact overview topology, labels, edge endpoints, and repeat counts are preserved.
    The compiler may add provenance-tagged *view-only* nodes/edges for a detailed
    proposed-block panel when verified internal structure is available.
    """
    # First build a paper-facing view graph. The exact parser IR remains untouched
    # on disk and every view node carries provenance back to the exact graph.
    out = compile_publication_view(deepcopy(spec), show_diagnostics=bool(spec.get("figure", {}).get("show_diagnostics", False)))
    out = compile_detail_panels(out)
    out = compile_visual_spec(out)
    out = compile_scientific_illustrations(out, max_auto=int(spec.get("style", {}).get("illustration_budget", 6)))
    family = str(out.get("metadata", {}).get("architecture_family", "generic"))
    out.setdefault("metadata", {})["publication_design_version"] = "2.0"
    fig = out.setdefault("figure", {})
    style = out.setdefault("style", {})

    _set_if_missing(fig, "target", "paper")
    _set_if_missing(fig, "width_mm", 178)
    _set_if_missing(fig, "font", "Arial")
    _set_if_missing(fig, "theme", "publication")
    _set_if_missing(fig, "show_title", False)
    _set_if_missing(fig, "panel_headers", "minimal")
    _set_if_missing(style, "publication_engine", True)
    _set_if_missing(style, "module_fill", "white-accent")
    _set_if_missing(style, "line_language", "fine")
    _set_if_missing(style, "show_shapes", True)
    _set_if_missing(style, "show_repeat_badges", True)
    _set_if_missing(style, "decorative_icons", False)

    if family == "transformer":
        _decorate_transformer(out)
    elif family == "unet":
        _decorate_unet(out)
        fig["layout_preset"] = "publication_unet"
    elif family == "multimodal":
        input_count = sum(1 for n in out.get("nodes", []) if n.get("role") == "input")
        head_count = sum(1 for n in out.get("nodes", []) if n.get("role") == "head")
        if input_count >= 3 and (head_count >= 2 or len(out.get("nodes", [])) >= 12):
            _decorate_framework(out)
        else:
            _decorate_multimodal(out)
    elif family == "operator":
        _decorate_operator(out)
    elif family == "cnn":
        _decorate_cnn(out)
        fig.setdefault("layout_preset", "publication_flow")
    else:
        # Generic publication mode still removes slide-like decoration and uses fine linework.
        fig.setdefault("layout_preset", fig.get("layout_preset") or "publication_flow")

    _apply_emphasis(out)
    return out
