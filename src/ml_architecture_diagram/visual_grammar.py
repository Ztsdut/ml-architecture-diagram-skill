from __future__ import annotations

from copy import deepcopy
import re
from collections import Counter


FAMILY_KEYWORDS = {
    "transformer": ("transformer", "attention", "multi-head", "mha", "token", "position embedding", "ffn"),
    "unet": ("u-net", "unet", "encoder-decoder", "bottleneck", "decoder", "skip"),
    "gnn": ("graph neural", "gnn", "message passing", "message-passing", "graph readout", "node", "edge"),
    "moe": ("mixture-of-experts", "mixture of experts", "moe", "expert", "router", "top-k", "routing"),
    "diffusion": ("diffusion", "denoiser", "noise", "timestep", "score network", "sampler"),
    "rnn": ("lstm", "gru", "rnn", "recurrent", "bidirectional", "sequence"),
    "cnn": ("cnn", "conv", "convolution", "pool", "resnet", "feature map", "vision backbone"),
    "operator": ("operator learning", "fourier", "spectral", "neural operator", "fno", "deeponet", "branch net", "trunk net"),
    "multimodal": ("multimodal", "multi-modal", "fusion", "modality", "cross-modal"),
}


VISUAL_TYPES = {
    "generic_module", "data_stack", "feature_map_stack", "token_strip", "token_matrix",
    "transformer_stack", "attention_heads", "ffn_block", "norm_bar", "graph_input",
    "graph_message", "graph_pool", "router_gate", "expert_fan", "weighted_merge",
    "sequence_strip", "recurrent_cells", "unet_stage", "bottleneck", "diffusion_noise",
    "diffusion_denoiser", "time_condition", "spectral_operator", "operator_branch",
    "operator_trunk", "fusion_hub", "output_card", "operator_glyph", "merge_glyph",
    "modality_card", "loss_card", "embedding_card", "pooling_glyph", "classifier_head",
    "input_tensor", "feature_tensor", "field_tensor", "field_tensor_output",
    "feature_vector", "encoder_module", "emphasis_module", "transformer_macro",
    "attention_block", "ffn_block_publication", "norm_bar_publication", "add_node",
    "sequence_port", "token_strip_publication", "fusion_node_publication",
    "spectral_operator_publication", "linear_map_publication", "pooling_bar_publication",
}


def _text_blob(spec: dict) -> str:
    chunks = [str(spec.get("figure", {}).get("title", "")), str(spec.get("metadata", {}).get("architecture_family", ""))]
    for n in spec.get("nodes", []):
        chunks.extend([
            str(n.get("label", "")), str(n.get("subtitle", "")), str(n.get("shape", "")),
            " ".join(map(str, n.get("details", []) or [])),
        ])
    for e in spec.get("edges", []):
        chunks.append(str(e.get("label", "")))
    return " ".join(chunks).lower()


def infer_architecture_family(spec: dict) -> str:
    """Infer a broad family for visual design, never for architectural truth."""
    meta = spec.get("metadata", {}) or {}
    explicit = meta.get("architecture_family") or spec.get("figure", {}).get("architecture_family")
    if explicit:
        return str(explicit).strip().lower().replace(" ", "_")

    blob = _text_blob(spec)
    scores = Counter()
    for family, kws in FAMILY_KEYWORDS.items():
        for kw in kws:
            if kw in blob:
                scores[family] += 2 if " " in kw or "-" in kw else 1

    # Strong structural cues.
    labels = {str(n.get("label", "")).lower() for n in spec.get("nodes", [])}
    if any("encoder" in x for x in labels) and any("decoder" in x for x in labels) and any(e.get("type") == "residual" for e in spec.get("edges", [])):
        scores["unet"] += 4
    if any("router" in x or "gate" in x for x in labels) and any("expert" in x for x in labels):
        scores["moe"] += 5
    if any("message" in x and "pass" in x for x in labels):
        scores["gnn"] += 5
    if any("attention" in x for x in labels):
        scores["transformer"] += 4
    if any("conv" in x for x in labels):
        scores["cnn"] += 3

    # Structural override for real multi-branch systems. Attention is an operator, not
    # sufficient evidence that the *whole architecture* is a Transformer. A model with
    # three or more independent inputs and several fusion/attention points is better
    # composed as a multimodal framework overview.
    input_count = sum(1 for n in spec.get("nodes", []) if n.get("role") == "input")
    fusionish = sum(1 for n in spec.get("nodes", []) if n.get("role") == "fusion" or n.get("kind") == "merge")
    attention_count = sum(1 for n in spec.get("nodes", []) if "attention" in str(n.get("label", "")).lower())
    head_count = sum(1 for n in spec.get("nodes", []) if n.get("role") == "head")
    if input_count >= 3 and (fusionish >= 2 or attention_count >= 2 or head_count >= 2):
        return "multimodal"

    if not scores:
        indegree = Counter(e.get("to") for e in spec.get("edges", []))
        if any(v >= 2 for v in indegree.values()) and input_count >= 2:
            return "multimodal"
        return "generic"
    return scores.most_common(1)[0][0]


def _label(node: dict) -> str:
    return " ".join([
        str(node.get("label", "")), str(node.get("subtitle", "")), str(node.get("shape", "")),
        " ".join(map(str, node.get("details", []) or [])),
    ]).lower()


def _shape_numbers(node: dict) -> dict:
    """Extract lightweight visual hints from shape strings without asserting tensor semantics."""
    shape = str(node.get("shape", ""))
    nums = [int(x) for x in re.findall(r"(?<![A-Za-z])\d+(?![A-Za-z])", shape)]
    out: dict[str, int | list[int]] = {}
    if nums:
        out["shape_numbers"] = nums[:4]
    return out


def _visual_for_node(node: dict, family: str) -> dict:
    existing = node.get("visual")
    if isinstance(existing, dict) and existing.get("type"):
        return dict(existing)
    if isinstance(existing, str):
        return {"type": existing}

    text = _label(node)
    kind = node.get("kind", "module")
    role = node.get("role", "neutral")
    repeat = int(node.get("repeat", 1) or 1)
    hints = _shape_numbers(node)

    # Family-specific semantic primitives that may intentionally use kind=merge/operator in the IR.
    if family == "gnn" and ("readout" in text or "graph pool" in text or "global pool" in text):
        return {"type": "graph_pool", **hints}
    if family == "multimodal" and (role == "fusion" or "fusion" in text or "concat" in text):
        return {"type": "fusion_hub", **hints}
    if family == "moe" and ("weighted" in text or "merge" in text):
        return {"type": "weighted_merge", **hints}

    # Universal semantic primitives first.
    if kind == "merge":
        if "concat" in text:
            return {"type": "merge_glyph", "symbol": "∥", **hints}
        if "weighted" in text or "merge" in text:
            return {"type": "weighted_merge", **hints}
        return {"type": "merge_glyph", "symbol": str(node.get("label", "+")), **hints}
    if kind == "operator":
        if "pool" in text:
            return {"type": "pooling_glyph", **hints}
        if "norm" in text:
            return {"type": "norm_bar", **hints}
        return {"type": "operator_glyph", **hints}
    if kind == "loss":
        return {"type": "loss_card", **hints}
    if role == "output" or kind == "output":
        return {"type": "output_card", **hints}

    if family == "cnn":
        if kind == "data":
            return {"type": "feature_map_stack" if any(k in text for k in ("image", "feature", "map", "tensor")) else "data_stack", **hints}
        if any(k in text for k in ("conv", "residual", "resnet", "stage", "backbone", "feature")):
            return {"type": "feature_map_stack", "layers": min(4, max(2, repeat)), **hints}
        if "pool" in text:
            return {"type": "pooling_glyph", **hints}
        if any(k in text for k in ("classifier", "head", "linear", "fc")):
            return {"type": "classifier_head", **hints}

    if family == "transformer":
        if kind == "data" and any(k in text for k in ("token", "sequence", "patch")):
            return {"type": "token_strip", **hints}
        if "embedding" in text:
            return {"type": "embedding_card", **hints}
        if "attention" in text or "mha" in text:
            return {"type": "attention_heads", "heads": 4, **hints}
        if "feed-forward" in text or "feed forward" in text or "ffn" in text or "mlp" in text:
            return {"type": "ffn_block", **hints}
        if "norm" in text:
            return {"type": "norm_bar", **hints}
        if "transformer" in text or "encoder" in text or "decoder" in text:
            return {"type": "transformer_stack", "layers": min(4, max(2, repeat)), **hints}
        if "pool" in text:
            return {"type": "pooling_glyph", **hints}

    if family == "unet":
        if kind == "data":
            return {"type": "feature_map_stack", **hints}
        if "bottleneck" in text:
            return {"type": "bottleneck", **hints}
        if "encoder" in text or "decoder" in text:
            return {"type": "unet_stage", **hints}

    if family == "gnn":
        if kind == "data" or "graph input" in text:
            return {"type": "graph_input", **hints}
        if "message" in text or "graph conv" in text or "aggregation" in text:
            return {"type": "graph_message", "layers": min(4, max(2, repeat)), **hints}
        if "readout" in text or "pool" in text:
            return {"type": "graph_pool", **hints}

    if family == "moe":
        if "router" in text or "gate" in text:
            return {"type": "router_gate", **hints}
        if "expert" in text:
            return {"type": "expert_fan", "experts": min(6, max(3, repeat)), "total_experts": repeat, **hints}
        if "merge" in text:
            return {"type": "weighted_merge", **hints}

    if family == "diffusion":
        if kind == "data" and "noise" in text:
            return {"type": "diffusion_noise", **hints}
        if "time" in text or "timestep" in text:
            return {"type": "time_condition", **hints}
        if any(k in text for k in ("denoiser", "unet", "score", "backbone")):
            return {"type": "diffusion_denoiser", **hints}

    if family == "rnn":
        if kind == "data" or "sequence" in text:
            return {"type": "sequence_strip", **hints}
        if any(k in text for k in ("lstm", "gru", "rnn", "recurrent")):
            return {"type": "recurrent_cells", "cells": min(5, max(3, repeat)), **hints}

    if family == "operator":
        if "branch" in text:
            return {"type": "operator_branch", **hints}
        if "trunk" in text:
            return {"type": "operator_trunk", **hints}
        if any(k in text for k in ("spectral", "fourier", "operator")):
            return {"type": "spectral_operator", **hints}

    if family == "multimodal":
        if kind == "data" or role == "input":
            return {"type": "modality_card", **hints}
        if role == "fusion" or "fusion" in text or "concat" in text:
            return {"type": "fusion_hub", **hints}

    if kind == "data":
        return {"type": "data_stack", **hints}
    if "embedding" in text:
        return {"type": "embedding_card", **hints}
    if role == "fusion":
        return {"type": "fusion_hub", **hints}
    if role == "head":
        return {"type": "classifier_head", **hints}
    return {"type": "generic_module", **hints}


def _infer_layout_preset(spec: dict, family: str) -> str:
    fig = spec.get("figure", {})
    explicit = fig.get("layout_preset")
    if explicit:
        return str(explicit)
    if family == "unet":
        return "unet"
    if family == "moe":
        return "branch_fan"
    if family == "multimodal":
        return "multi_lane"
    if family == "gnn":
        return "semantic_flow"
    if family == "transformer":
        return "block_diagram"
    if family == "diffusion":
        return "conditioned_flow"
    return "semantic_flow"


def compile_visual_spec(spec: dict) -> dict:
    """Return a deep-copied spec enriched with visual semantics and layout hints.

    Visual inference never changes nodes, edges, labels, repeat counts, or architectural
    truth. It only adds presentation metadata.
    """
    out = deepcopy(spec)
    family = infer_architecture_family(out)
    out.setdefault("metadata", {})["architecture_family"] = family
    out.setdefault("metadata", {})["visual_grammar_version"] = "1.0"
    out.setdefault("figure", {})["architecture_family"] = family
    out["figure"].setdefault("layout_preset", _infer_layout_preset(out, family))
    out.setdefault("style", {})
    out["style"].setdefault("visual_grammar", True)
    for node in out.get("nodes", []):
        node["visual"] = _visual_for_node(node, family)
    return out


def visual_inventory(spec: dict) -> dict[str, int]:
    compiled = compile_visual_spec(spec)
    counts = Counter(str((n.get("visual") or {}).get("type", "generic_module")) for n in compiled.get("nodes", []))
    return dict(sorted(counts.items()))
