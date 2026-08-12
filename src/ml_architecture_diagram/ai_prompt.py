from __future__ import annotations

from collections import defaultdict
from .publication_design import compile_publication_spec


ROLE_WORDS = {
    "input": "input data / observed features",
    "preprocess": "preprocessing",
    "representation": "learned representation / encoder",
    "backbone": "main backbone",
    "novel": "proposed or novel module",
    "auxiliary": "auxiliary / conditioning branch",
    "operator": "mathematical or neural operator",
    "fusion": "feature fusion",
    "head": "prediction head",
    "output": "model output",
    "training": "training-only component",
    "neutral": "general module",
}

EDGE_WORDS = {
    "main": "main forward flow",
    "residual": "residual / skip connection",
    "auxiliary": "auxiliary branch",
    "conditioning": "conditioning path",
    "training": "training-only path",
}


def _node_line(node: dict) -> str:
    role = ROLE_WORDS.get(node.get("role", "neutral"), node.get("role", "module"))
    parts = [f"{node['id']}: {node.get('label', node['id'])}", f"role={role}"]
    if node.get("subtitle"):
        parts.append(f"subtitle={node['subtitle']}")
    if node.get("shape"):
        parts.append(f"tensor/shape={node['shape']}")
    if int(node.get("repeat", 1) or 1) > 1:
        parts.append(f"repeated ×{int(node['repeat'])}")
    if node.get("shared"):
        parts.append("shared weights")
    if node.get("details"):
        parts.append("details=" + ", ".join(map(str, node["details"][:6])))
    visual = node.get("visual") or {}
    if isinstance(visual, dict) and visual.get("type"):
        parts.append(f"visual_grammar={visual['type']}")
    illustration = node.get("illustration") or {}
    if isinstance(illustration, dict) and illustration.get("type"):
        item = f"scientific_illustration={illustration['type']}"
        if illustration.get("composition"):
            item += f"/{illustration['composition']}"
        parts.append(item)
        if illustration.get("evidence"):
            parts.append(f"illustration_evidence={illustration['evidence']}")
    return " | ".join(parts)


def _panel_summary(spec: dict) -> list[str]:
    panels = spec.get("panels") or []
    if not panels:
        return []
    out = []
    for p in panels:
        ids = p.get("node_ids") or [n["id"] for n in spec.get("nodes", []) if n.get("panel") == p.get("id")]
        title = " ".join(x for x in [str(p.get("label", "")), str(p.get("title", ""))] if x).strip()
        out.append(f"- panel {p['id']} ({title or 'untitled'}): {', '.join(ids)}")
    return out


def architecture_prompt(spec: dict, *, mode: str = "generate", style: str = "publication-rich") -> str:
    """Build a fidelity-first prompt for an image-generation model.

    The prompt intentionally repeats topology constraints because image models may
    otherwise beautify by changing the graph. `reference` mode assumes a precise
    deterministic diagram is supplied as an input image.
    """
    spec = compile_publication_spec(spec)
    fig = spec.get("figure", {})
    title = str(fig.get("title", "Machine-learning architecture"))
    direction = fig.get("direction", "LR")
    target = fig.get("target", "paper")
    metadata = spec.get("metadata", {}) or {}
    family = metadata.get("architecture_family") or metadata.get("family") or "general machine-learning model"

    nodes = spec.get("nodes", [])
    edges = spec.get("edges", [])
    node_lines = "\n".join(f"- {_node_line(n)}" for n in nodes)
    edge_lines = "\n".join(
        f"- {e['from']} -> {e['to']} | {EDGE_WORDS.get(e.get('type', 'main'), e.get('type', 'main'))}"
        + (f" | label={e['label']}" if e.get("label") else "")
        for e in edges
    )
    panel_lines = "\n".join(_panel_summary(spec))

    ref_intro = ""
    if mode == "reference":
        ref_intro = (
            "A reference image containing the exact architecture layout is attached. "
            "Treat it as a structural blueprint. Preserve every node, label, repetition count, panel, and arrow direction. "
            "Do not move modules enough to change the topology. Improve only visual design, hierarchy, grouping, iconography, and dimensional cues.\n\n"
        )

    style_rules = """
Create an original, publication-quality scientific figure suitable for an IEEE/ACM/Nature-style machine-learning paper.
The visual language should be richer than a generic flowchart but still precise and restrained:
- white background, manuscript-scale composition, compact purposeful whitespace, aligned grid, strong visual hierarchy;
- do NOT make a presentation slide: no oversized title, no banner, no decorative subtitle, no huge empty canvas;
- 3–5 restrained low-saturation semantic accents, with mostly white module interiors;
- follow each node's visual_grammar hint as the preferred scientific glyph while preserving that node as exactly one architecture node;
- CNN/spatial tensors may use feature-map stacks; token sequences may use token strips/matrices; attention may use multi-head glyphs; graph modules may use node-edge glyphs; MoE experts may use an expert fan; recurrent modules may use symbolic cell chains;
- learned modules should use restrained scientific primitives: fine outlines, square/slightly rounded corners, small accent bands; avoid generic dashboard cards;
- use subtle 2.5D tensor depth only when the supplied visual grammar indicates tensor/spatial structure;
- use compact mathematical operator glyphs for add/concat/gating when present;
- visually distinguish main flow, residual/skip paths, conditioning paths, and training-only paths;
- novel/proposed modules may be visually emphasized and can include a small internal schematic if details are provided;
- repeated blocks should be represented cleanly with ×N, braces, or a short stack, not by copying dozens of blocks;
- when a node includes scientific_illustration metadata, reproduce that symbolic scientific concept inside the node without changing its architecture identity;
- scientific mini-illustrations should look like compact paper sketches (geometry, graph, field, sensor, spectrum, uncertainty), not decorative stock icons;
- avoid decorative icons by default for nodes without an explicit scientific illustration; represent data with tensor, token, vector, graph, field, or sequence geometry instead;
- typography must be crisp, compact, and readable at 178 mm two-column manuscript width; labels should sit close to the objects they describe;
- no photorealistic scene, no glossy corporate infographic, no neon cyberpunk styling, no decorative AI brain, no random circuitry.
""".strip()

    fidelity = """
NON-NEGOTIABLE STRUCTURAL CONSTRAINTS:
1. Do not invent, delete, merge, split, rename, or reorder architecture nodes.
2. Do not invent attention, convolution, transformer, physics, residual, recurrent, graph, or spectral operations unless explicitly present below.
3. Preserve all directed connections exactly. A residual edge must remain a skip/residual path; an auxiliary or conditioning edge must not become a main-path edge.
4. Preserve every explicit ×N repetition count and every tensor/shape annotation that appears below.
5. Training-only components must be visually separated from inference flow.
6. If any label cannot be rendered perfectly, keep the box location and leave the label simple rather than substituting a different technical term.
""".strip()

    layout_text = "left-to-right" if direction == "LR" else "top-to-bottom"
    extra = []
    if metadata.get("novelty"):
        extra.append(f"Scientific emphasis / novelty: {metadata['novelty']}")
    if metadata.get("caption"):
        extra.append(f"Figure intent: {metadata['caption']}")
    if spec.get("annotations"):
        extra.append("Annotations: " + "; ".join(map(str, spec["annotations"][:8])))

    return f"""{ref_intro}TITLE: {title}
ARCHITECTURE FAMILY: {family}
TARGET: {target}
READING DIRECTION: {layout_text}
STYLE PROFILE: {style}

{style_rules}

{fidelity}

EXACT NODES:
{node_lines}

EXACT EDGES:
{edge_lines or '- none'}

PANELS:
{panel_lines or '- single overall panel'}

{chr(10).join(extra)}

Produce one coherent scientific architecture diagram. The architecture, labels, counts, and connections must match the specification above exactly; visual sophistication must never override technical fidelity.
""".strip()
