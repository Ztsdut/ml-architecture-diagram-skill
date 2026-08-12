from __future__ import annotations

"""Automatic proposed-block selection and detail-panel compilation.

The publication overview and the detailed proposed block serve different purposes.
This module keeps them linked while avoiding unsupported invention: a detailed
panel is synthesized only when its internal operations are explicit in the IR or
when a conservative, well-defined architecture template is available.
"""

from copy import deepcopy
from typing import Iterable


NOVELTY_TOKENS = {
    "proposed": 4.0,
    "novel": 4.0,
    "operator": 3.0,
    "spectral": 3.0,
    "attention": 2.5,
    "transformer": 3.0,
    "graph": 2.0,
    "fusion": 1.5,
    "gate": 1.5,
    "latent": 1.5,
    "physics": 2.0,
    "neural field": 2.0,
}


def _score_node(node: dict, edges: Iterable[dict]) -> float:
    role = str(node.get("role", ""))
    text = " ".join(
        str(node.get(k, "")) for k in ("label", "subtitle", "shape")
    ).lower()
    score = 0.0
    if role == "novel":
        score += 8.0
    if node.get("detail"):
        score += 20.0
    rep = int(node.get("repeat", 1) or 1)
    if rep > 1:
        score += min(4.0, 1.0 + 0.6 * rep)
    for token, weight in NOVELTY_TOKENS.items():
        if token in text:
            score += weight
    if node.get("illustration"):
        score += 1.0
    nid = node.get("id")
    degree = sum(1 for e in edges if e.get("from") == nid or e.get("to") == nid)
    score += min(2.0, degree * 0.25)
    if role in {"input", "output", "head"}:
        score -= 4.0
    if node.get("kind") in {"output", "loss"}:
        score -= 4.0
    return score


def select_proposed_block(spec: dict) -> dict | None:
    """Return the highest-value node for a paper detail panel."""
    nodes = list(spec.get("nodes", []))
    edges = list(spec.get("edges", []))
    if not nodes:
        return None
    explicit = spec.get("figure", {}).get("detail_node")
    if explicit:
        return next((n for n in nodes if n.get("id") == explicit), None)
    ranked = sorted(nodes, key=lambda n: (_score_node(n, edges), str(n.get("id"))), reverse=True)
    best = ranked[0]
    return best if _score_node(best, edges) >= 4.0 else None


def _prefix_detail(candidate_id: str, local_id: str) -> str:
    safe = str(local_id).strip().replace(" ", "_") or "op"
    return f"detail__{candidate_id}__{safe}"


def _instantiate_explicit_detail(candidate: dict) -> tuple[list[dict], list[dict], dict] | None:
    detail = candidate.get("detail")
    if not isinstance(detail, dict):
        return None
    raw_nodes = detail.get("nodes") or []
    raw_edges = detail.get("edges") or []
    if not raw_nodes:
        return None
    cid = str(candidate["id"])
    id_map: dict[str, str] = {}
    nodes: list[dict] = []
    for i, raw in enumerate(raw_nodes):
        if isinstance(raw, str):
            raw = {"id": f"op{i+1}", "label": raw, "kind": "module", "role": "backbone"}
        local = str(raw.get("id") or f"op{i+1}")
        gid = _prefix_detail(cid, local)
        id_map[local] = gid
        node = deepcopy(raw)
        node["id"] = gid
        node["panel"] = f"detail_{cid}"
        node["_detail_panel"] = True
        node["provenance"] = {
            "detail_of": cid,
            "local_id": local,
            "source": "explicit_detail",
        }
        nodes.append(node)
    edges: list[dict] = []
    for raw in raw_edges:
        a, b = str(raw.get("from", "")), str(raw.get("to", ""))
        if a not in id_map or b not in id_map:
            continue
        e = deepcopy(raw)
        e["from"], e["to"] = id_map[a], id_map[b]
        e["_detail_panel"] = True
        edges.append(e)
    # If the user supplied a sequence of nodes without edges, connect them as a safe chain.
    if len(nodes) > 1 and not edges:
        for a, b in zip(nodes, nodes[1:]):
            edges.append({"from": a["id"], "to": b["id"], "type": "main", "_detail_panel": True})
    panel = {
        "id": f"detail_{cid}",
        "label": detail.get("label", "(b)"),
        "title": detail.get("title") or f"Detailed structure of {candidate.get('label', 'proposed block')}",
        "direction": detail.get("direction", "TB"),
        "node_ids": [n["id"] for n in nodes],
        "detail_of": cid,
    }
    return nodes, edges, panel


def _transformer_template(candidate: dict) -> tuple[list[dict], list[dict], dict]:
    cid = str(candidate["id"])
    pid = f"detail_{cid}"
    defs = [
        ("input", "Block input", "input", "data", "sequence_port"),
        ("norm1", "Layer Normalization", "preprocess", "operator", "norm_bar_publication"),
        ("attn", "Multi-Head Self-Attention", "novel", "module", "attention_block"),
        ("add1", "+", "fusion", "merge", "add_node"),
        ("norm2", "Layer Normalization", "preprocess", "operator", "norm_bar_publication"),
        ("ffn", "Feed-Forward Network", "backbone", "module", "ffn_block_publication"),
        ("add2", "+", "fusion", "merge", "add_node"),
        ("output", "Block output", "output", "output", "sequence_port"),
    ]
    nodes=[]
    for local,label,role,kind,vtype in defs:
        nodes.append({
            "id": _prefix_detail(cid, local), "label": label, "role": role, "kind": kind,
            "panel": pid, "_detail_panel": True,
            "visual": {"type": vtype},
            "provenance": {"detail_of": cid, "local_id": local, "source": "transformer_template"},
        })
    ids={d[0]: _prefix_detail(cid,d[0]) for d in defs}
    edges=[
        {"from":ids["input"],"to":ids["norm1"],"type":"main"},
        {"from":ids["norm1"],"to":ids["attn"],"type":"main"},
        {"from":ids["attn"],"to":ids["add1"],"type":"main"},
        {"from":ids["input"],"to":ids["add1"],"type":"residual"},
        {"from":ids["add1"],"to":ids["norm2"],"type":"main"},
        {"from":ids["norm2"],"to":ids["ffn"],"type":"main"},
        {"from":ids["ffn"],"to":ids["add2"],"type":"main"},
        {"from":ids["add1"],"to":ids["add2"],"type":"residual"},
        {"from":ids["add2"],"to":ids["output"],"type":"main"},
    ]
    for e in edges: e["_detail_panel"] = True
    panel={
        "id":pid,"label":"(b)","title":f"Detailed structure of {candidate.get('label','Transformer block')}",
        "direction":"TB","node_ids":[n["id"] for n in nodes],"detail_of":cid,
    }
    return nodes,edges,panel


def _known_template(spec: dict, candidate: dict):
    family = str(spec.get("metadata", {}).get("architecture_family", "")).lower()
    text = str(candidate.get("label", "")).lower()
    if family == "transformer" or "transformer" in text:
        return _transformer_template(candidate)
    return None


def compile_detail_panels(spec: dict) -> dict:
    """Add an overall + proposed-block detail composition when evidence permits.

    The exact overview nodes/edges are preserved. Detail nodes are *additional view
    nodes* with provenance linking them to the selected macro block. Unknown custom
    blocks are never expanded by guesswork; instead metadata records which block the
    agent should inspect and enrich during semantic review.
    """
    out = deepcopy(spec)
    fig = out.setdefault("figure", {})
    mode = str(fig.get("auto_detail", "auto")).lower()
    if mode in {"off", "false", "0", "none"}:
        return out

    # Respect a hand-authored multi-panel figure unless force is explicit.
    existing = out.get("panels") or []
    if len(existing) > 1 and mode != "force":
        out.setdefault("metadata", {}).setdefault("detail_panel", {})["status"] = "existing_panels_preserved"
        return out

    candidate = select_proposed_block(out)
    if not candidate:
        out.setdefault("metadata", {}).setdefault("detail_panel", {})["status"] = "no_candidate"
        return out

    built = _instantiate_explicit_detail(candidate) or _known_template(out, candidate)
    meta = out.setdefault("metadata", {}).setdefault("detail_panel", {})
    meta["candidate_id"] = candidate.get("id")
    meta["candidate_label"] = candidate.get("label")
    meta["candidate_score"] = round(_score_node(candidate, out.get("edges", [])), 3)

    if built is None:
        meta["status"] = "semantic_review_required"
        meta["instruction"] = (
            "Inspect the selected block implementation and add node.detail.nodes/edges "
            "during semantic review; do not invent unsupported internal operations."
        )
        return out

    detail_nodes, detail_edges, detail_panel = built
    detail_ids = {n["id"] for n in detail_nodes}
    # Defensive uniqueness if a repeated compile is applied.
    if detail_ids & {n.get("id") for n in out.get("nodes", [])}:
        meta["status"] = "already_compiled"
        return out

    original_ids = [n["id"] for n in out.get("nodes", [])]
    if existing:
        overall = deepcopy(existing[0])
        overall.setdefault("id", "overall")
        overall.setdefault("label", "(a)")
        overall.setdefault("title", "Overall architecture")
        overall["node_ids"] = [nid for nid in overall.get("node_ids", original_ids) if nid in original_ids]
    else:
        overall = {"id":"overall","label":"(a)","title":"Overall architecture","direction":fig.get("direction","LR"),"node_ids":original_ids}
    out["panels"] = [overall, detail_panel]
    out["nodes"].extend(detail_nodes)
    out.setdefault("edges", []).extend(detail_edges)
    candidate["detail_panel_ref"] = detail_panel["label"]
    candidate.setdefault("visual", {})["detail_emphasis"] = True
    fig["panel_layout"] = fig.get("panel_layout") or "vertical"
    fig.setdefault("panel_gap", 54.0)
    meta.update({
        "status": "compiled",
        "detail_panel_id": detail_panel["id"],
        "detail_source": "explicit" if candidate.get("detail") else "template",
        "detail_node_count": len(detail_nodes),
        "detail_edge_count": len(detail_edges),
    })
    return out


def detail_review_skeleton(spec: dict) -> dict:
    """Return a semantic-review patch skeleton for the selected custom block."""
    candidate = select_proposed_block(spec)
    if not candidate:
        return {"notes": ["No strong proposed-block candidate was detected."]}
    cid = str(candidate["id"])
    return {
        "nodes": {
            cid: {
                "detail": {
                    "label": "(b)",
                    "title": f"Detailed structure of {candidate.get('label', 'proposed block')}",
                    "direction": "TB",
                    "nodes": [
                        {"id": "input", "label": "Block input", "role": "input", "kind": "data"},
                        {"id": "op1", "label": "REPLACE WITH IMPLEMENTED OPERATION", "role": "novel", "kind": "module"},
                        {"id": "output", "label": "Block output", "role": "output", "kind": "output"},
                    ],
                    "edges": [
                        {"from": "input", "to": "op1", "type": "main"},
                        {"from": "op1", "to": "output", "type": "main"},
                    ],
                }
            }
        },
        "notes": [
            f"Selected proposed-block candidate: {cid}. Replace placeholders only with operations verified from code/config/method description."
        ],
    }
