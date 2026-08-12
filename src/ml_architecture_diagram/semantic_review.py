from __future__ import annotations

"""Apply an agent- or human-authored semantic review patch to Architecture IR.

The static parser is deliberately conservative. A paper figure often benefits from a
small review step that adds domain-facing names, tensor shapes, concise subtitles,
and overview-only routing hints while preserving the parsed computation graph.
"""

from copy import deepcopy
from typing import Any


def _merge_mapping(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge_mapping(dst[key], value)
        else:
            dst[key] = deepcopy(value)


def apply_semantic_review(spec: dict, patch: dict) -> dict:
    """Return a reviewed copy of *spec*.

    Supported patch fields:
      figure: mapping merged into figure metadata
      metadata: mapping merged into metadata
      nodes: mapping keyed by node id, or a list with ``id`` fields
      edges: list of edge patches matched by ``from``/``to`` and optional ``label``;
             fields under ``set`` are merged into the matched edge
      notes: free-form review notes stored in metadata.semantic_review.notes

    Edge patches may set ``overview: hide``. The exact reviewed IR keeps that edge;
    the publication-view compiler omits it only from the overview figure.
    """
    out = deepcopy(spec)
    if patch.get("figure"):
        _merge_mapping(out.setdefault("figure", {}), patch["figure"])
    if patch.get("metadata"):
        _merge_mapping(out.setdefault("metadata", {}), patch["metadata"])

    node_map = {n.get("id"): n for n in out.get("nodes", [])}
    node_patch = patch.get("nodes") or {}
    if isinstance(node_patch, list):
        node_patch = {n.get("id"): {k: v for k, v in n.items() if k != "id"} for n in node_patch if n.get("id")}
    for nid, changes in node_patch.items():
        if nid not in node_map:
            raise ValueError(f"Semantic review references unknown node {nid!r}")
        if not isinstance(changes, dict):
            raise ValueError(f"Node review for {nid!r} must be a mapping")
        _merge_mapping(node_map[nid], changes)

    matched_edge_patches = []
    for ep in patch.get("edges") or []:
        src, dst = ep.get("from"), ep.get("to")
        if not src or not dst:
            raise ValueError("Edge review entries require 'from' and 'to'")
        candidates = [e for e in out.get("edges", []) if e.get("from") == src and e.get("to") == dst]
        if "label" in ep:
            candidates = [e for e in candidates if e.get("label", "") == ep.get("label", "")]
        if not candidates:
            raise ValueError(f"Semantic review edge {src!r}->{dst!r} did not match the Architecture IR")
        changes = ep.get("set") or {k: v for k, v in ep.items() if k not in {"from", "to", "label"}}
        for e in candidates:
            _merge_mapping(e, changes)
        matched_edge_patches.append({"from": src, "to": dst, "count": len(candidates)})

    meta = out.setdefault("metadata", {})
    review = meta.setdefault("semantic_review", {})
    review["applied"] = True
    review["node_overrides"] = sorted(node_patch)
    review["edge_overrides"] = matched_edge_patches
    if patch.get("notes"):
        review["notes"] = deepcopy(patch["notes"])
    return out
