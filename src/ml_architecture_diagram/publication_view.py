from __future__ import annotations

"""Publication-view compiler.

The exact Architecture IR is intentionally richer than a paper figure. This module
creates a *view graph* that may collapse implementation-detail merge nodes while
preserving provenance back to the exact graph. It is a presentation transform, not
a model parser.
"""

from copy import deepcopy
from collections import defaultdict
from typing import Any


_GENERIC_MERGE_LABELS = {"concat", "concatenate", "stack", "+", "add"}
_DIAGNOSTIC_OUTPUT_TERMS = {"attention", "weights", "feature weights", "map attention"}


def _dedupe_edges(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out=[]; seen=set()
    for e in edges:
        key=(e.get("from"), e.get("to"), e.get("type","main"), e.get("label",""))
        if key in seen or not e.get("from") or not e.get("to") or e.get("from")==e.get("to"):
            continue
        seen.add(key); out.append(e)
    return out


def compile_publication_view(spec: dict, *, show_diagnostics: bool = False) -> dict:
    """Return a compact paper-facing view with provenance.

    Rules are conservative:
    * collapse generic concat/add/stack nodes when they only feed one downstream node;
    * keep semantically named fusion nodes;
    * optionally hide diagnostic-only outputs such as attention-weight returns;
    * every remaining view node records ``source_nodes`` and the view metadata records
      which exact nodes were collapsed/hidden.
    """
    out = deepcopy(spec)
    nodes = [deepcopy(n) for n in out.get("nodes", [])]
    original_edges = [deepcopy(e) for e in out.get("edges", [])]
    overview_hidden_edges = [e for e in original_edges if str(e.get("overview", "")).lower() in {"hide", "omit"}]
    edges = [e for e in original_edges if e not in overview_hidden_edges]
    node_map = {n["id"]: n for n in nodes}
    collapsed: list[dict[str, Any]] = []
    hidden: list[str] = []

    for n in nodes:
        n.setdefault("source_nodes", [n["id"]])

    # Repeatedly collapse low-value merge nodes. This commonly converts code-level
    # torch.cat()/residual bookkeeping into direct semantic inputs to the next module.
    changed = True
    while changed:
        changed = False
        incoming=defaultdict(list); outgoing=defaultdict(list)
        for e in edges:
            incoming[e.get("to")].append(e); outgoing[e.get("from")].append(e)
        for n in list(nodes):
            nid=n["id"]; label=str(n.get("label","")).strip().lower()
            if n.get("kind") != "merge" or label not in _GENERIC_MERGE_LABELS:
                continue
            ins=incoming.get(nid,[]); outs=outgoing.get(nid,[])
            # Standard case: merge -> one semantic child. Also collapse a unary add/concat
            # even when it fans out (e.g. a learned query used by two attention blocks).
            if not outs or not (len(outs) == 1 or len(ins) <= 1):
                continue
            child_ids=list(dict.fromkeys(e.get("to") for e in outs if e.get("to")))
            children=[node_map.get(cid) for cid in child_ids]
            # High-fan-in concatenations immediately before a prediction head are useful
            # paper-level semantics: they explain where multi-source features are fused.
            # Keep them in the publication view and give them a meaningful presentation
            # label rather than collapsing all incoming arrows directly into the head.
            if len(ins) >= 3 and len(children) == 1 and children[0] and children[0].get("role") == "head" and label in {"concat", "concatenate", "stack"}:
                child_label=str(children[0].get("label", "Prediction")).replace(" Head", "").replace(" head", "")
                n["label"] = f"{child_label} feature fusion"
                n["subtitle"] = "Concatenate"
                n["role"] = "fusion"
                n["kind"] = "merge"
                continue
            if not children or any(c is None for c in children):
                continue
            safe = all(
                c.get("role") in {"backbone","novel","fusion","head"}
                or any(k in str(c.get("label","")).lower() for k in ("encoder","attention","gate","head"))
                for c in children
            )
            if not safe:
                continue
            new_edges=[]
            for child_id, child in zip(child_ids, children):
                child_outs=[oe for oe in outs if oe.get("to")==child_id]
                inherited_out_label = child_outs[0].get("label") if len(child_outs)==1 else None
                for e in ins:
                    ne={"from":e.get("from"),"to":child_id,"type":e.get("type","main")}
                    if label == "+" and ne["type"] == "residual":
                        ne["type"] = "conditioning"
                    # Preserve semantically important port labels from either side of
                    # the collapsed implementation node. This is how Q survives the
                    # learned-query residual add before MultiheadAttention.
                    if e.get("label"):
                        ne["label"]=e["label"]
                    elif inherited_out_label:
                        ne["label"]=inherited_out_label
                    new_edges.append(ne)
                child.setdefault("source_nodes", [child_id])
                child["source_nodes"].extend(x for x in n.get("source_nodes",[nid]) if x not in child["source_nodes"])
            edges=[e for e in edges if e.get("from")!=nid and e.get("to")!=nid]
            edges.extend(new_edges)
            collapsed.append({"node":nid,"into":child_ids,"label":n.get("label","")})
            nodes.remove(n); node_map.pop(nid,None)
            changed=True
            break

    # Compact long raw-input reuse paths in the *overview* only. If a raw input
    # feeds an encoder and the resulting representation also reaches the same later
    # semantic target, keep the encoded path as the visible route and record the raw
    # reuse as a detail on the target. This is especially valuable for multi-input
    # gates/fusion heads where exact concatenation would otherwise create spaghetti.
    incoming=defaultdict(list); outgoing=defaultdict(list)
    for e in edges:
        incoming[e.get("to")].append(e); outgoing[e.get("from")].append(e)
    input_ids={n["id"] for n in nodes if n.get("role")=="input"}
    encoder_ids={n["id"] for n in nodes if "encoder" in str(n.get("label","")).lower()}

    # Adjacency excluding the direct raw edge under consideration.
    def _has_path(src: str, dst: str, skip_edge: tuple[str,str]) -> bool:
        q=[src]; seen={src}
        while q:
            cur=q.pop(0)
            for oe in outgoing.get(cur,[]):
                a,b=oe.get("from"),oe.get("to")
                if (a,b)==skip_edge or not b:
                    continue
                if b==dst:
                    return True
                if b not in seen:
                    seen.add(b); q.append(b)
        return False

    compacted_raw=[]
    semantic_targets={n["id"]:n for n in nodes if n.get("role") in {"head","fusion","novel"} or n.get("kind")=="merge"}
    for tid,target in list(semantic_targets.items()):
        raw_edges=[e for e in incoming.get(tid,[]) if e.get("from") in input_ids]
        for re in raw_edges:
            iid=re.get("from")
            encs={e.get("to") for e in outgoing.get(iid,[]) if e.get("to") in encoder_ids}
            redundant=any(_has_path(enc,tid,(iid,tid)) for enc in encs)
            if not redundant:
                continue
            edges=[e for e in edges if e is not re]
            label=node_map.get(iid,{}).get("label",iid)
            details=target.setdefault("details",[])
            phrase=f"also uses raw {label}"
            if phrase not in details:
                details.append(phrase)
            compacted_raw.append({"from":iid,"to":tid,"recorded_on":tid})

    # Rebuild adjacency for downstream diagnostics after compaction.
    incoming=defaultdict(list); outgoing=defaultdict(list)
    for e in edges:
        incoming[e.get("to")].append(e); outgoing[e.get("from")].append(e)

    if not show_diagnostics:
        incoming=defaultdict(list)
        for e in edges: incoming[e.get("to")].append(e)
        for n in list(nodes):
            if n.get("role")!="output" and n.get("kind")!="output":
                continue
            nid=n["id"]
            text=str(n.get("label","")).lower()
            hidden_by_review = (
                not incoming.get(nid)
                and any(e.get("to") == nid for e in overview_hidden_edges)
            )
            if hidden_by_review or any(term in text for term in _DIAGNOSTIC_OUTPUT_TERMS):
                hidden.append(nid)
                edges=[e for e in edges if e.get("from")!=nid and e.get("to")!=nid]
                nodes.remove(n); node_map.pop(nid,None)

    out["nodes"]=nodes
    out["edges"]=_dedupe_edges(edges)
    meta=out.setdefault("metadata",{})
    meta["publication_view_version"]="1.2"
    meta["publication_view"]={
        "collapsed_nodes": collapsed,
        "compacted_raw_edges": compacted_raw,
        "overview_hidden_edges": [{"from": e.get("from"), "to": e.get("to"), "note": e.get("overview_note", "")} for e in overview_hidden_edges],
        "hidden_nodes": hidden,
        "show_diagnostics": bool(show_diagnostics),
    }
    return out
