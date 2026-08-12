from __future__ import annotations

from collections import defaultdict, OrderedDict
from copy import deepcopy
import hashlib
import json
import math
import networkx as nx

from .routing import route_panel_edges
from .joint_layout import optimize_publication_layout


_LAYOUT_CACHE: OrderedDict[str, dict] = OrderedDict()
_LAYOUT_CACHE_MAX = 8

def _layout_cache_key(spec: dict) -> str:
    payload=json.dumps(spec,sort_keys=True,separators=(",",":"),ensure_ascii=False,default=str).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()

def _cache_layout(key: str, value: dict) -> dict:
    _LAYOUT_CACHE[key]=deepcopy(value); _LAYOUT_CACHE.move_to_end(key)
    while len(_LAYOUT_CACHE)>_LAYOUT_CACHE_MAX:
        _LAYOUT_CACHE.popitem(last=False)
    return value


# Canvas-space units. Renderers scale these to the target format.
BASE_NODE_H = 78.0
X_GAP = 78.0
Y_GAP = 48.0
PANEL_PAD_X = 52.0
PANEL_PAD_TOP = 78.0
PANEL_PAD_BOTTOM = 48.0
GROUP_PAD = 16.0


def node_size(node: dict) -> tuple[float, float]:
    """Size nodes using the Scientific Visual Grammar primitive."""
    kind = node.get("kind", "module")
    label = str(node.get("label", ""))
    subtitle = str(node.get("subtitle") or node.get("shape") or "")
    visual = node.get("visual") or {}
    vtype = visual.get("type") if isinstance(visual, dict) else str(visual)
    illustration = node.get("illustration") or None

    visual_sizes = {
        "feature_map_stack": (186.0, 112.0),
        "token_strip": (190.0, 100.0),
        "token_matrix": (185.0, 108.0),
        "transformer_stack": (214.0, 118.0),
        "attention_heads": (210.0, 112.0),
        "ffn_block": (196.0, 102.0),
        "norm_bar": (144.0, 60.0),
        "graph_input": (190.0, 124.0),
        "graph_message": (208.0, 126.0),
        "graph_pool": (150.0, 82.0),
        "router_gate": (164.0, 102.0),
        "expert_fan": (230.0, 148.0),
        "weighted_merge": (120.0, 78.0),
        "sequence_strip": (190.0, 100.0),
        "recurrent_cells": (216.0, 112.0),
        "unet_stage": (184.0, 112.0),
        "bottleneck": (184.0, 104.0),
        "diffusion_noise": (176.0, 108.0),
        "diffusion_denoiser": (210.0, 118.0),
        "time_condition": (154.0, 82.0),
        "spectral_operator": (208.0, 112.0),
        "operator_branch": (190.0, 104.0),
        "operator_trunk": (190.0, 104.0),
        "fusion_hub": (132.0, 88.0),
        "output_card": (162.0, 82.0),
        "operator_glyph": (146.0, 72.0),
        "merge_glyph": (74.0, 66.0),
        "modality_card": (178.0, 100.0),
        "loss_card": (166.0, 82.0),
        "embedding_card": (194.0, 96.0),
        "pooling_glyph": (150.0, 74.0),
        "classifier_head": (174.0, 88.0),
        # Publication primitives are deliberately denser than slide-style cards.
        "input_tensor": (164.0, 132.0),
        "input_card_publication": (232.0, 104.0),
        "feature_tensor": (162.0, 124.0),
        "field_tensor": (166.0, 118.0),
        "field_tensor_output": (166.0, 118.0),
        "feature_vector": (154.0, 108.0),
        "encoder_module": (188.0, 108.0),
        "emphasis_module": (198.0, 108.0),
        "transformer_macro": (194.0, 142.0),
        "attention_block": (208.0, 124.0),
        "ffn_block_publication": (190.0, 82.0),
        "norm_bar_publication": (156.0, 58.0),
        "add_node": (46.0, 46.0),
        "sequence_port": (142.0, 52.0),
        "token_strip_publication": (172.0, 110.0),
        "fusion_node_publication": (76.0, 76.0),
        "fusion_bar_publication": (166.0, 72.0),
        "spectral_operator_publication": (218.0, 126.0),
        "linear_map_publication": (154.0, 74.0),
        "pooling_bar_publication": (118.0, 48.0),
    }
    if vtype in visual_sizes:
        base_w, base_h = visual_sizes[vtype]
        text_need = min(280.0, max(base_w, 7.4 * len(label) + 34.0))
        extra = 10.0 if subtitle and len(subtitle) > 28 else 0.0
        if node.get("subtitle") and node.get("shape"):
            extra += 14.0
        if illustration:
            comp = str(illustration.get("composition", "illustration-top"))
            if comp in {"illustration-top", "illustration-bottom", "illustration-center"}:
                base_h = max(base_h, 154.0 if vtype == "attention_block" else 142.0)
                text_need = max(text_need, 190.0)
            else:
                base_h = max(base_h, 112.0)
                text_need = max(text_need, 224.0 if vtype == "input_card_publication" else 210.0)
        return text_need, base_h + extra

    if kind == "merge":
        w = 66.0 if len(label) <= 2 else 92.0
        return w, 62.0
    if kind == "operator":
        base = 142.0
    elif kind == "data":
        base = 174.0
    elif kind == "output":
        base = 158.0
    elif kind == "loss":
        base = 166.0
    else:
        base = 196.0
    text_need = min(270.0, max(base, 8.1 * len(label) + 44.0, 6.0 * len(subtitle) + 34.0 if subtitle else base))
    h = BASE_NODE_H + (12.0 if subtitle and len(subtitle) > 26 else 0.0)
    if illustration:
        comp = str(illustration.get("composition", "illustration-top"))
        if comp in {"illustration-top", "illustration-bottom", "illustration-center"}:
            h = max(h, 142.0); text_need = max(text_need, 190.0)
        else:
            h = max(h, 112.0); text_need = max(text_need, 210.0)
    return text_need, h


def panel_definitions(spec: dict) -> list[dict]:
    nodes = spec.get("nodes", [])
    panels = spec.get("panels") or []
    if not panels:
        return [{"id": "overall", "label": "", "title": "", "node_ids": [n["id"] for n in nodes]}]

    assigned = set()
    out = []
    for p in panels:
        ids = list(p.get("node_ids") or [])
        if not ids:
            ids = [n["id"] for n in nodes if n.get("panel") == p["id"]]
        assigned.update(ids)
        out.append({**p, "node_ids": ids})

    unassigned = [n["id"] for n in nodes if n["id"] not in assigned and not n.get("panel")]
    if unassigned:
        out[0]["node_ids"].extend(unassigned)
    return out


def _graph(node_ids: list[str], edges: list[dict]) -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_nodes_from(node_ids)
    for e in edges:
        if e.get("from") in g and e.get("to") in g and e.get("type", "main") != "residual":
            g.add_edge(e["from"], e["to"])
    return g


def _levels(node_ids: list[str], edges: list[dict]) -> dict[str, int]:
    g = _graph(node_ids, edges)
    if not nx.is_directed_acyclic_graph(g):
        # Recurrent/dynamic loops are valid semantically. Keep a stable fallback.
        return {node: i for i, node in enumerate(node_ids)}
    level = {n: 0 for n in g.nodes}
    for n in nx.topological_sort(g):
        preds = list(g.predecessors(n))
        if preds:
            level[n] = max(level[p] + 1 for p in preds)
    return level


def _crossing_reduced_order(node_ids: list[str], levels: dict[str, int], edges: list[dict]) -> dict[str, int]:
    """Small Sugiyama-style barycentric sweeps for publication diagrams."""
    buckets: dict[int, list[str]] = defaultdict(list)
    original_index = {n: i for i, n in enumerate(node_ids)}
    for n in node_ids:
        buckets[levels[n]].append(n)

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for e in edges:
        if e.get("type") == "residual":
            continue
        a, b = e.get("from"), e.get("to")
        if a in levels and b in levels:
            incoming[b].append(a)
            outgoing[a].append(b)

    # Four forward/backward sweeps are sufficient for small paper figures.
    for _ in range(4):
        pos = {n: i for lev in sorted(buckets) for i, n in enumerate(buckets[lev])}
        for lev in sorted(buckets):
            def fscore(n: str):
                ps = [pos[p] for p in incoming[n] if p in pos]
                return (sum(ps) / len(ps), original_index[n]) if ps else (1e6, original_index[n])
            buckets[lev].sort(key=fscore)
        pos = {n: i for lev in sorted(buckets) for i, n in enumerate(buckets[lev])}
        for lev in sorted(buckets, reverse=True):
            def bscore(n: str):
                ss = [pos[s] for s in outgoing[n] if s in pos]
                return (sum(ss) / len(ss), original_index[n]) if ss else (1e6, original_index[n])
            buckets[lev].sort(key=bscore)

    return {n: i for lev in buckets for i, n in enumerate(buckets[lev])}


def _assign_residual_lanes(edges: list[dict], levels: dict[str, int]) -> list[dict]:
    """Greedily separate overlapping residual bypasses into lanes."""
    out = [dict(e) for e in edges]
    residuals = []
    for idx, e in enumerate(out):
        if e.get("type") != "residual":
            continue
        a, b = e.get("from"), e.get("to")
        if a not in levels or b not in levels:
            continue
        lo, hi = sorted((levels[a], levels[b]))
        residuals.append((idx, lo, hi))
    lanes: list[list[tuple[int, int]]] = []
    for idx, lo, hi in sorted(residuals, key=lambda x: (x[1], x[2] - x[1])):
        lane = 0
        while lane < len(lanes) and any(not (hi < a or lo > b) for a, b in lanes[lane]):
            lane += 1
        if lane == len(lanes):
            lanes.append([])
        lanes[lane].append((lo, hi))
        out[idx]["_route_lane"] = lane
    return out


def _group_boxes(spec: dict, node_ids: list[str], positions: dict[str, dict]) -> list[dict]:
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    grouped: dict[str, list[str]] = defaultdict(list)
    for nid in node_ids:
        group = node_map.get(nid, {}).get("group")
        if group:
            grouped[group].append(nid)
    boxes = []
    group_meta = {str(g.get("id")): g for g in (spec.get("groups") or []) if isinstance(g, dict) and g.get("id")}
    for gid, ids in grouped.items():
        xs = [positions[n]["x"] for n in ids]
        ys = [positions[n]["y"] for n in ids]
        rights = [positions[n]["x"] + positions[n]["w"] for n in ids]
        bottoms = [positions[n]["y"] + positions[n]["h"] for n in ids]
        meta = group_meta.get(gid, {})
        box = {
            "id": gid,
            "label": meta.get("label", gid.replace("_", " ").title()),
            "x": min(xs) - GROUP_PAD,
            "y": min(ys) - GROUP_PAD - 17,
            "w": max(rights) - min(xs) + 2 * GROUP_PAD,
            "h": max(bottoms) - min(ys) + 2 * GROUP_PAD + 17,
        }
        for k, v in meta.items():
            if k not in {"id", "label"}:
                box[k] = v
        boxes.append(box)
    return boxes


def _layout_conditioned_panel(spec: dict, panel: dict) -> dict | None:
    """Main inference lane with conditioning branches above/below it."""
    node_ids = panel["node_ids"]
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    edges0 = [e for e in spec.get("edges", []) if e.get("from") in node_ids and e.get("to") in node_ids]
    main_inputs = [nid for nid in node_ids if node_map[nid].get("role") == "input"]
    targets = [nid for nid in node_ids if node_map[nid].get("role") in {"novel", "backbone"}]
    outputs = [nid for nid in node_ids if node_map[nid].get("role") == "output" or node_map[nid].get("kind") == "output"]
    if not main_inputs or not targets or not outputs:
        return None
    target = targets[-1]
    inp = main_inputs[0]
    outp = outputs[-1]
    sizes = {nid: node_size(node_map[nid]) for nid in node_ids}
    positions = {}
    main_y = PANEL_PAD_TOP + 210.0
    iw,ih=sizes[inp]; tw,th=sizes[target]; ow,oh=sizes[outp]
    positions[inp]={"x":PANEL_PAD_X,"y":main_y+(th-ih)/2,"w":iw,"h":ih}
    positions[target]={"x":PANEL_PAD_X+iw+430.0,"y":main_y,"w":tw,"h":th}
    positions[outp]={"x":positions[target]["x"]+tw+150.0,"y":main_y+(th-oh)/2,"w":ow,"h":oh}

    # Discover conditioning chains that end at the target using non-main edge types.
    aux_roots=[nid for nid in node_ids if node_map[nid].get("role")=="auxiliary" and nid not in positions]
    incoming={nid:[] for nid in node_ids}; outgoing={nid:[] for nid in node_ids}
    for e in edges0:
        outgoing.setdefault(e["from"],[]).append(e); incoming.setdefault(e["to"],[]).append(e)
    lanes=[PANEL_PAD_TOP+12.0, main_y+th+125.0]
    used=set(positions)
    for lane_idx,root in enumerate(aux_roots[:2]):
        y=lanes[lane_idx]
        rw,rh=sizes[root]; positions[root]={"x":PANEL_PAD_X,"y":y,"w":rw,"h":rh}; used.add(root)
        cur=root; x=PANEL_PAD_X+rw+120.0; guard=0
        while guard<8:
            guard+=1
            candidates=[e for e in outgoing.get(cur,[]) if e.get("to")!=target and e.get("to") not in used]
            if not candidates: break
            nxt=candidates[0]["to"]
            nw,nh=sizes[nxt]; positions[nxt]={"x":x,"y":y+(rh-nh)/2,"w":nw,"h":nh}; used.add(nxt)
            x+=nw+105.0; cur=nxt
    # Place any remaining nodes conservatively, excluding target/output already set.
    extras=[nid for nid in node_ids if nid not in positions]
    exx=PANEL_PAD_X+210.0
    for nid in extras:
        w,h=sizes[nid]; positions[nid]={"x":exx,"y":main_y+th+280.0,"w":w,"h":h}; exx+=w+70.0
    levels={nid:i for i,nid in enumerate(node_ids)}
    edges=_assign_residual_lanes(edges0,levels)
    width=max(v["x"]+v["w"] for v in positions.values())+PANEL_PAD_X
    height=max(v["y"]+v["h"] for v in positions.values())+PANEL_PAD_BOTTOM
    return {"positions":positions,"width":width,"height":height,"edges":edges,"levels":levels,"groups":_group_boxes(spec,node_ids,positions)}


def _layout_moe_panel(spec: dict, panel: dict) -> dict | None:
    """Sparse MoE layout: router above the token-to-expert main lane."""
    node_ids = panel["node_ids"]
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    def find(pred):
        return [nid for nid in node_ids if pred(str(node_map[nid].get("label", "")).lower(), node_map[nid])]
    routers = find(lambda t,n: "router" in t or "gate" in t)
    experts = find(lambda t,n: "expert" in t)
    merges = find(lambda t,n: "merge" in t or (n.get("role") == "fusion" and "expert" not in t))
    inputs = [nid for nid in node_ids if node_map[nid].get("role") == "input"]
    outputs = [nid for nid in node_ids if node_map[nid].get("role") == "output" or node_map[nid].get("kind") == "output"]
    if not routers or not experts or not inputs or not outputs:
        return None
    router, expert, inp, outp = routers[0], experts[0], inputs[0], outputs[-1]
    merge = merges[0] if merges else None
    sizes = {nid: node_size(node_map[nid]) for nid in node_ids}
    positions = {}
    main_y = PANEL_PAD_TOP + 155.0
    iw, ih = sizes[inp]
    positions[inp] = {"x": PANEL_PAD_X, "y": main_y, "w": iw, "h": ih}
    # Leave a generous routing corridor between token input and experts.
    ew, eh = sizes[expert]
    expert_x = positions[inp]["x"] + iw + 300.0
    positions[expert] = {"x": expert_x, "y": main_y, "w": ew, "h": eh}
    cursor = expert_x + ew + 150.0
    if merge:
        mw, mh = sizes[merge]
        positions[merge] = {"x": cursor, "y": main_y + (eh-mh)/2, "w": mw, "h": mh}
        cursor += mw + 112.0
    ow, oh = sizes[outp]
    positions[outp] = {"x": cursor, "y": main_y + (eh-oh)/2, "w": ow, "h": oh}
    rw,rh=sizes[router]
    positions[router] = {"x": positions[inp]["x"] + iw + 70.0, "y": PANEL_PAD_TOP + 4.0, "w": rw, "h": rh}
    x = cursor + ow + 112.0
    # Place any extra nodes using a conservative row below the main lane.
    extras=[nid for nid in node_ids if nid not in positions]
    exx=PANEL_PAD_X
    for nid in extras:
        w,h=sizes[nid]; positions[nid]={"x":exx,"y":main_y+190.0,"w":w,"h":h}; exx+=w+70.0
    edges0=[e for e in spec.get("edges",[]) if e.get("from") in node_ids and e.get("to") in node_ids]
    # Synthetic levels only drive residual-lane assignment; topology is untouched.
    levels={inp:0,router:1,expert:2,outp:4}
    if merge: levels[merge]=3
    for i,nid in enumerate(extras,5): levels[nid]=i
    edges=_assign_residual_lanes(edges0,levels)
    width=max(v["x"]+v["w"] for v in positions.values())+PANEL_PAD_X
    height=max(v["y"]+v["h"] for v in positions.values())+PANEL_PAD_BOTTOM
    return {"positions":positions,"width":width,"height":height,"edges":edges,"levels":levels,"groups":_group_boxes(spec,node_ids,positions)}


def _layout_unet_panel(spec: dict, panel: dict) -> dict | None:
    """Dedicated U-shaped layout for encoder-decoder diagrams.

    It is intentionally conservative: if a clear encoder/bottleneck/decoder sequence
    is not present, return None and fall back to the generic DAG layout.
    """
    node_ids = panel["node_ids"]
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    enc = [nid for nid in node_ids if "encoder" in str(node_map[nid].get("label", "")).lower()]
    dec = [nid for nid in node_ids if "decoder" in str(node_map[nid].get("label", "")).lower()]
    bott = [nid for nid in node_ids if "bottleneck" in str(node_map[nid].get("label", "")).lower()]
    if not enc or not dec or not bott:
        return None

    edges0 = [e for e in spec.get("edges", []) if e.get("from") in node_ids and e.get("to") in node_ids]
    g = _graph(node_ids, edges0)
    if nx.is_directed_acyclic_graph(g):
        topo = list(nx.topological_sort(g))
    else:
        topo = list(node_ids)
    order = {nid: i for i, nid in enumerate(topo)}
    enc.sort(key=lambda n: order.get(n, 10**6))
    dec.sort(key=lambda n: order.get(n, 10**6))
    bottleneck = min(bott, key=lambda n: order.get(n, 10**6))

    sequence = [n for n in topo if n not in set(enc + dec + [bottleneck])]
    before = [n for n in sequence if order.get(n, 0) < order.get(enc[0], 0)]
    after = [n for n in sequence if order.get(n, 0) > order.get(dec[-1], 0)]
    chain = before + enc + [bottleneck] + dec + after
    if len(chain) != len(set(chain)):
        return None

    sizes = {nid: node_size(node_map[nid]) for nid in node_ids}
    x = PANEL_PAD_X
    positions: dict[str, dict] = {}
    step_gap = 70.0
    max_depth = max(len(enc), len(dec))
    base_y = PANEL_PAD_TOP + 34.0
    depth_step = 84.0

    enc_depth = {nid: i for i, nid in enumerate(enc)}
    dec_depth = {nid: max(0, len(dec)-1-i) for i, nid in enumerate(dec)}
    for nid in chain:
        w, h = sizes[nid]
        if nid in enc_depth:
            depth = enc_depth[nid]
        elif nid == bottleneck:
            depth = max_depth
        elif nid in dec_depth:
            depth = dec_depth[nid]
        elif nid in before or nid in after:
            depth = 0
        else:
            depth = 0
        y = base_y + depth * depth_step
        positions[nid] = {"x": x, "y": y, "w": w, "h": h}
        x += w + step_gap

    levels = {nid: i for i, nid in enumerate(chain)}
    edges = _assign_residual_lanes(edges0, levels)
    width = x - step_gap + PANEL_PAD_X
    height = base_y + max_depth * depth_step + max((sizes[n][1] for n in node_ids), default=BASE_NODE_H) + PANEL_PAD_BOTTOM
    return {
        "positions": positions, "width": width, "height": height, "edges": edges,
        "levels": levels, "groups": _group_boxes(spec, node_ids, positions),
    }




def _stack_required(heights: list[float], gap: float) -> float:
    if not heights:
        return 0.0
    return sum(heights) + gap * max(0, len(heights) - 1)


def _stack_centers(heights: list[float], top: float, height: float, gap: float) -> list[float]:
    """Return non-overlapping vertical centers, centered inside the available lane."""
    if not heights:
        return []
    required = _stack_required(heights, gap)
    y = top + max(0.0, (height - required) / 2.0)
    centers = []
    for h in heights:
        centers.append(y + h / 2.0)
        y += h + gap
    return centers


def _pack_preferred_centers(
    preferred: list[float], heights: list[float], top: float, height: float, gap: float
) -> list[float]:
    """Pack boxes near preferred centers while guaranteeing vertical separation.

    This is used for encoder lanes: predecessor alignment is desirable, but overlap is
    never allowed.  The packing pass is deterministic and remains inside the stage.
    """
    if not preferred:
        return []
    order = sorted(range(len(preferred)), key=lambda i: preferred[i])
    centers = [0.0] * len(preferred)
    cursor = top
    for idx in order:
        h = heights[idx]
        cy = max(float(preferred[idx]), cursor + h / 2.0)
        centers[idx] = cy
        cursor = cy + h / 2.0 + gap
    overflow = cursor - gap - (top + height)
    if overflow > 0:
        for idx in order:
            centers[idx] -= overflow
    # A reverse pass prevents the upward shift from creating top-bound violations.
    cursor = top + height
    for idx in reversed(order):
        h = heights[idx]
        cy = min(centers[idx], cursor - h / 2.0)
        centers[idx] = cy
        cursor = cy - h / 2.0 - gap
    under = top - (cursor + gap)
    if under > 0:
        for idx in order:
            centers[idx] += under
    return centers


def _framework_stage_widths(stages: dict, sizes: dict, levels: dict[str, int]) -> tuple[dict[str, float], str, dict[int, list[str]]]:
    """Derive stage widths from actual node geometry instead of fixed columns."""
    side_pad = 28.0
    inner_gap = 38.0
    widths: dict[str, float] = {}
    widths['stage_inputs'] = max([sizes[n][0] for n in stages['stage_inputs']] or [190.0]) + 2 * side_pad
    widths['stage_encoders'] = max([sizes[n][0] for n in stages['stage_encoders']] or [180.0]) + 2 * side_pad

    core = stages['stage_interaction']
    buckets: dict[int, list[str]] = defaultdict(list)
    for nid in core:
        buckets[levels.get(nid, 0)].append(nid)
    levs = sorted(buckets)
    horizontal_need = 2 * side_pad
    if levs:
        for l in levs:
            horizontal_need += max(sizes[n][0] for n in buckets[l])
        horizontal_need += inner_gap * max(0, len(levs) - 1)
    row_need = max(
        [sum(sizes[n][0] for n in buckets[l]) + inner_gap * max(0, len(buckets[l]) - 1) for l in levs] or [210.0]
    ) + 2 * side_pad
    # Three or more sequential levels are far more legible as a top-to-bottom method
    # narrative inside the central stage (the common Figure-1 pattern).
    core_mode = 'vertical' if len(levs) >= 3 or horizontal_need > 620.0 else 'horizontal'
    widths['stage_interaction'] = max(286.0, row_need if core_mode == 'vertical' else horizontal_need)

    heads = [n for n in stages['stage_outputs'] if n]
    # Output stage reserves two independent columns when heads and terminal outputs coexist.
    # The exact split is finalized in the placement pass.
    widths['stage_outputs'] = max([sizes[n][0] for n in heads] or [190.0]) + 2 * side_pad
    return widths, core_mode, buckets


def _layout_publication_framework(spec: dict, panel: dict) -> dict | None:
    """Adaptive Figure-1 composition for complex multi-input systems.

    Stage geometry avoids fixed stage coordinates and is derived
    from actual node sizes, labels, illustrations, graph depth and branch count before
    any node is placed.  The layout may therefore grow or change the central-stage
    orientation for a complex model rather than allowing overlap.
    """
    ids = panel['node_ids']
    node_map = {n['id']: n for n in spec.get('nodes', [])}
    edges0 = [e for e in spec.get('edges', []) if e.get('from') in ids and e.get('to') in ids]
    stages = {
        'stage_inputs': [i for i in ids if node_map[i].get('group') == 'stage_inputs'],
        'stage_encoders': [i for i in ids if node_map[i].get('group') == 'stage_encoders'],
        'stage_interaction': [i for i in ids if node_map[i].get('group') == 'stage_interaction'],
        'stage_outputs': [i for i in ids if node_map[i].get('group') == 'stage_outputs'],
    }
    if len(stages['stage_inputs']) < 2 or not stages['stage_outputs']:
        return None

    sizes = {nid: node_size(node_map[nid]) for nid in ids}
    positions: dict[str, dict] = {}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for e in edges0:
        incoming[e['to']].append(e['from'])
        outgoing[e['from']].append(e['to'])

    core = stages['stage_interaction']
    core_edges = [e for e in edges0 if e['from'] in core and e['to'] in core]
    core_levels = _levels(core, core_edges) if core else {}
    sw, core_mode, core_buckets = _framework_stage_widths(stages, sizes, core_levels)

    heads = [n for n in stages['stage_outputs'] if node_map[n].get('role') == 'head']
    terms = [n for n in stages['stage_outputs'] if n not in heads]
    stage_side_pad = 28.0
    output_col_gap = 48.0
    if heads and terms:
        head_w = max(sizes[n][0] for n in heads)
        term_w = max(sizes[n][0] for n in terms)
        sw['stage_outputs'] = head_w + term_w + output_col_gap + 2 * stage_side_pad

    # Height is content-driven.  The largest stage determines the shared vertical lane.
    stack_gap = 38.0
    input_req = _stack_required([sizes[n][1] for n in stages['stage_inputs']], stack_gap)
    encoder_req = _stack_required([sizes[n][1] for n in stages['stage_encoders']], stack_gap)
    output_req = max(
        _stack_required([sizes[n][1] for n in heads], stack_gap),
        _stack_required([sizes[n][1] for n in terms], stack_gap),
    )
    if core_mode == 'vertical':
        core_rows = []
        for lev in sorted(core_buckets):
            core_rows.append(max(sizes[n][1] for n in core_buckets[lev]))
        core_req = _stack_required(core_rows, 34.0)
    else:
        core_req = max(
            [_stack_required([sizes[n][1] for n in core_buckets[lev]], 28.0) for lev in core_buckets] or [0.0]
        )
    H = max(540.0, input_req, encoder_req, core_req, output_req) + 24.0
    top = PANEL_PAD_TOP + 58.0

    # Stage x-coordinates are sequential and cannot overlap by construction.
    stage_gap = max(46.0, float(spec.get('figure', {}).get('stage_gap', 52.0)))
    sx: dict[str, float] = {}
    xcur = PANEL_PAD_X
    for stage in ('stage_inputs', 'stage_encoders', 'stage_interaction', 'stage_outputs'):
        sx[stage] = xcur
        xcur += sw[stage] + stage_gap

    # Inputs: stable source lanes, automatically expanded for illustration-rich cards.
    roots = stages['stage_inputs']
    root_centers = _stack_centers([sizes[n][1] for n in roots], top, H, stack_gap)
    for nid, cy in zip(roots, root_centers):
        w, h = sizes[nid]
        positions[nid] = {'x': sx['stage_inputs'] + (sw['stage_inputs'] - w) / 2, 'y': cy - h / 2, 'w': w, 'h': h}

    # Encoders stay close to their input lane but are packed geometrically, never nudged
    # by an unbounded while-loop that can push them outside the stage.
    encs = stages['stage_encoders']
    preferred = []
    for j, nid in enumerate(encs):
        preds = [p for p in incoming.get(nid, []) if p in positions]
        if preds:
            preferred.append(sum(positions[p]['y'] + positions[p]['h'] / 2 for p in preds) / len(preds))
        else:
            preferred.append(top + H * (j + 1) / (len(encs) + 1))
    enc_centers = _pack_preferred_centers(preferred, [sizes[n][1] for n in encs], top, H, stack_gap)
    for nid, cy in zip(encs, enc_centers):
        w, h = sizes[nid]
        positions[nid] = {'x': sx['stage_encoders'] + (sw['stage_encoders'] - w) / 2, 'y': cy - h / 2, 'w': w, 'h': h}

    # Central scientific-method stage: choose its local orientation from graph depth.
    if core:
        levs = sorted(core_buckets)
        if core_mode == 'vertical':
            row_heights = [max(sizes[n][1] for n in core_buckets[l]) for l in levs]
            row_centers = _stack_centers(row_heights, top, H, 34.0)
            for lev, cy in zip(levs, row_centers):
                arr = list(core_buckets[lev])
                # Order a row by predecessor y so incoming edges cross less often.
                arr.sort(key=lambda n: sum(
                    positions[p]['y'] + positions[p]['h'] / 2 for p in incoming.get(n, []) if p in positions
                ) / max(1, len([p for p in incoming.get(n, []) if p in positions])))
                row_w = sum(sizes[n][0] for n in arr) + 38.0 * max(0, len(arr) - 1)
                xx = sx['stage_interaction'] + (sw['stage_interaction'] - row_w) / 2
                for nid in arr:
                    w, h = sizes[nid]
                    positions[nid] = {'x': xx, 'y': cy - h / 2, 'w': w, 'h': h}
                    xx += w + 38.0
        else:
            col_widths = [max(sizes[n][0] for n in core_buckets[l]) for l in levs]
            xx = sx['stage_interaction'] + stage_side_pad
            for lev, cw in zip(levs, col_widths):
                arr = list(core_buckets[lev])
                arr.sort(key=lambda n: sum(
                    positions[p]['y'] + positions[p]['h'] / 2 for p in incoming.get(n, []) if p in positions
                ) / max(1, len([p for p in incoming.get(n, []) if p in positions])))
                centers = _stack_centers([sizes[n][1] for n in arr], top, H, 28.0)
                for nid, cy in zip(arr, centers):
                    w, h = sizes[nid]
                    positions[nid] = {'x': xx + (cw - w) / 2, 'y': cy - h / 2, 'w': w, 'h': h}
                xx += cw + 38.0

    # Output stage: independent head/output columns sized from their real content.
    if heads and terms:
        head_w = max(sizes[n][0] for n in heads)
        term_w = max(sizes[n][0] for n in terms)
        hx = sx['stage_outputs'] + stage_side_pad
        tx = sx['stage_outputs'] + sw['stage_outputs'] - stage_side_pad - term_w
        term_centers = _stack_centers([sizes[n][1] for n in terms], top, H, stack_gap)
        for nid, cy in zip(terms, term_centers):
            w, h = sizes[nid]
            positions[nid] = {'x': tx + (term_w - w) / 2, 'y': cy - h / 2, 'w': w, 'h': h}
        for j, nid in enumerate(heads):
            succ = [q for q in outgoing.get(nid, []) if q in positions]
            if succ:
                preferred_cy = sum(positions[q]['y'] + positions[q]['h'] / 2 for q in succ) / len(succ)
            else:
                preferred_cy = top + H * (j + 1) / (len(heads) + 1)
            # Multiple heads still need a collision-safe pack.
            # Defer exact placement until all preferred centers are known.
        hprefs = []
        for j, nid in enumerate(heads):
            succ = [q for q in outgoing.get(nid, []) if q in positions]
            hprefs.append(
                sum(positions[q]['y'] + positions[q]['h'] / 2 for q in succ) / len(succ)
                if succ else top + H * (j + 1) / (len(heads) + 1)
            )
        hcenters = _pack_preferred_centers(hprefs, [sizes[n][1] for n in heads], top, H, stack_gap)
        for nid, cy in zip(heads, hcenters):
            w, h = sizes[nid]
            positions[nid] = {'x': hx + (head_w - w) / 2, 'y': cy - h / 2, 'w': w, 'h': h}
    else:
        outs = heads or terms
        centers = _stack_centers([sizes[n][1] for n in outs], top, H, stack_gap)
        for nid, cy in zip(outs, centers):
            w, h = sizes[nid]
            positions[nid] = {'x': sx['stage_outputs'] + (sw['stage_outputs'] - w) / 2, 'y': cy - h / 2, 'w': w, 'h': h}

    # Rare unclassified nodes are appended below the central stage and force the canvas
    # to grow; they are never placed on top of existing content.
    extras = [n for n in ids if n not in positions]
    extra_y = top + H + 30.0
    for nid in extras:
        w, h = sizes[nid]
        positions[nid] = {'x': sx['stage_interaction'] + (sw['stage_interaction'] - w) / 2, 'y': extra_y, 'w': w, 'h': h}
        extra_y += h + 24.0

    levels = _levels(ids, edges0)
    edges = _assign_residual_lanes(edges0, levels)
    layout_meta = {'framework_core_mode': core_mode, 'stage_widths': sw, 'stage_x': sx}
    groups = _group_boxes(spec, ids, positions)
    if spec.get('figure', {}).get('joint_optimization', True):
        positions, _routed_preview, _joint_groups, joint_meta = optimize_publication_layout(
            spec, positions, edges, layout_meta,
            max_passes=int(spec.get('figure', {}).get('joint_optimization_passes', 2) or 2),
        )
        # The normal layout_figure pass performs the authoritative route after all
        # panel geometry is known.  Here routing is used only as the optimization
        # objective; groups are recomputed from the accepted node geometry.
        groups = _group_boxes(spec, ids, positions)
        layout_meta['joint_optimization'] = joint_meta
    width = max(v['x'] + v['w'] for v in positions.values()) + PANEL_PAD_X
    height = max(top + H + PANEL_PAD_BOTTOM, max(v['y'] + v['h'] for v in positions.values()) + PANEL_PAD_BOTTOM)
    return {
        'positions': positions, 'width': width, 'height': height, 'edges': edges,
        'levels': levels, 'groups': groups,
        'layout_meta': layout_meta,
    }

def _layout_publication_transformer(spec: dict, panel: dict) -> dict | None:
    """Compose Transformer figures like paper schematics, not generic DAGs."""
    ids = panel["node_ids"]
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    edges0 = [e for e in spec.get("edges", []) if e.get("from") in ids and e.get("to") in ids]
    labels = " ".join(str(node_map[i].get("label", "")).lower() for i in ids)
    detail = ("attention" in labels or "mha" in labels) and ("feed-forward" in labels or "ffn" in labels)
    sizes = {nid: node_size(node_map[nid]) for nid in ids}
    positions: dict[str, dict] = {}
    if detail:
        # A narrow, vertical computational block with visible residual bypasses.
        g = _graph(ids, edges0)
        topo = list(nx.topological_sort(g)) if nx.is_directed_acyclic_graph(g) else list(ids)
        cx = 280.0
        y = PANEL_PAD_TOP + 26.0
        for nid in topo:
            w,h = sizes[nid]
            positions[nid] = {"x": cx-w/2, "y": y, "w": w, "h": h}
            y += h + 20.0
        levels = {nid:i for i,nid in enumerate(topo)}
        edges = _assign_residual_lanes(edges0, levels)
        # Give residual loops lateral room without wasting a full screen width.
        width = 560.0
        height = y + PANEL_PAD_BOTTOM - 18.0
        return {"positions":positions,"width":width,"height":height,"edges":edges,"levels":levels,"groups":_group_boxes(spec,ids,positions)}

    # Overall transformer: compact left-to-right ribbon, encoder stack is the visual anchor.
    g = _graph(ids, edges0)
    topo = list(nx.topological_sort(g)) if nx.is_directed_acyclic_graph(g) else list(ids)
    x = PANEL_PAD_X
    center_y = PANEL_PAD_TOP + 54.0
    for nid in topo:
        w,h=sizes[nid]
        positions[nid]={"x":x,"y":center_y-h/2,"w":w,"h":h}
        x += w + 54.0
    levels={nid:i for i,nid in enumerate(topo)}
    edges=_assign_residual_lanes(edges0,levels)
    return {"positions":positions,"width":x-54.0+PANEL_PAD_X,"height":PANEL_PAD_TOP+150.0+PANEL_PAD_BOTTOM,"edges":edges,"levels":levels,"groups":_group_boxes(spec,ids,positions)}


def _layout_publication_multilane(spec: dict, panel: dict) -> dict | None:
    ids=panel["node_ids"]
    node_map={n["id"]:n for n in spec.get("nodes",[])}
    edges0=[e for e in spec.get("edges",[]) if e.get("from") in ids and e.get("to") in ids]
    fusion_candidates=[i for i in ids if (node_map[i].get("visual") or {}).get("type")=="fusion_node_publication" or node_map[i].get("role")=="fusion"]
    roots=[i for i in ids if node_map[i].get("role")=="input"]
    if not fusion_candidates or len(roots)<2:
        return None
    fusion=fusion_candidates[0]
    sizes={nid:node_size(node_map[nid]) for nid in ids}
    incoming=defaultdict(list); outgoing=defaultdict(list)
    for e in edges0:
        outgoing[e["from"]].append(e["to"]); incoming[e["to"]].append(e["from"])
    positions={}
    lane_gap=128.0
    top=PANEL_PAD_TOP+28.0
    input_x=PANEL_PAD_X
    encoder_x=268.0
    fusion_x=515.0
    lane_centers=[top+i*lane_gap for i in range(len(roots))]
    for root,cy in zip(roots,lane_centers):
        w,h=sizes[root]; positions[root]={"x":input_x,"y":cy-h/2,"w":w,"h":h}
        # Walk the lane until fusion; usually one encoder but handles short chains.
        cur=root; x=encoder_x; guard=0
        while guard<5:
            guard+=1
            nxts=[n for n in outgoing.get(cur,[]) if n!=fusion and n not in positions]
            if not nxts: break
            nxt=nxts[0]; w,h=sizes[nxt]; positions[nxt]={"x":x,"y":cy-h/2,"w":w,"h":h}; x+=w+42; cur=nxt
    fw,fh=sizes[fusion]; mid=sum(lane_centers)/len(lane_centers)
    positions[fusion]={"x":fusion_x,"y":mid-fh/2,"w":fw,"h":fh}
    # Place descendants as one compact central ribbon.
    cur=fusion; x=fusion_x+fw+86.0; used=set(positions)
    guard=0
    while guard<10:
        guard+=1
        nxts=[n for n in outgoing.get(cur,[]) if n not in used]
        if not nxts: break
        nxt=nxts[0]; w,h=sizes[nxt]; positions[nxt]={"x":x,"y":mid-h/2,"w":w,"h":h}; x+=w+58; used.add(nxt); cur=nxt
    # Conservative fallback for extras.
    extras=[n for n in ids if n not in positions]
    for j,nid in enumerate(extras):
        w,h=sizes[nid]; positions[nid]={"x":encoder_x,"y":top+len(roots)*lane_gap+j*(h+24),"w":w,"h":h}
    levels=_levels(ids,edges0); edges=_assign_residual_lanes(edges0,levels)
    width=max(v["x"]+v["w"] for v in positions.values())+PANEL_PAD_X
    height=max(v["y"]+v["h"] for v in positions.values())+PANEL_PAD_BOTTOM
    return {"positions":positions,"width":width,"height":height,"edges":edges,"levels":levels,"groups":_group_boxes(spec,ids,positions)}


def _layout_publication_operator(spec: dict, panel: dict) -> dict | None:
    ids=panel["node_ids"]
    node_map={n["id"]:n for n in spec.get("nodes",[])}
    edges0=[e for e in spec.get("edges",[]) if e.get("from") in ids and e.get("to") in ids]
    g=_graph(ids,edges0)
    if not nx.is_directed_acyclic_graph(g): return None
    topo=list(nx.topological_sort(g)); sizes={nid:node_size(node_map[nid]) for nid in ids}
    positions={}; x=PANEL_PAD_X; cy=PANEL_PAD_TOP+66.0
    for nid in topo:
        w,h=sizes[nid]; positions[nid]={"x":x,"y":cy-h/2,"w":w,"h":h}; x+=w+(72.0 if (node_map[nid].get("visual") or {}).get("type")=="spectral_operator_publication" else 54.0)
    levels={n:i for i,n in enumerate(topo)}; edges=_assign_residual_lanes(edges0,levels)
    return {"positions":positions,"width":x-54+PANEL_PAD_X,"height":PANEL_PAD_TOP+160+PANEL_PAD_BOTTOM,"edges":edges,"levels":levels,"groups":_group_boxes(spec,ids,positions)}

def layout_panel(spec: dict, panel: dict, direction: str = "LR") -> dict:
    preset = str(spec.get("figure", {}).get("layout_preset", ""))
    if preset == "publication_framework" and direction == "LR":
        special = _layout_publication_framework(spec, panel)
        if special is not None:
            return special
    if preset == "publication_transformer" and direction == "LR":
        special = _layout_publication_transformer(spec, panel)
        if special is not None:
            return special
    if preset == "publication_multilane" and direction == "LR":
        special = _layout_publication_multilane(spec, panel)
        if special is not None:
            return special
    if preset == "publication_operator" and direction == "LR":
        special = _layout_publication_operator(spec, panel)
        if special is not None:
            return special
    if preset == "publication_unet" and direction == "LR":
        special = _layout_unet_panel(spec, panel)
        if special is not None:
            return special
    if preset == "unet" and direction == "LR":
        special = _layout_unet_panel(spec, panel)
        if special is not None:
            return special
    if preset == "branch_fan" and direction == "LR":
        special = _layout_moe_panel(spec, panel)
        if special is not None:
            return special
    if preset == "conditioned_flow" and direction == "LR":
        special = _layout_conditioned_panel(spec, panel)
        if special is not None:
            return special
    node_ids = panel["node_ids"]
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    edges0 = [e for e in spec.get("edges", []) if e.get("from") in node_ids and e.get("to") in node_ids]
    levels = _levels(node_ids, edges0)
    edges = _assign_residual_lanes(edges0, levels)
    order = _crossing_reduced_order(node_ids, levels, edges)

    per_level: dict[int, list[str]] = defaultdict(list)
    for n in node_ids:
        per_level[levels[n]].append(n)
    for lev in per_level:
        per_level[lev].sort(key=lambda n: order[n])

    level_width: dict[int, float] = {}
    level_height: dict[int, float] = {}
    sizes = {nid: node_size(node_map[nid]) for nid in node_ids}
    for lev, ids in per_level.items():
        if direction == "LR":
            level_width[lev] = max(sizes[n][0] for n in ids)
            level_height[lev] = sum(sizes[n][1] for n in ids) + max(0, len(ids) - 1) * Y_GAP
        else:
            level_width[lev] = sum(sizes[n][0] for n in ids) + max(0, len(ids) - 1) * X_GAP
            level_height[lev] = max(sizes[n][1] for n in ids)

    max_level = max(levels.values(), default=0)
    positions: dict[str, dict] = {}
    if direction == "LR":
        x_at: dict[int, float] = {}
        cursor = PANEL_PAD_X
        for lev in range(max_level + 1):
            x_at[lev] = cursor
            cursor += level_width.get(lev, 0) + (X_GAP if lev < max_level else 0)
        content_h = max(level_height.values(), default=BASE_NODE_H)
        residual_headroom = 34.0 * (1 + max((e.get("_route_lane", -1) for e in edges if e.get("type") == "residual"), default=-1))
        top = PANEL_PAD_TOP + residual_headroom
        for lev, ids in per_level.items():
            total_h = level_height[lev]
            y = top + (content_h - total_h) / 2
            for nid in ids:
                w, h = sizes[nid]
                x = x_at[lev] + (level_width[lev] - w) / 2
                positions[nid] = {"x": x, "y": y, "w": w, "h": h}
                y += h + Y_GAP
        width = cursor + PANEL_PAD_X
        height = top + content_h + PANEL_PAD_BOTTOM
    else:
        y_at: dict[int, float] = {}
        cursor = PANEL_PAD_TOP
        for lev in range(max_level + 1):
            y_at[lev] = cursor
            cursor += level_height.get(lev, 0) + (Y_GAP if lev < max_level else 0)
        content_w = max(level_width.values(), default=196.0)
        for lev, ids in per_level.items():
            total_w = level_width[lev]
            x = PANEL_PAD_X + (content_w - total_w) / 2
            for nid in ids:
                w, h = sizes[nid]
                y = y_at[lev] + (level_height[lev] - h) / 2
                positions[nid] = {"x": x, "y": y, "w": w, "h": h}
                x += w + X_GAP
        width = PANEL_PAD_X * 2 + content_w
        height = cursor + PANEL_PAD_BOTTOM

    return {
        "positions": positions,
        "width": width,
        "height": height,
        "edges": edges,
        "levels": levels,
        "groups": _group_boxes(spec, node_ids, positions),
    }


def layout_figure(spec: dict) -> dict:
    # Multi-format exports often request the same expensive routed layout several
    # times in one process. Cache by immutable spec content so joint optimization
    # runs once rather than once per renderer.
    cache_key=_layout_cache_key(spec)
    if cache_key in _LAYOUT_CACHE:
        _LAYOUT_CACHE.move_to_end(cache_key)
        return deepcopy(_LAYOUT_CACHE[cache_key])
    fig = spec.get("figure", {})
    direction = fig.get("direction", "LR")
    panels = panel_definitions(spec)
    gap = float(fig.get("panel_gap", 58.0))
    arrangement = fig.get("panel_layout", "vertical")
    panel_layouts = [dict(layout_panel(spec, p, direction=str(p.get("direction") or direction)), panel=p) for p in panels]
    # Route only after all node/group geometry is final.  This makes edge
    # routing a deterministic consequence of the content-driven layout instead of
    # letting renderers improvise curves that may pass through modules.
    for lay in panel_layouts:
        lay["edges"] = route_panel_edges(lay.get("positions", {}), lay.get("edges", []), lay.get("groups", []))
        route_pts=[p for e in lay["edges"] for p in (e.get("_route_points") or [])]
        if route_pts:
            lay["width"] = max(float(lay.get("width", 0.0)), max(float(p[0]) for p in route_pts) + PANEL_PAD_X)
            lay["height"] = max(float(lay.get("height", 0.0)), max(float(p[1]) for p in route_pts) + PANEL_PAD_BOTTOM)

    # Horizontal panel composition is opt-in. Vertical is safer for journal figures.
    if arrangement == "horizontal" and len(panel_layouts) > 1:
        x_offset = 0.0
        max_h = 0.0
        for lay in panel_layouts:
            lay["x_offset"] = x_offset
            lay["y_offset"] = 0.0
            x_offset += lay["width"] + gap
            max_h = max(max_h, lay["height"])
        return _cache_layout(cache_key, {"panels": panel_layouts, "width": max(0.0, x_offset - gap), "height": max_h})

    y_offset = 0.0
    max_width = max((lay["width"] for lay in panel_layouts), default=0.0)
    for lay in panel_layouts:
        # Center narrow detail panels under wider overview panels. This removes the
        # slide-like empty right half common in generic vertical composition.
        lay["x_offset"] = max(0.0, (max_width - lay["width"]) / 2.0)
        lay["y_offset"] = y_offset
        y_offset += lay["height"] + gap
    return _cache_layout(cache_key, {"panels": panel_layouts, "width": max_width, "height": max(0.0, y_offset - gap)})
