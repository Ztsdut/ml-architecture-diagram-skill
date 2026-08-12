from __future__ import annotations

"""Joint node/edge layout refinement for publication figures.

The earlier node-first layout strategy produced collision-free node geometry first and routed edges
second.  This module closes that loop: it perturbs legal node orderings, reroutes the
full graph, and accepts changes only when a publication-oriented objective improves.
The topology never changes.
"""

from collections import defaultdict
from copy import deepcopy
from math import hypot
from typing import Iterable

from .routing import route_panel_edges, count_route_crossings, route_node_intersections, route_stage_intersections


GROUP_PAD = 16.0


def _center(box: dict) -> tuple[float, float]:
    return float(box["x"] + box["w"] / 2), float(box["y"] + box["h"] / 2)


def _group_boxes(spec: dict, positions: dict[str, dict]) -> list[dict]:
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    grouped: dict[str, list[str]] = defaultdict(list)
    for nid in positions:
        gid = node_map.get(nid, {}).get("group")
        if gid:
            grouped[str(gid)].append(nid)
    meta = {str(g.get("id")): g for g in (spec.get("groups") or []) if isinstance(g, dict) and g.get("id")}
    out: list[dict] = []
    for gid, ids in grouped.items():
        xs = [positions[n]["x"] for n in ids]
        ys = [positions[n]["y"] for n in ids]
        rs = [positions[n]["x"] + positions[n]["w"] for n in ids]
        bs = [positions[n]["y"] + positions[n]["h"] for n in ids]
        m = meta.get(gid, {})
        box = {
            "id": gid,
            "label": m.get("label", gid.replace("_", " ").title()),
            "x": min(xs) - GROUP_PAD,
            "y": min(ys) - GROUP_PAD - 17,
            "w": max(rs) - min(xs) + 2 * GROUP_PAD,
            "h": max(bs) - min(ys) + 2 * GROUP_PAD + 17,
        }
        for k, v in m.items():
            if k not in {"id", "label"}:
                box[k] = v
        out.append(box)
    return out


def _route_length(edges: Iterable[dict]) -> float:
    total = 0.0
    for e in edges:
        pts = e.get("_route_points") or []
        for p, q in zip(pts, pts[1:]):
            total += abs(float(q[0]) - float(p[0])) + abs(float(q[1]) - float(p[1]))
    return total


def _bend_count(edges: Iterable[dict]) -> int:
    bends = 0
    for e in edges:
        pts = e.get("_route_points") or []
        for a, b, c in zip(pts, pts[1:], pts[2:]):
            d1 = (round(float(b[0]) - float(a[0]), 5), round(float(b[1]) - float(a[1]), 5))
            d2 = (round(float(c[0]) - float(b[0]), 5), round(float(c[1]) - float(b[1]), 5))
            h1 = abs(d1[1]) < 1e-6; h2 = abs(d2[1]) < 1e-6
            if h1 != h2:
                bends += 1
    return bends


def _bbox_area(positions: dict[str, dict]) -> float:
    if not positions:
        return 1.0
    minx = min(float(b["x"]) for b in positions.values())
    miny = min(float(b["y"]) for b in positions.values())
    maxx = max(float(b["x"] + b["w"]) for b in positions.values())
    maxy = max(float(b["y"] + b["h"]) for b in positions.values())
    return max(1.0, (maxx - minx) * (maxy - miny))


def _rect_overlap(a: dict, b: dict, pad: float = 0.0) -> bool:
    return not (
        float(a['x'] + a['w']) + pad <= float(b['x']) or
        float(b['x'] + b['w']) + pad <= float(a['x']) or
        float(a['y'] + a['h']) + pad <= float(b['y']) or
        float(b['y'] + b['h']) + pad <= float(a['y'])
    )


def _stage_overlap_count(groups: list[dict]) -> int:
    count=0
    for i,a in enumerate(groups):
        for b in groups[i+1:]:
            if _rect_overlap(a,b,pad=3.0): count += 1
    return count


def score_layout(spec: dict, positions: dict[str, dict], edges: list[dict]) -> tuple[float, dict, list[dict], list[dict]]:
    """Route a candidate and return a deterministic publication objective.

    Crossings are deliberately much more expensive than moderate extra route length.
    Any connector passing through a node or unrelated stage is effectively forbidden.
    """
    groups = _group_boxes(spec, positions)
    routed = route_panel_edges(positions, edges, groups)
    crossings = count_route_crossings(routed)
    length = _route_length(routed)
    bends = _bend_count(routed)
    node_hits = sum(len(route_node_intersections(e, positions, pad=1.0)) for e in routed)
    stage_hits = sum(len(route_stage_intersections(e, positions, groups, pad=.5)) for e in routed)
    stage_overlaps = _stage_overlap_count(groups)
    area = _bbox_area(positions)
    # Weights are in canvas units.  A single illegal intersection dominates any
    # compactness gain; one clean crossing is worth roughly 180 px of extra routing.
    score = (
        crossings * 520.0
        + node_hits * 100000.0
        + stage_hits * 70000.0
        + stage_overlaps * 120000.0
        + length * 0.72
        + bends * 18.0
        + area * 0.0009
    )
    metrics = {
        "objective": round(score, 3),
        "edge_crossings": int(crossings),
        "route_length": round(length, 3),
        "route_bends": int(bends),
        "edge_through_node": int(node_hits),
        "edge_through_stage": int(stage_hits),
        "stage_overlaps": int(stage_overlaps),
        "bbox_area": round(area, 3),
    }
    return score, metrics, routed, groups


def _cluster(values: list[tuple[str, float]], tolerance: float) -> list[list[str]]:
    if not values:
        return []
    values = sorted(values, key=lambda z: z[1])
    out: list[list[str]] = [[values[0][0]]]
    anchor = values[0][1]
    for nid, v in values[1:]:
        if abs(v - anchor) <= tolerance:
            out[-1].append(nid)
            anchor = sum(next(x for n, x in values if n == k) for k in out[-1]) / len(out[-1])
        else:
            out.append([nid]); anchor = v
    return out


def _lanes(spec: dict, positions: dict[str, dict], layout_meta: dict | None) -> list[tuple[str, list[str]]]:
    node_map = {n["id"]: n for n in spec.get("nodes", [])}
    by_group: dict[str, list[str]] = defaultdict(list)
    for nid in positions:
        gid = node_map.get(nid, {}).get("group")
        if gid:
            by_group[str(gid)].append(nid)
    lanes: list[tuple[str, list[str]]] = []
    for gid in ("stage_inputs", "stage_encoders"):
        ids = sorted(by_group.get(gid, []), key=lambda n: _center(positions[n])[1])
        if len(ids) > 1:
            lanes.append(("vertical", ids))

    # Output stages often contain a left head column and a right terminal column.
    outs = by_group.get("stage_outputs", [])
    if len(outs) > 1:
        clusters = _cluster([(n, _center(positions[n])[0]) for n in outs], tolerance=62.0)
        for arr in clusters:
            arr.sort(key=lambda n: _center(positions[n])[1])
            if len(arr) > 1:
                lanes.append(("vertical", arr))

    core = by_group.get("stage_interaction", [])
    if len(core) > 1:
        mode = str((layout_meta or {}).get("framework_core_mode", "horizontal"))
        if mode == "vertical":
            clusters = _cluster([(n, _center(positions[n])[1]) for n in core], tolerance=54.0)
            for arr in clusters:
                arr.sort(key=lambda n: _center(positions[n])[0])
                if len(arr) > 1:
                    lanes.append(("horizontal", arr))
        else:
            clusters = _cluster([(n, _center(positions[n])[0]) for n in core], tolerance=62.0)
            for arr in clusters:
                arr.sort(key=lambda n: _center(positions[n])[1])
                if len(arr) > 1:
                    lanes.append(("vertical", arr))
    return lanes


def _repack(order: list[str], positions: dict[str, dict], orientation: str, gap: float = 28.0) -> dict[str, dict]:
    cand = deepcopy(positions)
    if orientation == "vertical":
        top = min(float(positions[n]["y"]) for n in order)
        bottom = max(float(positions[n]["y"] + positions[n]["h"]) for n in order)
        required = sum(float(positions[n]["h"]) for n in order) + gap * (len(order) - 1)
        span = max(required, bottom - top)
        y = top + max(0.0, (span - required) / 2.0)
        # Preserve each lane's x center while changing only vertical order.
        for n in order:
            h = float(cand[n]["h"])
            cand[n]["y"] = y
            y += h + gap
    else:
        left = min(float(positions[n]["x"]) for n in order)
        right = max(float(positions[n]["x"] + positions[n]["w"]) for n in order)
        required = sum(float(positions[n]["w"]) for n in order) + gap * (len(order) - 1)
        span = max(required, right - left)
        x = left + max(0.0, (span - required) / 2.0)
        for n in order:
            w = float(cand[n]["w"])
            cand[n]["x"] = x
            x += w + gap
    return cand


def _neighbor_barycenter(nid: str, positions: dict[str, dict], edges: list[dict], axis: int) -> float | None:
    vals: list[float] = []
    for e in edges:
        if e.get("from") == nid and e.get("to") in positions:
            vals.append(_center(positions[e["to"]])[axis])
        elif e.get("to") == nid and e.get("from") in positions:
            vals.append(_center(positions[e["from"]])[axis])
    return sum(vals) / len(vals) if vals else None


def _shift_later_stages(spec: dict, positions: dict[str, dict], boundary: int, dx: float) -> dict[str, dict]:
    order=['stage_inputs','stage_encoders','stage_interaction','stage_outputs']
    later=set(order[boundary+1:])
    node_map={n['id']:n for n in spec.get('nodes',[])}
    cand=deepcopy(positions)
    for nid,b in cand.items():
        if node_map.get(nid,{}).get('group') in later:
            b['x']=float(b['x'])+dx
    return cand


def optimize_publication_layout(
    spec: dict,
    positions: dict[str, dict],
    edges: list[dict],
    layout_meta: dict | None = None,
    *,
    max_passes: int = 2,
) -> tuple[dict[str, dict], list[dict], list[dict], dict]:
    """Jointly refine legal node order and edge routing.

    The search is intentionally conservative: only nodes sharing an existing layout
    lane may reorder, so the visual narrative and stage assignment stay intact.
    """
    best_pos = deepcopy(positions)
    best_score, before, best_edges, best_groups = score_layout(spec, best_pos, edges)
    accepted = 0
    evaluated = 1

    for _ in range(max_passes):
        improved = False
        lanes = _lanes(spec, best_pos, layout_meta)
        # First try a barycentric lane order, then adjacent swaps.  The exact routed
        # objective decides whether either proposal is retained.
        for orientation, lane in lanes:
            axis = 1 if orientation == "vertical" else 0
            bary = []
            for idx, n in enumerate(lane):
                b = _neighbor_barycenter(n, best_pos, edges, axis)
                bary.append((n, b if b is not None else _center(best_pos[n])[axis], idx))
            proposed = [n for n, _, _ in sorted(bary, key=lambda z: (z[1], z[2]))]
            candidates: list[list[str]] = []
            if proposed != lane:
                candidates.append(proposed)
            base = proposed if proposed != lane else lane
            for i in range(len(base) - 1):
                arr = list(base); arr[i], arr[i + 1] = arr[i + 1], arr[i]
                candidates.append(arr)

            local_best = None
            for order in candidates:
                cand = _repack(order, best_pos, orientation, gap=28.0)
                score, metrics, routed, groups = score_layout(spec, cand, edges)
                evaluated += 1
                if score + 1e-6 < best_score and (local_best is None or score < local_best[0]):
                    local_best = (score, cand, metrics, routed, groups)
            if local_best is not None:
                best_score, best_pos, _, best_edges, best_groups = local_best
                accepted += 1; improved = True
        if not improved:
            break

    # Jointly tune inter-stage spacing after node order stabilizes.  The search is
    # deliberately small: publication figures benefit from consistent gutters, so we
    # only test local contractions/expansions around the content-derived baseline.
    stage_spacing_moves=0
    for boundary in range(3):
        local_best=None
        for dx in (-18.0,-10.0,12.0,24.0):
            cand=_shift_later_stages(spec,best_pos,boundary,dx)
            score,metrics,routed,groups=score_layout(spec,cand,edges)
            evaluated += 1
            if score + 1e-6 < best_score and (local_best is None or score < local_best[0]):
                local_best=(score,cand,metrics,routed,groups,dx)
        if local_best is not None:
            best_score,best_pos,_,best_edges,best_groups,_=local_best
            accepted += 1; stage_spacing_moves += 1

    # Re-score to expose final metrics with the accepted geometry.
    best_score, after, best_edges, best_groups = score_layout(spec, best_pos, edges)
    meta = {
        "enabled": True,
        "algorithm": "routed-lane-local-search",
        "evaluated_candidates": evaluated,
        "accepted_moves": accepted,
        "stage_spacing_moves": stage_spacing_moves,
        "before": before,
        "after": after,
        "improvement_percent": round(max(0.0, (before["objective"] - after["objective"]) / max(before["objective"], 1e-9) * 100.0), 3),
    }
    return best_pos, best_edges, best_groups, meta
