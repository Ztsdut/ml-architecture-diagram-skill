from __future__ import annotations

"""Obstacle-aware orthogonal routing for publication architecture figures.

The router is deliberately independent from model semantics.  It receives already
placed node bounding boxes and returns Manhattan polylines that avoid unrelated
nodes.  Stage gutters are treated as preferred corridors so long cross-stage
connections tend to read like intentional buses rather than arbitrary detours.
"""

from dataclasses import dataclass
from heapq import heappop, heappush
import math
from typing import Iterable

Point = tuple[float, float]
Segment = tuple[Point, Point]


@dataclass(frozen=True)
class Rect:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def from_box(cls, box: dict, pad: float = 0.0) -> "Rect":
        return cls(
            float(box["x"]) - pad,
            float(box["y"]) - pad,
            float(box["x"] + box["w"]) + pad,
            float(box["y"] + box["h"]) + pad,
        )

    def contains_open(self, p: Point, eps: float = 1e-6) -> bool:
        x, y = p
        return self.left + eps < x < self.right - eps and self.top + eps < y < self.bottom - eps


def _anchor_pair(a: dict, b: dict, edge_type: str = "main", local_stage: bool = False) -> tuple[Point, Point, str, str]:
    acx, acy = a["x"] + a["w"] / 2, a["y"] + a["h"] / 2
    bcx, bcy = b["x"] + b["w"] / 2, b["y"] + b["h"] / 2
    dx, dy = bcx - acx, bcy - acy
    # Inside one semantic stage, geometry should dominate over the global reading
    # direction.  This produces the short vertical arrows common in paper block
    # diagrams (attention → gate → fusion) instead of large rectangular detours.
    if local_stage and abs(dy) > max(34.0, abs(dx) * 0.78):
        if dy >= 0:
            return (acx, a["y"] + a["h"]), (bcx, b["y"]), "bottom", "top"
        return (acx, a["y"]), (bcx, b["y"] + b["h"]), "top", "bottom"
    # Conditioning/query branches are semantically secondary.  When they originate
    # clearly above/below a downstream module, enter through the corresponding port
    # instead of masquerading as another main left-to-right trunk.
    if edge_type == "conditioning" and dx > 24.0 and abs(dy) > 38.0:
        if dy < 0:  # source is below target
            return (a["x"] + a["w"], acy), (bcx, b["y"] + b["h"]), "right", "bottom"
        return (a["x"] + a["w"], acy), (bcx, b["y"]), "right", "top"
    # Strongly prefer left-to-right reading when horizontal separation is meaningful.
    if dx > max(24.0, abs(dy) * 0.34):
        return (a["x"] + a["w"], acy), (b["x"], bcy), "right", "left"
    if dx < -max(24.0, abs(dy) * 0.34):
        return (a["x"], acy), (b["x"] + b["w"], bcy), "left", "right"
    if dy >= 0:
        return (acx, a["y"] + a["h"]), (bcx, b["y"]), "bottom", "top"
    return (acx, a["y"]), (bcx, b["y"] + b["h"]), "top", "bottom"


def _stub(p: Point, side: str, distance: float) -> Point:
    x, y = p
    if side == "right":
        return x + distance, y
    if side == "left":
        return x - distance, y
    if side == "bottom":
        return x, y + distance
    return x, y - distance


def _between(v: float, a: float, b: float, eps: float = 1e-7) -> bool:
    lo, hi = sorted((a, b))
    return lo - eps <= v <= hi + eps


def _segment_hits_rect(p1: Point, p2: Point, r: Rect, eps: float = 1e-6) -> bool:
    x1, y1 = p1
    x2, y2 = p2
    if abs(y1 - y2) <= eps:  # horizontal
        y = y1
        if not (r.top + eps < y < r.bottom - eps):
            return False
        lo, hi = sorted((x1, x2))
        return max(lo, r.left + eps) < min(hi, r.right - eps)
    if abs(x1 - x2) <= eps:  # vertical
        x = x1
        if not (r.left + eps < x < r.right - eps):
            return False
        lo, hi = sorted((y1, y2))
        return max(lo, r.top + eps) < min(hi, r.bottom - eps)
    return True


def _clear_segment(p1: Point, p2: Point, obstacles: Iterable[Rect]) -> bool:
    return not any(_segment_hits_rect(p1, p2, r) for r in obstacles)


def _segments_cross(a: Segment, b: Segment, eps: float = 1e-6) -> tuple[bool, bool]:
    """Return (crosses, overlaps_collinearly). Endpoint touching is not penalized."""
    (x1, y1), (x2, y2) = a
    (u1, v1), (u2, v2) = b
    ah = abs(y1 - y2) <= eps
    bh = abs(v1 - v2) <= eps
    if ah and bh:
        if abs(y1 - v1) > eps:
            return False, False
        lo1, hi1 = sorted((x1, x2)); lo2, hi2 = sorted((u1, u2))
        overlap = min(hi1, hi2) - max(lo1, lo2)
        return (overlap > eps, overlap > eps)
    if (not ah) and (not bh):
        if abs(x1 - u1) > eps:
            return False, False
        lo1, hi1 = sorted((y1, y2)); lo2, hi2 = sorted((v1, v2))
        overlap = min(hi1, hi2) - max(lo1, lo2)
        return (overlap > eps, overlap > eps)
    if ah:
        hx1, hx2 = sorted((x1, x2)); vy1, vy2 = sorted((v1, v2))
        cx, cy = u1, y1
    else:
        hx1, hx2 = sorted((u1, u2)); vy1, vy2 = sorted((y1, y2))
        cx, cy = x1, v1
    if hx1 + eps < cx < hx2 - eps and vy1 + eps < cy < vy2 - eps:
        return True, False
    return False, False


def _simplify(points: list[Point]) -> list[Point]:
    if len(points) <= 2:
        return points
    out = [points[0]]
    for p in points[1:]:
        if math.dist(p, out[-1]) < 1e-6:
            continue
        out.append(p)
    changed = True
    while changed and len(out) > 2:
        changed = False
        new = [out[0]]
        for i in range(1, len(out) - 1):
            a, b, c = new[-1], out[i], out[i + 1]
            if (abs(a[0] - b[0]) < 1e-6 and abs(b[0] - c[0]) < 1e-6) or (
                abs(a[1] - b[1]) < 1e-6 and abs(b[1] - c[1]) < 1e-6
            ):
                changed = True
                continue
            new.append(b)
        new.append(out[-1])
        out = new
    return out


def _route_one(
    source_id: str,
    target_id: str,
    boxes: dict[str, dict],
    previous_segments: list[Segment],
    gutter_xs: list[float],
    *,
    edge_type: str = "main",
    local_stage: bool = False,
    extra_obstacles: list[Rect] | None = None,
    clearance: float = 13.0,
    stub: float = 11.0,
    bend_penalty: float = 34.0,
    crossing_penalty: float = 86.0,
) -> list[Point]:
    a, b = boxes[source_id], boxes[target_id]
    start_anchor, end_anchor, sside, tside = _anchor_pair(a, b, edge_type=edge_type, local_stage=local_stage)
    start = _stub(start_anchor, sside, stub)
    end = _stub(end_anchor, tside, stub)
    obstacles = [Rect.from_box(box, clearance) for nid, box in boxes.items() if nid not in {source_id, target_id}]
    obstacles.extend(extra_obstacles or [])

    # Direct orthogonal dogleg first.  This keeps short local connections compact.
    doglegs = [
        [start, (end[0], start[1]), end],
        [start, (start[0], end[1]), end],
    ]
    for pts in doglegs:
        segs=list(zip(pts,pts[1:]))
        no_reuse=not any(_segments_cross(seg, old)[1] for seg in segs for old in previous_segments)
        if no_reuse and all(_clear_segment(p, q, obstacles) for p, q in segs):
            return _simplify([start_anchor, *pts, end_anchor])

    xs = {round(start[0], 3), round(end[0], 3)}
    ys = {round(start[1], 3), round(end[1], 3)}
    for r in obstacles:
        xs.update((round(r.left, 3), round(r.right, 3)))
        ys.update((round(r.top, 3), round(r.bottom, 3)))
    xs.update(round(x, 3) for x in gutter_xs)
    # Existing routes create parallel-lane candidates.  Without these coordinates an
    # A* search can be forced to reuse the exact same horizontal/vertical segment,
    # making two semantically different arrows visually indistinguishable.
    lane_sep = max(10.0, clearance * 0.92)
    for (p1, p2) in previous_segments:
        if abs(p1[1] - p2[1]) < 1e-6:
            ys.update((round(p1[1]-lane_sep,3), round(p1[1]+lane_sep,3)))
        elif abs(p1[0] - p2[0]) < 1e-6:
            xs.update((round(p1[0]-lane_sep,3), round(p1[0]+lane_sep,3)))
    # Outer escape corridors make a route possible even for dense nested compositions.
    if boxes:
        minx = min(v["x"] for v in boxes.values()) - clearance * 2.2
        maxx = max(v["x"] + v["w"] for v in boxes.values()) + clearance * 2.2
        miny = min(v["y"] for v in boxes.values()) - clearance * 2.2
        maxy = max(v["y"] + v["h"] for v in boxes.values()) + clearance * 2.2
        xs.update((round(minx, 3), round(maxx, 3)))
        ys.update((round(miny, 3), round(maxy, 3)))

    xs = sorted(xs); ys = sorted(ys)
    valid: set[Point] = set()
    for x in xs:
        for y in ys:
            p = (x, y)
            if not any(r.contains_open(p) for r in obstacles):
                valid.add(p)
    valid.add((round(start[0], 3), round(start[1], 3)))
    valid.add((round(end[0], 3), round(end[1], 3)))

    neighbors: dict[Point, list[tuple[Point, str]]] = {p: [] for p in valid}
    by_y: dict[float, list[Point]] = {}
    by_x: dict[float, list[Point]] = {}
    for p in valid:
        by_y.setdefault(p[1], []).append(p)
        by_x.setdefault(p[0], []).append(p)
    for arr in by_y.values():
        arr.sort()
        for p, q in zip(arr, arr[1:]):
            seg=(p,q)
            reused=any(_segments_cross(seg, old)[1] for old in previous_segments)
            if (not reused) and _clear_segment(p, q, obstacles):
                neighbors[p].append((q, "H")); neighbors[q].append((p, "H"))
    for arr in by_x.values():
        arr.sort(key=lambda p: p[1])
        for p, q in zip(arr, arr[1:]):
            seg=(p,q)
            reused=any(_segments_cross(seg, old)[1] for old in previous_segments)
            if (not reused) and _clear_segment(p, q, obstacles):
                neighbors[p].append((q, "V")); neighbors[q].append((p, "V"))

    start = (round(start[0], 3), round(start[1], 3)); end = (round(end[0], 3), round(end[1], 3))
    heap: list[tuple[float, float, Point, str | None]] = [(0.0, 0.0, start, None)]
    best: dict[tuple[Point, str | None], float] = {(start, None): 0.0}
    parent: dict[tuple[Point, str | None], tuple[Point, str | None] | None] = {(start, None): None}
    finish: tuple[Point, str | None] | None = None
    while heap:
        _, cost, p, prev_dir = heappop(heap)
        state = (p, prev_dir)
        if cost > best.get(state, float("inf")) + 1e-8:
            continue
        if p == end:
            finish = state
            break
        for q, d in neighbors.get(p, []):
            seg = (p, q)
            length = abs(q[0] - p[0]) + abs(q[1] - p[1])
            extra = bend_penalty if prev_dir and prev_dir != d else 0.0
            # Long vertical runs in a stage gutter are visually desirable.
            if d == "V" and any(abs(p[0] - gx) < 1.0 for gx in gutter_xs):
                length *= 0.78
            for old in previous_segments:
                crosses, overlap = _segments_cross(seg, old)
                if crosses:
                    # Collinear reuse is worse than a clean perpendicular crossing: it
                    # visually merges two independent information paths into one line.
                    extra += crossing_penalty * (1.65 if overlap else 1.0)
            nc = cost + length + extra
            ns = (q, d)
            if nc + 1e-8 < best.get(ns, float("inf")):
                best[ns] = nc
                parent[ns] = state
                heuristic = abs(end[0] - q[0]) + abs(end[1] - q[1])
                heappush(heap, (nc + heuristic, nc, q, d))

    if finish is None:
        # Safe perimeter fallback.  Never trade a failed search for a connector that
        # cuts through a node: test all four outer corridors and take the shortest
        # obstacle-free route.  Route overlap is allowed here as a last resort, but
        # node/stage intersection is not.
        left = min([r.left for r in obstacles] + [start[0], end[0]]) - clearance
        right = max([r.right for r in obstacles] + [start[0], end[0]]) + clearance
        top = min([r.top for r in obstacles] + [start[1], end[1]]) - clearance
        bottom = max([r.bottom for r in obstacles] + [start[1], end[1]]) + clearance
        candidates=[
            [start,(start[0],top),(end[0],top),end],
            [start,(start[0],bottom),(end[0],bottom),end],
            [start,(left,start[1]),(left,end[1]),end],
            [start,(right,start[1]),(right,end[1]),end],
        ]
        safe=[]
        for pts in candidates:
            if all(_clear_segment(p,q,obstacles) for p,q in zip(pts,pts[1:])):
                length=sum(abs(q[0]-p[0])+abs(q[1]-p[1]) for p,q in zip(pts,pts[1:]))
                safe.append((length,pts))
        if safe:
            _,pts=min(safe,key=lambda z:z[0])
            return _simplify([start_anchor,*pts,end_anchor])
        # This should be exceptionally rare.  Return anchors directly so the quality
        # linter can fail loudly rather than silently pretending the route is valid.
        return _simplify([start_anchor,start,end,end_anchor])

    rev: list[Point] = []
    cur = finish
    while cur is not None:
        rev.append(cur[0])
        cur = parent[cur]
    routed = list(reversed(rev))
    return _simplify([start_anchor, *routed, end_anchor])


def _gutter_xs(groups: list[dict]) -> list[float]:
    horizontal = sorted(groups, key=lambda g: float(g["x"]))
    out: list[float] = []
    for a, b in zip(horizontal, horizontal[1:]):
        ar = float(a["x"] + a["w"]); bl = float(b["x"])
        if bl - ar > 8.0:
            out.append((ar + bl) / 2)
    return out


def _box_center(box: dict) -> Point:
    return float(box["x"] + box["w"] / 2), float(box["y"] + box["h"] / 2)


def _group_contains(group: dict, box: dict) -> bool:
    x,y=_box_center(box)
    return float(group["x"]) <= x <= float(group["x"]+group["w"]) and float(group["y"]) <= y <= float(group["y"]+group["h"])


def route_panel_edges(positions: dict[str, dict], edges: list[dict], groups: list[dict] | None = None) -> list[dict]:
    """Return edge copies enriched with ``_route_points``.

    Residual connections keep their renderer-specific bypass convention because that
    notation has semantic meaning.  Other edges receive obstacle-aware routes.
    """
    groups = groups or []
    gutters = _gutter_xs(groups)
    previous: list[Segment] = []
    enriched = [dict(e) for e in edges]

    def priority(e: dict) -> tuple[int, float]:
        a, b = positions[e["from"]], positions[e["to"]]
        ac = (a["x"] + a["w"] / 2, a["y"] + a["h"] / 2)
        bc = (b["x"] + b["w"] / 2, b["y"] + b["h"] / 2)
        distance = abs(ac[0] - bc[0]) + abs(ac[1] - bc[1])
        typ = e.get("type", "main")
        # Long auxiliary/conditioning paths are routed first so they can reserve a clean gutter.
        type_rank = {"conditioning": 0, "auxiliary": 0, "main": 1, "training": 2}.get(typ, 1)
        return type_rank, -distance

    index_order = sorted(range(len(enriched)), key=lambda i: priority(enriched[i]))
    for i in index_order:
        e = enriched[i]
        if e.get("type") == "residual":
            continue
        if e.get("from") not in positions or e.get("to") not in positions:
            continue
        et=e.get("type", "main")
        source_groups={g.get("id") for g in groups if _group_contains(g, positions[e["from"]])}
        target_groups={g.get("id") for g in groups if _group_contains(g, positions[e["to"]])}
        same_stage=bool(source_groups & target_groups)
        extra=[]
        if et in {"conditioning", "training", "auxiliary"} and groups:
            for g in groups:
                if g.get("id") in source_groups or g.get("id") in target_groups:
                    continue
                extra.append(Rect.from_box(g, 7.0))
        prior_segments=[] if same_stage else previous
        points = _route_one(e["from"], e["to"], positions, prior_segments, gutters, edge_type=et, local_stage=same_stage, extra_obstacles=extra)
        e["_route_points"] = [[round(x, 3), round(y, 3)] for x, y in points]
        previous.extend(list(zip(points, points[1:])))
    return enriched


def segment_intersects_box(p1: Point, p2: Point, box: dict, pad: float = 2.0) -> bool:
    return _segment_hits_rect(p1, p2, Rect.from_box(box, pad))


def route_node_intersections(edge: dict, positions: dict[str, dict], pad: float = 2.0) -> list[str]:
    points = edge.get("_route_points") or []
    if len(points) < 2:
        return []
    hits = []
    ignored = {edge.get("from"), edge.get("to")}
    for nid, box in positions.items():
        if nid in ignored:
            continue
        if any(segment_intersects_box(tuple(p), tuple(q), box, pad=pad) for p, q in zip(points, points[1:])):
            hits.append(nid)
    return hits


def count_route_crossings(edges: list[dict]) -> int:
    segments_by_edge: list[list[Segment]] = []
    for e in edges:
        pts = [tuple(p) for p in (e.get("_route_points") or [])]
        segments_by_edge.append(list(zip(pts, pts[1:])))
    count = 0
    for i, aa in enumerate(segments_by_edge):
        ei=edges[i]
        for j, bb in enumerate(segments_by_edge[i + 1:], start=i+1):
            ej=edges[j]
            # Fan-in/fan-out connections intentionally meet near their common source or
            # target and should not be reported as visual crossings.
            if {ei.get("from"),ei.get("to")} & {ej.get("from"),ej.get("to")}:
                continue
            if any(_segments_cross(a, b)[0] and not _segments_cross(a, b)[1] for a in aa for b in bb):
                count += 1
    return count


def route_stage_intersections(edge: dict, positions: dict[str, dict], groups: list[dict], pad: float = 1.0) -> list[str]:
    """Return unrelated stage containers crossed by a secondary routed edge."""
    if edge.get("type") not in {"conditioning", "auxiliary", "training"}:
        return []
    pts=[tuple(p) for p in (edge.get("_route_points") or [])]
    if len(pts)<2 or edge.get("from") not in positions or edge.get("to") not in positions:
        return []
    source_groups={g.get("id") for g in groups if _group_contains(g, positions[edge["from"]])}
    target_groups={g.get("id") for g in groups if _group_contains(g, positions[edge["to"]])}
    hits=[]
    for g in groups:
        if g.get("id") in source_groups or g.get("id") in target_groups:
            continue
        if any(segment_intersects_box(p,q,g,pad=pad) for p,q in zip(pts,pts[1:])):
            hits.append(str(g.get("id")))
    return hits


def fanin_bundle_geometry(edges: list[dict], boxes: dict[str, dict], target_id: str, offset: float = 20.0) -> dict | None:
    """Create a compact publication fan-in bus for a high-indegree target.

    This is a rendering abstraction only: each original edge remains in the IR.  The
    returned geometry contains branch polylines plus one shared trunk with the sole
    arrowhead.  Orientation follows the spatial distribution of the sources.
    """
    if target_id not in boxes or len(edges) < 3:
        return None
    target=boxes[target_id]
    tcx,tcy=_box_center(target)
    src=[]
    for e in edges:
        if e.get('from') not in boxes:
            continue
        b=boxes[e['from']]; cx,cy=_box_center(b)
        src.append((e,b,cx,cy))
    if len(src)<3:
        return None
    above=sum(cy < target['y']-4 for _,_,_,cy in src)
    below=sum(cy > target['y']+target['h']+4 for _,_,_,cy in src)
    left=sum(cx < target['x']-4 for _,_,cx,_ in src)
    right=sum(cx > target['x']+target['w']+4 for _,_,cx,_ in src)
    branches=[]
    if above >= max(2, len(src)-1):
        bus_y=target['y']-offset
        xs=[]
        for e,b,cx,cy in src:
            if b['y']+b['h'] <= bus_y:
                p0=(cx,b['y']+b['h'])
            else:
                # A source level with the bus exits from the nearest side then turns.
                p0=(b['x']+b['w'],cy) if cx < tcx else (b['x'],cy)
            bx=p0[0]
            branches.append({'edge':e,'points':[p0,(bx,bus_y)]})
            xs.append(bx)
        x1=min(xs+[tcx]); x2=max(xs+[tcx])
        return {'orientation':'top','bus':[(x1,bus_y),(x2,bus_y)],'branches':branches,'trunk':[(tcx,bus_y),(tcx,target['y'])]}
    if below >= max(2, len(src)-1):
        bus_y=target['y']+target['h']+offset
        xs=[]
        for e,b,cx,cy in src:
            p0=(cx,b['y']) if b['y'] >= bus_y else ((b['x']+b['w'],cy) if cx < tcx else (b['x'],cy))
            bx=p0[0]; branches.append({'edge':e,'points':[p0,(bx,bus_y)]}); xs.append(bx)
        x1=min(xs+[tcx]); x2=max(xs+[tcx])
        return {'orientation':'bottom','bus':[(x1,bus_y),(x2,bus_y)],'branches':branches,'trunk':[(tcx,bus_y),(tcx,target['y']+target['h'])]}
    if right > left:
        bus_x=target['x']+target['w']+offset; ys=[]
        for e,b,cx,cy in src:
            p0=(b['x'],cy) if b['x'] >= bus_x else (b['x']+b['w'],cy)
            by=p0[1]; branches.append({'edge':e,'points':[p0,(bus_x,by)]}); ys.append(by)
        y1=min(ys+[tcy]); y2=max(ys+[tcy])
        return {'orientation':'right','bus':[(bus_x,y1),(bus_x,y2)],'branches':branches,'trunk':[(bus_x,tcy),(target['x']+target['w'],tcy)]}
    bus_x=target['x']-offset; ys=[]
    for e,b,cx,cy in src:
        p0=(b['x']+b['w'],cy) if b['x']+b['w'] <= bus_x else (b['x'],cy)
        by=p0[1]; branches.append({'edge':e,'points':[p0,(bus_x,by)]}); ys.append(by)
    y1=min(ys+[tcy]); y2=max(ys+[tcy])
    return {'orientation':'left','bus':[(bus_x,y1),(bus_x,y2)],'branches':branches,'trunk':[(bus_x,tcy),(target['x'],tcy)]}


def fanout_bundle_geometry(edges: list[dict], boxes: dict[str, dict], source_id: str, offset: float = 20.0) -> dict | None:
    """Create a compact publication fan-out bus for a high-outdegree source.

    Exact edges remain unchanged in the IR.  The source contributes one trunk to a
    shared bus; each target gets an independent arrowed branch from that bus.
    """
    if source_id not in boxes or len(edges) < 3:
        return None
    source = boxes[source_id]
    scx, scy = _box_center(source)
    tgts=[]
    for e in edges:
        tid=e.get('to')
        if tid not in boxes:
            continue
        b=boxes[tid]; cx,cy=_box_center(b)
        tgts.append((e,b,cx,cy))
    if len(tgts)<3:
        return None
    above=sum(cy < source['y']-4 for _,_,_,cy in tgts)
    below=sum(cy > source['y']+source['h']+4 for _,_,_,cy in tgts)
    left=sum(cx < source['x']-4 for _,_,cx,_ in tgts)
    right=sum(cx > source['x']+source['w']+4 for _,_,cx,_ in tgts)
    branches=[]
    if right >= max(2, len(tgts)-1):
        bus_x=source['x']+source['w']+offset; ys=[]
        for e,b,cx,cy in tgts:
            p1=(b['x'],cy) if b['x'] >= bus_x else ((b['x']+b['w']/2,b['y']) if cy>scy else (b['x']+b['w']/2,b['y']+b['h']))
            by=p1[1]; branches.append({'edge':e,'points':[(bus_x,by),p1]}); ys.append(by)
        y1=min(ys+[scy]); y2=max(ys+[scy])
        return {'orientation':'right','trunk':[(source['x']+source['w'],scy),(bus_x,scy)],'bus':[(bus_x,y1),(bus_x,y2)],'branches':branches}
    if left >= max(2, len(tgts)-1):
        bus_x=source['x']-offset; ys=[]
        for e,b,cx,cy in tgts:
            p1=(b['x']+b['w'],cy) if b['x']+b['w'] <= bus_x else ((b['x']+b['w']/2,b['y']) if cy>scy else (b['x']+b['w']/2,b['y']+b['h']))
            by=p1[1]; branches.append({'edge':e,'points':[(bus_x,by),p1]}); ys.append(by)
        y1=min(ys+[scy]); y2=max(ys+[scy])
        return {'orientation':'left','trunk':[(source['x'],scy),(bus_x,scy)],'bus':[(bus_x,y1),(bus_x,y2)],'branches':branches}
    if below >= max(2, len(tgts)-1):
        bus_y=source['y']+source['h']+offset; xs=[]
        for e,b,cx,cy in tgts:
            p1=(cx,b['y']) if b['y'] >= bus_y else ((b['x'],cy) if cx>scx else (b['x']+b['w'],cy))
            bx=p1[0]; branches.append({'edge':e,'points':[(bx,bus_y),p1]}); xs.append(bx)
        x1=min(xs+[scx]); x2=max(xs+[scx])
        return {'orientation':'bottom','trunk':[(scx,source['y']+source['h']),(scx,bus_y)],'bus':[(x1,bus_y),(x2,bus_y)],'branches':branches}
    if above >= max(2, len(tgts)-1):
        bus_y=source['y']-offset; xs=[]
        for e,b,cx,cy in tgts:
            p1=(cx,b['y']+b['h']) if b['y']+b['h'] <= bus_y else ((b['x'],cy) if cx>scx else (b['x']+b['w'],cy))
            bx=p1[0]; branches.append({'edge':e,'points':[(bx,bus_y),p1]}); xs.append(bx)
        x1=min(xs+[scx]); x2=max(xs+[scx])
        return {'orientation':'top','trunk':[(scx,source['y']),(scx,bus_y)],'bus':[(x1,bus_y),(x2,bus_y)],'branches':branches}
    return None
