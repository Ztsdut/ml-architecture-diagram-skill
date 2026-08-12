from __future__ import annotations

from dataclasses import dataclass, asdict

from .publication_design import compile_publication_spec
from .layout import layout_figure
from .scientific_illustration import validate_illustration
from .routing import route_node_intersections, route_stage_intersections, count_route_crossings


@dataclass
class QualityIssue:
    severity: str
    code: str
    message: str

    def to_dict(self):
        return asdict(self)



def _rect_overlap(a: dict, b: dict, pad: float = 0.0) -> bool:
    return not (
        a['x'] + a['w'] + pad <= b['x'] or
        b['x'] + b['w'] + pad <= a['x'] or
        a['y'] + a['h'] + pad <= b['y'] or
        b['y'] + b['h'] + pad <= a['y']
    )

def lint_publication(spec: dict) -> list[QualityIssue]:
    pub = compile_publication_spec(spec)
    layout = layout_figure(pub)
    issues: list[QualityIssue] = []
    fig = pub.get('figure', {})

    if fig.get('show_title', False):
        issues.append(QualityIssue('warning','figure-title','Paper figures normally rely on the manuscript caption; disable the large in-figure title unless it is necessary.'))

    width=max(float(layout.get('width',0)),1.0); height=max(float(layout.get('height',0)),1.0)
    if width/height > 4.8:
        issues.append(QualityIssue('warning','extreme-aspect','The figure is extremely wide; consider a second panel or a line break in the main architecture.'))
    if height/width > 2.2:
        issues.append(QualityIssue('warning','extreme-aspect','The figure is extremely tall; consider side-by-side panels or compressing detail.'))

    total_node_area=0.0
    for pl in layout.get('panels',[]):
        total_node_area += sum(float(b['w'])*float(b['h']) for b in pl.get('positions',{}).values())
    density=total_node_area/(width*height)
    if density < 0.055:
        issues.append(QualityIssue('warning','low-density',f'Only {density:.1%} of the canvas is occupied by architecture primitives; the figure may feel slide-like or under-composed.'))

    # Geometry is a hard quality constraint for publication-framework layouts.
    # A renderer may enlarge the canvas, but it must never resolve complexity by
    # placing modules or stage containers on top of each other.
    for pidx, pl in enumerate(layout.get('panels', [])):
        items = list(pl.get('positions', {}).items())
        for i in range(len(items)):
            ida, a = items[i]
            for j in range(i + 1, len(items)):
                idb, b = items[j]
                if _rect_overlap(a, b, pad=2.0):
                    issues.append(QualityIssue('error','node-overlap',f"Panel {pidx}: nodes '{ida}' and '{idb}' overlap; the layout engine must expand or reflow the composition."))
        groups = list(pl.get('groups', []))
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if _rect_overlap(groups[i], groups[j], pad=2.0):
                    issues.append(QualityIssue('error','stage-overlap',f"Panel {pidx}: stage containers '{groups[i].get('id')}' and '{groups[j].get('id')}' overlap; recompute stage widths/gaps."))
        # Edge geometry is part of publication quality.  A valid figure must not
        # solve a dense layout by letting a connector pass through an unrelated module.
        for e in pl.get('edges', []):
            hits=route_node_intersections(e, pl.get('positions', {}), pad=1.0)
            if hits:
                issues.append(QualityIssue('error','edge-through-node',f"Panel {pidx}: edge '{e.get('from')} → {e.get('to')}' passes through {', '.join(hits)}; reroute through a stage gutter."))
            stage_hits=route_stage_intersections(e, pl.get('positions', {}), pl.get('groups', []), pad=.5)
            if stage_hits:
                issues.append(QualityIssue('error','edge-through-stage',f"Panel {pidx}: secondary edge '{e.get('from')} → {e.get('to')}' crosses unrelated stage(s) {', '.join(stage_hits)}; use an outer gutter or stage bus."))
        crossings=count_route_crossings(pl.get('edges', []))
        note_threshold=max(2, len(pl.get('edges', []))//7)
        warn_threshold=max(24, len(pl.get('edges', [])))
        if crossings > warn_threshold:
            issues.append(QualityIssue('warning','edge-crossings',f'Panel {pidx}: {crossings} routed edge crossings remain; consider reordering lanes or enlarging stage gutters.'))
        elif crossings > note_threshold:
            issues.append(QualityIssue('note','edge-crossings',f'Panel {pidx}: {crossings} routed edge crossings remain; the figure is valid but could benefit from more lane separation.'))

    panels=layout.get('panels',[])
    if len(panels)>1:
        widths=[float(p['width']) for p in panels]
        if min(widths)/max(widths) < 0.28:
            issues.append(QualityIssue('note','panel-imbalance','One panel is much narrower than another. Verify that its labels remain readable at final manuscript width.'))

    illustrated_count=0
    for node in pub.get('nodes',[]):
        label=str(node.get('label',''))
        if node.get('illustration'):
            illustrated_count += 1
            for err in validate_illustration(node['illustration']):
                issues.append(QualityIssue('warning','illustration-invalid',f"Node '{node.get('id')}' illustration: {err}"))
            if node.get('illustration',{}).get('type') == 'custom_dsl' and len(node.get('illustration',{}).get('primitives') or []) > 28:
                issues.append(QualityIssue('note','illustration-complexity',f"Node '{node.get('id')}' custom illustration has many primitives; simplify it for manuscript-scale readability."))
        if len(label)>42:
            issues.append(QualityIssue('warning','long-label',f"Node '{node.get('id')}' has a long label ({len(label)} characters); shorten it or move detail to the caption."))
        if int(node.get('repeat',1) or 1)>=5 and (node.get('visual') or {}).get('type') not in {'transformer_macro','expert_fan','feature_tensor','spectral_operator_publication'}:
            issues.append(QualityIssue('note','repeat-encoding',f"Node '{node.get('id')}' repeats ×{node.get('repeat')}; verify the repeated-block encoding is visually compact."))

    budget=int(pub.get('style',{}).get('illustration_budget',6) or 6)
    if illustrated_count > budget + 2:
        issues.append(QualityIssue('note','illustration-density',f'{illustrated_count} nodes contain scientific mini-illustrations; verify the figure is not becoming icon-heavy.'))

    # Staged framework overviews can remain readable with more nodes than a generic
    # single-panel DAG because the stage containers provide strong visual chunking.
    complexity_limit = 24 if fig.get('layout_preset') == 'publication_framework' else 16
    if len(pub.get('nodes',[])) > complexity_limit and len(pub.get('panels',[]) or []) <= 1:
        issues.append(QualityIssue('warning','single-panel-complexity',f'More than {complexity_limit} nodes are shown in one panel; consider overall + detail composition.'))

    return issues
