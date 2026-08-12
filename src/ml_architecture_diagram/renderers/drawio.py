from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET
from ..layout import layout_figure
from ..theme import theme_for_spec
from ..publication_design import compile_publication_spec


def render_drawio(spec: dict, output: str | Path) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    spec = compile_publication_spec(spec)
    layout = layout_figure(spec)
    fig = spec.get("figure", {})
    theme = theme_for_spec(spec)
    node_map = {n["id"]: n for n in spec.get("nodes", [])}

    mxfile = ET.Element("mxfile", host="app.diagrams.net", version="24.7.17")
    diagram = ET.SubElement(mxfile, "diagram", id="architecture", name="Architecture")
    model = ET.SubElement(diagram, "mxGraphModel", dx="1200", dy="800", grid="1", gridSize="10", guides="1", tooltips="1", connect="1", arrows="1", fold="1", page="1", pageScale="1", pageWidth="1600", pageHeight="1200", math="0", shadow="0")
    root = ET.SubElement(model, "root")
    ET.SubElement(root, "mxCell", id="0")
    ET.SubElement(root, "mxCell", id="1", parent="0")

    cell_id = 2
    id_map = {}
    route_map = {}
    for pl in layout["panels"]:
        xoff = pl.get("x_offset", 0.0)
        off = pl.get("y_offset", 0.0) + 30
        for e in pl.get("edges", []):
            if e.get("_route_points"):
                key=(e.get("from"),e.get("to"),e.get("label",""),e.get("type","main"))
                route_map[key]=[(float(q[0])+xoff,float(q[1])+off) for q in e["_route_points"]]
        for nid, box0 in pl["positions"].items():
            node = node_map[nid]
            box = {**box0, "x": box0["x"] + xoff, "y": box0["y"] + off}
            if node.get("kind") == "merge":
                target_w = 86.0 if len(str(node.get("label", ""))) > 2 else 64.0
                box["x"] = box["x"] + (box["w"] - target_w) / 2
                box["w"] = target_w
            role = node.get("role", "neutral")
            fill = theme.get(role, theme["neutral"])
            label = node.get("label", "")
            if node.get("repeat", 1) > 1:
                label += f"\n×{node['repeat']}"
            if node.get("subtitle"):
                label += f"\n{node['subtitle']}"
            elif node.get("shape"):
                label += f"\n{node['shape']}"
            kind = node.get("kind", "module")
            visual = node.get("visual") or {}
            vtype = visual.get("type", "generic_module") if isinstance(visual, dict) else str(visual)
            base = f"whiteSpace=wrap;html=1;fillColor={fill};strokeColor={theme['stroke']};fontColor={theme['text']};fontSize=14;"
            if vtype in {"merge_glyph", "graph_pool"} or kind == "merge":
                style = "ellipse;" + base + "fontStyle=1;"
            elif vtype in {"router_gate", "weighted_merge"}:
                style = "rhombus;" + base + "fontStyle=1;"
            elif vtype in {"data_stack", "feature_map_stack", "modality_card", "token_strip", "token_matrix", "sequence_strip"}:
                style = "shape=process;rounded=1;arcSize=10;" + base
            elif vtype == "expert_fan":
                style = "shape=process;rounded=1;arcSize=8;" + base + "fontStyle=1;"
            elif vtype in {"fusion_hub"}:
                style = "hexagon;perimeter=hexagonPerimeter2;" + base + "fontStyle=1;"
            elif vtype in {"norm_bar", "pooling_glyph", "operator_glyph"}:
                style = "rounded=1;arcSize=35;" + base
            elif kind == "output" or vtype == "output_card":
                style = "rounded=1;arcSize=20;" + base + "fontStyle=1;"
            else:
                style = "rounded=1;arcSize=12;" + base
            c = ET.SubElement(root, "mxCell", id=str(cell_id), value=label, style=style, vertex="1", parent="1")
            ET.SubElement(c, "mxGeometry", x=f"{box['x']:.1f}", y=f"{box['y']:.1f}", width=f"{box['w']:.1f}", height=f"{box['h']:.1f}", **{"as": "geometry"})
            id_map[nid] = str(cell_id)
            cell_id += 1

    for e in spec.get("edges", []):
        if e.get("from") not in id_map or e.get("to") not in id_map:
            continue
        et = e.get("type", "main")
        dashed = "1" if et in {"auxiliary", "conditioning", "training"} else "0"
        width = "2" if et == "main" else "1.5"
        style = f"edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor={theme['edge']};strokeWidth={width};dashed={dashed};"
        c = ET.SubElement(root, "mxCell", id=str(cell_id), value=str(e.get("label", "")), style=style, edge="1", parent="1", source=id_map[e["from"]], target=id_map[e["to"]])
        geom=ET.SubElement(c, "mxGeometry", relative="1", **{"as": "geometry"})
        key=(e.get("from"),e.get("to"),e.get("label",""),e.get("type","main"))
        pts=route_map.get(key)
        if pts and len(pts)>2:
            arr=ET.SubElement(geom,"Array",**{"as":"points"})
            for x,y in pts[1:-1]:
                ET.SubElement(arr,"mxPoint",x=f"{x:.1f}",y=f"{y:.1f}")
        cell_id += 1

    tree = ET.ElementTree(mxfile)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output
