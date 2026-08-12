from __future__ import annotations

from html import escape
from pathlib import Path
import math

from ..layout import layout_figure
from ..theme import theme_for_spec
from ..visual_grammar import compile_visual_spec


def _dash(edge_type: str) -> str:
    if edge_type in {"auxiliary", "conditioning"}:
        return ' stroke-dasharray="8 6"'
    if edge_type == "training":
        return ' stroke-dasharray="3 6"'
    return ""


def _edge_width(edge_type: str) -> float:
    return 2.25 if edge_type == "main" else 1.55


def _wrap_label(text: str, max_chars: int = 24) -> list[str]:
    words = text.split()
    if not words or len(text) <= max_chars:
        return [text]
    lines, cur = [], ""
    for word in words:
        trial = f"{cur} {word}".strip()
        if len(trial) > max_chars and cur:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines[:2]


def _label_svg(node: dict, box: dict, theme: dict, font: str, *, y_ratio: float = 0.78, small: bool = False) -> str:
    title = str(node.get("label", ""))
    subtitle = str(node.get("subtitle") or node.get("shape") or "")
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    fs = 11.5 if small else 13.2
    lines = _wrap_label(title, max_chars=max(16, int(w / (7.8 if small else 8.4))))
    center = y + h * y_ratio
    if len(lines) == 1:
        ys = [center]
    else:
        ys = [center - 7, center + 8]
    out = []
    for line, yy in zip(lines, ys):
        out.append(f'<text x="{x+w/2:.1f}" y="{yy:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="{font}" font-size="{fs}" font-weight="650" fill="{theme["text"]}">{escape(line)}</text>')
    if subtitle:
        out.append(f'<text x="{x+w/2:.1f}" y="{y+h-9:.1f}" text-anchor="middle" font-family="{font}" font-size="9.0" fill="{theme["text"]}" opacity="0.72">{escape(subtitle)}</text>')
    return "\n".join(out)


def _repeat_badge(node: dict, box: dict, theme: dict, font: str) -> str:
    repeat = int(node.get("repeat", 1) or 1)
    if repeat <= 1:
        return ""
    x, y, w = box["x"], box["y"], box["w"]
    bx, by = x + w - 40, y + 8
    return (
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="32" height="19" rx="9.5" fill="#FFFFFF" fill-opacity="0.94" stroke="{theme["stroke"]}" stroke-width="0.9"/>'
        f'<text x="{bx+16:.1f}" y="{by+10:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="{font}" font-size="9.6" font-weight="700" fill="{theme["text"]}">×{repeat}</text>'
    )


def _standard_card(node: dict, box: dict, theme: dict, font: str, *, rx: float = 13.0) -> str:
    role = node.get("role", "neutral")
    fill = theme.get(role, theme["neutral"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.55"/>']
    out.append(f'<line x1="{x+12:.1f}" y1="{y+10:.1f}" x2="{x+w-12:.1f}" y2="{y+10:.1f}" stroke="{stroke}" stroke-width="3.6" stroke-linecap="round" opacity="0.17"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.50))
    out.append(_repeat_badge(node, box, theme, font))
    if node.get("shared"):
        out.append(f'<text x="{x+10:.1f}" y="{y+h-8:.1f}" font-family="{font}" font-size="8.3" font-style="italic" fill="{theme["text"]}" opacity="0.62">shared weights</text>')
    return "\n".join(filter(None, out))


def _data_stack(node: dict, box: dict, theme: dict, font: str, feature: bool = False) -> str:
    role = node.get("role", "input")
    fill = theme.get(role, theme["input"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    top_h = h * 0.57
    card_w = w * 0.70
    card_h = top_h * 0.73
    cx = x + w / 2
    base_x = cx - card_w / 2
    base_y = y + 11
    out = []
    layers = 4 if feature else 3
    for i in range(layers - 1, -1, -1):
        dx, dy = i * 6.0, -i * 4.0
        opacity = 0.38 + (layers - i) * 0.14
        out.append(f'<rect x="{base_x+dx:.1f}" y="{base_y+dy:.1f}" width="{card_w:.1f}" height="{card_h:.1f}" rx="5" fill="{fill}" fill-opacity="{min(opacity,0.92):.2f}" stroke="{stroke}" stroke-width="1.0"/>')
    if feature:
        gx = base_x + 9
        gy = base_y + 7
        gw = card_w - 18
        gh = card_h - 14
        for q in (0.33, 0.66):
            out.append(f'<line x1="{gx+gw*q:.1f}" y1="{gy:.1f}" x2="{gx+gw*q:.1f}" y2="{gy+gh:.1f}" stroke="{stroke}" stroke-width="0.55" opacity="0.28"/>')
            out.append(f'<line x1="{gx:.1f}" y1="{gy+gh*q:.1f}" x2="{gx+gw:.1f}" y2="{gy+gh*q:.1f}" stroke="{stroke}" stroke-width="0.55" opacity="0.28"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.77))
    out.append(_repeat_badge(node, box, theme, font))
    return "\n".join(filter(None, out))


def _token_strip(node: dict, box: dict, theme: dict, font: str, matrix: bool = False) -> str:
    fill = theme.get(node.get("role", "input"), theme["input"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = []
    ix, iy, iw, ih = x + 17, y + 12, w - 34, h * 0.45
    out.append(f'<rect x="{ix:.1f}" y="{iy:.1f}" width="{iw:.1f}" height="{ih:.1f}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    cols = 7
    rows = 3 if matrix else 1
    gap = 4
    cw = (iw - 14 - gap * (cols - 1)) / cols
    ch = (ih - 12 - gap * (rows - 1)) / rows
    for r in range(rows):
        for c in range(cols):
            alpha = 0.13 + 0.05 * ((r + c) % 3)
            out.append(f'<rect x="{ix+7+c*(cw+gap):.1f}" y="{iy+6+r*(ch+gap):.1f}" width="{cw:.1f}" height="{ch:.1f}" rx="2.5" fill="{stroke}" fill-opacity="{alpha:.2f}"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.76))
    return "\n".join(out)


def _embedding_card(node: dict, box: dict, theme: dict, font: str) -> str:
    fill = theme.get(node.get("role", "representation"), theme["representation"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
    bx, by, bw = x + 18, y + 15, w - 36
    for i in range(5):
        yy = by + i * 8
        width = bw * (0.58 + (i % 3) * 0.15)
        out.append(f'<rect x="{bx:.1f}" y="{yy:.1f}" width="{width:.1f}" height="4.5" rx="2.2" fill="{stroke}" fill-opacity="{0.15+0.03*i:.2f}"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.69))
    return "\n".join(out)


def _transformer_stack(node: dict, box: dict, theme: dict, font: str) -> str:
    fill = theme.get(node.get("role", "novel"), theme["novel"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = []
    layers = min(4, max(3, int((node.get("visual") or {}).get("layers", 3))))
    bw, bh = w * 0.74, h * 0.47
    bx, by = x + w * 0.13, y + 10
    for i in range(layers - 1, -1, -1):
        dx, dy = i * 7.0, -i * 4.0
        out.append(f'<rect x="{bx+dx:.1f}" y="{by+dy:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="7" fill="{fill}" fill-opacity="{0.46+0.12*(layers-i):.2f}" stroke="{stroke}" stroke-width="1.0"/>')
        if i == 0:
            mid = by + bh / 2
            out.append(f'<line x1="{bx+10:.1f}" y1="{mid:.1f}" x2="{bx+bw-10:.1f}" y2="{mid:.1f}" stroke="{stroke}" stroke-width="0.8" opacity="0.35"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.77))
    out.append(_repeat_badge(node, box, theme, font))
    return "\n".join(filter(None, out))


def _attention_heads(node: dict, box: dict, theme: dict, font: str) -> str:
    fill = theme.get(node.get("role", "novel"), theme["novel"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = []
    heads = int((node.get("visual") or {}).get("heads", 4))
    heads = min(6, max(3, heads))
    cy = y + 28
    radius = 10
    xs = [x + w * (0.20 + 0.60 * i / (heads - 1)) for i in range(heads)]
    target_x, target_y = x + w/2, y + 58
    for i, hx in enumerate(xs):
        out.append(f'<circle cx="{hx:.1f}" cy="{cy:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="1.1"/>')
        out.append(f'<line x1="{hx:.1f}" y1="{cy+radius:.1f}" x2="{target_x:.1f}" y2="{target_y:.1f}" stroke="{stroke}" stroke-width="0.8" opacity="0.50"/>')
    out.append(f'<rect x="{target_x-34:.1f}" y="{target_y-5:.1f}" width="68" height="11" rx="5.5" fill="{stroke}" fill-opacity="0.16"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.77, small=True))
    return "\n".join(out)


def _ffn(node: dict, box: dict, theme: dict, font: str) -> str:
    fill = theme.get(node.get("role", "backbone"), theme["backbone"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="1.45"/>']
    y0 = y + 17
    for j, widths in enumerate(([34, 48, 34], [32, 54, 32])):
        yy = y0 + j * 20
        total = sum(widths) + 10
        xx = x + (w-total)/2
        for i, ww in enumerate(widths):
            out.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{ww}" height="9" rx="4.5" fill="{stroke}" fill-opacity="{0.12+0.05*i:.2f}"/>')
            xx += ww + 5
    out.append(_label_svg(node, box, theme, font, y_ratio=0.75))
    return "\n".join(out)


def _norm_bar(node: dict, box: dict, theme: dict, font: str) -> str:
    fill = theme.get(node.get("role", "preprocess"), theme["preprocess"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = [f'<rect x="{x+8:.1f}" y="{y+h*0.27:.1f}" width="{w-16:.1f}" height="{h*0.34:.1f}" rx="{h*0.17:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.3"/>']
    out.append(_label_svg(node, box, theme, font, y_ratio=0.46, small=True))
    return "\n".join(out)


def _graph_glyph(node: dict, box: dict, theme: dict, font: str, message: bool = False) -> str:
    fill = theme.get(node.get("role", "novel" if message else "input"), theme["novel" if message else "input"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="1.35"/>']
    cx, cy = x+w/2, y+38
    pts = [(cx-45,cy+3),(cx-20,cy-19),(cx+10,cy-13),(cx+38,cy+5),(cx+4,cy+22),(cx-31,cy+24)]
    edges = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,5),(2,4)]
    for a,b in edges:
        x1,y1=pts[a]; x2,y2=pts[b]
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{1.2 if message else 0.9}" opacity="0.52"/>')
    for i,(px,py) in enumerate(pts):
        rr = 6.4 if message and i in {2,4} else 5.3
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rr}" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.2"/>')
    if message:
        for px,py in (pts[2],pts[4]):
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.2" fill="{stroke}" opacity="0.55"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.79, small=True))
    out.append(_repeat_badge(node, box, theme, font))
    return "\n".join(filter(None,out))


def _graph_pool(node: dict, box: dict, theme: dict, font: str) -> str:
    fill = theme.get(node.get("role", "fusion"), theme["fusion"])
    stroke = theme["stroke"]
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    out=[]
    xs=[x+w*0.28,x+w*0.40,x+w*0.52,x+w*0.64,x+w*0.76]
    cy=y+23
    for px in xs:
        out.append(f'<circle cx="{px:.1f}" cy="{cy:.1f}" r="5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        out.append(f'<line x1="{px:.1f}" y1="{cy+5:.1f}" x2="{x+w/2:.1f}" y2="{y+43:.1f}" stroke="{stroke}" stroke-width="0.8" opacity="0.45"/>')
    out.append(f'<rect x="{x+w/2-28:.1f}" y="{y+39:.1f}" width="56" height="10" rx="5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    out.append(_label_svg(node, box, theme, font, y_ratio=0.76, small=True))
    return "\n".join(out)


def _router(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","auxiliary"),theme["auxiliary"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    cx,cy=x+w/2,y+37; rw,rh=37,25
    pts=f"{cx:.1f},{cy-rh:.1f} {cx+rw:.1f},{cy:.1f} {cx:.1f},{cy+rh:.1f} {cx-rw:.1f},{cy:.1f}"
    out=[f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>']
    for dy in (-12,0,12):
        out.append(f'<line x1="{cx-rw+9:.1f}" y1="{cy+dy:.1f}" x2="{cx+rw-9:.1f}" y2="{cy+dy/2:.1f}" stroke="{stroke}" stroke-width="0.8" opacity="0.38"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.81,small=True))
    return "\n".join(out)


def _expert_fan(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","novel"),theme["novel"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    visual=node.get("visual") or {}; count=min(6,max(3,int(visual.get("experts",4))))
    out=[]; card_w=w*0.52; card_h=18; start=y+10
    for i in range(count):
        yy=start+i*18; xx=x+w*0.20+(i%2)*6
        out.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{card_w:.1f}" height="{card_h:.1f}" rx="5" fill="{fill}" fill-opacity="{0.58+0.06*i:.2f}" stroke="{stroke}" stroke-width="0.9"/>')
        out.append(f'<text x="{xx+10:.1f}" y="{yy+12:.1f}" font-family="{font}" font-size="8.3" fill="{theme["text"]}" opacity="0.75">E{i+1}</text>')
    total=int(visual.get("total_experts",node.get("repeat",1) or 1))
    if total>count:
        out.append(f'<text x="{x+w*0.77:.1f}" y="{y+h*0.47:.1f}" font-family="{font}" font-size="11" font-weight="700" fill="{theme["text"]}">… ×{total}</text>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.82,small=True))
    return "\n".join(out)


def _merge(node: dict, box: dict, theme: dict, font: str, weighted: bool = False) -> str:
    fill=theme.get(node.get("role","fusion"),theme["fusion"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    cx,cy=x+w/2,y+h/2-3
    if weighted:
        pts=f"{cx:.1f},{cy-24:.1f} {cx+34:.1f},{cy:.1f} {cx:.1f},{cy+24:.1f} {cx-34:.1f},{cy:.1f}"
        out=[f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>']
        out.append(f'<text x="{cx:.1f}" y="{cy+3:.1f}" text-anchor="middle" font-family="{font}" font-size="10" font-weight="700" fill="{theme["text"]}">Σw</text>')
    else:
        r=min(w,h)/2-4
        out=[f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>']
        symbol=(node.get("visual") or {}).get("symbol",node.get("label","+"))
        out.append(f'<text x="{cx:.1f}" y="{cy+1:.1f}" text-anchor="middle" dominant-baseline="middle" font-family="{font}" font-size="{17 if len(str(symbol))<=2 else 11}" font-weight="700" fill="{theme["text"]}">{escape(str(symbol))}</text>')
    return "\n".join(out)


def _recurrent(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","backbone"),theme["backbone"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    cells=int((node.get("visual") or {}).get("cells",4)); cells=min(5,max(3,cells))
    out=[]; cell_w=30; gap=8; total=cells*cell_w+(cells-1)*gap; sx=x+(w-total)/2; yy=y+20
    for i in range(cells):
        xx=sx+i*(cell_w+gap)
        out.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{cell_w}" height="28" rx="7" fill="{fill}" stroke="{stroke}" stroke-width="1.0"/>')
        if i<cells-1:
            out.append(f'<line x1="{xx+cell_w:.1f}" y1="{yy+14:.1f}" x2="{xx+cell_w+gap:.1f}" y2="{yy+14:.1f}" stroke="{stroke}" stroke-width="1.0"/>')
    # symbolic recurrence loop over central cell
    midx=sx+(cells//2)*(cell_w+gap)+cell_w/2
    out.append(f'<path d="M {midx-9:.1f},{yy:.1f} C {midx-13:.1f},{yy-12:.1f} {midx+13:.1f},{yy-12:.1f} {midx+9:.1f},{yy:.1f}" fill="none" stroke="{stroke}" stroke-width="1.0"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.76,small=True))
    out.append(_repeat_badge(node,box,theme,font))
    return "\n".join(filter(None,out))


def _unet_stage(node: dict, box: dict, theme: dict, font: str, bottleneck: bool = False) -> str:
    # Visual resolution is encoded by stack width/depth; no fake tensor dimension is asserted.
    fill=theme.get(node.get("role","backbone"),theme["backbone"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    out=[]; is_dec="decoder" in str(node.get("label","")).lower()
    card_w=w*(0.46 if bottleneck else 0.62); card_h=h*0.43; bx=x+(w-card_w)/2; by=y+12
    layers=4 if bottleneck else 3
    for i in range(layers-1,-1,-1):
        dx=i*5; dy=-i*3
        out.append(f'<rect x="{bx+dx:.1f}" y="{by+dy:.1f}" width="{card_w:.1f}" height="{card_h:.1f}" rx="4" fill="{fill}" fill-opacity="{0.45+0.13*(layers-i):.2f}" stroke="{stroke}" stroke-width="0.9"/>')
    # small scale-direction chevron
    if not bottleneck:
        cx=x+w*0.83; cy=y+32
        if is_dec:
            pts=f"{cx-8},{cy+5} {cx},{cy-5} {cx+8},{cy+5}"
        else:
            pts=f"{cx-8},{cy-5} {cx},{cy+5} {cx+8},{cy-5}"
        out.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" stroke-width="1.4" opacity="0.58"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.78,small=True))
    return "\n".join(out)


def _noise(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","input"),theme["input"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    out=[]
    for i in range(24):
        px=x+25+((i*37)%int(max(40,w-50))); py=y+15+((i*23)%int(max(30,h*0.46)))
        rr=1.7+(i%3)*0.7
        out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{rr:.1f}" fill="{stroke}" fill-opacity="{0.18+0.08*(i%4):.2f}"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.78,small=True))
    return "\n".join(out)


def _spectral(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","novel"),theme["novel"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="13" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>']
    pts=[]
    for i in range(50):
        xx=x+18+(w-36)*i/49; yy=y+36+12*math.sin(i/49*math.pi*4)
        pts.append(f"{xx:.1f},{yy:.1f}")
    out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{stroke}" stroke-width="1.4" opacity="0.62"/>')
    for i,hh in enumerate((9,18,28,17,10)):
        xx=x+w*0.36+i*11
        out.append(f'<rect x="{xx:.1f}" y="{y+56-hh:.1f}" width="6" height="{hh}" rx="2" fill="{stroke}" fill-opacity="0.18"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.79,small=True))
    out.append(_repeat_badge(node,box,theme,font))
    return "\n".join(filter(None,out))


def _fusion(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","fusion"),theme["fusion"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    cx,cy=x+w/2,y+h*0.40; out=[]
    for ang in (-150,-90,-30):
        a=math.radians(ang); x1=cx+34*math.cos(a); y1=cy+26*math.sin(a)
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{cx:.1f}" y2="{cy:.1f}" stroke="{stroke}" stroke-width="1.2" opacity="0.5"/>')
        out.append(f'<circle cx="{x1:.1f}" cy="{y1:.1f}" r="5.5" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="15" fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.78,small=True))
    return "\n".join(out)


def _output(node: dict, box: dict, theme: dict, font: str) -> str:
    fill=theme.get(node.get("role","output"),theme["output"]); stroke=theme["stroke"]
    x,y,w,h=box["x"],box["y"],box["w"],box["h"]
    out=[f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="16" fill="{fill}" stroke="{stroke}" stroke-width="1.55"/>']
    out.append(f'<path d="M {x+18:.1f},{y+22:.1f} L {x+29:.1f},{y+33:.1f} L {x+48:.1f},{y+15:.1f}" fill="none" stroke="{stroke}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" opacity="0.45"/>')
    out.append(_label_svg(node,box,theme,font,y_ratio=0.55))
    return "\n".join(out)


def _node_svg(node: dict, box: dict, theme: dict, font: str) -> str:
    visual=node.get("visual") or {}; vtype=visual.get("type","generic_module") if isinstance(visual,dict) else str(visual)
    if vtype == "data_stack": return _data_stack(node,box,theme,font,False)
    if vtype == "feature_map_stack": return _data_stack(node,box,theme,font,True)
    if vtype in {"token_strip","sequence_strip"}: return _token_strip(node,box,theme,font,False)
    if vtype == "token_matrix": return _token_strip(node,box,theme,font,True)
    if vtype == "embedding_card": return _embedding_card(node,box,theme,font)
    if vtype == "transformer_stack": return _transformer_stack(node,box,theme,font)
    if vtype == "attention_heads": return _attention_heads(node,box,theme,font)
    if vtype == "ffn_block": return _ffn(node,box,theme,font)
    if vtype == "norm_bar": return _norm_bar(node,box,theme,font)
    if vtype == "graph_input": return _graph_glyph(node,box,theme,font,False)
    if vtype == "graph_message": return _graph_glyph(node,box,theme,font,True)
    if vtype == "graph_pool": return _graph_pool(node,box,theme,font)
    if vtype == "router_gate": return _router(node,box,theme,font)
    if vtype == "expert_fan": return _expert_fan(node,box,theme,font)
    if vtype == "weighted_merge": return _merge(node,box,theme,font,True)
    if vtype == "merge_glyph": return _merge(node,box,theme,font,False)
    if vtype == "recurrent_cells": return _recurrent(node,box,theme,font)
    if vtype == "unet_stage": return _unet_stage(node,box,theme,font,False)
    if vtype == "bottleneck": return _unet_stage(node,box,theme,font,True)
    if vtype == "diffusion_noise": return _noise(node,box,theme,font)
    if vtype == "spectral_operator": return _spectral(node,box,theme,font)
    if vtype == "fusion_hub": return _fusion(node,box,theme,font)
    if vtype == "output_card": return _output(node,box,theme,font)
    # Useful graceful fallbacks preserve editability and semantics.
    if vtype == "modality_card": return _data_stack(node,box,theme,font,False)
    if vtype == "pooling_glyph": return _standard_card(node,box,theme,font,rx=20)
    if vtype == "classifier_head": return _standard_card(node,box,theme,font,rx=15)
    if vtype == "diffusion_denoiser": return _standard_card(node,box,theme,font,rx=18)
    if vtype == "time_condition": return _standard_card(node,box,theme,font,rx=20)
    if vtype in {"operator_branch","operator_trunk"}: return _standard_card(node,box,theme,font)
    if vtype == "operator_glyph": return _standard_card(node,box,theme,font,rx=20)
    if vtype == "loss_card": return _standard_card(node,box,theme,font,rx=9)
    return _standard_card(node,box,theme,font)


def _anchor(box: dict, side: str) -> tuple[float,float]:
    if side=="left": return box["x"], box["y"]+box["h"]/2
    if side=="right": return box["x"]+box["w"], box["y"]+box["h"]/2
    if side=="top": return box["x"]+box["w"]/2, box["y"]
    return box["x"]+box["w"]/2, box["y"]+box["h"]


def _edge_svg(edge: dict, a: dict, b: dict, theme: dict, font: str, direction: str) -> str:
    et=edge.get("type","main"); label=escape(str(edge.get("label","")))
    if direction=="LR": x1,y1=_anchor(a,"right"); x2,y2=_anchor(b,"left")
    else: x1,y1=_anchor(a,"bottom"); x2,y2=_anchor(b,"top")
    if et=="residual":
        lane=int(edge.get("_route_lane",0)); top=min(a["y"],b["y"])-30-lane*28
        # skip paths leave from upper corners to emphasize bypass semantics
        x1=a["x"]+a["w"]*0.72; y1=a["y"]
        x2=b["x"]+b["w"]*0.28; y2=b["y"]
        d=f"M {x1:.1f},{y1:.1f} C {x1+28:.1f},{top:.1f} {x2-28:.1f},{top:.1f} {x2:.1f},{y2:.1f}"
    elif direction=="LR":
        mid=(x1+x2)/2; d=f"M {x1:.1f},{y1:.1f} C {mid:.1f},{y1:.1f} {mid:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"
    else:
        mid=(y1+y2)/2; d=f"M {x1:.1f},{y1:.1f} C {x1:.1f},{mid:.1f} {x2:.1f},{mid:.1f} {x2:.1f},{y2:.1f}"
    opacity="0.84" if et!="main" else "0.96"
    s=f'<path d="{d}" fill="none" stroke="{theme["edge"]}" stroke-opacity="{opacity}" stroke-width="{_edge_width(et):.1f}" stroke-linecap="round" marker-end="url(#arrow)"{_dash(et)}/>'
    if label:
        lx=(x1+x2)/2; ly=min(y1,y2)-12 if et=="residual" else (y1+y2)/2-10
        ww=max(52,7.2*len(label)+14)
        s+=f'\n<rect x="{lx-ww/2:.1f}" y="{ly-10:.1f}" width="{ww:.1f}" height="17" rx="5" fill="{theme["background"]}" fill-opacity="0.94"/>'
        s+=f'\n<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" font-family="{font}" font-size="9.2" fill="{theme["text"]}">{label}</text>'
    return s


def render_svg(spec: dict, output: str | Path) -> Path:
    # Paper targets default to the Publication Design Engine.
    # Set figure.renderer: legacy to use the legacy diagram renderer.
    fig0 = spec.get("figure", {}) or {}
    if fig0.get("target", "paper") == "paper" and fig0.get("renderer", "publication") != "legacy":
        from .publication_svg import render_publication_svg
        return render_publication_svg(spec, output)
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    spec=compile_visual_spec(spec)
    layout=layout_figure(spec); fig=spec.get("figure",{}); theme=theme_for_spec(spec)
    font=fig.get("font","Arial")+", Helvetica, sans-serif"
    top_margin=44.0 if fig.get("title") else 16.0
    width,height=layout["width"],layout["height"]+top_margin+14
    node_map={n["id"]:n for n in spec.get("nodes",[])}
    family=escape(str(spec.get("metadata",{}).get("architecture_family","generic")).replace("_"," ").title())
    parts=[
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{theme["edge"]}"/></marker></defs>',
        f'<rect x="0" y="0" width="100%" height="100%" fill="{theme["background"]}"/>',
    ]
    title=escape(str(fig.get("title","")))
    if title:
        parts.append(f'<text x="{width/2:.1f}" y="22" text-anchor="middle" font-family="{font}" font-size="16" font-weight="700" fill="{theme["text"]}">{title}</text>')
        parts.append(f'<text x="{width/2:.1f}" y="36" text-anchor="middle" font-family="{font}" font-size="8.4" font-weight="600" letter-spacing="0.8" fill="{theme["text"]}" opacity="0.46">{family.upper()} · SCIENTIFIC VISUAL GRAMMAR</text>')
    for i,pl in enumerate(layout["panels"]):
        xoff=pl.get("x_offset",0.0); yoff=pl.get("y_offset",0.0)+top_margin; p=pl["panel"]
        if p.get("label") or p.get("title"):
            label=escape(str(p.get("label",""))); ptitle=escape(str(p.get("title","")))
            parts.append(f'<text x="{xoff+18:.1f}" y="{yoff+27:.1f}" font-family="{font}" font-size="12.6" font-weight="700" fill="{theme["text"]}">{label} {ptitle}</text>')
        for gb in pl.get("groups",[]):
            parts.append(f'<rect x="{gb["x"]+xoff:.1f}" y="{gb["y"]+yoff:.1f}" width="{gb["w"]:.1f}" height="{gb["h"]:.1f}" rx="14" fill="#F8FAFB" stroke="{theme["panel"]}" stroke-width="1" stroke-dasharray="5 5"/>')
            parts.append(f'<text x="{gb["x"]+xoff+10:.1f}" y="{gb["y"]+yoff+13:.1f}" font-family="{font}" font-size="9" font-weight="700" fill="{theme["text"]}" opacity="0.52">{escape(str(gb["label"]).upper())}</text>')
        boxes={nid:{**box,"x":box["x"]+xoff,"y":box["y"]+yoff} for nid,box in pl["positions"].items()}
        for e in pl["edges"]:
            parts.append(_edge_svg(e,boxes[e["from"]],boxes[e["to"]],theme,font,fig.get("direction","LR")))
        for nid,box in boxes.items():
            node = node_map[nid]
            node_label = escape(str(node.get("label", nid)), quote=True)
            parts.append(f'<g id="node-{escape(str(nid), quote=True)}" data-label="{node_label}">')
            parts.append(_node_svg(node,box,theme,font))
            parts.append('</g>')
        if fig.get("panel_layout","vertical")!="horizontal" and i<len(layout["panels"])-1:
            ysep=yoff+pl["height"]+28
            parts.append(f'<line x1="22" y1="{ysep:.1f}" x2="{width-22:.1f}" y2="{ysep:.1f}" stroke="{theme["panel"]}" stroke-width="0.9"/>')
    parts.append('</svg>'); output.write_text("\n".join(parts),encoding="utf-8"); return output
