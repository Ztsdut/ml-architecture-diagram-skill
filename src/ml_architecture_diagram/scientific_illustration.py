from __future__ import annotations

"""Scientific mini-illustration planning and safe vector primitives.

The Architecture IR is the source of structural truth.  This module adds a
paper-facing illustration layer only.  It never changes nodes, edges, repeats,
or tensor dimensions.

A node may carry::

    illustration:
      type: spectral_sphere
      composition: illustration-top
      params: {...}

or a tool/agent-authored vector DSL::

    illustration:
      type: custom_dsl
      composition: illustration-left
      primitives:
        - {shape: circle, cx: 25, cy: 50, r: 9, fill: accent}
        - {shape: line, x1: 34, y1: 50, x2: 72, y2: 30}

DSL coordinates are normalized to a 0..100 local canvas.  Only a deliberately
small primitive set is supported so the output stays editable and auditable.
"""

from copy import deepcopy
import math
from typing import Any


BUILTIN_TYPES = {
    "timeseries",
    "feature_grid",
    "sequence",
    "globe_coordinates",
    "coordinate_axes",
    "fibonacci_sphere",
    "graph_network",
    "spectral_sphere",
    "attention_fan",
    "uncertainty_curve",
    "satellite_occultation",
    "radar_dish",
    "field_map",
    "sun_geomagnetic",
    "point_cloud",
    "custom_dsl",
}


def _text(node: dict) -> str:
    values = [node.get("label", ""), node.get("subtitle", ""), node.get("shape", "")]
    values += list(node.get("details") or []) if isinstance(node.get("details"), list) else []
    return " ".join(str(v) for v in values if v).lower()


def infer_illustration(node: dict) -> dict | None:
    """Conservatively infer a mini-illustration from verified semantic text.

    The inference is intentionally semantic and sparse.  It does not invent a
    physical object when the label is generic.  Agent-authored ``illustration``
    always wins.
    """
    if node.get("illustration"):
        return deepcopy(node["illustration"])

    text = _text(node)
    role = str(node.get("role", ""))
    kind = str(node.get("kind", ""))

    # Domain/geometry cues first.
    rules: list[tuple[tuple[str, ...], str, str]] = [
        (("radio occult", "occultation", " ro ", "ro context"), "satellite_occultation", "illustration-left"),
        (("ionosonde", "digisonde", "radar dish", "sounder"), "radar_dish", "illustration-left"),
        (("solar", "geomagnetic", "sym-h", "dst", "f10.7"), "sun_geomagnetic", "illustration-left"),
        (("fibonacci sphere", "sphere nodes", "spherical latent", "global latent prior"), "fibonacci_sphere", "illustration-top"),
        (("spherical spectral", "spectral operator", "fourier", "spherical harmonic"), "spectral_sphere", "illustration-top"),
        (("graph propagation", "message passing", "graph network", "gnn", "graph convolution"), "graph_network", "illustration-top"),
        (("cross-attention", "cross attention", "self-attention", "self attention", "attention"), "attention_fan", "illustration-top"),
        (("student-t", "student t", "uncertainty", "variance", "distribution"), "uncertainty_curve", "illustration-bottom"),
        (("latitude", "longitude", "coordinate", "query information", "spatiotemporal query"), "globe_coordinates", "illustration-left"),
        (("global field", "spatial field", "output field", "heatmap", "map field"), "field_map", "illustration-bottom"),
        (("point cloud", "particles", "samples"), "point_cloud", "illustration-top"),
        (("history", "time series", "temporal", "tcn", "rnn", "lstm", "causal driver", "sequence encoder"), "timeseries", "illustration-left"),
        (("image", "feature map", "spatial features", "grid", "map encoder"), "feature_grid", "illustration-left"),
        (("token", "sequence"), "sequence", "illustration-left"),
    ]
    padded = f" {text} "
    # Ordinary encoders/backbone blocks should normally remain visually quiet;
    # the data card already explains the input modality.  Keep automatic
    # illustration for encoders only when the label itself names a distinctive
    # scientific operator/geometry.
    quiet_encoder = ("encoder" in padded and role in {"representation", "backbone", "preprocess"})
    distinctive = any(k in padded for k in ("attention", "graph", "spectral", "fourier", "sphere", "coordinate"))
    for terms, typ, comp in rules:
        if any(term in padded for term in terms):
            if quiet_encoder and not distinctive:
                continue
            return {"type": typ, "composition": comp, "source": "auto-semantic"}

    if role == "output" and kind == "output":
        return None
    return None


def _illustration_score(node: dict, illustration: dict) -> int:
    text = _text(node)
    typ = illustration.get("type", "")
    score = 0
    if node.get("role") == "novel":
        score += 5
    if typ in {"fibonacci_sphere", "spectral_sphere", "graph_network", "satellite_occultation", "radar_dish", "field_map"}:
        score += 4
    if typ in {"attention_fan", "uncertainty_curve", "globe_coordinates", "sun_geomagnetic"}:
        score += 3
    if node.get("role") in {"input", "output"}:
        score += 1
    if any(k in text for k in ("proposed", "latent", "operator", "physics", "geometry")):
        score += 2
    return score


def compile_scientific_illustrations(spec: dict, *, max_auto: int = 6) -> dict:
    """Add a sparse scientific-illustration plan without mutating topology."""
    out = deepcopy(spec)
    meta = out.setdefault("metadata", {})
    meta["scientific_illustration_version"] = "1.0"
    style = out.setdefault("style", {})
    style.setdefault("scientific_illustrations", True)
    style.setdefault("illustration_budget", max_auto)

    explicit_ids: list[str] = []
    candidates: list[tuple[int, dict, dict]] = []
    for node in out.get("nodes", []):
        if node.get("illustration"):
            explicit_ids.append(str(node.get("id")))
            continue
        ill = infer_illustration(node)
        if ill:
            candidates.append((_illustration_score(node, ill), node, ill))

    # Explicit agent/human plans are never removed.  Automatic decorations are
    # budgeted so a paper figure does not become icon-heavy.
    remaining = max(0, int(max_auto) - len(explicit_ids))
    candidates.sort(key=lambda x: (-x[0], str(x[1].get("id", ""))))
    auto_ids: list[str] = []
    for _, node, ill in candidates[:remaining]:
        node["illustration"] = ill
        auto_ids.append(str(node.get("id")))

    meta["scientific_illustration"] = {
        "explicit_nodes": explicit_ids,
        "auto_nodes": auto_ids,
        "budget": int(max_auto),
    }
    return out


def validate_illustration(ill: dict) -> list[str]:
    errors: list[str] = []
    typ = str(ill.get("type", ""))
    if typ not in BUILTIN_TYPES:
        errors.append(f"unknown illustration type: {typ}")
    if typ == "custom_dsl":
        primitives = ill.get("primitives")
        if not isinstance(primitives, list) or not primitives:
            errors.append("custom_dsl requires a non-empty primitives list")
        else:
            allowed = {"circle", "ellipse", "rect", "line", "polyline", "polygon", "text"}
            for i, p in enumerate(primitives):
                if not isinstance(p, dict) or p.get("shape") not in allowed:
                    errors.append(f"primitive {i} uses an unsupported shape")
    return errors


# ---------- SVG primitive rendering -------------------------------------------------

def _hex_role(value: Any, theme: dict, default: str) -> str:
    if not value:
        return default
    value = str(value)
    if value == "accent":
        return theme.get("novel", default)
    if value == "stroke":
        return theme.get("stroke", default)
    if value == "text":
        return theme.get("text", default)
    if value == "input":
        return theme.get("input", default)
    if value == "output":
        return theme.get("output", default)
    if value.startswith("#"):
        return value
    return theme.get(value, default)


def _mapx(x: float, w: float, v: float) -> float:
    return x + w * float(v) / 100.0


def _mapy(y: float, h: float, v: float) -> float:
    return y + h * float(v) / 100.0


def svg_illustration(ill: dict, x: float, y: float, w: float, h: float, theme: dict, font: str = "Arial") -> str:
    """Render a scientific mini-illustration into an SVG fragment."""
    typ = str(ill.get("type", ""))
    stroke = theme.get("stroke", "#374151")
    accent = theme.get("novel", "#D97706")
    secondary = theme.get("input", "#2563EB")
    green = theme.get("output", "#4D7C0F")
    out: list[str] = []

    def line(x1, y1, x2, y2, *, color=stroke, sw=1.1, dash=None, opacity=1.0):
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        out.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.2f}"{dash_attr}/>' )

    def circle(cx, cy, r, *, fill="#FFFFFF", color=stroke, sw=.9, opacity=1.0):
        out.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}" fill="{fill}" stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.2f}"/>')

    if typ == "custom_dsl":
        for p in ill.get("primitives") or []:
            shp = p.get("shape")
            fill = _hex_role(p.get("fill"), theme, "none")
            color = _hex_role(p.get("stroke"), theme, stroke)
            sw = float(p.get("stroke_width", .9))
            opacity = float(p.get("opacity", 1.0))
            if shp == "circle":
                circle(_mapx(x,w,p.get("cx",50)), _mapy(y,h,p.get("cy",50)), min(w,h)*float(p.get("r",8))/100.0, fill=fill, color=color, sw=sw, opacity=opacity)
            elif shp == "ellipse":
                out.append(f'<ellipse cx="{_mapx(x,w,p.get("cx",50)):.1f}" cy="{_mapy(y,h,p.get("cy",50)):.1f}" rx="{w*float(p.get("rx",10))/100:.1f}" ry="{h*float(p.get("ry",7))/100:.1f}" fill="{fill}" stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.2f}"/>')
            elif shp == "rect":
                out.append(f'<rect x="{_mapx(x,w,p.get("x",10)):.1f}" y="{_mapy(y,h,p.get("y",10)):.1f}" width="{w*float(p.get("w",20))/100:.1f}" height="{h*float(p.get("h",20))/100:.1f}" rx="{min(w,h)*float(p.get("rx",2))/100:.1f}" fill="{fill}" stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.2f}"/>')
            elif shp == "line":
                line(_mapx(x,w,p.get("x1",0)), _mapy(y,h,p.get("y1",0)), _mapx(x,w,p.get("x2",100)), _mapy(y,h,p.get("y2",100)), color=color, sw=sw, dash=p.get("dash"), opacity=opacity)
            elif shp in {"polyline", "polygon"}:
                pts = p.get("points") or []
                coords = " ".join(f"{_mapx(x,w,a):.1f},{_mapy(y,h,b):.1f}" for a,b in pts)
                tag = shp
                out.append(f'<{tag} points="{coords}" fill="{fill}" stroke="{color}" stroke-width="{sw:.2f}" opacity="{opacity:.2f}"/>')
            elif shp == "text":
                txt = str(p.get("text", ""))
                fs = float(p.get("font_size", 8))
                out.append(f'<text x="{_mapx(x,w,p.get("x",50)):.1f}" y="{_mapy(y,h,p.get("y",50)):.1f}" text-anchor="middle" font-family="{font}" font-size="{fs:.1f}" fill="{color}">{txt}</text>')
        return "\n".join(out)

    if typ in {"timeseries", "sequence"}:
        bx, by = x+w*.10, y+h*.15
        line(bx, y+h*.76, x+w*.91, y+h*.76, sw=.8, opacity=.6)
        line(bx, y+h*.76, bx, y+h*.14, sw=.8, opacity=.6)
        pts=[]
        for i in range(26):
            xx=bx+(w*.76)*i/25
            yy=y+h*(.50-.13*math.sin(i*.72)-.07*math.sin(i*1.91))
            pts.append(f"{xx:.1f},{yy:.1f}")
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{secondary}" stroke-width="1.4"/>')
        return "\n".join(out)

    if typ == "feature_grid":
        for j in range(3, -1, -1):
            xx=x+w*.16+j*w*.04; yy=y+h*.18-j*h*.04
            out.append(f'<rect x="{xx:.1f}" y="{yy:.1f}" width="{w*.56:.1f}" height="{h*.55:.1f}" fill="{secondary}" fill-opacity="{.10+.08*j:.2f}" stroke="{stroke}" stroke-width=".75"/>')
        return "\n".join(out)

    if typ in {"globe_coordinates", "fibonacci_sphere", "spectral_sphere", "field_map"}:
        cx=x+w*.50; cy=y+h*.47; r=min(w,h)*.32
        circle(cx,cy,r,fill="#FFFFFF",color=stroke,sw=.85)
        for frac in (-.55,0,.55):
            ry=r*math.sqrt(max(0.0,1-frac*frac))
            out.append(f'<ellipse cx="{cx:.1f}" cy="{cy+r*frac:.1f}" rx="{ry:.1f}" ry="{r*.12:.1f}" fill="none" stroke="{stroke}" stroke-width=".45" opacity=".45"/>')
        for frac in (-.55,0,.55):
            out.append(f'<ellipse cx="{cx:.1f}" cy="{cy:.1f}" rx="{r*(.18+.58*abs(frac)):.1f}" ry="{r:.1f}" fill="none" stroke="{stroke}" stroke-width=".45" opacity=".38"/>')
        if typ == "fibonacci_sphere":
            n=int((ill.get("params") or {}).get("nodes",22))
            phi=(1+5**.5)/2
            for i in range(n):
                z=1-2*(i+.5)/n; rr=(1-z*z)**.5; theta=2*math.pi*i/phi
                px=cx+r*.83*rr*math.cos(theta); py=cy-r*.83*z
                circle(px,py,max(1.2,r*.026),fill=accent,color=accent,sw=.3)
        elif typ == "spectral_sphere":
            for k,(color,phase) in enumerate(((accent,0),(green,.9),(secondary,1.8))):
                pts=[]
                for i in range(40):
                    xx=cx-r*.78+2*r*.78*i/39
                    yy=cy+(k-1)*r*.27+r*.07*math.sin(i/39*math.pi*4+phase)
                    pts.append(f"{xx:.1f},{yy:.1f}")
                out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="1.0"/>')
        elif typ == "field_map":
            # Three low-saturation latitudinal bands suggest a spatial scalar field.
            for off,color,op in ((-.28,secondary,.18),(0,green,.20),(.28,accent,.24)):
                out.append(f'<ellipse cx="{cx:.1f}" cy="{cy+r*off:.1f}" rx="{r*.83:.1f}" ry="{r*.22:.1f}" fill="{color}" fill-opacity="{op:.2f}" stroke="none"/>')
        elif typ == "globe_coordinates":
            circle(cx+r*.42,cy-r*.08,r*.08,fill=accent,color=accent,sw=.4)
            line(cx+r*.42,cy, cx+r*.42,cy+r*.27,color=accent,sw=1.0)
        return "\n".join(out)

    if typ == "coordinate_axes":
        ox=x+w*.35; oy=y+h*.68
        line(ox,oy,x+w*.82,oy,sw=1.0); line(ox,oy,ox,y+h*.18,sw=1.0); line(ox,oy,x+w*.15,y+h*.84,sw=1.0)
        return "\n".join(out)

    if typ == "graph_network":
        pts=[(.18,.55),(.34,.28),(.50,.50),(.68,.22),(.78,.58),(.48,.78),(.28,.76)]
        xy=[(x+w*a,y+h*b) for a,b in pts]
        for a,b in ((0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,0),(1,6),(2,5),(2,4)):
            line(*xy[a],*xy[b],sw=.7,opacity=.58,dash="3 2" if (a+b)%3==0 else None)
        for i,(cx,cy) in enumerate(xy):
            circle(cx,cy,min(w,h)*.045,fill=accent if i in {2,5} else "#FFFFFF",color=stroke,sw=.8)
        return "\n".join(out)

    if typ == "attention_fan":
        q=(x+w*.50,y+h*.75)
        left=[(x+w*.17,y+h*(.20+i*.16)) for i in range(3)]
        right=[(x+w*.83,y+h*(.20+i*.16)) for i in range(3)]
        for pt in left+right:
            line(q[0],q[1],pt[0],pt[1],sw=.65,dash="3 2",opacity=.55)
            circle(pt[0],pt[1],min(w,h)*.035,fill=secondary if pt in left else green,color=stroke,sw=.6)
        circle(q[0],q[1],min(w,h)*.045,fill=accent,color=stroke,sw=.7)
        return "\n".join(out)

    if typ == "uncertainty_curve":
        line(x+w*.12,y+h*.78,x+w*.90,y+h*.78,sw=.75,opacity=.55)
        pts=[]
        for i in range(50):
            t=-3+6*i/49; val=math.exp(-.5*t*t)
            pts.append(f"{x+w*(.14+.72*i/49):.1f},{y+h*(.75-.50*val):.1f}")
        out.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{secondary}" stroke-width="1.35"/>')
        for frac in (.38,.62):
            xx=x+w*frac; line(xx,y+h*.45,xx,y+h*.79,sw=.65,dash="3 2",opacity=.55)
        return "\n".join(out)

    if typ == "satellite_occultation":
        # Satellite body + panels.
        sx=x+w*.23; sy=y+h*.24
        out.append(f'<rect x="{sx:.1f}" y="{sy:.1f}" width="{w*.13:.1f}" height="{h*.10:.1f}" fill="{secondary}" fill-opacity=".42" stroke="{stroke}" stroke-width=".7"/>')
        for d in (-1,1):
            out.append(f'<rect x="{sx+d*w*.11-(w*.08 if d<0 else 0):.1f}" y="{sy-h*.01:.1f}" width="{w*.08:.1f}" height="{h*.12:.1f}" fill="{secondary}" fill-opacity=".24" stroke="{stroke}" stroke-width=".6"/>')
        # Earth limb.
        out.append(f'<path d="M {x+w*.10:.1f},{y+h*.87:.1f} Q {x+w*.50:.1f},{y+h*.56:.1f} {x+w*.92:.1f},{y+h*.87:.1f}" fill="none" stroke="{green}" stroke-width="2.0" opacity=".55"/>')
        for i in range(3):
            out.append(f'<path d="M {sx+w*.06:.1f},{sy+h*.08:.1f} Q {x+w*(.48+i*.07):.1f},{y+h*(.42+i*.06):.1f} {x+w*(.71+i*.05):.1f},{y+h*.80:.1f}" fill="none" stroke="{stroke}" stroke-width=".65" stroke-dasharray="3 2" opacity=".55"/>')
        return "\n".join(out)

    if typ == "radar_dish":
        cx=x+w*.28; cy=y+h*.43; rr=min(w,h)*.19
        out.append(f'<path d="M {cx-rr:.1f},{cy-rr*.45:.1f} Q {cx:.1f},{cy+rr*.70:.1f} {cx+rr:.1f},{cy-rr*.45:.1f}" fill="#FFFFFF" stroke="{stroke}" stroke-width="1.0"/>')
        line(cx,cy+rr*.45,cx-w*.02,y+h*.78,sw=1.1); line(cx,cy+rr*.45,cx+w*.12,y+h*.78,sw=1.1)
        for j in range(3):
            out.append(f'<path d="M {cx+rr*.50:.1f},{cy-rr*.45:.1f} Q {cx+rr*(1.25+j*.35):.1f},{cy-rr*(1.05+j*.15):.1f} {cx+rr*(1.45+j*.42):.1f},{cy-rr*.15:.1f}" fill="none" stroke="{secondary}" stroke-width=".8" opacity=".65"/>')
        return "\n".join(out)

    if typ == "sun_geomagnetic":
        scx=x+w*.25; scy=y+h*.45; sr=min(w,h)*.12
        circle(scx,scy,sr,fill="#FFF7D6",color=accent,sw=1.0)
        for i in range(8):
            a=i*math.pi/4; line(scx+math.cos(a)*sr*1.35,scy+math.sin(a)*sr*1.35,scx+math.cos(a)*sr*1.85,scy+math.sin(a)*sr*1.85,color=accent,sw=.85)
        ecx=x+w*.67; ecy=y+h*.47; er=min(w,h)*.10
        circle(ecx,ecy,er,fill="#FFFFFF",color=secondary,sw=.9)
        for s in (-1,1):
            out.append(f'<path d="M {ecx:.1f},{ecy-er*1.7:.1f} C {ecx+s*er*2.2:.1f},{ecy-er*.9:.1f} {ecx+s*er*2.2:.1f},{ecy+er*.9:.1f} {ecx:.1f},{ecy+er*1.7:.1f}" fill="none" stroke="{secondary}" stroke-width=".75" opacity=".7"/>')
        return "\n".join(out)

    if typ == "point_cloud":
        pts=[(.18,.28),(.30,.60),(.43,.36),(.55,.70),(.68,.25),(.80,.54),(.60,.45),(.35,.78),(.73,.75)]
        for i,(a,b) in enumerate(pts):
            circle(x+w*a,y+h*b,min(w,h)*(.022+.006*(i%3)),fill=accent if i%3==0 else secondary,color="none",sw=0)
        return "\n".join(out)

    return ""


def illustration_region(box: dict, ill: dict) -> tuple[float,float,float,float]:
    """Return a local illustration rectangle based on composition."""
    x,y,w,h = box["x"],box["y"],box["w"],box["h"]
    comp = str(ill.get("composition", "illustration-top"))
    if comp == "illustration-left":
        return x+w*.05, y+h*.18, w*.31, h*.62
    if comp == "illustration-right":
        return x+w*.64, y+h*.18, w*.31, h*.62
    if comp == "illustration-bottom":
        return x+w*.18, y+h*.57, w*.64, h*.34
    if comp == "illustration-center":
        return x+w*.14, y+h*.22, w*.72, h*.58
    return x+w*.16, y+h*.10, w*.68, h*.43
