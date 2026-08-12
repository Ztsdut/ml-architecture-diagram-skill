from __future__ import annotations

import argparse
from pathlib import Path
from .spec import load_spec, validate_spec
from .renderers.svg import render_svg
from .renderers.drawio import render_drawio
from .ai_prompt import architecture_prompt


def _render(spec: dict, source: Path, outdir: Path, formats: list[str]) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    stem = source.stem
    outputs: list[Path] = []
    svg_path = outdir / f"{stem}.svg"
    need_svg = any(f in {"svg", "png", "pdf"} for f in formats)
    if need_svg:
        render_svg(spec, svg_path)
        if "svg" in formats:
            outputs.append(svg_path)
    if "drawio" in formats:
        p = outdir / f"{stem}.drawio"
        render_drawio(spec, p)
        outputs.append(p)
    if "pptx" in formats:
        from .renderers.pptx import render_pptx
        p = outdir / f"{stem}.pptx"
        render_pptx(spec, p)
        outputs.append(p)
    if "png" in formats or "pdf" in formats:
        try:
            import cairosvg
        except ImportError as exc:
            raise RuntimeError("PNG/PDF export requires CairoSVG. Install with: pip install -e '.[export]'") from exc
        if "png" in formats:
            p = outdir / f"{stem}.png"
            cairosvg.svg2png(url=str(svg_path), write_to=str(p), output_width=2400)
            outputs.append(p)
        if "pdf" in formats:
            p = outdir / f"{stem}.pdf"
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(p))
            outputs.append(p)
    if need_svg and "svg" not in formats and svg_path.exists():
        svg_path.unlink()
    return outputs


def cmd_validate(args) -> int:
    spec = load_spec(args.spec)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print("OK: architecture spec is valid")
    return 0


def cmd_render(args) -> int:
    source = Path(args.spec)
    spec = load_spec(source)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    outs = _render(spec, source, Path(args.outdir), args.format)
    for p in outs:
        print(p)
    return 0



def cmd_prompt(args) -> int:
    spec = load_spec(args.spec)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print(architecture_prompt(spec, mode=args.mode, style=args.style))
    return 0


def cmd_ai_render(args) -> int:
    source = Path(args.spec)
    spec = load_spec(source)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    from .renderers.gpt_image import render_gpt_image
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.gpt-image.png"
    p = render_gpt_image(
        spec, output, mode=args.mode, model=args.model, quality=args.quality,
        size=args.size, style=args.style, reference_image=args.reference_image,
        prompt_output=args.prompt_output, reference_output=args.reference_output,
    )
    print(p)
    return 0



def cmd_parse_pytorch(args) -> int:
    from .parsers.pytorch_ast import parse_pytorch_file, dump_spec
    spec = parse_pytorch_file(args.model, class_name=args.class_name, detail=args.detail)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    source = Path(args.model)
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.architecture.yaml"
    dump_spec(spec, output)
    print(output)
    unresolved = spec.get("metadata", {}).get("unresolved", [])
    for item in unresolved:
        print(f"NOTE: {item}")
    if args.render:
        outs = _render(spec, output, Path(args.outdir), args.render)
        for out in outs:
            print(out)
    return 0



def cmd_parse_keras(args) -> int:
    from .parsers.keras_ast import parse_keras_file
    import yaml
    spec = parse_keras_file(args.model, class_name=args.class_name, detail=args.detail)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    source = Path(args.model)
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.architecture.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    for item in spec.get("metadata", {}).get("unresolved", []):
        print(f"NOTE: {item}")
    if args.render:
        outs = _render(spec, output, Path(args.outdir), args.render)
        for out in outs:
            print(out)
    return 0



def cmd_prepare_ai(args) -> int:
    spec = load_spec(args.spec)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    from .ai_bundle import prepare_ai_bundle
    source = Path(args.spec)
    outputs = prepare_ai_bundle(spec, args.outdir, stem=source.stem, mode=args.mode, style=args.style, width=args.width)
    for p in outputs.values():
        print(p)
    return 0



def cmd_lint_publication(args) -> int:
    from .publication_quality import lint_publication
    spec = load_spec(args.spec)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    issues = lint_publication(spec)
    if not issues:
        print("OK: no publication-design warnings")
        return 0
    for item in issues:
        print(f"{item.severity.upper()} [{item.code}]: {item.message}")
    return 1 if args.strict and any(x.severity == "warning" for x in issues) else 0



def cmd_layout_metrics(args) -> int:
    import json
    from .publication_design import compile_publication_spec
    from .layout import layout_figure
    from .routing import count_route_crossings
    spec = load_spec(args.spec)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    pub = compile_publication_spec(spec)
    layout = layout_figure(pub)
    panels=[]
    for i,pl in enumerate(layout.get("panels", [])):
        jm=(pl.get("layout_meta") or {}).get("joint_optimization") or {}
        panels.append({
            "panel": i,
            "width": round(float(pl.get("width",0)),3),
            "height": round(float(pl.get("height",0)),3),
            "edge_crossings": int(count_route_crossings(pl.get("edges",[]))),
            "joint_optimization": jm,
        })
    payload={
        "figure_width": round(float(layout.get("width",0)),3),
        "figure_height": round(float(layout.get("height",0)),3),
        "panels": panels,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0

def cmd_compile_publication(args) -> int:
    import yaml
    from .publication_design import compile_publication_spec
    source = Path(args.spec)
    spec = load_spec(source)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    if args.family:
        spec.setdefault("metadata", {})["architecture_family"] = args.family
    enriched = compile_publication_spec(spec)
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.publication.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(enriched, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    print("family:", enriched.get("metadata", {}).get("architecture_family", "generic"))
    print("preset:", enriched.get("figure", {}).get("layout_preset", "publication_flow"))
    return 0



def cmd_compile_illustrations(args) -> int:
    import yaml
    from .scientific_illustration import compile_scientific_illustrations, validate_illustration
    source = Path(args.spec)
    spec = load_spec(source)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    enriched = compile_scientific_illustrations(spec, max_auto=args.budget)
    for node in enriched.get("nodes", []):
        if node.get("illustration"):
            for err in validate_illustration(node["illustration"]):
                print(f"ERROR [{node.get('id')}]: {err}")
                return 1
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.illustrated.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(enriched, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    plan=enriched.get("metadata", {}).get("scientific_illustration", {})
    print("illustrations:", ", ".join(plan.get("explicit_nodes", []) + plan.get("auto_nodes", [])) or "none")
    return 0

def cmd_compile_visual(args) -> int:
    import yaml
    from .visual_grammar import compile_visual_spec, visual_inventory
    source = Path(args.spec)
    spec = load_spec(source)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    if args.family:
        spec.setdefault("metadata", {})["architecture_family"] = args.family
    if args.preset:
        spec.setdefault("figure", {})["layout_preset"] = args.preset
    enriched = compile_visual_spec(spec)
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.visual.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(enriched, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    print("family:", enriched.get("metadata", {}).get("architecture_family", "generic"))
    print("visuals:", ", ".join(f"{k}={v}" for k, v in visual_inventory(enriched).items()))
    return 0


def cmd_apply_review(args) -> int:
    import yaml
    from .semantic_review import apply_semantic_review
    source = Path(args.spec)
    spec = load_spec(source)
    patch = load_spec(args.review)
    reviewed = apply_semantic_review(spec, patch)
    errors = validate_spec(reviewed)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.reviewed.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(reviewed, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    return 0



def cmd_detail_review(args) -> int:
    import yaml
    from .detail_panels import detail_review_skeleton, select_proposed_block
    source = Path(args.spec)
    spec = load_spec(source)
    errors = validate_spec(spec)
    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    candidate = select_proposed_block(spec)
    patch = detail_review_skeleton(spec)
    output = Path(args.output) if args.output else Path(args.outdir) / f"{source.stem}.detail-review.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(patch, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output)
    if candidate:
        print("candidate:", candidate.get("id"), "-", candidate.get("label"))
    else:
        print("candidate: none")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ml-arch", description="Render publication-oriented ML architecture diagrams")
    sub = p.add_subparsers(dest="command", required=True)
    v = sub.add_parser("validate", help="validate an Architecture IR YAML/JSON file")
    v.add_argument("spec")
    v.set_defaults(func=cmd_validate)
    pp = sub.add_parser("parse-pytorch", help="statically inspect a PyTorch model and create Architecture IR")
    pp.add_argument("model", help="path to a Python source file")
    pp.add_argument("--class", dest="class_name", default=None, help="nn.Module class to parse; auto-detected when omitted")
    pp.add_argument("--detail", choices=["system", "block", "operation"], default="system")
    pp.add_argument("--output", default=None, help="output YAML path")
    pp.add_argument("--outdir", default="build")
    pp.add_argument("--render", nargs="+", choices=["svg", "drawio", "pptx", "png", "pdf"], default=None, help="optionally render immediately")
    pp.set_defaults(func=cmd_parse_pytorch)

    pk = sub.add_parser("parse-keras", help="statically inspect a TensorFlow/Keras model and create Architecture IR")
    pk.add_argument("model", help="path to a Python source file")
    pk.add_argument("--class", dest="class_name", default=None, help="keras.Model/Layer subclass to parse; functional/Sequential is auto-detected when omitted")
    pk.add_argument("--detail", choices=["system", "block", "operation"], default="system")
    pk.add_argument("--output", default=None, help="output YAML path")
    pk.add_argument("--outdir", default="build")
    pk.add_argument("--render", nargs="+", choices=["svg", "drawio", "pptx", "png", "pdf"], default=None, help="optionally render immediately")
    pk.set_defaults(func=cmd_parse_keras)

    lp = sub.add_parser("lint-publication", help="check manuscript composition before final rendering")
    lp.add_argument("spec")
    lp.add_argument("--strict", action="store_true", help="return non-zero when warnings are present")
    lp.set_defaults(func=cmd_lint_publication)

    lm = sub.add_parser("layout-metrics", help="report routed publication-layout metrics and joint-optimization improvement")
    lm.add_argument("spec")
    lm.set_defaults(func=cmd_layout_metrics)

    ar = sub.add_parser("apply-review", help="apply an agent/human semantic-review patch without changing parsed topology")
    ar.add_argument("spec", help="parsed Architecture IR")
    ar.add_argument("review", help="semantic review patch YAML/JSON")
    ar.add_argument("--output", default=None)
    ar.add_argument("--outdir", default="build")
    ar.set_defaults(func=cmd_apply_review)

    dr = sub.add_parser("detail-review", help="select the proposed-block candidate and write a semantic-review skeleton for its verified internals")
    dr.add_argument("spec")
    dr.add_argument("--output", default=None)
    dr.add_argument("--outdir", default="build")
    dr.set_defaults(func=cmd_detail_review)

    cp = sub.add_parser("compile-publication", help="derive a compact paper-facing Publication Design Blueprint")
    cp.add_argument("spec")
    cp.add_argument("--family", default=None, help="override architecture family")
    cp.add_argument("--output", default=None)
    cp.add_argument("--outdir", default="build")
    cp.set_defaults(func=cmd_compile_publication)

    cv = sub.add_parser("compile-visual", help="infer Scientific Visual Grammar and write an editable enriched IR")
    cv.add_argument("spec")
    cv.add_argument("--family", default=None, help="override inferred architecture family, e.g. transformer, cnn, unet, gnn, moe")
    cv.add_argument("--preset", default=None, help="override layout preset")
    cv.add_argument("--output", default=None)
    cv.add_argument("--outdir", default="build")
    cv.set_defaults(func=cmd_compile_visual)

    ci = sub.add_parser("compile-illustrations", help="infer or preserve scientific mini-illustrations and write an enriched IR")
    ci.add_argument("spec")
    ci.add_argument("--budget", type=int, default=6, help="maximum number of automatic mini-illustrations; explicit review plans are preserved")
    ci.add_argument("--output", default=None)
    ci.add_argument("--outdir", default="build")
    ci.set_defaults(func=cmd_compile_illustrations)

    r = sub.add_parser("render", help="render an Architecture IR YAML/JSON file")
    r.add_argument("spec")
    r.add_argument("--format", nargs="+", choices=["svg", "drawio", "pptx", "png", "pdf"], default=["svg"])
    r.add_argument("--outdir", default="build")
    r.set_defaults(func=cmd_render)

    pr = sub.add_parser("prompt", help="print the image-generation prompt derived from Architecture IR")
    pr.add_argument("spec")
    pr.add_argument("--mode", choices=["generate", "reference"], default="reference")
    pr.add_argument("--style", default="publication-rich")
    pr.set_defaults(func=cmd_prompt)

    pa = sub.add_parser("prepare-ai", help="create a reference PNG/SVG and constrained prompt without calling an image API")
    pa.add_argument("spec")
    pa.add_argument("--mode", choices=["generate", "reference"], default="reference")
    pa.add_argument("--style", default="publication-rich")
    pa.add_argument("--width", type=int, default=2048)
    pa.add_argument("--outdir", default="build/ai-bundle")
    pa.set_defaults(func=cmd_prepare_ai)

    ai = sub.add_parser("ai-render", help="render a richer raster diagram with GPT Image")
    ai.add_argument("spec")
    ai.add_argument("--mode", choices=["generate", "reference"], default="reference")
    ai.add_argument("--model", default="gpt-image-2")
    ai.add_argument("--quality", choices=["low", "medium", "high", "auto"], default="medium")
    ai.add_argument("--size", default="2048x1152")
    ai.add_argument("--style", default="publication-rich")
    ai.add_argument("--reference-image", default=None, help="optional existing reference PNG/JPEG; otherwise one is rendered automatically")
    ai.add_argument("--prompt-output", default=None, help="save the exact prompt used for reproducibility")
    ai.add_argument("--reference-output", default=None, help="keep the deterministic reference PNG instead of using a temporary file")
    ai.add_argument("--output", default=None)
    ai.add_argument("--outdir", default="build")
    ai.set_defaults(func=cmd_ai_render)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
