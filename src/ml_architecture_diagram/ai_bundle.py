from __future__ import annotations

from pathlib import Path

from .ai_prompt import architecture_prompt
from .renderers.svg import render_svg


def prepare_ai_bundle(
    spec: dict,
    outdir: str | Path,
    *,
    stem: str = "architecture",
    mode: str = "reference",
    style: str = "publication-rich",
    width: int = 2048,
) -> dict[str, Path]:
    """Create a portable AI-render bundle without calling any paid API.

    The bundle contains the deterministic structural blueprint and the exact prompt.
    It can be used with GPT Image 2 or another image-edit backend.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    svg = outdir / f"{stem}.reference.svg"
    png = outdir / f"{stem}.reference.png"
    prompt_path = outdir / f"{stem}.prompt.txt"
    render_svg(spec, svg)
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError("AI bundle PNG creation requires CairoSVG. Install with: pip install -e '.[export]'") from exc
    cairosvg.svg2png(url=str(svg), write_to=str(png), output_width=width)
    prompt_path.write_text(architecture_prompt(spec, mode=mode, style=style) + "\n", encoding="utf-8")
    return {"svg": svg, "png": png, "prompt": prompt_path}
