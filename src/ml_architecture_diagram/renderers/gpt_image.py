from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from ..ai_prompt import architecture_prompt
from .svg import render_svg


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "GPT Image rendering requires the OpenAI Python SDK. "
            "Install with: pip install -e '.[ai]'"
        ) from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")
    return OpenAI()


def _save_result(result, output: Path) -> Path:
    data = getattr(result, "data", None)
    if not data:
        raise RuntimeError("Image API returned no image data.")
    b64 = getattr(data[0], "b64_json", None)
    if not b64:
        raise RuntimeError("Image API response did not contain b64_json.")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(base64.b64decode(b64))
    return output


def _reference_png(spec: dict, path: Path, width: int = 2048) -> Path:
    try:
        import cairosvg
    except ImportError as exc:
        raise RuntimeError(
            "Reference-guided GPT Image rendering requires CairoSVG. "
            "Install with: pip install -e '.[ai]'"
        ) from exc
    svg = path.with_suffix(".svg")
    render_svg(spec, svg)
    cairosvg.svg2png(url=str(svg), write_to=str(path), output_width=width)
    try:
        svg.unlink()
    except OSError:
        pass
    return path


def render_gpt_image(
    spec: dict,
    output: str | Path,
    *,
    mode: str = "reference",
    model: str = "gpt-image-2",
    quality: str = "medium",
    size: str = "2048x1152",
    style: str = "publication-rich",
    reference_image: str | Path | None = None,
) -> Path:
    """Render an Architecture IR with a GPT Image model.

    Modes:
      - generate: prompt-only generation; visually flexible, topology less reliable.
      - reference: edit/reference workflow using a deterministic diagram as blueprint.

    The output is raster by design. Keep SVG/draw.io/PPTX exports as the source of
    truth when exact topology or editability matters.
    """
    if mode not in {"generate", "reference"}:
        raise ValueError("mode must be 'generate' or 'reference'")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt = architecture_prompt(spec, mode=mode, style=style)
    client = _client()

    kwargs = {
        "model": model,
        "prompt": prompt,
        "quality": quality,
        "size": size,
    }

    if mode == "generate":
        result = client.images.generate(**kwargs)
        return _save_result(result, output)

    temp_created = False
    ref_path: Path
    if reference_image is not None:
        ref_path = Path(reference_image)
        if not ref_path.exists():
            raise FileNotFoundError(ref_path)
    else:
        fd, tmp = tempfile.mkstemp(prefix="ml_arch_reference_", suffix=".png")
        os.close(fd)
        ref_path = Path(tmp)
        _reference_png(spec, ref_path)
        temp_created = True

    try:
        with ref_path.open("rb") as image_file:
            result = client.images.edit(image=image_file, **kwargs)
        return _save_result(result, output)
    finally:
        if temp_created:
            try:
                ref_path.unlink()
            except OSError:
                pass
