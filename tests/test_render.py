from pathlib import Path
from ml_architecture_diagram.spec import load_spec
from ml_architecture_diagram.renderers.svg import render_svg
from ml_architecture_diagram.renderers.drawio import render_drawio

ROOT = Path(__file__).resolve().parents[1]


def test_svg_and_drawio_render(tmp_path):
    spec = load_spec(ROOT / "examples" / "multimodal_fusion.yaml")
    svg = render_svg(spec, tmp_path / "figure.svg")
    drawio = render_drawio(spec, tmp_path / "figure.drawio")
    assert svg.exists() and svg.stat().st_size > 500
    assert drawio.exists() and drawio.stat().st_size > 500
    assert "Feature Fusion" in svg.read_text(encoding="utf-8")
    assert "mxGraphModel" in drawio.read_text(encoding="utf-8")
