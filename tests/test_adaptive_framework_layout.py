from pathlib import Path

from ml_architecture_diagram.spec import load_spec
from ml_architecture_diagram.publication_design import compile_publication_spec
from ml_architecture_diagram.layout import layout_figure
from ml_architecture_diagram.publication_quality import lint_publication

ROOT = Path(__file__).resolve().parents[1]


def overlap(a, b, pad=0.0):
    return not (
        a['x'] + a['w'] + pad <= b['x'] or
        b['x'] + b['w'] + pad <= a['x'] or
        a['y'] + a['h'] + pad <= b['y'] or
        b['y'] + b['h'] + pad <= a['y']
    )


def test_scientific_illustration_demo_has_no_node_or_stage_overlap():
    raw = load_spec(ROOT / 'examples' / 'scientific_illustration_framework.yaml')
    pub = compile_publication_spec(raw)
    layout = layout_figure(pub)
    pl = layout['panels'][0]
    boxes = list(pl['positions'].items())
    for i, (ida, a) in enumerate(boxes):
        for idb, b in boxes[i + 1:]:
            assert not overlap(a, b, 2.0), f'{ida} overlaps {idb}'
    groups = pl['groups']
    for i, a in enumerate(groups):
        for b in groups[i + 1:]:
            assert not overlap(a, b, 2.0), f"{a['id']} overlaps {b['id']}"
    assert pl.get('layout_meta', {}).get('framework_core_mode') == 'vertical'


def test_framework_stage_width_grows_for_long_decoder_and_outputs():
    raw = load_spec(ROOT / 'examples' / 'scientific_illustration_framework.yaml')
    pub = compile_publication_spec(raw)
    layout = layout_figure(pub)
    pl = layout['panels'][0]
    meta = pl['layout_meta']
    # Output stage must reserve separate columns rather than placing decoder on top of outputs.
    assert meta['stage_widths']['stage_outputs'] > 430.0
    decoder = pl['positions']['decoder']
    field = pl['positions']['field_output']
    uncertainty = pl['positions']['uncertainty_output']
    assert decoder['x'] + decoder['w'] < field['x']
    assert decoder['x'] + decoder['w'] < uncertainty['x']


def test_publication_lint_reports_no_geometry_errors_for_demo():
    raw = load_spec(ROOT / 'examples' / 'scientific_illustration_framework.yaml')
    codes = [i.code for i in lint_publication(raw)]
    assert 'node-overlap' not in codes
    assert 'stage-overlap' not in codes
