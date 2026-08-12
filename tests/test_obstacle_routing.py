from pathlib import Path

from ml_architecture_diagram.spec import load_spec
from ml_architecture_diagram.publication_design import compile_publication_spec
from ml_architecture_diagram.layout import layout_figure
from ml_architecture_diagram.publication_quality import lint_publication
from ml_architecture_diagram.routing import route_node_intersections, route_stage_intersections

ROOT=Path(__file__).resolve().parents[1]


def test_demo_routes_avoid_unrelated_nodes_and_stages():
    raw=load_spec(ROOT/'examples'/'scientific_illustration_framework.yaml')
    pub=compile_publication_spec(raw)
    pl=layout_figure(pub)['panels'][0]
    for edge in pl['edges']:
        assert not route_node_intersections(edge,pl['positions'])
        assert not route_stage_intersections(edge,pl['positions'],pl['groups'])
    cond=next(e for e in pl['edges'] if e['from']=='coordinate_query' and e['to']=='decoder')
    assert cond.get('_route_points') and len(cond['_route_points']) >= 4
    # Query conditioning should enter through the decoder's vertical port rather than
    # becoming visually indistinguishable from the left-to-right main trunk.
    last=cond['_route_points'][-1]
    decoder=pl['positions']['decoder']
    assert abs(last[0]-(decoder['x']+decoder['w']/2)) < 1e-3
    assert abs(last[1]-(decoder['y']+decoder['h'])) < 1e-3


def test_linter_has_no_routing_errors_for_demo():
    raw=load_spec(ROOT/'examples'/'scientific_illustration_framework.yaml')
    errors=[i for i in lint_publication(raw) if i.severity=='error']
    assert not errors
