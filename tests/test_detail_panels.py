from pathlib import Path

from ml_architecture_diagram.detail_panels import compile_detail_panels, select_proposed_block
from ml_architecture_diagram.publication_design import compile_publication_spec
from ml_architecture_diagram.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]


def _overview_topology(spec):
    detail_ids = {n['id'] for n in spec.get('nodes', []) if n.get('_detail_panel')}
    nodes = {n['id'] for n in spec.get('nodes', []) if n['id'] not in detail_ids}
    edges = {(e['from'], e['to'], e.get('type','main')) for e in spec.get('edges', []) if e['from'] in nodes and e['to'] in nodes}
    return nodes, edges


def test_selects_explicit_novel_detail_candidate():
    spec = {
        'nodes': [
            {'id':'x','label':'Input','role':'input','kind':'data'},
            {'id':'block','label':'Proposed spectral operator','role':'novel','kind':'module','repeat':4,
             'detail': {'nodes':[{'id':'in','label':'Block input'},{'id':'out','label':'Block output'}]}},
            {'id':'y','label':'Output','role':'output','kind':'output'},
        ],
        'edges': [{'from':'x','to':'block'},{'from':'block','to':'y'}],
    }
    assert select_proposed_block(spec)['id'] == 'block'


def test_explicit_detail_adds_second_panel_without_changing_overview():
    raw = load_spec(ROOT / 'examples' / 'proposed_block_detail.yaml')
    before = _overview_topology(raw)
    pub = compile_publication_spec(raw)
    assert before == _overview_topology(pub)
    assert len(pub.get('panels', [])) == 2
    assert pub['metadata']['detail_panel']['status'] == 'compiled'
    assert pub['metadata']['detail_panel']['detail_source'] == 'explicit'
    assert any(n.get('_detail_panel') for n in pub['nodes'])
    assert pub['figure']['panel_layout'] == 'vertical'


def test_unknown_custom_block_requests_semantic_review_instead_of_inventing():
    spec = {
        'figure': {}, 'metadata': {'architecture_family':'generic'},
        'nodes': [
            {'id':'x','label':'Input','role':'input','kind':'data'},
            {'id':'m','label':'Custom proposed block','role':'novel','kind':'module'},
            {'id':'y','label':'Output','role':'output','kind':'output'},
        ],
        'edges': [{'from':'x','to':'m'},{'from':'m','to':'y'}],
    }
    out = compile_detail_panels(spec)
    assert out['metadata']['detail_panel']['status'] == 'semantic_review_required'
    assert not out.get('panels')


def test_transformer_single_panel_gets_safe_detail_template():
    spec = {
        'figure': {'direction':'LR'}, 'metadata': {'architecture_family':'transformer'},
        'nodes': [
            {'id':'x','label':'Tokens','role':'input','kind':'data'},
            {'id':'m','label':'Transformer Encoder','role':'novel','kind':'module','repeat':6},
            {'id':'y','label':'Prediction','role':'output','kind':'output'},
        ],
        'edges': [{'from':'x','to':'m'},{'from':'m','to':'y'}],
    }
    out = compile_detail_panels(spec)
    assert len(out['panels']) == 2
    assert out['metadata']['detail_panel']['detail_source'] == 'template'
    assert sum(1 for n in out['nodes'] if n.get('_detail_panel')) == 8
