from pathlib import Path

from ml_architecture_diagram.parsers.pytorch_ast import parse_pytorch_file
from ml_architecture_diagram.semantic_review import apply_semantic_review
from ml_architecture_diagram.spec import load_spec, validate_spec
from ml_architecture_diagram.visual_grammar import infer_architecture_family
from ml_architecture_diagram.publication_design import compile_publication_spec
from ml_architecture_diagram.publication_quality import lint_publication

ROOT = Path(__file__).resolve().parents[1]


def _topology(spec):
    return (
        [n['id'] for n in spec['nodes']],
        [(e['from'], e['to'], e.get('type','main')) for e in spec['edges']],
    )


def test_complex_static_parse_and_review_preserve_exact_topology():
    model = ROOT / 'examples' / 'pytorch' / 'complex_multibranch.py'
    raw = parse_pytorch_file(model, class_name='ComplexFusionNet', detail='system')
    assert validate_spec(raw) == []
    assert not any(n.get('label') == 'Conditional' for n in raw['nodes'])
    assert not any(n.get('id') == 'query_encoder' and n.get('repeat', 1) > 1 for n in raw['nodes'])
    labels = {n['label'] for n in raw['nodes']}
    assert 'Spatial Attention' in labels and 'History Attention' in labels
    assert {'Main Logits', 'Aux Logits', 'Gate', 'Attention Weights'} <= labels
    assert infer_architecture_family(raw) == 'multimodal'

    patch = load_spec(ROOT / 'examples' / 'pytorch' / 'complex_multibranch.review.yaml')
    reviewed = apply_semantic_review(raw, patch)
    assert _topology(reviewed) == _topology(raw)
    assert reviewed['metadata']['semantic_review']['applied'] is True


def test_publication_view_compacts_implementation_detail_with_provenance():
    model = ROOT / 'examples' / 'pytorch' / 'complex_multibranch.py'
    raw = parse_pytorch_file(model, class_name='ComplexFusionNet', detail='system')
    patch = load_spec(ROOT / 'examples' / 'pytorch' / 'complex_multibranch.review.yaml')
    reviewed = apply_semantic_review(raw, patch)
    pub = compile_publication_spec(reviewed)
    assert pub['figure']['layout_preset'] == 'publication_framework'
    assert pub['metadata']['publication_view']['collapsed_nodes']
    assert any('feature fusion' in n.get('label','').lower() for n in pub['nodes'])
    # Diagnostic attention weights remain in exact IR but are hidden from overview.
    assert any(n['label'] == 'Attention Weights' for n in reviewed['nodes'])
    assert not any(n['label'] == 'Attention Weights' for n in pub['nodes'])
    assert any(n['label'] == 'Gate' for n in reviewed['nodes'])
    assert not any(n['label'] == 'Gate' for n in pub['nodes'])
    assert not [i for i in lint_publication(pub) if i.severity == 'warning']
