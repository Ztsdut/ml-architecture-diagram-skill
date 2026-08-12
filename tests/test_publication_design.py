from pathlib import Path

from ml_architecture_diagram.spec import load_spec
from ml_architecture_diagram.publication_design import compile_publication_spec
from ml_architecture_diagram.renderers.svg import render_svg
from ml_architecture_diagram.ai_prompt import architecture_prompt

ROOT = Path(__file__).resolve().parents[1]


def _topology(spec):
    return (
        [(n['id'], n.get('label'), n.get('repeat', 1)) for n in spec['nodes']],
        [(e['from'], e['to'], e.get('type', 'main'), e.get('label')) for e in spec['edges']],
    )


def test_publication_compile_preserves_transformer_topology():
    raw = load_spec(ROOT / 'examples' / 'transformer_encoder.yaml')
    pub = compile_publication_spec(raw)
    assert _topology(raw) == _topology(pub)
    assert pub['metadata']['publication_design_version'] == '2.0'
    assert pub['figure']['layout_preset'] == 'publication_transformer'
    assert pub['figure']['show_title'] is False


def test_unet_gets_scale_aware_feature_tensors():
    raw = load_spec(ROOT / 'examples' / 'unet_encoder_decoder.yaml')
    pub = compile_publication_spec(raw)
    enc = [n for n in pub['nodes'] if 'Encoder' in n['label']]
    assert all(n['visual']['type'] == 'feature_tensor' for n in enc)
    assert enc[1]['visual']['spatial_scale'] < enc[0]['visual']['spatial_scale']
    assert enc[1]['visual']['channel_scale'] > enc[0]['visual']['channel_scale']


def test_publication_svg_omits_slide_banner(tmp_path):
    raw = load_spec(ROOT / 'examples' / 'transformer_encoder.yaml')
    out = render_svg(raw, tmp_path / 'x.svg')
    text = out.read_text(encoding='utf-8')
    assert 'SCIENTIFIC VISUAL GRAMMAR' not in text
    assert 'Transformer encoder model</text>' not in text
    assert '(a) Overall architecture' in text
    assert 'Multi-Head' in text and 'Self-Attention' in text


def test_ai_prompt_uses_publication_direction():
    raw = load_spec(ROOT / 'examples' / 'multimodal_fusion.yaml')
    prompt = architecture_prompt(raw, mode='reference')
    assert 'do NOT make a presentation slide' in prompt
    assert 'publication_multilane' not in prompt  # layout internals are not user-facing prose
    assert 'visual_grammar=input_tensor' in prompt
