from pathlib import Path
from ml_architecture_diagram.spec import load_spec
from ml_architecture_diagram.publication_quality import lint_publication

ROOT=Path(__file__).resolve().parents[1]


def test_linter_returns_structured_issues():
    spec=load_spec(ROOT/'examples'/'transformer_encoder.yaml')
    issues=lint_publication(spec)
    assert all(i.code and i.severity and i.message for i in issues)
    assert not any(i.code=='figure-title' for i in issues)
