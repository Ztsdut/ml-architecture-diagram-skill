# Contributing

Thanks for helping improve ML Architecture Diagram Skill.

## Good contributions

Useful contributions include:

- architecture examples that reveal a missing visual pattern;
- renderer improvements that preserve editability;
- better DAG layout and edge routing;
- framework-specific static/runtime parser adapters and model-inspection guidance;
- accessibility and grayscale improvements;
- tests for residual, branching, recurrent, shared-weight, or training-only graphs;
- fixes to the Architecture IR schema.

## Design constraints

Please keep these principles intact:

1. The real computation path is more important than a model class listing.
2. Paper figures should compress implementation detail deliberately.
3. The IR must remain framework-neutral.
4. SVG/draw.io/PPTX should remain editable.
5. Renderers should not silently invent missing model information.
6. Do not copy third-party figure assets into the repository unless licensing and attribution are explicit and compatible.
7. Static parsers must not execute the target model file by default.
8. Unsupported dynamic topology should produce an unresolved note or require agent/manual review rather than be guessed.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -e '.[dev,export]'
pytest
```

Render all examples:

```bash
python scripts/render_examples.py
```

## Pull requests

Please include:

- a short description of the architecture/renderer issue;
- before/after screenshots or SVGs for visual changes;
- a test or example spec when practical;
- confirmation that `pytest` passes.

For changes to `SKILL.md`, add at least one example/eval prompt that would fail under the old wording and should improve under the new wording.
