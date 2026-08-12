# AI-assisted rendering with GPT Image

The deterministic SVG/draw.io/PPTX renderers are the **structural source of truth**. Image-generation models are an optional visual layer for users who want a richer publication aesthetic.

## Why use two rendering paths?

Deterministic rendering is excellent at exact topology, text, repeated-block counts, editable connectors, and reproducibility. Image generation is much stronger at visual hierarchy, semantic pictograms, dimensional cues, refined grouping, and overall aesthetic coherence. Neither should pretend to replace the other.

## Modes

### `vector`
Use SVG/draw.io/PPTX when exact labels and editability are the priority.

### `ai-generate`
Architecture IR is converted into a fidelity-first prompt and sent to `gpt-image-2`. This can produce the richest visual exploration, but it may change technical details. Treat it as a concept render unless verified manually.

### `ai-reference` — recommended
The renderer first creates a deterministic architecture image, then sends that image plus the Architecture IR prompt to the GPT Image edit/reference workflow. The prompt explicitly requires the model to preserve all nodes, labels, counts, panels, and directed connections while improving visual design.

This mode is substantially safer than free generation, but it is still a generative image. The final raster must be checked against the Architecture IR.

## Validation rule

Never use an AI-rendered architecture figure as the only representation of the model without checking:

- node inventory;
- label spelling;
- repeated-block counts;
- edge direction and type;
- tensor dimensions;
- panel membership;
- inference vs training-only paths.

For publication workflows, keep an editable vector export alongside the AI render.

## API key

Set `OPENAI_API_KEY` in the environment. Do not store keys in YAML specs, source files, screenshots, commits, or example notebooks.
