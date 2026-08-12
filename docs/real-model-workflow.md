# Real-model workflow

For non-trivial source code, use this pipeline:

```text
source code
   ↓
static parser
   ↓
exact Architecture IR
   ↓
semantic review checkpoint
   ↓
reviewed exact IR
   ↓
scientific illustration plan
   ↓
publication-view compiler
   ↓
staged/architecture-specific composition
   ↓
publication lint
   ↓
SVG / PPTX / draw.io / PNG / PDF
```

Example:

```bash
ml-arch parse-pytorch model.py --output build/model.yaml
ml-arch apply-review build/model.yaml review.yaml --output build/model.reviewed.yaml
# Optional inspection step; compile-publication also performs conservative illustration inference.
ml-arch compile-illustrations build/model.reviewed.yaml --output build/model.illustrated.yaml
ml-arch compile-publication build/model.reviewed.yaml --output build/model.paper.yaml
ml-arch lint-publication build/model.paper.yaml
ml-arch render build/model.paper.yaml --format svg pptx png
```

The raw parser YAML is a debugging artifact for complex architectures, not automatically the final figure specification.

## Illustration planning for real models

The review patch should normally include a sparse illustration plan for modules whose scientific meaning is clearer visually than textually. The agent should decide this from the source code and the method description. Do not ask the user to supply icons when the concept can be drawn symbolically.

Use built-in vector primitives when they fit; otherwise author a `custom_dsl` sketch. Keep the exact model graph unchanged. See `references/scientific-illustration.md`.


## Proposed-block detail checkpoint

After semantic review, ask the tool to nominate the main proposed block:

```bash
ml-arch detail-review architecture.reviewed.yaml --output proposed-block-review.yaml
```

Codex/Agent must inspect the selected block implementation and replace the placeholder operations in the review skeleton. Apply the patch, then run `compile-publication`. If the internal graph cannot be verified, do not create panel `(b)`.
