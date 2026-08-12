# Publication Design Engine

## Goal

Turn a structurally correct Architecture IR into a manuscript figure rather than a slide-style flowchart.

The Publication Design Engine works from an immutable reviewed Architecture IR. Before layout, it may derive a smaller **publication view graph** that collapses implementation-only merge bookkeeping, compacts redundant raw-feature reuse, or hides diagnostic-only outputs. Every view transformation must retain provenance in `metadata.publication_view`. The exact reviewed IR is never overwritten, and the publication view must never invent a computation, reverse data flow, or alter repetition counts.

## Design pipeline

```text
Model code / config / prose
        ↓
Architecture IR                 exact structural truth
        ↓
Semantic review                 verified scientific labels / shapes
        ↓
Publication view                paper-facing abstraction + provenance
        ↓
Scientific Visual Grammar      what each visible node can look like
        ↓
Scientific Illustration        small domain-aware vector sketches
        ↓
Publication Design Engine      composition + hierarchy + typography
        ↓
Design Blueprint
        ↓
SVG / PPTX / draw.io / AI reference
```

## Manuscript principles

1. Design for the final printed width, normally about 85–90 mm (single column) or 175–180 mm (double column), not for a presentation slide.
2. Do not place a large figure title or promotional subtitle inside the artwork by default. The manuscript caption supplies the title.
3. Prefer compact purposeful whitespace. Whitespace should separate concepts, not make the figure look empty.
4. Use mostly white module interiors with fine outlines and small low-saturation semantic accents.
5. Give the scientifically important module the strongest visual weight. Do not make every node equal in size and style.
6. Prefer architecture-specific geometry over decorative icons.
7. Show tensor scale changes geometrically when they are scientifically meaningful and known.
8. Keep arrows thin and directional. Residual/skip paths should bypass the main trunk cleanly.
9. Repetition is shown with `×N`, a short stack, or a brace; do not duplicate dozens of blocks.
10. Panel labels such as `(a)` and `(b)` are small and functional.

## Scientific mini-illustrations and local composition

The Publication Design Engine may reserve space inside selected nodes for a scientific mini-illustration. This is intentionally different from a decorative icon. The mini-illustration should communicate data modality, geometry, a physical measurement, a proposed operator, or an output distribution/field.

For a complex Figure-1 framework, use a restrained illustration budget. A typical composition uses richer mini-illustrations in the input cards and scientifically novel central modules, while ordinary encoders and MLP heads remain simple.

The Agent should decide the illustration semantics during review and store them in the node's `illustration` mapping. The renderer then allocates local space according to `illustration-left`, `illustration-top`, `illustration-bottom`, or `illustration-center`. If no built-in scientific primitive fits, the Agent may author a safe `custom_dsl` vector sketch.

Read `references/scientific-illustration.md`.

## Architecture-specific composition

### Transformer

- Overall panel: token representation → embedding → repeated Transformer macro-block → pooling/head/output.
- Detail panel: vertical computational path with normalization, attention, add/residual, feed-forward, add/residual.
- Residual paths route along the side of the block instead of crossing the main path.
- Avoid large decorative attention icons; use compact head lanes or matrices.

### U-Net / encoder-decoder

- Use a true U-shaped composition.
- Encoder tensors shrink in spatial scale while channel depth may increase when supported by the specification.
- Decoder stages mirror the encoder.
- Skip paths travel over the U without obscuring the main path.

### Multimodal fusion

- Give each modality its own aligned lane.
- For complex Figure-1 overviews, use numbered stage containers such as inputs, representation encoders, context/fusion, and prediction/outputs.
- Use compact input cards with a small semantic data glyph plus concise scientific text; avoid free-floating decorative icons.
- Keep high-fan-in feature fusion visible when it explains the method; route dense fan-in with a shared bus rather than spaghetti arrows.
- Use different data geometry only when it reflects the modality: image tensor, token strip, feature vector, graph, field, etc.
- Lanes converge at the actual fusion operator or reviewed paper-level representation; do not invent a fusion stage.

### Neural/operator learning

- Input and output fields use field/tensor geometry.
- The operator block is the visual center when it is the methodological contribution.
- Spectral/operator glyphs must remain symbolic unless exact internal operations are present in the IR.

## Typography

- Default: Arial/Helvetica-compatible sans serif.
- Use compact labels; avoid paragraph text inside modules.
- Use smaller secondary text for tensor dimensions and subtitles.
- Keep labels close to the visual primitive they describe.
- Do not use all-caps banner subtitles such as “SCIENTIFIC VISUAL GRAMMAR” inside a manuscript figure.

## Color

Color encodes semantic role, not individual layer identity.

Recommended behavior:

- white background;
- mostly white modules;
- small colored accent bands or lightly tinted tensors;
- 3–5 low-saturation semantic accents;
- dark neutral strokes and arrows;
- grayscale distinguishability must not rely only on hue.

## AI-assisted design

For image-generation backends, use the deterministic publication SVG as a structural reference. The image model may improve composition, hierarchy, tensor rendering, and visual polish, but the result must be checked against Architecture IR before use.

The image model is a **visual art director**, not the source of architectural truth.


## Adaptive framework geometry

For staged Figure-1 frameworks, layout is **content driven**. Do not use fixed x coordinates or fixed stage widths.

The required order is:

1. measure every node after visual/illustration compilation;
2. estimate the minimum width and vertical stack height for each stage;
3. inspect graph depth inside the method/interaction stage;
4. select a local orientation (horizontal layered or vertical method chain);
5. allocate independent head/output columns when needed;
6. place stages sequentially with a minimum gutter;
7. route edges only after all boxes are fixed;
8. run geometry lint for node and stage-container intersections.

The renderer is allowed to enlarge the canvas. A fixed target aspect ratio is never a reason to overlap boxes, clip labels, or place one stage inside another. If the central stage contains three or more sequential levels, a vertical method-chain composition is usually preferred because it mirrors how scientific framework figures narrate a method.
