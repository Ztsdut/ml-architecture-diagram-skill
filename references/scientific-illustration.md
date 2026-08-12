# Scientific Illustration Composer

## Purpose

A publication architecture figure should not be only a graph of boxes and arrows. When a module has a meaningful physical, geometric, statistical, or data-representation interpretation, add a small **scientific mini-illustration** inside that module or input card.

The illustration layer is paper-facing metadata. It must never change the Architecture IR topology.

## Core rule

**Reason from the model code and the user's method description first; draw second.**

The Agent/Codex is expected to decide which modules benefit from illustration. Do not wait for the user to name an icon. A useful illustration answers at least one of these questions:

- What kind of data enters this branch?
- What geometry does the proposed representation live on?
- What physical measurement or sensor does this context represent?
- What operation is scientifically distinctive (graph propagation, spectral transform, attention, coordinate decoding, uncertainty, etc.)?
- What does the predicted field/distribution look like conceptually?

## Illustration budget

For a Figure-1 overview, normally use **3–6 mini-illustrations**. Explicit user/author choices may exceed this, but automatic decoration should stay sparse.

Prioritize, in order:

1. the proposed/novel representation or operator;
2. unusual data geometry or physical observation modality;
3. important fusion/attention geometry;
4. output field or uncertainty representation;
5. only then generic sequence/image/vector inputs.

Do **not** decorate plain Linear/MLP/LayerNorm/Dropout blocks unless their visual meaning is essential to the method.

## Evidence requirement

Every non-generic scientific illustration must be justified by supplied code, configuration, or prose. Store the reason when useful:

```yaml
illustration:
  type: spectral_sphere
  composition: illustration-top
  evidence: "The module applies a low-order spherical spectral operator."
```

Never draw a satellite, molecule, physical equation, globe, radar, spectrum, or probability distribution merely because it looks plausible.

## Local composition

A node is allowed to be a **local scientific composition**, not just a label in a rectangle.

Supported composition intents:

- `illustration-left` — visual on the left, title/annotation on the right; ideal for input cards and sensor modalities;
- `illustration-right` — the inverse;
- `illustration-top` — illustration above, title/subtitle below; ideal for operators and latent representations;
- `illustration-bottom` — label first, visual below; ideal for output uncertainty/field summaries;
- `illustration-center` — visual dominates the node; use for a major proposed representation with minimal text.

The figure renderer may allocate a larger node box when the illustration is scientifically important.

## Built-in vector illustrations

The renderer currently supports these original, editable vector primitives:

- `timeseries`
- `sequence`
- `feature_grid`
- `globe_coordinates`
- `coordinate_axes`
- `fibonacci_sphere`
- `graph_network`
- `spectral_sphere`
- `attention_fan`
- `uncertainty_curve`
- `satellite_occultation`
- `radar_dish`
- `field_map`
- `sun_geomagnetic`
- `point_cloud`

These are semantic drawing primitives, not a closed list of supported research domains.

## New scientific concepts: Custom Illustration DSL

If the method needs a visual that is not in the built-in library, the Agent should **draw it itself** using the safe vector DSL instead of silently falling back to a generic box.

Coordinates are normalized to a local 0–100 canvas. Supported primitives are deliberately limited to preserve editability and portability:

- `circle`
- `ellipse`
- `rect`
- `line`
- `polyline`
- `polygon`
- `text`

Example — a custom sensor-to-volume sampling concept:

```yaml
illustration:
  type: custom_dsl
  composition: illustration-top
  evidence: "The code samples a 3-D latent field along several measurement rays."
  primitives:
    - {shape: rect, x: 58, y: 18, w: 26, h: 48, fill: input, opacity: 0.18}
    - {shape: line, x1: 10, y1: 80, x2: 65, y2: 25, stroke: stroke, dash: "3 2"}
    - {shape: line, x1: 22, y1: 82, x2: 74, y2: 40, stroke: stroke, dash: "3 2"}
    - {shape: circle, cx: 10, cy: 80, r: 4, fill: accent}
    - {shape: circle, cx: 22, cy: 82, r: 4, fill: accent}
```

The DSL describes a **symbolic scientific sketch**, not measured data. Do not fabricate realistic data curves, maps, spectra, or instrument values.

## How the Agent should create an illustration plan

During semantic review:

1. inspect the active computation path and method description;
2. identify the 3–6 modules where a mini-illustration improves scientific comprehension;
3. choose a built-in primitive when it genuinely fits;
4. otherwise author `custom_dsl` primitives;
5. select the local composition;
6. add a short `evidence` string for domain-specific choices when useful;
7. render a preview and inspect whether the visual actually clarifies the module;
8. remove illustrations that are decorative rather than explanatory.

The plan can be written directly into `semantic-review.yaml`, because `apply-review` can add paper-facing node metadata without altering topology.

Example:

```yaml
nodes:
  driver_encoder:
    illustration:
      type: timeseries
      composition: illustration-left
      evidence: "The encoder consumes a causal multihour driver history."

  latent_nodes:
    illustration:
      type: fibonacci_sphere
      composition: illustration-top
      params: {nodes: 24}
      evidence: "The latent state is defined on distributed spherical nodes."

  proposed_operator:
    illustration:
      type: custom_dsl
      composition: illustration-top
      evidence: "The proposed block alternates local neighborhood propagation and a global low-order transform."
      primitives:
        - {shape: circle, cx: 18, cy: 58, r: 5, fill: accent}
        - {shape: circle, cx: 35, cy: 30, r: 5, fill: accent}
        - {shape: circle, cx: 50, cy: 60, r: 5, fill: accent}
        - {shape: line, x1: 18, y1: 58, x2: 35, y2: 30}
        - {shape: line, x1: 35, y1: 30, x2: 50, y2: 60}
        - {shape: polyline, points: [[60,45],[68,35],[76,50],[84,37],[92,49]], stroke: input}
```

## Automatic inference

`ml-arch compile-illustrations` can conservatively infer common primitives from verified semantic labels. It is a fallback, not a replacement for Agent reasoning.

```bash
ml-arch compile-illustrations architecture.reviewed.yaml \
  --budget 6 \
  --output architecture.illustrated.yaml
```

`compile-publication` also runs the conservative illustration compiler automatically. Explicit Agent/human illustration plans always take precedence over automatic hints.

## Figure-1 framework style

For a complex method overview, scientific mini-illustrations work best with stage containers:

- Inputs — measurement/data mini-sketches inside cards;
- Encoders — mostly restrained modules, with an illustration only when the encoder's data geometry matters;
- Core representation/operator — larger mini-illustrations and nested local composition;
- Decoder/outputs — coordinate geometry, field map, or uncertainty sketch where appropriate.

This is the mechanism that allows a paper overview to communicate the method visually rather than looking like a software flowchart.

## Originality and licensing

Create original vector sketches from primitives. Do not copy ML Visuals, paper figures, logos, or third-party icon assets verbatim. External imagery should be used only when the user explicitly supplies/licences it and it materially improves the figure.

## Quality checks

Before delivery, verify:

- the illustration is supported by the model/method semantics;
- it does not imply a computation that is absent from the Architecture IR;
- it is legible at manuscript width;
- it does not collide with labels, tensor dimensions, or arrows;
- it remains vector/editable in SVG/PPTX when using deterministic rendering;
- it is symbolic rather than fabricated measured data;
- the overall illustration budget remains restrained.
