---
name: ml-architecture-diagram
description: Create accurate, publication-ready, editable machine-learning and deep-learning architecture diagrams from model code, configuration files, model summaries, graph exports, or written specifications. Use for neural-network figures, architecture schematics, model block diagrams, training/inference diagrams, paper figures, and editable SVG/draw.io/PPTX outputs. Recover the real computation graph first, compress unimportant implementation detail, emphasize the scientific contribution, and never invent modules for aesthetics.
---

# ML Architecture Diagram

Create architecture figures that are both **faithful to the model** and **useful to a scientific reader**.

The job has two distinct stages:

1. recover and normalize the architecture into the repository's Architecture IR;
2. render or redesign that IR into a publication-ready figure, choosing deterministic vector rendering, AI-assisted rendering, or both.

Do not jump directly from a class listing to a pretty diagram.

## 1. Determine the requested figure

Infer the user's goal from the request and supplied files.

Common modes:

- **paper-overview** — compact model overview for a manuscript;
- **paper-detail** — overall architecture plus one or more custom-block details;
- **training-pipeline** — architecture plus loss/objective/training-only paths;
- **inference-pipeline** — runtime data flow only;
- **presentation** — more explanatory spacing and labels;
- **exhaustive-debug** — more implementation detail for engineering review.

Default to `paper-overview` unless the request clearly implies another mode.

## 2. Recover the true computation graph

Use the executed inference path as the source of truth.

### General rules

Identify:

- inputs and their semantic meaning;
- preprocessing and embeddings;
- serial and parallel branches;
- repeated blocks;
- residual/skip routes;
- merge operations such as add, concatenate, gating, multiplication, routing, or cross-attention;
- recurrent, convolutional, attention, graph, spectral, state-space, operator-learning, physics-informed, diffusion, or other custom modules;
- heads and outputs;
- tensor dimensions only when reliable and useful;
- training-only components when the requested figure includes training.

Do not infer topology from class names alone. A registered module may never be executed, and the same module may be reused multiple times.

When the architecture spans multiple files, follow imports and configuration only as far as necessary to reconstruct the active path.

### Framework-specific inspection

Read `references/model-inspection.md` when framework details matter.

At minimum:

- **PyTorch**: inspect `nn.Module`, construction, `forward()`, active config, and repeated containers. Optional tracing can corroborate but not replace source inspection.
- **TensorFlow/Keras**: inspect Functional graph connectivity or subclassed `call()`; use `model.summary()` only as supporting evidence.
- **JAX/Flax/Haiku**: inspect `__call__`, scans/repetition, residual operations, and explicit transforms.
- **ONNX**: use the graph as a structural source, then semantically group low-level operators.
- **other/custom frameworks**: reconstruct data flow from the actual call path and normalize it into the same IR.

If code cannot be executed safely or dependencies are unavailable, use static inspection. Do not fabricate missing runtime facts.

## 3. Separate architecture fidelity from visual abstraction

A paper figure should not enumerate every implementation operator.

Create three possible detail levels:

- **Level 1 — system**: inputs → branches/backbone → fusion → heads → outputs;
- **Level 2 — block**: internal structure of novel or important repeated modules;
- **Level 3 — operation**: norm/linear/conv/activation/dropout/etc. only when scientifically important.

For manuscripts, prefer Level 1 plus selective Level 2.

Compress repeated structures with `×N` unless individual stages have different roles or resolutions.

Treat shape changes as annotations, not as separate nodes, unless the shape transition is conceptually important.

## 4. Build Architecture IR before rendering

For non-trivial models, first normalize the architecture using `templates/architecture_spec.yaml` and `templates/architecture.schema.json`.

The IR is intentionally framework-neutral. It represents:

- figure metadata;
- panels;
- inputs, internal nodes, and outputs;
- semantic roles;
- edges and edge types;
- repeats and tensor shapes;
- groups and annotations;
- unresolved/uncertain facts.

Use stable, human-readable node IDs. Keep labels short.

### Uncertainty handling

If an architectural fact is uncertain:

- record it in `metadata.unresolved`;
- avoid presenting it as certain in the final figure;
- omit low-value uncertain detail rather than inventing it.

## 5. Semantic review checkpoint for real models

For non-trivial source code, **do not render the raw static-parser output as the final paper figure**. The parser is a structural draft. Before publication compilation, inspect the active `forward()`/`call()` path and resolve paper-facing semantics.

This review is mandatory when any of the following is true:

- `metadata.unresolved` is non-empty;
- the model has three or more independent inputs;
- there are multiple attention/fusion/gating branches;
- a dictionary/tuple exposes several outputs;
- ablation flags change the active route;
- the raw IR contains generic `Concat`, `Add`, `Conditional`, or duplicated generic labels that obscure the method.

During semantic review:

- verify constructor/config defaults that determine the active inference path;
- replace code-centric labels with concise scientific labels only when the meaning is supported by source/config;
- add reliable tensor shapes and short subtitles;
- distinguish diagnostic outputs from manuscript-facing outputs;
- keep exact topology intact in the reviewed IR;
- use overview-only hints to suppress redundant long raw-feature reuse paths rather than deleting them from the exact graph.

Prefer an explicit review patch so the correction is auditable:

```bash
ml-arch parse-pytorch model.py --output architecture.yaml
# Agent/human inspects model.py + architecture.yaml and writes semantic-review.yaml
ml-arch apply-review architecture.yaml semantic-review.yaml \
  --output architecture.reviewed.yaml
```

`apply-review` may change labels, subtitles, shapes, semantic roles, and overview hints, but it must not add/delete nodes or edges. The reviewed IR remains the exact structural source of truth. Read `references/semantic-review.md`.

## 6. Choose the figure hierarchy and proposed-block detail

Read `references/proposed-block-detail.md` for the proposed-block detail workflow. For manuscripts, the default hierarchy should be **overall + selective block detail** whenever one custom/repeated module carries the main methodological novelty or contains roughly four or more meaningful operations.

### Automatic proposed-block selection

Rank candidate macro modules using evidence from the reviewed Architecture IR:

- explicit `role: novel` or user-provided novelty emphasis;
- custom/proposed/operator/attention/graph/physics semantics;
- repeated use (`×N`);
- connectivity/fusion importance;
- scientific mini-illustration or domain-specific geometry;
- an explicit `figure.detail_node` override, which always wins.

Do **not** automatically expand ordinary input/output/head blocks merely because they are large.

### Verify internals before drawing panel (b)

For a custom selected block, **the Agent/Codex must perform this inspection itself**: inspect its actual implementation (`forward`, `call`, `__call__`, nested module class, config) and record the verified internal operations under that macro node's `detail` mapping. Never infer internal layers from a class name alone.

Recommended Agent/Codex path:

```bash
ml-arch detail-review architecture.reviewed.yaml \
  --output proposed-block-review.yaml
```

The generated review file is a **skeleton**, not a factual architecture. The Agent/Codex should fill it automatically from the inspected code/config/method description; do not ask the user to enumerate internals that are already available in the source. Replace placeholders only with verified operations, then apply it:

```bash
ml-arch apply-review architecture.reviewed.yaml proposed-block-review.yaml \
  --output architecture.detail-reviewed.yaml
```

Supported detail structure:

```yaml
nodes:
  proposed_operator:
    detail:
      label: "(b)"
      title: "Detailed structure of the proposed operator block"
      direction: TB
      nodes:
        - {id: input, label: "Block input", role: input, kind: data}
        - {id: local, label: "Local graph propagation", role: novel, kind: module}
        - {id: spectral, label: "Low-order spectral transform", role: novel, kind: module}
        - {id: add, label: "+", role: fusion, kind: merge}
        - {id: output, label: "Block output", role: output, kind: output}
      edges:
        - {from: input, to: local, type: main}
        - {from: local, to: spectral, type: main}
        - {from: spectral, to: add, type: main}
        - {from: input, to: add, type: residual}
        - {from: add, to: output, type: main}
```

### Detail-panel invariants

- Panel `(a)` keeps the reviewed overview topology exactly.
- Panel `(b)` contains provenance-tagged **view-only** nodes linked to the selected macro block.
- Residual, concat/add, gate, and repeated-block semantics must match the implementation order.
- The selected macro node in `(a)` receives a small `(b)` reference marker.
- Panel directions may differ: overview is commonly LR; the block detail may be TB.
- If verified internals are unavailable, keep a single-panel figure and record `semantic_review_required` rather than inventing a detail graph.
- If the user already supplied a deliberate multi-panel figure, preserve it unless `figure.auto_detail: force` is explicit.

A conservative Transformer detail template may be used when the block semantics are unambiguous. Do not generalize that exception to unknown custom modules.

## 7. Compile Scientific Visual Grammar

Read `references/visual-grammar.md` and `references/style-guide.md`.

Do not force every architecture into generic rounded rectangles. After Architecture IR is stable, choose architecture-appropriate visual primitives.

Current families include CNN, Transformer, U-Net/encoder-decoder, GNN, mixture-of-experts, diffusion, recurrent models, operator learning, multimodal models, and generic DAGs.

Examples of visual primitives:

- spatial/CNN representations → `feature_map_stack`;
- token sequences → `token_strip` or `token_matrix`;
- Transformer stacks → `transformer_stack`;
- self/cross attention → `attention_heads` only when attention actually exists;
- feed-forward sublayers → `ffn_block`;
- graph data/message passing/readout → `graph_input`, `graph_message`, `graph_pool`;
- MoE → `router_gate`, `expert_fan`, `weighted_merge`;
- recurrent layers → `recurrent_cells`;
- U-Net stages → `unet_stage` with a dedicated U-shaped layout;
- spectral/operator-learning modules → `spectral_operator` only when supported by the model;
- multimodal fusion → `modality_card` + `fusion_hub`.

**Invariant:** visual compilation may add presentation metadata but must not add/delete/reorder nodes, mutate edges, alter repeat counts, rename architecture modules, or invent tensor dimensions.

When the repository CLI is available, inspect automatic inference with:

```bash
ml-arch compile-visual architecture.yaml --output architecture.visual.yaml
```

If the inferred family or glyph is inappropriate, edit `metadata.architecture_family`, `figure.layout_preset`, or a node's explicit `visual.type`. Prefer a small manual correction over forcing an incorrect automatic abstraction.

Default style principles remain:

- light neutral background;
- restrained semantic palette, normally 3–5 role colors;
- consistent color by function, not by individual layer;
- solid arrows for main inference flow;
- visually separate residual paths;
- dashed arrows for conditioning or auxiliary information;
- dotted or clearly labeled paths for training-only connections;
- minimal decorative icons;
- no gradients or heavy shadows by default.

The visual language may be inspired by common ML-paper conventions, but the drawing must be original. Do not copy third-party figure assets verbatim.

## 8. Compose Scientific Mini-Illustrations

Read `references/scientific-illustration.md`. A paper architecture figure is allowed to contain small, original scientific illustrations **inside selected modules** when they improve comprehension of the data, geometry, physical measurement, operator, or output.

This step is not optional decoration. For a complex `paper-overview`, the Agent/Codex should inspect the model code **and** the supplied method description and decide which 3–6 modules deserve a mini-illustration. Typical high-value candidates include unusual observation modalities, spherical/graph/geometric latent spaces, spectral or graph operators, coordinate decoders, physical drivers, spatial fields, and uncertainty outputs. Plain Linear/MLP/Norm blocks normally remain text-only.

The Agent should create the illustration plan itself rather than waiting for the user to name icons. Prefer a built-in editable vector primitive when semantically correct, for example:

- causal/temporal history → `timeseries`;
- gridded/spatial data → `feature_grid`;
- spherical coordinates → `globe_coordinates`;
- distributed spherical latent nodes → `fibonacci_sphere`;
- local message passing → `graph_network`;
- spherical/Fourier operator → `spectral_sphere`;
- self/cross attention → `attention_fan`;
- probabilistic output → `uncertainty_curve`;
- radio-occultation geometry → `satellite_occultation`;
- sounding/radar observation → `radar_dish`;
- spatial scalar field → `field_map`.

If the scientific concept is not in the built-in library, **draw a new symbolic illustration** with the safe `custom_dsl` primitives (`circle`, `ellipse`, `rect`, `line`, `polyline`, `polygon`, `text`) using normalized local coordinates. This is the general mechanism for new domains such as biomedical imaging, molecular graphs, remote sensing geometry, sensor networks, PDE operators, or custom physics. Do not silently fall back to a generic box simply because a ready-made icon does not exist.

Add the plan to selected nodes during semantic review:

```yaml
nodes:
  latent_state:
    illustration:
      type: fibonacci_sphere
      composition: illustration-top
      params: {nodes: 24}
      evidence: "The method defines the latent state on distributed spherical nodes."

  proposed_operator:
    illustration:
      type: custom_dsl
      composition: illustration-top
      evidence: "The block combines neighborhood propagation with a global transform."
      primitives:
        - {shape: circle, cx: 20, cy: 55, r: 5, fill: accent}
        - {shape: circle, cx: 38, cy: 30, r: 5, fill: accent}
        - {shape: line, x1: 20, y1: 55, x2: 38, y2: 30}
        - {shape: polyline, points: [[58,45],[67,35],[76,50],[85,37],[93,48]], stroke: input}
```

Use local composition intentionally:

- `illustration-left` for input/sensor cards;
- `illustration-top` for operators/latent representations;
- `illustration-bottom` for field/uncertainty outputs;
- `illustration-center` when a scientifically novel representation should dominate the module.

Illustrations are symbolic explanations, not fabricated observations. Never invent realistic time series, maps, spectra, physical values, or instrument geometry unsupported by the supplied material.

When the CLI is available, inspect or auto-fill a conservative plan with:

```bash
ml-arch compile-illustrations architecture.reviewed.yaml \
  --budget 6 \
  --output architecture.illustrated.yaml
```

`compile-publication` also performs conservative illustration inference automatically; explicit Agent/human plans always override automatic hints.

**Illustration invariant:** mini-illustrations may change only local visual composition. They must not add architecture nodes, remove edges, alter block counts, or imply a computation not present in the reviewed IR.

## 9. Compile Publication Design

Read `references/publication-design.md`. For paper targets, do not stop after assigning glyphs. Compile a manuscript composition layer that decides visual hierarchy, panel balance, tensor scale geometry, emphasis, and typography. The exact reviewed Architecture IR stays untouched; the compiler may derive a smaller **publication view graph** for Figure-1 communication.

When the CLI is available, use:

```bash
ml-arch compile-publication architecture.yaml --output architecture.publication.yaml
```

Default paper behavior:

- no large in-figure title or banner subtitle;
- mostly white modules with fine outlines and small semantic accents;
- compact purposeful whitespace rather than slide-scale spacing;
- architecture-specific composition instead of generic DAG placement;
- visual emphasis concentrated on the scientifically important module;
- tensor/field/token/vector geometry used only when supported by the model semantics;
- detailed Transformer blocks use a vertical computational path with side residuals;
- U-Net uses scale-aware feature tensors in a true U-shaped composition;
- multimodal models use aligned modality lanes that meet at the real fusion point;
- operator-learning models place the operator block at the visual center without inventing unspecified internals.
- staged framework figures use **content-driven adaptive geometry**: compute node sizes first, then stage widths/heights, then node positions and edge routes. Never start from fixed stage coordinates.
- if a central method stage has several sequential levels or cannot fit horizontally at manuscript scale, reflow it vertically inside the stage rather than shrinking or overlapping modules.
- decoder/head/output stages reserve independent columns when their combined minimum widths exceed a single-column stage.
- the canvas may grow when the model is complex; preserving non-overlap and manuscript readability has priority over a predetermined aspect ratio.
- route edges **after** node/stage geometry is final. Use obstacle-aware orthogonal routing rather than renderer-specific freehand curves.
- a connector must never pass through an unrelated node. For conditioning/training/auxiliary paths, unrelated stage containers are soft obstacles; prefer stage gutters or outer buses.
- inside one stage, choose ports from local geometry (e.g. bottom→top for vertically stacked attention/gate/fusion modules) instead of forcing every edge to read left→right.
- when a fusion/merge node has three or more incoming paper-facing edges, prefer a shared **fan-in bus** with one final arrowhead over a bundle of independent arrowheads. For a representation block with three or more outgoing paper-facing edges, a shared **fan-out bus** may be used when it reduces clutter. These are visual routing abstractions only; all exact IR edges remain present.
- for `publication_framework` figures, jointly optimize legal node order and stage spacing against the **routed** graph rather than accepting the first collision-free layout. Read `references/joint-layout-optimization.md`. Reorder only within existing lanes/rows; never change semantic stage membership or topology.
- preserve the lower routed objective when choosing between candidate compositions: prioritize zero hard intersections, fewer crossings, shorter routes, fewer bends, then compact occupied area.
- route growth may enlarge the canvas. Never clip a legal detour merely to preserve a fixed aspect ratio.

**Publication-view invariant:** the exact reviewed IR is immutable. The paper-facing view may collapse implementation-only merge nodes, compact redundant raw-feature reuse paths, or hide diagnostic-only outputs **only with provenance recorded in `metadata.publication_view`**. It must never invent a computation, change repeat counts, reverse data flow, or hide a scientifically important branch without an explicit review decision.

Before final paper export, run:

```bash
ml-arch lint-publication architecture.publication.yaml
```

If it reports slide-like titles, extreme aspect ratio, low canvas density, panel imbalance, long labels, **node/stage overlap**, **edge-through-node**, **edge-through-stage**, or excessive single-panel complexity, revise the publication blueprint and render again. Use `ml-arch layout-metrics architecture.publication.yaml` on dense figures to inspect joint-optimization gains and remaining crossing pressure. Geometry and routing intersections are hard failures: expand/reflow the layout or reroute through a gutter in code; never accept them to preserve a fixed canvas. Remaining edge crossings are a softer composition metric and should be reduced when practical. Do not treat the first automatic layout as final merely because it is technically valid.

## 10. Typography and manuscript fit

Default to a portable sans-serif stack such as Arial/Helvetica/Liberation Sans.

Design for final printed size, not only for a full-screen preview.

For scientific papers:

- keep labels concise;
- prefer readable text over showing every layer;
- use consistent capitalization;
- place obscure abbreviations in the caption or legend;
- preserve legibility in grayscale by combining color with labels, outline, or geometry.

## 11. Tensor dimensions

Show dimensions only when they explain representation changes.

Useful examples:

- `B × T × F`
- `B × C × H × W`
- `d_model = 512`
- `64 → 128 → 256`

Do not annotate every arrow.

Never invent dimensions. If inferred from config, verify the active configuration.

## 12. Special architecture families

Use architecture-appropriate visual abstractions rather than forcing every model into the same boxes.

### CNN / vision backbones

Show stage-level resolution/channel progression when important. Avoid drawing individual feature-map cubes unless spatial scale is central to the method.

### RNN / temporal models

Show recurrence or sequence processing symbolically. Do not draw dozens of unrolled timesteps unless recurrence itself is being explained.

### Transformer / attention models

Distinguish self-attention, cross-attention, feed-forward, residual, normalization, and repeated blocks only to the level needed by the paper.

### Graph neural networks

Separate graph construction/input, message passing/aggregation, pooling/readout, and task head. A graph icon is optional, not mandatory.

### U-Net / encoder-decoder

Emphasize scale hierarchy and skip links. Keep encoder and decoder stages aligned.

### Diffusion / generative models

Separate denoiser/backbone architecture from the iterative sampling/training process. If both matter, use separate panels.

### Physics-informed / hybrid models

Clearly distinguish learned components, deterministic operators, constraints, and physical/context inputs. Do not visually imply that a deterministic equation is trainable.

### Mixture-of-experts / routing

Show the router/gate, experts, sparse/dense selection, and merge operation. Compress homogeneous expert sets with `×N` where appropriate.

## 13. Rendering strategy

Use one of three rendering modes based on the user's goal.

### A. Deterministic editable rendering — source of truth

Use SVG/draw.io/PPTX when architecture fidelity, exact labels, or editability matters. This is the default for scientific delivery.

### B. AI concept rendering — visual exploration

When the user explicitly wants a more sophisticated, less schematic, or more visually expressive figure and an image-generation capability is available, the Architecture IR may be converted into a constrained image-generation prompt. Use this for visual exploration, not as unquestioned architectural truth.

The prompt must enumerate exact nodes, edges, repetition counts, tensor annotations, panels, and edge semantics, and explicitly prohibit invented or deleted modules.

### C. Reference-guided AI rendering — recommended rich mode

For a rich paper-quality raster figure, first render the Architecture IR deterministically, then use that exact diagram as the reference image in an image-edit/reference workflow. Ask the image model to improve only visual hierarchy, grouping, dimensional cues, restrained iconography, and module styling while preserving topology.

When the OpenAI API backend is used, `gpt-image-2` is the default model in this repository. Read `references/ai-rendering.md`.

AI-generated architecture figures are not guaranteed to preserve text or topology perfectly. Before delivery, compare the AI render against the Architecture IR. If fidelity is uncertain, deliver the editable deterministic figure as the canonical version and label the AI render as a visual concept/polish.

## 14. Rendering and deliverables

When file creation tools are available, prefer an editable source plus preview/export.

Recommended order:

1. `.svg` — portable vector manuscript source;
2. `.drawio` — maximum diagram editability;
3. `.pptx` — editable PowerPoint shapes for manual polishing;
4. `.png` — preview;
5. `.pdf` — manuscript export.

This repository includes a deterministic renderer for the Architecture IR. Use it when available:

```bash
python scripts/render_architecture.py examples/transformer_encoder.yaml --format svg drawio pptx png pdf
```

For paper figures, compile and inspect the publication blueprint before final rendering:

```bash
ml-arch compile-publication examples/transformer_encoder.yaml --output build/transformer.publication.yaml
ml-arch lint-publication build/transformer.publication.yaml
```

Use `compile-visual` only when you need lower-level control of family/glyph inference.

For a real complex model, the default Codex/Agent workflow is:

```bash
ml-arch parse-pytorch model.py --output architecture.yaml
ml-arch apply-review architecture.yaml semantic-review.yaml --output architecture.reviewed.yaml
ml-arch compile-publication architecture.reviewed.yaml --output architecture.paper.yaml
ml-arch lint-publication architecture.paper.yaml
ml-arch render architecture.paper.yaml --format svg pptx png
```

Never skip the semantic-review checkpoint merely because the parser produced valid YAML.

For optional GPT Image rendering when the API key and dependency are available:

```bash
ml-arch ai-render examples/transformer_encoder.yaml --mode reference --model gpt-image-2 --quality medium --size 2048x1152
```

If a host environment has better native diagram tooling, it may render the same IR independently, but it should preserve the semantic rules and fidelity checks in this skill.

## 15. Validation before delivery

Check all of the following:

- every major drawn path exists in the implementation/specification;
- arrow direction matches execution;
- add/concat/gate/multiply are not conflated;
- repeated-block counts are correct;
- tensor dimensions are consistent with active config;
- training-only operations are not shown as inference modules unless requested;
- losses are separated from inference architecture unless a training view is requested;
- shared/reused weights are not falsely drawn as independent learned modules when that distinction matters;
- the claimed novel component is visible but not exaggerated;
- no aesthetic-only module has been invented;
- if an AI render was used, its node inventory, labels, block counts, and edge topology have been checked against the IR;
- the figure remains understandable at its intended size.

## 16. Failure modes

Do not:

- draw a generic AI infographic disconnected from the real model;
- use a module list as if it were the execution graph;
- show every low-level op in a paper overview;
- introduce many unrelated colors;
- overuse 3-D tensor cubes or neuron-by-neuron illustrations;
- hide residual/branch logic behind decorative connectors;
- rasterize the whole figure and call the PPTX editable;
- silently guess tensor shapes or block counts;
- copy a third-party visual asset when an original equivalent can be created;
- treat an AI-generated raster as structurally authoritative without checking it against the IR.

## 17. Output summary

When delivering a figure, briefly state:

- which model path was represented;
- what detail was intentionally compressed;
- which outputs are editable;
- any unresolved architectural uncertainty.

Do not burden the user with low-level rendering details unless they ask.
