# Scientific Visual Grammar

Scientific Visual Grammar is the presentation layer between Architecture IR and a renderer.

Its purpose is to make a paper figure visually communicate *what kind of computation is happening* without inventing architecture.

## Invariant

The visual compiler may add:

- `metadata.architecture_family`;
- `figure.layout_preset`;
- `node.visual` metadata.

It must never change:

- node IDs;
- node labels;
- node count;
- edge source/target;
- edge type;
- repeat count;
- tensor annotations supplied by the user/parser.

## Family inference

Current families:

- `cnn`
- `transformer`
- `unet`
- `gnn`
- `moe`
- `diffusion`
- `rnn`
- `operator`
- `multimodal`
- `generic`

Inference uses semantic labels and structural cues. It is intentionally presentation-oriented, not a classifier of model scientific identity.

For custom models, override it explicitly:

```yaml
metadata:
  architecture_family: operator
```

or:

```bash
ml-arch compile-visual model.yaml --family operator
```

## Visual types

### Data and tensors

- `data_stack`
- `feature_map_stack`
- `token_strip`
- `token_matrix`
- `sequence_strip`
- `modality_card`
- `output_card`

### Transformer

- `embedding_card`
- `transformer_stack`
- `attention_heads`
- `ffn_block`
- `norm_bar`

### Graph models

- `graph_input`
- `graph_message`
- `graph_pool`

### Mixture of experts

- `router_gate`
- `expert_fan`
- `weighted_merge`

### Recurrent models

- `recurrent_cells`

### Encoder–decoder

- `unet_stage`
- `bottleneck`

### Operator learning

- `spectral_operator`
- `operator_branch`
- `operator_trunk`

### Generic scientific primitives

- `fusion_hub`
- `merge_glyph`
- `operator_glyph`
- `pooling_glyph`
- `classifier_head`
- `loss_card`
- `generic_module`

## Manual overrides

A user or agent may override any visual inference:

```yaml
- id: custom_block
  label: Local Spectral Block
  role: novel
  kind: module
  visual:
    type: spectral_operator
```

Renderers must honor explicit `visual.type` before automatic inference.

## Layout presets

Current presets include:

- `semantic_flow` — general DAG flow;
- `block_diagram` — compact repeated-block models;
- `unet` — U-shaped multiscale encoder–decoder;
- `branch_fan` — sparse routing / MoE;
- `multi_lane` — multimodal branch fusion;
- `conditioned_flow` — main model plus conditioning branches.

A preset is a composition rule, not topology.

## Renderer expectations

### SVG

Specialized glyphs should remain vector primitives and text should remain editable/selectable.

### PowerPoint

Use native PowerPoint shapes whenever practical. A complex node may consist of multiple independently editable shapes.

### draw.io

Use native diagram primitives and semantic shapes. The draw.io backend can be visually simpler than SVG/PPTX but should preserve editability.

### AI reference

The prompt may include `visual.type` as a stylistic hint. It must explicitly say that each visual glyph still represents exactly one Architecture IR node.

## What not to do

Do not infer technical operations merely because they look good. For example:

- do not add attention heads to a generic module;
- do not draw feature-map stacks for a scalar MLP input;
- do not label a spectral icon as FFT unless the model actually uses/claims Fourier operations;
- do not expand experts into independent graph nodes when the IR represents them as one repeated homogeneous module;
- do not add U-Net skip paths that are absent from the Architecture IR.

## Relationship to Scientific Illustration Composer

`visual.type` describes the architecture-aware glyph/geometry of a node. `illustration` describes an optional **scientific mini-illustration inside that node**. They are separate layers.

For example, a node can remain an `encoder_module` while carrying `illustration: {type: timeseries, composition: illustration-left}`. A proposed spectral block can use an emphasized publication module with `illustration: {type: spectral_sphere}`.

When `illustration` is present, publication renderers may prioritize that local scientific composition over the lower-level glyph. The node ID, edges, repeat count, and architecture semantics remain unchanged.
