# Semantic Review for Real Model Code

Static parsing is a structural aid, not a substitute for reading the active model path. Complex paper figures should pass through a short semantic review before publication compilation.

## Why the checkpoint exists

A parser can correctly discover calls and still produce a poor paper figure. Typical real-model failure modes include heterogeneous `Sequential` containers mistaken for repeated blocks, chained tensor methods that obscure the producer, constructor ablation flags that create fake branches, tuple/dict outputs that look generic, and attention modules that cause the whole network to be mislabeled as a Transformer.

The semantic review separates these concerns:

- **exact IR** answers “what computation is executed?”;
- **review patch** answers “what are these tensors/modules scientifically called?”;
- **publication view** answers “which exact implementation details deserve their own visible object in Figure 1?”;
- **renderer** answers “how should that view be composed?”

## Required review for complex models

Review the source when `metadata.unresolved` is non-empty or when the model contains multiple inputs, branches, attention/fusion/gates, several heads, or protocol/ablation switches.

Check the active constructor/config defaults and actual `forward()`/`call()` route. Verify tensor dimensions only when directly supported by code/config. Shorten labels for manuscript reading. Identify diagnostic outputs such as returned attention maps. Do not invent a semantic module merely to simplify the drawing.

## Patch format

A patch is deliberately smaller than the IR:

```yaml
figure:
  stage_labels:
    inputs: "1) Inputs"
    encoders: "2) Representation encoders"
    interaction: "3) Cross-context fusion"
    outputs: "4) Prediction"

nodes:
  map_encoder:
    subtitle: "3-D conv + recurrent aggregation"
    shape: "spatial tokens + global state"
  cross_attention:
    subtitle: "Q: future · K,V: spatial"

edges:
  - from: diagnostic_source
    to: diagnostic_output
    set:
      overview: hide
      overview_note: "diagnostic output; retained in exact IR"
```

Apply it with:

```bash
ml-arch apply-review architecture.yaml semantic-review.yaml \
  --output architecture.reviewed.yaml
```

`apply-review` rejects unknown node IDs or unmatched edges. It does not add/delete nodes or edges. The reviewed file records the applied overrides under `metadata.semantic_review`.

## Publication-view compression

After review, `compile-publication` may create a smaller view graph. Conservative automatic rules include collapsing low-value generic `Concat`/`Add` bookkeeping, keeping high-fan-in feature fusion before prediction heads when it improves explanation, compacting redundant long raw-feature reuse paths into node annotations, and hiding diagnostic-only outputs by default.

Every such change is recorded in `metadata.publication_view`, including collapsed nodes, compacted raw edges, explicitly overview-hidden edges, and diagnostic outputs. The reviewed exact IR remains available as the structural source of truth.

## Agent/Codex behavior

When using this repository as a Skill, the agent should not immediately render a complex parser result. It should read parser notes, inspect the relevant source blocks, write/apply a semantic review patch, compile the publication view, lint it, and only then render. If a semantic fact cannot be verified, leave it unresolved rather than guessing.

## Scientific mini-illustrations

For `paper-overview` and `paper-detail`, semantic review should also decide whether scientifically meaningful nodes need a mini-illustration. This is **paper-facing metadata only** and does not alter topology.

Add an `illustration` mapping to selected nodes, for example:

```yaml
nodes:
  spatial_context:
    illustration:
      type: feature_grid
      composition: illustration-left
      evidence: "The branch consumes a gridded spatial context sequence."
  proposed_graph_operator:
    illustration:
      type: graph_network
      composition: illustration-top
      evidence: "The forward path performs neighborhood graph propagation."
```

If no built-in primitive describes the scientific concept, author a `custom_dsl` illustration from normalized vector primitives. Read `references/scientific-illustration.md`.

Do not add scientific objects from aesthetic guesswork. The code, configuration, or user description must support the interpretation.
