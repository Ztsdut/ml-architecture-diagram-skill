# Overall + Proposed-Block Detail Panels

## Purpose

Many strong ML method figures use two abstraction levels:

- **(a) Overall architecture**: the scientific story, major inputs, interaction/fusion, and outputs.
- **(b) Detailed proposed block**: only the custom/repeated block whose internal operations support the paper's novelty claim.

The second panel is not an excuse to invent a plausible neural block. It must be traceable to source code, config, a structured method description, or an explicit user-supplied architecture.

## Candidate selection

The compiler scores overview nodes using:

1. `role: novel` or explicit `figure.detail_node`;
2. custom/proposed/operator/transformer/attention/graph/physics semantics;
3. repeat count;
4. graph connectivity;
5. scientific illustration / geometric significance.

Inputs, terminal outputs, losses, and ordinary prediction heads are down-weighted.

`figure.detail_node` is the deterministic override when the paper author knows which block should be expanded.

## Evidence-first detail expansion

A selected custom node may contain:

```yaml
detail:
  label: "(b)"
  title: "Detailed structure of the proposed block"
  direction: TB
  nodes: [...]
  edges: [...]
```

The nested IDs are local to the block. At compilation they are prefixed, converted to view-only nodes, assigned to a detail panel, and tagged with provenance back to the macro node.

The compiler will not fabricate detail operations for an unknown custom block. Instead it reports:

```yaml
metadata:
  detail_panel:
    status: semantic_review_required
```

Use `ml-arch detail-review` to create a review skeleton, inspect the selected implementation, fill the verified internals, then apply the review.

## Safe built-in template

A standard Transformer encoder block is the one conservative automatic exception. When the model semantics explicitly identify a Transformer encoder, the skill may show LayerNorm, self-attention, residual add, feed-forward network, and the second residual add as a standard block view. If the implementation differs, provide an explicit `detail` graph and it takes precedence.

## Layout rules

- Keep `(a)` left-to-right unless the architecture itself demands another orientation.
- Default `(b)` to top-to-bottom when the block has residual paths; this usually minimizes crossings.
- Center a narrower detail panel beneath the overall panel rather than stretching it to screen width.
- Preserve manuscript font sizes after scaling to the target width.
- Use local scientific mini-illustrations only when they clarify the operation (graph propagation, spectral transform, coordinate mapping, etc.).
- Mark the expanded macro node in panel `(a)` with the detail panel label, e.g. `(b)`.

## Topology invariants

The overview exact graph is immutable. Detail panel nodes are additional paper-view nodes; they never replace or mutate the macro node in the exact Architecture IR. This keeps parser/review truth separate from explanatory decomposition.
