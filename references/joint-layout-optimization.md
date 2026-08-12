# Joint Layout Optimization

Node placement and connector routing are treated as one coupled publication-design problem.

## Why

A collision-free node layout can still produce a poor paper figure when the subsequent router must create long detours or many crossings. Conversely, aggressively shortening edges can cause cramped stages. The joint optimizer therefore evaluates legal node-order and stage-spacing candidates using the routed graph itself.

## Invariants

The optimizer must never:

- add, remove, rename, merge, or split Architecture IR nodes;
- add/remove/reverse edges;
- change repeat counts;
- move a node to a different semantic stage;
- use overlap as a way to improve compactness.

Only paper-facing geometry may change.

## Objective

Candidate layouts are routed with the same obstacle-aware router used for final rendering. A deterministic score combines:

1. routed edge crossings — highest soft penalty;
2. illegal node/stage intersections — effectively forbidden;
3. Manhattan route length;
4. route bend count;
5. occupied bounding-box area.

This intentionally prefers a slightly larger but much cleaner Figure 1 over a compact line tangle.

The exact weights are implementation details and may evolve, but the priority order should remain stable.

## Search space

The default search is conservative and deterministic:

- reorder only nodes already sharing the same layout lane or row;
- propose barycentric ordering from connected-neighbor positions;
- test adjacent swaps around that order;
- reroute each proposal before accepting it;
- tune inter-stage gutters with a small coordinate-descent search;
- keep the best routed objective after each accepted move.

This avoids global combinatorial search while capturing the majority of crossings encountered in manuscript-scale multimodal figures.

## Fan-in and fan-out buses

High-degree connectivity may be visually compacted after geometry is selected:

- high-indegree fusion nodes may use a shared fan-in bus;
- high-outdegree representation blocks may use a shared fan-out bus;
- these are rendering abstractions only; exact edges remain in Architecture IR.

A bus must never hide which sources or targets participate.

## Scientific labels

Layout quality also depends on typography. Long hyphenated scientific compounds should wrap at semantic delimiters (`-`, `/`) before hard splitting. Do not allow a single token to overflow a module simply to preserve a fixed node width.

## Metrics

Use:

```bash
ml-arch layout-metrics architecture.paper.yaml
```

The report includes, per panel:

- pre/post objective;
- routed crossing count;
- route length;
- route bends;
- node/stage intersection counts;
- occupied area;
- evaluated candidates and accepted moves.

For complex models, prefer a lower post-optimization objective and fewer crossings, but still visually inspect the final manuscript-scale export.

## When to disable

Set:

```yaml
figure:
  joint_optimization: false
```

only when exact author-specified node order/geometry must be preserved. For normal `paper-overview` figures, joint optimization should remain enabled.
