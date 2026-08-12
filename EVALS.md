# Skill Evaluation

The most important failure mode is not ugly rendering; it is a **confident but incorrect architecture**.

Use `evals/cases.yaml` as a starter benchmark. Each case contains a user-style prompt and the structural facts that a correct response should preserve.

## Suggested evaluation dimensions

Score each generated result on a 0–2 scale for each dimension:

1. **Topology fidelity** — branches, merge types, skips, and execution direction are correct.
2. **Repeat/shared-weight fidelity** — repeated blocks and reused modules are represented correctly.
3. **Abstraction quality** — unimportant implementation ops are compressed without hiding the method.
4. **Uncertainty discipline** — unknown facts are omitted or marked unresolved rather than invented.
5. **Visual hierarchy** — the main information flow and novel component are easy to find.
6. **Scientific illustration fidelity** — mini-illustrations are evidence-backed, sparse, explanatory, and do not imply absent computation or fabricated observations.
7. **Editability** — requested vector/native-shape output is genuinely editable.

A release candidate should be tested on at least one serial model, one branching model, one residual model, one shared-weight model, one training graph, and one architecture with dynamic routing.


## Proposed-block detail

- A single-panel Transformer with an explicit Transformer macro must compile into `(a)` overall + `(b)` detail using the safe built-in template.
- A custom novel block with `node.detail` must compile the verified nested graph with provenance.
- A custom novel block without verified internals must return `semantic_review_required` rather than hallucinating a detail panel.
- The overview node/edge topology must remain identical before and after detail compilation.
- SVG/PPTX must show a compact detail reference marker on the selected macro node.
