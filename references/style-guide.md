# Publication Architecture Diagram Style Guide

## Goal

A reader should understand the model's main information flow and methodological contribution within roughly 10–15 seconds.

## Composition

- Favor left-to-right reading for ordinary architectures.
- Use top-to-bottom only when the architecture or page geometry benefits from it.
- Keep the principal inference trunk visually dominant.
- Put auxiliary/context/conditioning branches on separate lanes.
- Align modules to a grid.
- Keep connector crossings close to zero.
- Use whitespace instead of decorative separators.

## Visual grammar

| Semantic role | Preferred primitive |
|---|---|
| Input/data | thin card or stacked card |
| Preprocessing | compact neutral rounded rectangle |
| Representation/encoder | rounded rectangle |
| Backbone/repeated block | rounded rectangle + `×N` |
| Proposed/novel module | emphasized rounded rectangle |
| Deterministic operator | compact rectangle or operator node |
| Add | small circle with `+` |
| Concatenate | compact node labeled `Concat` |
| Multiply | small circle with `×` |
| Gate/router | diamond or compact node |
| Prediction head | rounded rectangle |
| Output | endpoint card |
| Loss | compact training-only node |
| Residual | bypass connector into the corresponding merge |

## Semantic roles and color

Use color for function, not for individual layers.

Recommended role families:

1. input/data — cool neutral / blue;
2. representation/backbone — cyan/teal;
3. novel/custom block — warm orange/gold;
4. auxiliary/context/conditioning — violet;
5. fusion/head/output — green;
6. training-only — muted rose/red, used sparingly.

A custom theme may replace the palette, but semantic roles should stay consistent.

## Edge language

- `main`: solid, visually strongest.
- `residual`: solid but thinner; route outside the main modules.
- `auxiliary` / `conditioning`: dashed.
- `training`: dotted or dashed and clearly separated from inference.

Labels should describe operations or conditions, not repeat the names of adjacent nodes.

## Hierarchy of detail

- Level 1: major modules and semantic data groups.
- Level 2: repeat counts and important representation dimensions.
- Level 3: internal operations of a custom block.

Do not show Level 3 throughout an entire paper overview.

## Architecture-specific conventions

### CNN

Prefer stage-level channel/resolution annotations to stacks of 3-D cubes unless spatial scale itself is the contribution.

### Transformer

Use one representative encoder/decoder block with `×N` when blocks are homogeneous. Show attention type explicitly when self- vs cross-attention matters.

### RNN

Use a recurrence marker or compact unrolling. Avoid long timestep chains.

### GNN

Separate graph input/construction, message passing, pooling/readout, and head.

### U-Net / encoder-decoder

Align matching resolutions; make skip links easy to trace.

### MoE

Show the router before experts and the merge after selection; compress homogeneous experts.

### Physics-informed / hybrid

Differentiate trainable modules from deterministic physics/operators by role, label, and/or outline.

## Typography

- Use a portable sans-serif font stack.
- Keep block labels to one or two short lines.
- Use sentence case or consistent title case.
- Do not use tiny text to force excessive detail into one panel.
- Use mathematical notation only when it adds precision.

## Grayscale and accessibility

A figure should remain understandable when printed in grayscale. Do not rely on color alone; preserve distinctions through labels, line styles, and geometry.

## Manuscript sizing

Design at the final intended width. Typical targets are approximately one-column (85–90 mm) or two-column (175–180 mm), but the journal template is authoritative.

## Icons

Icons are optional. Use them only when they clarify semantic input categories or real-world entities. Prefer simple original vector pictograms. Avoid decorative icon overload.
