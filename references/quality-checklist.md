# Quality Checklist

## Architectural fidelity

- [ ] Active forward/call path was inspected.
- [ ] All major branches are represented.
- [ ] Merge type is correct: add / concat / multiply / gate / attention / routing.
- [ ] Repeated block counts are correct.
- [ ] Shared weights are not accidentally shown as independent modules.
- [ ] Dimensions are verified or omitted.
- [ ] Training-only nodes are visually distinct.
- [ ] No aesthetic-only modules were invented.

## Scientific communication

- [ ] Inputs are grouped by meaning when useful.
- [ ] The model's key contribution is easy to locate.
- [ ] Unimportant low-level operations are compressed.
- [ ] Novel custom blocks receive enough detail.
- [ ] The output/task is explicit.
- [ ] Caption can explain the figure without restating every label.

## Visual quality

- [ ] Main flow is easy to follow.
- [ ] Edge crossings are minimized.
- [ ] Repeated modules use `×N` where appropriate.
- [ ] Color has semantic meaning.
- [ ] Grayscale remains understandable.
- [ ] Text is readable at intended publication size.
- [ ] Editable vector/source output exists.

## Export quality

- [ ] SVG text remains text where practical.
- [ ] draw.io objects remain independent/editable.
- [ ] PPTX is composed of native shapes/connectors rather than one raster image.
- [ ] PNG/PDF exports do not clip labels or arrows.
